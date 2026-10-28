/* Compares the time per step of the discrete contact solvers of
MultibodyPlant (DiscreteContactSolver::kSap and kIcf) on contact-rich scenes:
objects that fall into a bin. Real-time pacing is off. */

#include <memory>
#include <string>
#include <utility>

#include <benchmark/benchmark.h>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/tools/performance/fixture_common.h"
#include "drake/tools/performance/fixture_memory.h"

namespace drake {
namespace multibody {
namespace {

using Eigen::Vector3d;
using geometry::Box;
using geometry::ProximityProperties;
using geometry::Sphere;
using math::RigidTransformd;
using math::RollPitchYawd;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

constexpr double kTimeStep = 0.01;
// The objects fall and settle during this time, before the measurement.
constexpr double kWarmUpTime = 0.5;

// The first benchmark argument.
enum Scene {
  // Point contact between rigid objects.
  kPointContactBin = 0,
  // Hydroelastic contact between compliant objects and a rigid bin.
  kHydroelasticBin = 1,
};

// The second benchmark argument.
enum Solver {
  kSap = 0,
  kIcf = 1,
};

// Adds a bin and 3 layers of 3x3 objects, alternately boxes and spheres.
void PopulateBin(Scene scene, MultibodyPlant<double>* plant) {
  ProximityProperties rigid;
  geometry::AddContactMaterial(/* dissipation = */ 5.0,
                               /* point_stiffness = */ 1.0e5,
                               CoulombFriction<double>(0.5, 0.5), &rigid);
  ProximityProperties bin_props = rigid;
  ProximityProperties sphere_props = rigid;
  ProximityProperties box_props = rigid;
  if (scene == kHydroelasticBin) {
    plant->set_contact_model(ContactModel::kHydroelasticWithFallback);
    geometry::AddRigidHydroelasticProperties(&bin_props);
    geometry::AddCompliantHydroelasticProperties(
        /* resolution_hint = */ 0.02, /* hydroelastic_modulus = */ 1.0e6,
        &sphere_props);
    geometry::AddCompliantHydroelasticProperties(
        /* resolution_hint = */ 0.03, /* hydroelastic_modulus = */ 1.0e6,
        &box_props);
  } else {
    plant->set_contact_model(ContactModel::kPoint);
  }

  // The bin: a floor and four walls, welded to the world.
  const RigidBody<double>& world = plant->world_body();
  const double half_width = 0.2;
  const double wall_height = 0.3;
  plant->RegisterCollisionGeometry(world,
                                   RigidTransformd(Vector3d(0, 0, -0.05)),
                                   Box(0.6, 0.6, 0.1), "floor", bin_props);
  for (int i = 0; i < 4; ++i) {
    const double angle = i * M_PI / 2;
    const RigidTransformd X_WWall(
        RollPitchYawd(0, 0, angle),
        Vector3d(std::cos(angle) * (half_width + 0.025),
                 std::sin(angle) * (half_width + 0.025), wall_height / 2));
    plant->RegisterCollisionGeometry(
        world, X_WWall, Box(0.05, 2 * half_width + 0.1, wall_height),
        "wall" + std::to_string(i), bin_props);
  }

  // The objects.
  for (int k = 0; k < 27; ++k) {
    const std::string name = "object" + std::to_string(k);
    if (k % 2 == 0) {
      const RigidBody<double>& body = plant->AddRigidBody(
          name, SpatialInertia<double>::SolidCubeWithMass(0.1, 0.06));
      plant->RegisterCollisionGeometry(body, RigidTransformd(),
                                       Box(0.06, 0.06, 0.06), name, box_props);
    } else {
      const RigidBody<double>& body = plant->AddRigidBody(
          name, SpatialInertia<double>::SolidSphereWithMass(0.1, 0.03));
      plant->RegisterCollisionGeometry(body, RigidTransformd(), Sphere(0.03),
                                       name, sphere_props);
    }
  }
}

// Sets the objects in 3 layers of 3x3, with small offsets and rotations.
void SetInitialState(const MultibodyPlant<double>& plant,
                     Context<double>* context) {
  for (int k = 0; k < 27; ++k) {
    const int i = k % 3;
    const int j = (k / 3) % 3;
    const int layer = k / 9;
    const RigidTransformd X_WB(RollPitchYawd(0.1 * k, 0.2 * k, 0.3 * k),
                               Vector3d(0.11 * (i - 1) + 0.005 * layer,
                                        0.11 * (j - 1), 0.05 + 0.1 * layer));
    plant.SetFreeBodyPose(
        context, plant.GetBodyByName("object" + std::to_string(k)), X_WB);
  }
}

class DiscreteContactSolvers : public benchmark::Fixture {
 public:
  DiscreteContactSolvers() { tools::performance::AddMinMaxStatistics(this); }

  // NOLINTNEXTLINE(runtime/references)
  void SetUp(benchmark::State& state) override {
    const Scene scene = static_cast<Scene>(state.range(0));
    const Solver solver = static_cast<Solver>(state.range(1));
    DiagramBuilder<double> builder;
    auto [plant, scene_graph] =
        AddMultibodyPlantSceneGraph(&builder, kTimeStep);
    if (solver == kIcf) {
      plant.set_discrete_contact_solver(DiscreteContactSolver::kIcf);
    }
    PopulateBin(scene, &plant);
    plant.Finalize();
    plant_ = &plant;
    diagram_ = builder.Build();
    auto context = diagram_->CreateDefaultContext();
    SetInitialState(plant, &plant.GetMyMutableContextFromRoot(context.get()));
    simulator_ =
        std::make_unique<Simulator<double>>(*diagram_, std::move(context));
    simulator_->set_target_realtime_rate(0.0);
    simulator_->Initialize();
    // Advance to the middle of a step, thus the round-off error of the event
    // times has no effect on the number of steps.
    num_steps_ = static_cast<int>(kWarmUpTime / kTimeStep);
    simulator_->AdvanceTo((num_steps_ - 0.5) * kTimeStep);
    tools::performance::TareMemoryManager();
  }

  // NOLINTNEXTLINE(runtime/references)
  void TearDown(benchmark::State& state) override {
    const auto& plant_context =
        plant_->GetMyContextFromRoot(simulator_->get_context());
    state.counters["point_contacts"] =
        plant_->get_contact_results_output_port()
            .Eval<ContactResults<double>>(plant_context)
            .num_point_pair_contacts();
    state.counters["hydro_contacts"] =
        plant_->get_contact_results_output_port()
            .Eval<ContactResults<double>>(plant_context)
            .num_hydroelastic_contacts();
    simulator_.reset();
    diagram_.reset();
  }

 protected:
  // Advances the simulation by one time step.
  void Step() {
    ++num_steps_;
    simulator_->AdvanceTo((num_steps_ - 0.5) * kTimeStep);
  }

  std::unique_ptr<Diagram<double>> diagram_;
  const MultibodyPlant<double>* plant_{};
  std::unique_ptr<Simulator<double>> simulator_;
  int num_steps_{};
};

// NOLINTNEXTLINE(runtime/references)
BENCHMARK_DEFINE_F(DiscreteContactSolvers, Step)(benchmark::State& state) {
  for (auto _ : state) {
    Step();
  }
}
BENCHMARK_REGISTER_F(DiscreteContactSolvers, Step)
    ->Unit(benchmark::kMicrosecond)
    ->ArgNames({"scene", "icf"})
    ->Args({kPointContactBin, kSap})
    ->Args({kPointContactBin, kIcf})
    ->Args({kHydroelasticBin, kSap})
    ->Args({kHydroelasticBin, kIcf})
    // Each iteration is one step. With 300 steps per run, each run simulates
    // from 0.5 s to 3.5 s.
    ->Iterations(300);

}  // namespace
}  // namespace multibody
}  // namespace drake
