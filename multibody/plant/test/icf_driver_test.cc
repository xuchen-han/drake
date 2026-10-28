#include "drake/multibody/plant/icf_driver.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/cenic/cenic_integrator.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/test/compliant_contact_manager_tester.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {

// Friend class used to provide access to the discrete update manager.
class MultibodyPlantTester {
 public:
  // @pre The plant is discrete and finalized.
  static const internal::CompliantContactManager<double>& manager(
      const MultibodyPlant<double>& plant) {
    const auto* manager =
        dynamic_cast<const internal::CompliantContactManager<double>*>(
            plant.discrete_update_manager_.get());
    DRAKE_DEMAND(manager != nullptr);
    return *manager;
  }
};

namespace internal {
namespace {

using contact_solvers::internal::ContactSolverResults;
using Eigen::Vector3d;
using Eigen::VectorXd;
using geometry::Box;
using geometry::ProximityProperties;
using geometry::Sphere;
using math::RigidTransformd;
using math::RotationMatrixd;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;

constexpr double kTimeStep = 0.01;

// The scenes of the tests. Each scene covers a set of ICF features.
enum class Scene {
  // Two free boxes on the ground, with point contact. One box is registered
  // before the ground, and the other after it, thus the ground is body A of
  // one contact pair and body B of the other.
  kPointContact,
  // A compliant sphere and a compliant box on a rigid ground, with
  // hydroelastic contact.
  kHydroelastic,
  // A compliant sphere and two rigid boxes on a rigid hydroelastic ground, with
  // hydroelastic contact with fallback to point contact.
  kMixedContact,
  // An arm with three revolute joints: joint limits, joint damping, reflected
  // inertia, PD controlled actuators with effort limits, feed-forward
  // actuation and applied generalized forces.
  kArm,
  // The same arm, with its second joint locked.
  kArmWithLockedJoint,
  // Two free boxes that a weld constraint connects, which fall on the ground.
  kWeld,
};

std::string SceneName(Scene scene) {
  switch (scene) {
    case Scene::kPointContact:
      return "PointContact";
    case Scene::kHydroelastic:
      return "Hydroelastic";
    case Scene::kMixedContact:
      return "MixedContact";
    case Scene::kArm:
      return "Arm";
    case Scene::kArmWithLockedJoint:
      return "ArmWithLockedJoint";
    case Scene::kWeld:
      return "Weld";
  }
  DRAKE_UNREACHABLE();
}

ProximityProperties MakeRigidProperties() {
  ProximityProperties props;
  geometry::AddContactMaterial(/* dissipation = */ 5.0,
                               /* point_stiffness = */ 1.0e5,
                               CoulombFriction<double>(0.6, 0.4), &props);
  return props;
}

ProximityProperties MakeCompliantHydroProperties(double resolution_hint) {
  ProximityProperties props = MakeRigidProperties();
  geometry::AddCompliantHydroelasticProperties(
      resolution_hint, /* hydroelastic_modulus = */ 1.0e6, &props);
  return props;
}

const RigidBody<double>& AddBox(const std::string& name, double size,
                                const ProximityProperties& props,
                                MultibodyPlant<double>* plant) {
  const RigidBody<double>& body = plant->AddRigidBody(
      name, SpatialInertia<double>::SolidCubeWithMass(1.0, size));
  plant->RegisterCollisionGeometry(body, RigidTransformd(),
                                   Box(size, size, size), name, props);
  return body;
}

// Adds the arm of Scene::kArm.
void AddArm(MultibodyPlant<double>* plant) {
  const double length = 0.3;
  const SpatialInertia<double> M_BBo =
      SpatialInertia<double>::MakeFromCentralInertia(
          1.0, Vector3d(0, 0, -length / 2),
          RotationalInertia<double>(0.008, 0.008, 0.0004));
  const RigidBody<double>& link1 = plant->AddRigidBody("link1", M_BBo);
  const RigidBody<double>& link2 = plant->AddRigidBody("link2", M_BBo);
  const RigidBody<double>& link3 = plant->AddRigidBody("link3", M_BBo);
  const RigidTransformd X_PF(Vector3d(0, 0, -length));
  const auto& joint1 = plant->AddJoint<RevoluteJoint>(
      "joint1", plant->world_body(), RigidTransformd(Vector3d(0, 0, 1.0)),
      link1, std::nullopt, Vector3d::UnitY(), -1.0, 1.0, /* damping = */ 0.1);
  const auto& joint2 = plant->AddJoint<RevoluteJoint>(
      "joint2", link1, X_PF, link2, std::nullopt, Vector3d::UnitY(), -0.4, 0.4,
      /* damping = */ 0.1);
  const auto& joint3 = plant->AddJoint<RevoluteJoint>(
      "joint3", link2, X_PF, link3, std::nullopt, Vector3d::UnitY());
  // Actuator 1: PD control that saturates, with reflected inertia.
  JointActuator<double>& actuator1 = plant->get_mutable_joint_actuator(
      plant->AddJointActuator("actuator1", joint1, /* effort_limit = */ 3.0)
          .index());
  actuator1.set_controller_gains({.p = 50.0, .d = 5.0});
  actuator1.set_default_rotor_inertia(0.002);
  actuator1.set_default_gear_ratio(10.0);
  // Actuator 2: PD control that pushes the joint into its upper limit.
  JointActuator<double>& actuator2 = plant->get_mutable_joint_actuator(
      plant->AddJointActuator("actuator2", joint2, /* effort_limit = */ 100.0)
          .index());
  actuator2.set_controller_gains({.p = 200.0, .d = 10.0});
  // Actuator 3: feed-forward actuation only, which saturates.
  plant->AddJointActuator("actuator3", joint3, /* effort_limit = */ 0.3);
}

void PopulatePlant(Scene scene, MultibodyPlant<double>* plant) {
  plant->set_contact_surface_representation(
      geometry::HydroelasticContactRepresentation::kPolygon);
  const ProximityProperties rigid = MakeRigidProperties();
  ProximityProperties rigid_hydro = MakeRigidProperties();
  geometry::AddRigidHydroelasticProperties(&rigid_hydro);
  const RigidTransformd X_WG(Vector3d(0, 0, -0.5));
  const Box ground_shape(10, 10, 1);
  switch (scene) {
    case Scene::kPointContact: {
      plant->set_contact_model(ContactModel::kPoint);
      AddBox("box1", 0.1, rigid, plant);
      plant->RegisterCollisionGeometry(plant->world_body(), X_WG, ground_shape,
                                       "ground", rigid);
      AddBox("box2", 0.1, rigid, plant);
      break;
    }
    case Scene::kHydroelastic: {
      plant->set_contact_model(ContactModel::kHydroelastic);
      plant->RegisterCollisionGeometry(plant->world_body(), X_WG, ground_shape,
                                       "ground", rigid_hydro);
      const RigidBody<double>& sphere = plant->AddRigidBody(
          "sphere", SpatialInertia<double>::SolidSphereWithMass(1.0, 0.05));
      plant->RegisterCollisionGeometry(sphere, RigidTransformd(), Sphere(0.05),
                                       "sphere",
                                       MakeCompliantHydroProperties(0.02));
      AddBox("box", 0.1, MakeCompliantHydroProperties(0.05), plant);
      break;
    }
    case Scene::kMixedContact: {
      plant->set_contact_model(ContactModel::kHydroelasticWithFallback);
      AddBox("box1", 0.1, rigid, plant);
      plant->RegisterCollisionGeometry(plant->world_body(), X_WG, ground_shape,
                                       "ground", rigid_hydro);
      AddBox("box2", 0.1, rigid, plant);
      const RigidBody<double>& sphere = plant->AddRigidBody(
          "sphere", SpatialInertia<double>::SolidSphereWithMass(1.0, 0.05));
      plant->RegisterCollisionGeometry(sphere, RigidTransformd(), Sphere(0.05),
                                       "sphere",
                                       MakeCompliantHydroProperties(0.02));
      break;
    }
    case Scene::kArm:
    case Scene::kArmWithLockedJoint: {
      AddArm(plant);
      break;
    }
    case Scene::kWeld: {
      plant->set_contact_model(ContactModel::kPoint);
      plant->RegisterCollisionGeometry(plant->world_body(), X_WG, ground_shape,
                                       "ground", rigid);
      const RigidBody<double>& box1 = AddBox("box1", 0.1, rigid, plant);
      const RigidBody<double>& box2 = AddBox("box2", 0.1, rigid, plant);
      plant->AddWeldConstraint(box1, RigidTransformd(Vector3d(0, 0, 0.15)),
                               box2, RigidTransformd());
      break;
    }
  }
}

// Sets the initial state and the inputs of the plant.
void SetUpContext(Scene scene, const MultibodyPlant<double>& plant,
                  Context<double>* context) {
  auto set_free_body = [&](const char* name, const Vector3d& p_WB,
                           const Vector3d& w_WB, const Vector3d& v_WB) {
    const RigidBody<double>& body = plant.GetBodyByName(name);
    plant.SetFreeBodyPose(
        context, body,
        RigidTransformd(RotationMatrixd::MakeZRotation(0.3), p_WB));
    plant.SetFreeBodySpatialVelocity(context, body,
                                     SpatialVelocity<double>(w_WB, v_WB));
  };
  switch (scene) {
    case Scene::kPointContact: {
      set_free_body("box1", Vector3d(0, 0, 0.049), Vector3d(0, 0, 1),
                    Vector3d(0.5, 0, 0));
      set_free_body("box2", Vector3d(1, 0, 0.0495), Vector3d::Zero(),
                    Vector3d(0, 0.01, 0));
      break;
    }
    case Scene::kHydroelastic: {
      set_free_body("sphere", Vector3d(0, 0, 0.049), Vector3d(0, 1, 0),
                    Vector3d(0.3, 0, 0));
      set_free_body("box", Vector3d(1, 0, 0.049), Vector3d::Zero(),
                    Vector3d(0, 0.2, 0));
      break;
    }
    case Scene::kMixedContact: {
      set_free_body("box1", Vector3d(0, 0, 0.049), Vector3d(0, 0, 1),
                    Vector3d(0.5, 0, 0));
      set_free_body("box2", Vector3d(1, 0, 0.0495), Vector3d::Zero(),
                    Vector3d(0, 0.1, 0));
      set_free_body("sphere", Vector3d(2, 0, 0.049), Vector3d(0, 1, 0),
                    Vector3d(0.3, 0, 0));
      break;
    }
    case Scene::kArm:
    case Scene::kArmWithLockedJoint: {
      plant.SetPositions(context, Vector3d(0.2, 0.3, -0.5));
      plant.SetVelocities(context, Vector3d(0.5, 1.0, -0.2));
      // Desired state (qd, vd) for the three actuators. The third actuator has
      // no PD controller, thus its desired state has no effect.
      VectorXd xd(6);
      xd << 0.8, 0.9, 0.0, 0.2, -0.1, 0.0;
      plant.get_desired_state_input_port(default_model_instance())
          .FixValue(context, xd);
      plant.get_actuation_input_port().FixValue(context,
                                                Vector3d(0.3, -0.2, 0.5));
      plant.get_applied_generalized_force_input_port().FixValue(
          context, Vector3d(0.1, -0.05, 0.02));
      if (scene == Scene::kArmWithLockedJoint) {
        plant.GetJointByName("joint2").Lock(context);
      }
      break;
    }
    case Scene::kWeld: {
      set_free_body("box1", Vector3d(0, 0, 0.06), Vector3d(0, 0, 0.5),
                    Vector3d(0.2, 0, 0));
      set_free_body("box2", Vector3d(0.01, 0, 0.21), Vector3d::Zero(),
                    Vector3d::Zero());
      break;
    }
  }
}

// A plant in a diagram with its scene graph, and a context for the diagram.
// For a discrete plant, a simulator owns the context.
struct PlantSystem {
  std::unique_ptr<Diagram<double>> diagram;
  MultibodyPlant<double>* plant{};
  std::unique_ptr<Context<double>> context;
  std::unique_ptr<systems::Simulator<double>> simulator;
  int num_steps{0};

  Context<double>& root_context() {
    return simulator != nullptr ? simulator->get_mutable_context() : *context;
  }

  Context<double>& plant_context() {
    return plant->GetMyMutableContextFromRoot(&root_context());
  }
};

// Makes the system of the given scene. A discrete plant (time_step > 0) uses
// the given solver.
PlantSystem MakePlantSystem(Scene scene, double time_step,
                            DiscreteContactSolver solver,
                            bool use_sampled_output_ports = true) {
  PlantSystem result;
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, time_step);
  if (time_step > 0) {
    plant.set_discrete_contact_solver(solver);
    plant.SetUseSampledOutputPorts(use_sampled_output_ports);
  }
  PopulatePlant(scene, &plant);
  plant.Finalize();
  result.plant = &plant;
  result.diagram = builder.Build();
  result.context = result.diagram->CreateDefaultContext();
  SetUpContext(scene, plant, &result.plant_context());
  if (time_step > 0) {
    result.simulator = std::make_unique<systems::Simulator<double>>(
        *result.diagram, std::move(result.context));
    result.simulator->Initialize();
  }
  return result;
}

// Advances a discrete plant by one step. The plant updates its state at times
// 0, h, 2h, etc. We advance to the middle of a step, thus the round-off error
// of the event times has no effect on the number of updates.
void StepDiscrete(PlantSystem* system) {
  DRAKE_DEMAND(system->simulator != nullptr);
  ++system->num_steps;
  system->simulator->AdvanceTo((system->num_steps - 0.5) *
                               system->plant->time_step());
}

class IcfDriverTest : public ::testing::TestWithParam<Scene> {};

// The main contract: with no external systems, a discrete plant with the ICF
// driver at time step h gives the same trajectory as fixed step CENIC (no
// error control, step h) on a continuous copy of the same model.
TEST_P(IcfDriverTest, MatchesFixedStepCenic) {
  const Scene scene = GetParam();
  PlantSystem discrete =
      MakePlantSystem(scene, kTimeStep, DiscreteContactSolver::kIcf);
  PlantSystem continuous =
      MakePlantSystem(scene, 0.0, DiscreteContactSolver::kSap);

  CenicIntegrator<double> cenic(*continuous.diagram, continuous.context.get());
  cenic.set_maximum_step_size(kTimeStep);
  cenic.set_fixed_step_mode(true);
  cenic.Initialize();

  const MultibodyPlant<double>& plant_d = *discrete.plant;
  const MultibodyPlant<double>& plant_c = *continuous.plant;
  const int kNumSteps = 50;
  double max_v_error = 0;
  double max_pose_error = 0;
  for (int step = 0; step < kNumSteps; ++step) {
    StepDiscrete(&discrete);
    const double t = continuous.root_context().get_time();
    ASSERT_TRUE(cenic.IntegrateWithSingleFixedStepToTime(t + kTimeStep));

    const Context<double>& context_d = discrete.plant_context();
    const Context<double>& context_c = continuous.plant_context();
    const VectorXd v_d = plant_d.GetVelocities(context_d);
    const VectorXd v_c = plant_c.GetVelocities(context_c);
    max_v_error = std::max(max_v_error, (v_d - v_c).lpNorm<Eigen::Infinity>());
    // The discrete plant does not normalize the quaternions of free bodies,
    // but CENIC does. Thus we compare body poses, not generalized positions.
    for (BodyIndex b(0); b < plant_d.num_bodies(); ++b) {
      const RigidTransformd& X_WB_d =
          plant_d.EvalBodyPoseInWorld(context_d, plant_d.get_body(b));
      const RigidTransformd& X_WB_c =
          plant_c.EvalBodyPoseInWorld(context_c, plant_c.get_body(b));
      max_pose_error = std::max(
          max_pose_error, (X_WB_d.GetAsMatrix34() - X_WB_c.GetAsMatrix34())
                              .lpNorm<Eigen::Infinity>());
    }
  }
  // Make sure that the scene has motion and contact, thus the comparison is
  // not trivial.
  EXPECT_GT(plant_d.GetVelocities(discrete.plant_context()).norm(), 1e-3);
  if (scene != Scene::kArm && scene != Scene::kArmWithLockedJoint) {
    EXPECT_GT(MultibodyPlantTester::manager(plant_d)
                  .EvalDiscreteContactPairs(discrete.plant_context())
                  .size(),
              0);
  }

  // The two simulations solve the same convex problems. They differ only by
  // round-off error (for example, from quaternion normalization), which the
  // solver tolerance and contact can amplify.
  constexpr double kTolerance = 1e-8;
  EXPECT_LT(max_v_error, kTolerance) << SceneName(scene);
  EXPECT_LT(max_pose_error, kTolerance) << SceneName(scene);
  RecordProperty("max_v_error", fmt::format("{:.3g}", max_v_error));
  RecordProperty("max_pose_error", fmt::format("{:.3g}", max_pose_error));
}

// The forces of the discrete update (see
// DiscreteUpdateManager::EvalDiscreteUpdateMultibodyForces()) must satisfy the
// discrete momentum balance, for the velocities that are not locked. With
// a = (v - v₀)/δt, the drivers subtract the implicit term (R + δt⋅D)⋅a from the
// forces, where R is the reflected inertia and D the joint damping. Inverse
// dynamics includes R⋅a. Thus ID(a, forces) = R⋅a. We also check the SAP
// driver, to confirm that the check is correct.
TEST_P(IcfDriverTest, DiscreteUpdateForcesSatisfyMomentumBalance) {
  const Scene scene = GetParam();
  for (DiscreteContactSolver solver :
       {DiscreteContactSolver::kSap, DiscreteContactSolver::kIcf}) {
    PlantSystem system = MakePlantSystem(scene, kTimeStep, solver);
    const MultibodyPlant<double>& plant = *system.plant;
    // Take a few steps, to have contact.
    for (int step = 0; step < 5; ++step) {
      StepDiscrete(&system);
    }
    const Context<double>& context = system.plant_context();
    const CompliantContactManager<double>& manager =
        MultibodyPlantTester::manager(plant);
    const MultibodyForces<double>& forces =
        manager.EvalDiscreteUpdateMultibodyForces(context);
    const VectorXd& v_next = manager.EvalContactSolverResults(context).v_next;
    const VectorXd v0 = plant.GetVelocities(context);
    const VectorXd a = (v_next - v0) / kTimeStep;
    VectorXd reflected_inertia = VectorXd::Zero(plant.num_velocities());
    for (JointActuatorIndex index : plant.GetJointActuatorIndices()) {
      const JointActuator<double>& actuator = plant.get_joint_actuator(index);
      reflected_inertia[actuator.joint().velocity_start()] =
          actuator.default_reflected_inertia();
    }
    const VectorXd residual = plant.CalcInverseDynamics(context, a, forces) -
                              reflected_inertia.cwiseProduct(a);
    std::vector<int> unlocked;
    for (JointIndex j : plant.GetJointIndices()) {
      const Joint<double>& joint = plant.get_joint(j);
      if (!joint.is_locked(context)) {
        for (int k = 0; k < joint.num_velocities(); ++k) {
          unlocked.push_back(joint.velocity_start() + k);
        }
      }
    }
    double scale = 1.0;
    for (int i : unlocked) {
      scale = std::max(scale, std::abs(forces.generalized_forces()[i]));
    }
    for (int i : unlocked) {
      EXPECT_NEAR(residual[i], 0.0, 1e-5 * scale)
          << SceneName(scene) << " solver " << static_cast<int>(solver)
          << " dof " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Scenes, IcfDriverTest,
                         ::testing::Values(Scene::kPointContact,
                                           Scene::kHydroelastic,
                                           Scene::kMixedContact, Scene::kArm,
                                           Scene::kArmWithLockedJoint,
                                           Scene::kWeld),
                         [](const ::testing::TestParamInfo<Scene>& test_info) {
                           return SceneName(test_info.param);
                         });

// Proves the mapping from ICF contact pairs to the discrete contact pairs of
// the manager. For each discrete contact pair i, the manager gives the contact
// Jacobian Jᵢ (in the contact frame C). The ICF solver gives the contact force
// fᵢ, the contact velocity vcᵢ (both in C) and the generalized contact force
// τ_contact, which ICF computes from its own patch impulses. With a correct
// mapping:
//   τ_contact = ∑ᵢ Jᵢᵀ⋅fᵢ  and  vcᵢ = Jᵢ⋅v_next.
GTEST_TEST(IcfDriverContactResultsTest, MappingToDiscreteContactPairs) {
  PlantSystem system = MakePlantSystem(Scene::kMixedContact, kTimeStep,
                                       DiscreteContactSolver::kIcf);
  const MultibodyPlant<double>& plant = *system.plant;
  for (int step = 0; step < 5; ++step) {
    StepDiscrete(&system);
  }
  const Context<double>& context = system.plant_context();
  const CompliantContactManager<double>& manager =
      MultibodyPlantTester::manager(plant);
  const DiscreteContactData<DiscreteContactPair<double>>& pairs =
      manager.EvalDiscreteContactPairs(context);
  const ContactSolverResults<double>& results =
      manager.EvalContactSolverResults(context);

  // The scene has point and hydroelastic contact pairs, and the anchored ground
  // is body B of one of the point contact pairs. For that pair, ICF orders the
  // bodies in the opposite way.
  ASSERT_GT(pairs.num_point_contacts(), 1);
  ASSERT_GT(pairs.num_hydro_contacts(), 1);
  bool ground_is_body_B = false;
  bool ground_is_body_A = false;
  for (int i = 0; i < pairs.num_point_contacts(); ++i) {
    ground_is_body_B |= (pairs[i].object_B == plant.world_body().index());
    ground_is_body_A |= (pairs[i].object_A == plant.world_body().index());
  }
  EXPECT_TRUE(ground_is_body_B);
  EXPECT_TRUE(ground_is_body_A);

  const int nc = pairs.size();
  const Eigen::MatrixXd J_AcBc_C =
      CompliantContactManagerTester::CalcDenseJacobianMatrixInContactFrame(
          manager, pairs);
  VectorXd f_C(3 * nc);
  VectorXd vc_C(3 * nc);
  for (int i = 0; i < nc; ++i) {
    f_C.segment<3>(3 * i) << results.ft.segment<2>(2 * i), results.fn(i);
    vc_C.segment<3>(3 * i) << results.vt.segment<2>(2 * i), results.vn(i);
    // The normal force is never negative.
    EXPECT_GE(results.fn(i), 0.0);
  }
  // Each pair must have a non-trivial force, thus the check is not trivial.
  EXPECT_GT(results.fn.minCoeff(), 0.0);

  const double force_scale = results.tau_contact.norm();
  EXPECT_GT(force_scale, 1.0);
  EXPECT_TRUE(CompareMatrices(J_AcBc_C.transpose() * f_C, results.tau_contact,
                              1e-10 * force_scale));
  EXPECT_TRUE(CompareMatrices(J_AcBc_C * results.v_next, vc_C, 1e-12));

  // The contact results output uses the same values.
  const ContactResults<double>& contact_results =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          context);
  EXPECT_EQ(contact_results.num_point_pair_contacts(),
            pairs.num_point_contacts());
  EXPECT_EQ(contact_results.num_hydroelastic_contacts(), 1);
}

// Checks the net actuation: PD control with q = q₀ + δt⋅v, feed-forward
// actuation, effort limits, and locked joints.
GTEST_TEST(IcfDriverActuationTest, NetActuation) {
  for (Scene scene : {Scene::kArm, Scene::kArmWithLockedJoint}) {
    PlantSystem system =
        MakePlantSystem(scene, kTimeStep, DiscreteContactSolver::kIcf,
                        /* use_sampled_output_ports = */ false);
    const MultibodyPlant<double>& plant = *system.plant;
    const Context<double>& context = system.plant_context();
    const VectorXd q0 = plant.GetPositions(context);
    const VectorXd& v = MultibodyPlantTester::manager(plant)
                            .EvalContactSolverResults(context)
                            .v_next;
    const VectorXd net_actuation =
        plant.get_net_actuation_output_port().Eval(context);

    // These values are the same as in AddArm() and SetUpContext().
    const Vector3d qd(0.8, 0.9, 0.0);
    const Vector3d vd(0.2, -0.1, 0.0);
    const Vector3d u_ff(0.3, -0.2, 0.5);
    const Vector3d Kp(50.0, 200.0, 0.0);
    const Vector3d Kd(5.0, 10.0, 0.0);
    const Vector3d effort_limit(3.0, 100.0, 0.3);
    const bool joint2_locked = (scene == Scene::kArmWithLockedJoint);
    Vector3d expected;
    for (int i = 0; i < 3; ++i) {
      const bool has_pd = (i < 2) && !(i == 1 && joint2_locked);
      double u = u_ff[i];
      if (has_pd) {
        u += -Kp[i] * (q0[i] + kTimeStep * v[i] - qd[i]) -
             Kd[i] * (v[i] - vd[i]);
      }
      expected[i] = std::clamp(u, -effort_limit[i], effort_limit[i]);
    }
    EXPECT_TRUE(CompareMatrices(net_actuation, expected, 1e-12))
        << SceneName(scene);
    // The first and third actuators saturate.
    EXPECT_EQ(std::abs(net_actuation[0]), 3.0);
    EXPECT_EQ(std::abs(net_actuation[2]), 0.3);
    if (joint2_locked) {
      EXPECT_EQ(v[1], 0.0);
    }
  }
}

// ICF supports only T = double. A scalar converted plant can do other queries,
// but discrete updates throw.
GTEST_TEST(IcfDriverScalarTest, AutoDiffThrows) {
  PlantSystem system =
      MakePlantSystem(Scene::kArm, kTimeStep, DiscreteContactSolver::kIcf,
                      /* use_sampled_output_ports = */ false);
  const std::unique_ptr<Diagram<AutoDiffXd>> diagram_ad =
      systems::System<double>::ToAutoDiffXd(*system.diagram);
  const auto& plant_ad = dynamic_cast<const MultibodyPlant<AutoDiffXd>&>(
      diagram_ad->GetSubsystemByName(system.plant->get_name()));
  EXPECT_EQ(plant_ad.get_discrete_contact_solver(),
            DiscreteContactSolver::kIcf);
  auto context_ad = diagram_ad->CreateDefaultContext();
  const Context<AutoDiffXd>& plant_context_ad =
      plant_ad.GetMyContextFromRoot(*context_ad);
  // Kinematics work.
  EXPECT_NO_THROW(plant_ad.EvalBodyPoseInWorld(
      plant_context_ad, plant_ad.GetBodyByName("link3")));
  DRAKE_EXPECT_THROWS_MESSAGE(
      plant_ad.EvalUniquePeriodicDiscreteUpdate(plant_context_ad),
      ".*ICF solver.*only supported for T = double.*");
}

GTEST_TEST(IcfDriverSetupTest, SetterPreconditions) {
  MultibodyPlant<double> continuous_plant(0.0);
  EXPECT_THROW(
      continuous_plant.set_discrete_contact_solver(DiscreteContactSolver::kIcf),
      std::exception);

  MultibodyPlant<double> plant(kTimeStep);
  EXPECT_EQ(plant.get_discrete_contact_solver(), DiscreteContactSolver::kSap);
  plant.set_discrete_contact_solver(DiscreteContactSolver::kIcf);
  EXPECT_EQ(plant.get_discrete_contact_solver(), DiscreteContactSolver::kIcf);
  plant.Finalize();
  EXPECT_THROW(plant.set_discrete_contact_solver(DiscreteContactSolver::kSap),
               std::exception);
}

GTEST_TEST(IcfDriverSetupTest, TendonConstraintsThrow) {
  MultibodyPlant<double> plant(kTimeStep);
  plant.set_discrete_contact_solver(DiscreteContactSolver::kIcf);
  AddArm(&plant);
  plant.AddTendonConstraint({plant.GetJointByName("joint1").index()}, {1.0},
                            std::nullopt, -1.0, 1.0);
  DRAKE_EXPECT_THROWS_MESSAGE(plant.Finalize(),
                              ".*kIcf does not support tendon constraints.*");
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
