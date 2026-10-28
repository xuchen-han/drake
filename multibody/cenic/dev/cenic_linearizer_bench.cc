// Times IcfExternalSystemsLinearizer::LinearizeExternalSystem() and one
// simulation step on a robot with 29 actuated revolute joints plus free
// bodies. A firmware-style joint impedance controller (mass matrix,
// eigendecomposition, bias terms) drives the robot through the actuation input.
// The step uses fixed-step CENIC on a continuous plant, or SAP on a discrete
// plant with the same time step.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/geometry/shape_specification.h"
#include "drake/multibody/cenic/cenic_integrator.h"
#include "drake/multibody/contact_solvers/icf/icf_external_systems_linearizer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/quaternion_floating_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/matrix_gain.h"

DEFINE_string(integrator, "cenic", "cenic (continuous plant) or sap.");
DEFINE_string(damping, "none",
              "none; joint (joint damping on every joint); external "
              "(tau = -c v from a LeafSystem on the applied generalized force "
              "input); external_dense (the same damping from a dense "
              "MatrixGain on the full state).");
DEFINE_bool(contact, false,
            "Give the free bodies sphere collision geometry, resting on a "
            "ground box.");
DEFINE_double(time_step, 0.01, "Time step for CENIC and SAP.");
DEFINE_int32(num_free_bodies, 52, "Number of free bodies.");
DEFINE_int32(num_calls, 300, "Number of timed linearizer calls.");
DEFINE_int32(num_steps, 200, "Number of timed simulation steps.");

namespace drake {
namespace multibody {
namespace {

using contact_solvers::icf::internal::IcfExternalSystemsLinearizer;
using contact_solvers::icf::internal::IcfLinearFeedbackGains;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using math::RigidTransformd;
using systems::BasicVector;
using systems::Context;
using Clock = std::chrono::steady_clock;

constexpr double kDamping = 0.1;
constexpr double kRadius = 0.03;

double JointDamping() {
  return FLAGS_damping == "joint" ? kDamping : 0.0;
}

const RigidBody<double>& AddChain(MultibodyPlant<double>* plant,
                                  ModelInstanceIndex instance,
                                  const RigidBody<double>& parent,
                                  const std::string& prefix, int num_joints) {
  const RigidBody<double>* previous = &parent;
  for (int i = 0; i < num_joints; ++i) {
    const RigidBody<double>& body =
        plant->AddRigidBody(fmt::format("{}_link{}", prefix, i), instance,
                            SpatialInertia<double>::SolidCylinderWithMass(
                                1.0, 0.04, 0.2, Vector3d::UnitZ()));
    const Vector3d axis = (i % 2 == 0) ? Vector3d::UnitZ() : Vector3d::UnitY();
    const RevoluteJoint<double>& joint = plant->AddJoint<RevoluteJoint>(
        fmt::format("{}_joint{}", prefix, i), *previous,
        RigidTransformd(Vector3d(0, 0, 0.2)), body, RigidTransformd(), axis,
        JointDamping());
    const JointActuatorIndex actuator =
        plant->AddJointActuator(fmt::format("{}_actuator{}", prefix, i), joint)
            .index();
    // A geared motor: the reflected inertia is 1e-5 * 100^2 = 0.1 kg m^2. With
    // this inertia, the explicit (zero-order hold) controller on the discrete
    // SAP plant is stable at h = 0.01 s, as on a real geared robot.
    plant->get_mutable_joint_actuator(actuator).set_default_rotor_inertia(1e-5);
    plant->get_mutable_joint_actuator(actuator).set_default_gear_ratio(100);
    previous = &body;
  }
  return *previous;
}

// A 7-joint torso with two 11-joint arms: 29 actuated revolute joints.
ModelInstanceIndex AddRobot(MultibodyPlant<double>* plant) {
  const ModelInstanceIndex robot = plant->AddModelInstance("robot");
  const RigidBody<double>& torso =
      AddChain(plant, robot, plant->world_body(), "torso", 7);
  AddChain(plant, robot, torso, "left", 11);
  AddChain(plant, robot, torso, "right", 11);
  return robot;
}

// tau = -kp q - D v + C(q, v) v - tau_g(q), with D = 2 sqrt(kp) M^(1/2).
class ImpedanceController final : public systems::LeafSystem<double> {
 public:
  explicit ImpedanceController(const MultibodyPlant<double>* robot)
      : robot_(*robot), robot_context_(robot->CreateDefaultContext()) {
    const int n = robot_.num_velocities();
    DeclareVectorInputPort("state", 2 * n);
    DeclareVectorOutputPort("actuation", n,
                            &ImpedanceController::CalcActuation);
  }

  int num_evaluations() const { return num_evaluations_; }

 private:
  void CalcActuation(const Context<double>& context,
                     BasicVector<double>* output) const {
    ++num_evaluations_;
    const double kp = 100.0;
    const int n = robot_.num_velocities();
    const VectorXd& x = get_input_port().Eval(context);
    robot_.SetPositionsAndVelocities(robot_context_.get(), x);
    MatrixXd M(n, n);
    robot_.CalcMassMatrix(*robot_context_, &M);
    const Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(M);
    const VectorXd sqrt_lambda = eigen.eigenvalues().cwiseSqrt();
    const MatrixXd D = 2.0 * std::sqrt(kp) * eigen.eigenvectors() *
                       sqrt_lambda.asDiagonal() *
                       eigen.eigenvectors().transpose();
    VectorXd bias(n);
    robot_.CalcBiasTerm(*robot_context_, &bias);
    const VectorXd tau_g = robot_.CalcGravityGeneralizedForces(*robot_context_);
    output->SetFromVector(-kp * x.head(n) - D * x.tail(n) + bias - tau_g);
  }

  const MultibodyPlant<double>& robot_;
  std::unique_ptr<Context<double>> robot_context_;
  mutable int num_evaluations_{0};
};

// tau = -kDamping v, from the plant's full state.
class ViscousDamping final : public systems::LeafSystem<double> {
 public:
  ViscousDamping(int nq, int nv) : nq_(nq), nv_(nv) {
    DeclareVectorInputPort("state", nq + nv);
    DeclareVectorOutputPort("generalized_force", nv,
                            &ViscousDamping::CalcForce);
  }

 private:
  void CalcForce(const Context<double>& context,
                 BasicVector<double>* output) const {
    output->SetFromVector(-kDamping *
                          get_input_port().Eval(context).segment(nq_, nv_));
  }

  const int nq_;
  const int nv_;
};

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int do_main() {
  const bool use_sap = FLAGS_integrator == "sap";
  DRAKE_THROW_UNLESS(use_sap || FLAGS_integrator == "cenic");
  DRAKE_THROW_UNLESS(FLAGS_damping == "none" || FLAGS_damping == "joint" ||
                     FLAGS_damping == "external" ||
                     FLAGS_damping == "external_dense");
  const double h = FLAGS_time_step;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, use_sap ? h : 0.0);
  if (use_sap) {
    plant.set_discrete_contact_approximation(
        DiscreteContactApproximation::kSap);
  }
  const ModelInstanceIndex robot = AddRobot(&plant);
  const ModelInstanceIndex objects = plant.AddModelInstance("objects");
  const CoulombFriction<double> friction(0.5, 0.5);
  std::vector<const QuaternionFloatingJoint<double>*> object_joints;
  for (int i = 0; i < FLAGS_num_free_bodies; ++i) {
    const RigidBody<double>& body = plant.AddRigidBody(
        fmt::format("object{}", i), objects,
        SpatialInertia<double>::SolidSphereWithMass(0.1, kRadius));
    object_joints.push_back(&plant.AddJoint<QuaternionFloatingJoint>(
        fmt::format("object{}_joint", i), plant.world_body(), {}, body, {},
        JointDamping(), JointDamping()));
    if (FLAGS_contact) {
      plant.RegisterCollisionGeometry(body, RigidTransformd(),
                                      geometry::Sphere(kRadius), "sphere",
                                      friction);
    }
  }
  if (FLAGS_contact) {
    plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransformd(Vector3d(0, 0, -0.05)),
        geometry::Box(10, 10, 0.1), "ground", friction);
  }
  plant.Finalize();

  MultibodyPlant<double> robot_model(0.0);
  AddRobot(&robot_model);
  robot_model.Finalize();

  auto* controller = builder.AddSystem<ImpedanceController>(&robot_model);
  builder.Connect(plant.get_state_output_port(robot),
                  controller->get_input_port());
  builder.Connect(controller->get_output_port(),
                  plant.get_actuation_input_port(robot));
  const int nq = plant.num_positions();
  const int nv = plant.num_velocities();
  if (FLAGS_damping == "external") {
    auto* damping = builder.AddSystem<ViscousDamping>(nq, nv);
    builder.Connect(plant.get_state_output_port(), damping->get_input_port());
    builder.Connect(damping->get_output_port(),
                    plant.get_applied_generalized_force_input_port());
  } else if (FLAGS_damping == "external_dense") {
    MatrixXd D = MatrixXd::Zero(nv, nq + nv);
    D.rightCols(nv) = -kDamping * MatrixXd::Identity(nv, nv);
    auto* damping = builder.AddSystem<systems::MatrixGain<double>>(D);
    builder.Connect(plant.get_state_output_port(), damping->get_input_port());
    builder.Connect(damping->get_output_port(),
                    plant.get_applied_generalized_force_input_port());
  }
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(context.get());

  // An arbitrary non-zero robot state. The objects rest on the ground (in a
  // grid, apart from each other) and have zero velocity.
  plant.SetPositions(
      &plant_context, robot,
      VectorXd::LinSpaced(plant.num_positions(robot), -0.5, 0.5));
  plant.SetVelocities(
      &plant_context, robot,
      VectorXd::LinSpaced(plant.num_velocities(robot), -0.5, 0.5));
  for (int i = 0; i < FLAGS_num_free_bodies; ++i) {
    object_joints[i]->SetPose(
        &plant_context,
        RigidTransformd(
            math::RollPitchYawd(0.1 * i, 0.2, 0.3),
            Vector3d(1.0 + 0.2 * (i % 8), 1.0 + 0.2 * (i / 8), kRadius)));
  }
  const VectorXd x0 = plant.GetPositionsAndVelocities(plant_context);

  std::cout << fmt::format(
      "integrator = {}, damping = {}, contact = {}, h = {}, nq = {}, nv = {}, "
      "actuated velocities = {}\n",
      FLAGS_integrator, FLAGS_damping, FLAGS_contact, h, nq, nv,
      plant.num_actuated_dofs());

  // Time one controller evaluation.
  {
    std::vector<double> times;
    const Context<double>& controller_context =
        controller->GetMyContextFromRoot(*context);
    for (int k = 0; k < FLAGS_num_calls; ++k) {
      plant.SetPositionsAndVelocities(&plant_context, x0);
      const auto start = Clock::now();
      controller->get_output_port().Eval(controller_context);
      times.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - start)
              .count());
    }
    std::cout << fmt::format("controller evaluation: median {:.1f} us\n",
                             Median(times));
  }

  // Time the linearizer.
  if (!use_sap) {
    const IcfExternalSystemsLinearizer<double> linearizer(&plant);
    IcfLinearFeedbackGains<double> actuation;
    IcfLinearFeedbackGains<double> external;
    actuation.Resize(nv);
    external.Resize(nv);
    std::vector<double> times;
    const int evaluations_before = controller->num_evaluations();
    for (int k = 0; k < FLAGS_num_calls; ++k) {
      const auto start = Clock::now();
      linearizer.LinearizeExternalSystem(h, plant_context, &actuation,
                                         &external);
      times.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - start)
              .count());
    }
    std::cout << fmt::format(
        "LinearizeExternalSystem: median {:.1f} us, controller evaluations "
        "per call {:.1f}\n",
        Median(times),
        static_cast<double>(controller->num_evaluations() -
                            evaluations_before) /
            FLAGS_num_calls);
    std::cout << fmt::format("  Ku sum {:.12g}, bu sum {:.12g}\n",
                             actuation.K.sum(), actuation.b.sum());
    if (FLAGS_damping == "external" || FLAGS_damping == "external_dense") {
      std::cout << fmt::format("  Ke sum {:.12g}, be sum {:.12g}\n",
                               external.K.sum(), external.b.sum());
    }
  }

  // Time the simulation steps.
  {
    plant.SetPositionsAndVelocities(&plant_context, x0);
    systems::Simulator<double> simulator(*diagram, std::move(context));
    if (!use_sap) {
      auto& integrator = simulator.reset_integrator<CenicIntegrator<double>>();
      integrator.set_maximum_step_size(h);
      integrator.set_fixed_step_mode(true);
    }
    simulator.Initialize();
    simulator.AdvanceTo(10 * h);
    const int evaluations_before = controller->num_evaluations();
    const auto start = Clock::now();
    simulator.AdvanceTo((10 + FLAGS_num_steps) * h);
    const double total =
        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    std::cout << fmt::format(
        "{} step: mean {:.1f} us, controller evaluations per step {:.1f}\n",
        FLAGS_integrator, total / FLAGS_num_steps,
        static_cast<double>(controller->num_evaluations() -
                            evaluations_before) /
            FLAGS_num_steps);
  }
  return 0;
}

}  // namespace
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout.setf(std::ios::unitbuf);
  return drake::multibody::do_main();
}
