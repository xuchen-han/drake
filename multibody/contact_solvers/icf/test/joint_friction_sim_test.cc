/* Tests that CENIC models joint dry friction implicitly with ICF, and that a
fixed step CENIC simulation agrees with a discrete SAP plant at the same time
step. */
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "drake/multibody/cenic/cenic_integrator.h"
#include "drake/multibody/contact_solvers/icf/joint_friction_constraints_pool.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace {

using Eigen::VectorXd;
using multibody::CenicIntegrator;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

constexpr double kSigma =
    internal::JointFrictionConstraintsPool<double>::kSigma;
constexpr double kGravity = 9.81;  // Default plant gravity magnitude.
constexpr double kInf = std::numeric_limits<double>::infinity();

// A pendulum that rotates about the world's y axis, with its center of mass at
// distance kLength below the joint. The gravity torque is −m⋅g⋅L⋅sin(q).
constexpr double kMass = 1.0;
constexpr double kLength = 0.5;
constexpr double kInertia = 0.01 + kMass * kLength * kLength;  // About y.
constexpr char kPendulumXml[] = R"""(
  <?xml version="1.0"?>
  <mujoco model="pendulum">
    <worldbody>
      <body name="link">
        <joint name="joint" type="hinge" axis="0 1 0" damping="0.05"/>
        <inertial pos="0 0 -0.5" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
    </worldbody>
  </mujoco>
)""";
constexpr double kPendulumDamping = 0.05;

// A wheel that spins about the world's z axis, so that gravity does not act on
// it.
constexpr double kWheelInertia = 0.1;
constexpr char kWheelXml[] = R"""(
  <?xml version="1.0"?>
  <mujoco model="wheel">
    <worldbody>
      <body name="wheel">
        <joint name="wheel" type="hinge" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
)""";

// A double pendulum about the world's y axis.
constexpr char kDoublePendulumXml[] = R"""(
  <?xml version="1.0"?>
  <mujoco model="double_pendulum">
    <worldbody>
      <body name="link1">
        <joint name="joint1" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 -0.25" mass="1.0" diaginertia="0.01 0.01 0.01"/>
        <body name="link2" pos="0 0 -0.5">
          <joint name="joint2" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 -0.25" mass="1.0" diaginertia="0.01 0.01 0.01"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
)""";

// A planar three link arm about the world's y axis, mounted kArmHeight above
// the ground, and initially horizontal. The wrist link has a sphere of radius
// kTipRadius at distance kWristLength from the wrist. A free box falls on the
// ground next to the arm.
constexpr double kArmHeight = 0.15;
constexpr double kWristLength = 0.2;
constexpr double kTipRadius = 0.03;
constexpr char kArmXml[] = R"""(
  <?xml version="1.0"?>
  <mujoco model="arm">
    <worldbody>
      <geom name="ground" type="box" pos="0 0 -0.05" size="2 2 0.05"/>
      <body name="link1" pos="0 0 0.15">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0.15 0 0" mass="2.0" diaginertia="0.02 0.02 0.02"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
          <inertial pos="0.15 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
          <body name="link3" pos="0.3 0 0">
            <joint name="wrist" type="hinge" axis="0 1 0" damping="0.1"/>
            <inertial pos="0.1 0 0" mass="0.5" diaginertia="0.002 0.002 0.002"/>
            <geom name="tip" type="sphere" pos="0.2 0 0" size="0.03"/>
          </body>
        </body>
      </body>
      <body name="box" pos="-0.5 0 0.3">
        <freejoint name="box"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
        <geom name="box" type="box" size="0.05 0.05 0.05"/>
      </body>
    </worldbody>
  </mujoco>
)""";

// Options for a simulation of one of the models above.
struct SimulationOptions {
  // The model, in MJCF format.
  std::string xml;
  // Dry friction bound of each joint, by name.
  std::map<std::string, double> dry_friction;
  // Initial angle of each joint, by name.
  std::map<std::string, double> initial_angle;
  // Joints locked at their initial angle.
  std::vector<std::string> locked_joints;
  // For each joint, by name, adds an actuator with the given effort limit and
  // applies the given constant actuation (effort limit, actuation).
  std::map<std::string, std::pair<double, double>> actuation;
  // A positive time step makes a discrete SAP plant. A zero time step makes a
  // continuous plant integrated by CENIC.
  double time_step{0.0};
  // The step size of a fixed step CENIC integrator, or the maximum step size of
  // an error controlled CENIC integrator.
  double cenic_step{1.0e-3};
  // When true, CENIC uses error control with this target accuracy.
  bool error_control{false};
  double accuracy{1.0e-3};
  // The discrete contact approximation of a discrete plant.
  std::string discrete_contact_approximation{"sap"};
};

// A simulation of a model, made by MakeSimulation().
struct Simulation {
  std::unique_ptr<Diagram<double>> diagram;
  const MultibodyPlant<double>* plant{};
  std::unique_ptr<Simulator<double>> simulator;

  const Context<double>& plant_context() const {
    return plant->GetMyContextFromRoot(simulator->get_context());
  }
  double angle(const std::string& joint) const {
    return plant->GetJointByName<RevoluteJoint>(joint).get_angle(
        plant_context());
  }
  double rate(const std::string& joint) const {
    return plant->GetJointByName<RevoluteJoint>(joint).get_angular_rate(
        plant_context());
  }
  VectorXd positions() const { return plant->GetPositions(plant_context()); }
  VectorXd velocities() const { return plant->GetVelocities(plant_context()); }
};

Simulation MakeSimulation(const SimulationOptions& options) {
  DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = options.time_step;
  plant_config.contact_model = "point";
  plant_config.discrete_contact_approximation =
      options.discrete_contact_approximation;
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  Parser(&plant).AddModelsFromString(options.xml, "xml");
  for (const auto& [name, friction] : options.dry_friction) {
    plant.GetMutableJointByName<RevoluteJoint>(name).set_default_dry_friction(
        friction);
  }
  for (const auto& [name, effort] : options.actuation) {
    plant.AddJointActuator(name, plant.GetJointByName(name), effort.first);
  }
  plant.Finalize();

  Simulation result;
  result.plant = &plant;
  result.diagram = builder.Build();
  auto context = result.diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(context.get());
  for (const auto& [name, angle] : options.initial_angle) {
    plant.GetJointByName<RevoluteJoint>(name).set_angle(&plant_context, angle);
  }
  for (const std::string& name : options.locked_joints) {
    plant.GetJointByName(name).Lock(&plant_context);
  }
  if (plant.num_actuators() > 0) {
    VectorXd u = VectorXd::Zero(plant.num_actuated_dofs());
    for (const auto& [name, effort] : options.actuation) {
      u(plant.GetJointActuatorByName(name).input_start()) = effort.second;
    }
    plant.get_actuation_input_port().FixValue(&plant_context, u);
  }

  result.simulator =
      std::make_unique<Simulator<double>>(*result.diagram, std::move(context));
  if (options.time_step == 0.0) {
    auto& integrator =
        result.simulator->reset_integrator<CenicIntegrator<double>>();
    integrator.set_maximum_step_size(options.cenic_step);
    integrator.set_fixed_step_mode(!options.error_control);
    if (options.error_control) {
      integrator.set_target_accuracy(options.accuracy);
    }
  }
  result.simulator->Initialize();
  return result;
}

// Under a load below the friction bound, the joint sticks. It creeps at the
// velocity v = σ⋅w⋅δt⋅τₐ of the regularization, with w = 1/A the inverse of the
// linearized dynamics A = I + δt⋅d.
GTEST_TEST(JointFrictionSimulation, Stiction) {
  constexpr double kFriction = 0.5;
  constexpr double kTorque = 0.6 * kFriction;
  constexpr double kTimeStep = 1.0e-3;
  Simulation sim = MakeSimulation({.xml = kWheelXml,
                                   .dry_friction = {{"wheel", kFriction}},
                                   .actuation = {{"wheel", {kInf, kTorque}}},
                                   .cenic_step = kTimeStep});
  constexpr double kDuration = 1.0;
  sim.simulator->AdvanceTo(kDuration);

  const double creep = kSigma * kTimeStep * kTorque / kWheelInertia;
  // Without friction, the wheel would spin at this rate.
  const double free_rate = kDuration * kTorque / kWheelInertia;
  EXPECT_NEAR(sim.rate("wheel"), creep, 1.0e-6 * creep);
  EXPECT_NEAR(sim.angle("wheel"), kDuration * creep,
              1.0e-2 * kDuration * creep);
  EXPECT_LT(sim.rate("wheel"), 1.0e-6 * free_rate);
}

// Under a load above the friction bound, the joint slides with friction at its
// bound. The actuator saturates at its effort limit before friction acts, since
// friction and actuation are separate constraints.
GTEST_TEST(JointFrictionSimulation, Sliding) {
  constexpr double kFriction = 0.5;
  constexpr double kEffortLimit = 2.0 * kFriction;
  constexpr double kTimeStep = 1.0e-3;
  Simulation sim = MakeSimulation(
      {.xml = kWheelXml,
       .dry_friction = {{"wheel", kFriction}},
       .actuation = {{"wheel", {kEffortLimit, 3.0 * kEffortLimit}}},
       .cenic_step = kTimeStep});
  constexpr int kNumSteps = 200;
  sim.simulator->AdvanceTo(kNumSteps * kTimeStep);
  const double expected_rate =
      kNumSteps * kTimeStep * (kEffortLimit - kFriction) / kWheelInertia;
  EXPECT_NEAR(sim.rate("wheel"), expected_rate, 1.0e-10 * expected_rate);
}

// A spinning wheel decelerates uniformly under friction, comes to rest and does
// not reverse.
GTEST_TEST(JointFrictionSimulation, StoppingUnderFriction) {
  constexpr double kFriction = 0.5;
  constexpr double kInitialRate = 2.0;
  constexpr double kTimeStep = 1.0e-3;
  for (const bool error_control : {false, true}) {
    SCOPED_TRACE(fmt::format("error_control = {}", error_control));
    Simulation sim =
        MakeSimulation({.xml = kWheelXml,
                        .dry_friction = {{"wheel", kFriction}},
                        .cenic_step = error_control ? 0.1 : kTimeStep,
                        .error_control = error_control});
    Context<double>& context = sim.simulator->get_mutable_context();
    sim.plant->GetJointByName<RevoluteJoint>("wheel").set_angular_rate(
        &sim.plant->GetMyMutableContextFromRoot(&context), kInitialRate);

    const double deceleration = kFriction / kWheelInertia;
    const double stopping_time = kInitialRate / deceleration;
    const double stopping_angle =
        0.5 * kInitialRate * kInitialRate / deceleration;
    sim.simulator->AdvanceTo(0.5 * stopping_time);
    EXPECT_NEAR(sim.rate("wheel"), 0.5 * kInitialRate, 1.0e-8 * kInitialRate);

    sim.simulator->AdvanceTo(2.0 * stopping_time);
    EXPECT_GE(sim.rate("wheel"), 0.0);
    EXPECT_LT(sim.rate("wheel"), 1.0e-10);
    // The kinetic energy is dissipated over the stopping angle, up to the first
    // order error of the steps. Error control takes larger steps.
    EXPECT_NEAR(
        sim.angle("wheel"), stopping_angle,
        error_control ? 5.0e-2 * stopping_angle : kInitialRate * kTimeStep);
  }
}

// Returns the largest absolute differences between the positions and between
// the velocities of `sim1` and `sim2`, sampled every `sample_period` until
// `duration`.
std::pair<double, double> CompareTrajectories(Simulation* sim1,
                                              Simulation* sim2,
                                              double time_step,
                                              double sample_period,
                                              double duration) {
  const int steps_per_sample = std::round(sample_period / time_step);
  const int num_samples = std::round(duration / sample_period);
  double max_q_error = 0.0;
  double max_v_error = 0.0;
  for (int k = 1; k <= num_samples; ++k) {
    const double t = k * steps_per_sample * time_step;
    sim1->simulator->AdvanceTo(t);
    sim2->simulator->AdvanceTo(t);
    max_q_error = std::max(
        max_q_error,
        (sim1->positions() - sim2->positions()).lpNorm<Eigen::Infinity>());
    max_v_error = std::max(
        max_v_error,
        (sim1->velocities() - sim2->velocities()).lpNorm<Eigen::Infinity>());
  }
  return {max_q_error, max_v_error};
}

// A pendulum released above the angle at which friction can hold it swings,
// and sticks at an angle where the gravity torque is below the friction bound.
// A fixed step CENIC simulation agrees with a discrete SAP plant at the same
// time step, through the transitions between sliding and stiction.
GTEST_TEST(JointFrictionSimulation, PendulumMatchesSap) {
  constexpr double kFriction = 2.0;
  for (const double time_step : {1.0e-3, 1.0e-2}) {
    SCOPED_TRACE(fmt::format("time_step = {}", time_step));
    const SimulationOptions options{.xml = kPendulumXml,
                                    .dry_friction = {{"joint", kFriction}},
                                    .initial_angle = {{"joint", 1.2}},
                                    .cenic_step = time_step};
    Simulation cenic = MakeSimulation(options);
    SimulationOptions sap_options = options;
    sap_options.time_step = time_step;
    Simulation sap = MakeSimulation(sap_options);

    const auto [q_error, v_error] =
        CompareTrajectories(&cenic, &sap, time_step, 0.05, 3.0);
    EXPECT_LT(q_error, 1.0e-6);
    EXPECT_LT(v_error, 1.0e-5);

    // The pendulum is at rest, where friction can hold it.
    const double q = cenic.angle("joint");
    EXPECT_LT(std::abs(cenic.rate("joint")), 1.0e-4);
    EXPECT_LT(kMass * kGravity * kLength * std::abs(std::sin(q)), kFriction);
    // It did move from its initial angle.
    EXPECT_LT(std::abs(q), 1.0);
  }
}

// An error controlled CENIC simulation keeps a pendulum in stiction when
// gravity cannot overcome friction, and brings a released pendulum to rest at
// about the same angle as a discrete SAP plant with a small time step.
GTEST_TEST(JointFrictionSimulation, PendulumErrorControlled) {
  constexpr double kFriction = 2.0;
  constexpr double kMaxStep = 0.1;
  // In stiction the joint creeps at most at σ⋅w⋅δt⋅τ_c, with δt at most the
  // maximum step size, and w = 1/(I + δt⋅d).
  const double max_creep = kSigma * kMaxStep * kFriction / kInertia;
  {
    SCOPED_TRACE("stiction");
    constexpr double kInitialAngle = 0.3;
    const double gravity_torque =
        kMass * kGravity * kLength * std::sin(kInitialAngle);
    ASSERT_LT(gravity_torque, kFriction);
    Simulation cenic =
        MakeSimulation({.xml = kPendulumXml,
                        .dry_friction = {{"joint", kFriction}},
                        .initial_angle = {{"joint", kInitialAngle}},
                        .cenic_step = kMaxStep,
                        .error_control = true});
    constexpr double kDuration = 5.0;
    cenic.simulator->AdvanceTo(kDuration);
    const double creep = kSigma * kMaxStep * gravity_torque / kInertia;
    EXPECT_LT(std::abs(cenic.rate("joint")), creep);
    EXPECT_LT(std::abs(cenic.angle("joint") - kInitialAngle),
              kDuration * creep);
  }
  {
    SCOPED_TRACE("release");
    const SimulationOptions options{.xml = kPendulumXml,
                                    .dry_friction = {{"joint", kFriction}},
                                    .initial_angle = {{"joint", 1.2}},
                                    .cenic_step = kMaxStep,
                                    .error_control = true};
    Simulation cenic = MakeSimulation(options);
    SimulationOptions sap_options = options;
    sap_options.time_step = 1.0e-4;
    Simulation sap = MakeSimulation(sap_options);
    cenic.simulator->AdvanceTo(3.0);
    sap.simulator->AdvanceTo(3.0);
    EXPECT_LT(std::abs(cenic.rate("joint")), max_creep);
    EXPECT_NEAR(cenic.angle("joint"), sap.angle("joint"), 1.0e-2);
  }
}

// Friction on a locked joint has no effect, and friction on the other joints of
// the same tree still agrees with SAP. ICF removes the locked joint and its
// friction constraint with IcfModel::ReduceInto().
GTEST_TEST(JointFrictionSimulation, LockedJoint) {
  constexpr double kTimeStep = 1.0e-3;
  const SimulationOptions options{
      .xml = kDoublePendulumXml,
      .dry_friction = {{"joint1", 50.0}, {"joint2", 1.0}},
      .initial_angle = {{"joint1", 0.5}, {"joint2", 1.0}},
      .locked_joints = {"joint1"},
      .cenic_step = kTimeStep};
  Simulation cenic = MakeSimulation(options);
  SimulationOptions sap_options = options;
  sap_options.time_step = kTimeStep;
  Simulation sap = MakeSimulation(sap_options);

  const auto [q_error, v_error] =
      CompareTrajectories(&cenic, &sap, kTimeStep, 0.05, 2.0);
  EXPECT_LT(q_error, 1.0e-6);
  EXPECT_LT(v_error, 1.0e-5);
  EXPECT_EQ(cenic.angle("joint1"), 0.5);
  EXPECT_EQ(cenic.rate("joint1"), 0.0);
  // joint2 moved and came to rest.
  EXPECT_LT(cenic.angle("joint2"), 0.9);
  EXPECT_LT(std::abs(cenic.rate("joint2")), 1.0e-4);
}

// A robot-like arm with friction on all its joints. Friction holds the shoulder
// and the elbow against gravity, but not the wrist. The wrist falls until its
// tip rests on the ground. A free box falls next to the arm. Fixed step CENIC
// agrees with a discrete plant at the same time step, and error controlled
// CENIC comes to rest at about the same configuration.
GTEST_TEST(JointFrictionSimulation, ArmWithContact) {
  constexpr double kTimeStep = 1.0e-3;
  const SimulationOptions options{
      .xml = kArmXml,
      .dry_friction = {{"shoulder", 15.0}, {"elbow", 5.0}, {"wrist", 0.2}},
      .cenic_step = kTimeStep,
      .discrete_contact_approximation = "lagged"};
  Simulation cenic = MakeSimulation(options);
  SimulationOptions sap_options = options;
  sap_options.time_step = kTimeStep;
  Simulation sap = MakeSimulation(sap_options);
  SimulationOptions error_control_options = options;
  error_control_options.cenic_step = 0.1;
  error_control_options.error_control = true;
  Simulation cenic_error_control = MakeSimulation(error_control_options);

  const auto [q_error, v_error] =
      CompareTrajectories(&cenic, &sap, kTimeStep, 0.05, 2.0);
  EXPECT_LT(q_error, 1.0e-3);
  EXPECT_LT(v_error, 1.0e-2);
  cenic_error_control.simulator->AdvanceTo(2.0);

  // The wrist angle at which the tip touches the ground.
  const double touchdown_angle =
      std::asin((kArmHeight - kTipRadius) / kWristLength);
  for (const Simulation* sim : {&cenic, &cenic_error_control}) {
    // The shoulder and the elbow stick, and the wrist rests with its tip on
    // the ground.
    EXPECT_NEAR(sim->angle("shoulder"), 0.0, 1.0e-3);
    EXPECT_NEAR(sim->angle("elbow"), 0.0, 1.0e-3);
    EXPECT_NEAR(sim->angle("wrist"), touchdown_angle, 1.0e-2);
    for (const char* joint : {"shoulder", "elbow", "wrist"}) {
      SCOPED_TRACE(joint);
      EXPECT_LT(std::abs(sim->rate(joint)), 1.0e-3);
    }
  }
}

}  // namespace
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
