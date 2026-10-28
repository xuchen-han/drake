#include "drake/multibody/contact_solvers/icf/icf_external_systems_linearizer.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/controllers/pid_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/matrix_gain.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace internal {

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RollPitchYawd;
using systems::BasicVector;
using systems::ConstantVectorSource;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::FirstOrderTaylorApproximation;
using systems::MatrixGain;
using systems::controllers::PidController;

class LinearizerTestFixture {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LinearizerTestFixture);

  LinearizerTestFixture() = default;
  virtual ~LinearizerTestFixture() = default;

 protected:
  // Builds the diagram and creates a context.
  // The plant must already have been finalized.
  void Build() {
    diagram_ = builder_->Build();
    builder_.reset();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &plant_.GetMyMutableContextFromRoot(diagram_context_.get());
  }

  // A convenience wrapper around the DUT's LinearizeExternalSystem method that
  // uses optional return values instead of output arguments.
  struct Result {
    std::optional<IcfLinearFeedbackGains<double>> actuation_feedback;
    std::optional<IcfLinearFeedbackGains<double>> external_feedback;
  };
  Result LinearizeExternalSystem(double h) const {
    DRAKE_DEMAND(plant_context_ != nullptr);
    const IcfExternalSystemsLinearizer<double> dut(&plant_);
    const int nv = plant_.num_velocities();
    // The outputs start with NaN, so a test fails if the DUT does not write
    // every entry.
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    Result result;
    result.actuation_feedback.emplace();
    result.actuation_feedback->Resize(nv);
    result.actuation_feedback->K.setConstant(kNaN);
    result.actuation_feedback->b.setConstant(kNaN);
    result.external_feedback.emplace();
    result.external_feedback->Resize(nv);
    result.external_feedback->K.setConstant(kNaN);
    result.external_feedback->b.setConstant(kNaN);
    bool has_actuation_forces{};
    bool has_feedback_forces{};
    std::tie(has_actuation_forces, has_feedback_forces) =
        dut.LinearizeExternalSystem(h, *plant_context_,
                                    &result.actuation_feedback.value(),
                                    &result.external_feedback.value());
    if (!has_actuation_forces) {
      result.actuation_feedback.reset();
    }
    if (!has_feedback_forces) {
      result.external_feedback.reset();
    }
    return result;
  }

  // Only available prior to calling Build().
  std::unique_ptr<DiagramBuilder<double>> builder_{
      std::make_unique<DiagramBuilder<double>>()};

  // Always available.
  MultibodyPlant<double>& plant_{
      AddMultibodyPlantSceneGraph(builder_.get(), 0.0).plant};

  // Only available after calling Build().
  std::unique_ptr<Diagram<double>> diagram_;
  std::unique_ptr<Context<double>> diagram_context_;
  Context<double>* plant_context_{};
};

class IcfExternalSystemsLinearizerTest : public ::testing::Test,
                                         public LinearizerTestFixture {};

// Checks the short-circuit logic for a plant with no external input.
TEST_F(IcfExternalSystemsLinearizerTest, NoFeedback) {
  // Build a diagram with just the plant and scene graph (no controller).
  plant_.Finalize();
  Build();

  // No feedback.
  const auto result = LinearizeExternalSystem(/* h = */ 0.01);
  EXPECT_FALSE(result.actuation_feedback.has_value());
  EXPECT_FALSE(result.external_feedback.has_value());
}

// MJCF model of an actuated double pendulum.
constexpr char kActuatedPendulumXml[] = R"""(
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <body>
      <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
      <geom type="capsule" size="0.01 0.1"/>
      <body>
        <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 -0.1"/>
        <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" ctrlrange="-2 2"/>
    <motor joint="joint2" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
)""";

// A feedback "controller" that takes the plant's state_output_port as input and
// produces applied spatial forces as output. It assumes a 1-dof plant -- a ball
// with a prismatic joint to the world moving on the +x axis.
class SpatialForceFeedback final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpatialForceFeedback);

  static constexpr double Kp = 5;    // [N/m]
  static constexpr double Kd = 0.3;  // [Ns/m]

  explicit SpatialForceFeedback(BodyIndex body_index)
      : body_index_(body_index) {
    const int nq = 1;
    const int nv = 1;
    DeclareVectorInputPort("in", nq + nv);
    DeclareAbstractOutputPort("out", &SpatialForceFeedback::CalcOutput);
  }

 private:
  void CalcOutput(
      const Context<double>& context,
      std::vector<ExternallyAppliedSpatialForce<double>>* output) const {
    // Initialize the output.
    output->resize(1);
    ExternallyAppliedSpatialForce<double>& result = output->at(0);

    // Our input is connected to the the plant's state_output_port.
    auto x = get_input_port().Eval(context);
    const double q = x(0);
    const double v = x(1);

    // The feedback term is f = -(Kp q + Kd v).
    const double f = -(Kp * q + Kd * v);
    result.body_index = body_index_;
    result.p_BoBq_B = Vector3d::Zero();
    result.F_Bq_W = SpatialForce<double>(/* tau = */ Vector3d::Zero(),
                                         /* f = */ f * Vector3d::UnitX());
  }

  const BodyIndex body_index_;
};

// Run tests of the plant's applied_spatial_force input port.
TEST_F(IcfExternalSystemsLinearizerTest, ExternalSpatialForce) {
  const RigidBody<double>& ball =
      plant_.AddRigidBody("ball", default_model_instance(),
                          SpatialInertia<double>::SolidSphereWithMass(
                              /* mass = */ 1.0, /* radius = */ 0.01));
  plant_.AddJoint<PrismaticJoint>("prismatic", plant_.world_body(), {}, ball,
                                  {}, Vector3d::UnitX());

  plant_.Finalize();
  auto controller = builder_->AddSystem<SpatialForceFeedback>(ball.index());
  builder_->Connect(plant_.get_state_output_port(),
                    controller->get_input_port());
  builder_->Connect(controller->get_output_port(),
                    plant_.get_applied_spatial_force_input_port());
  Build();

  // Set an interesting initial state.
  const double q0 = 0.1;
  const double v0 = 0.2;
  plant_.SetPositions(plant_context_, Vector1d{q0});
  plant_.SetVelocities(plant_context_, Vector1d{v0});

  // Linearize the non-plant dynamics around the current state.
  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  ASSERT_TRUE(result.external_feedback.has_value());
  EXPECT_FALSE(result.actuation_feedback.has_value());
  const VectorXd& K = result.external_feedback->K;
  const VectorXd& b = result.external_feedback->b;

  // Check that the feedback was linearized as expected.
  // K = -df/dv  = -d/dv(-(Kp*(q0 + h*v) + Kd*v)) = Kd + h*Kp
  const Vector1d K_expected{SpatialForceFeedback::Kd +
                            h * SpatialForceFeedback::Kp};
  // b = f(q0 + h*v0, v0) + K*v0 = -(Kp*(q0 + h*v0) + Kd*v0) + K*v0
  const Vector1d b_expected{(-SpatialForceFeedback::Kp * (q0 + h * v0) -
                             SpatialForceFeedback::Kd * v0) +
                            K_expected(0) * v0};
  EXPECT_TRUE(CompareMatrices(K, K_expected, 1e-8));
  EXPECT_TRUE(CompareMatrices(b, b_expected, 1e-8));
}

// A feedback "controller" that takes the plant's state_output_port as input and
// produces applied spatial forces as output. It assumes a plant with a
// quaternion floating joint.
class QuaternionSpatialForceFeedback final
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuaternionSpatialForceFeedback);

  explicit QuaternionSpatialForceFeedback(BodyIndex body_index)
      : body_index_(body_index) {
    const int nq = 7;
    const int nv = 6;
    DeclareVectorInputPort("in", nq + nv);
    DeclareAbstractOutputPort("out",
                              &QuaternionSpatialForceFeedback::CalcOutput);
  }

  static Vector6d Kd() { return Vector6d::LinSpaced(0.95, 1.05); }

 private:
  void CalcOutput(
      const Context<double>& context,
      std::vector<ExternallyAppliedSpatialForce<double>>* output) const {
    // Initialize the output.
    output->resize(1);
    ExternallyAppliedSpatialForce<double>& result = output->at(0);

    // Our input is connected to the the plant's state_output_port.
    const Vector6d v = get_input_port().Eval(context).segment(7, 6);

    // The feedback term is f = -(Kd v).
    const Vector6d f = -Kd().array() * v.array();

    result.body_index = body_index_;
    result.p_BoBq_B = Vector3d::Zero();
    result.F_Bq_W = SpatialForce<double>(f);
  }

  const BodyIndex body_index_;
};

// Use a plant where qdot != v. We'll simply check for non-crashing (i.e., that
// the Eigen dimensions matched at runtime) and that Kd is recovered. We don't
// check the details of the the mapping math, assuming that so long as the sizes
// match the implementation is correctly formulated.
TEST_F(IcfExternalSystemsLinearizerTest, FloatingBall) {
  const RigidBody<double>& ball =
      plant_.AddRigidBody("ball", default_model_instance(),
                          SpatialInertia<double>::SolidSphereWithMass(
                              /* mass = */ 1.0, /* radius = */ 0.01));
  plant_.Finalize();
  auto controller =
      builder_->AddSystem<QuaternionSpatialForceFeedback>(ball.index());
  builder_->Connect(plant_.get_state_output_port(),
                    controller->get_input_port());
  builder_->Connect(controller->get_output_port(),
                    plant_.get_applied_spatial_force_input_port());
  Build();

  // Set an arbitrary non-zero initial state.
  plant_.SetFloatingBaseBodyPoseInWorldFrame(
      plant_context_, ball,
      RigidTransformd(RollPitchYawd(Vector3d::LinSpaced(-1, 3)),
                      Vector3d::LinSpaced(22, 33)));
  plant_.SetFreeBodySpatialVelocity(
      plant_context_, ball,
      SpatialVelocity<double>(Vector6d::LinSpaced(-10, 10)));

  // Linearize the non-plant dynamics around the current state.
  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  ASSERT_TRUE(result.external_feedback.has_value());
  EXPECT_FALSE(result.actuation_feedback.has_value());
  const VectorXd& K = result.external_feedback->K;
  const VectorXd& b = result.external_feedback->b;

  // Check that the feedback was linearized as expected. The forward differences
  // step size in the implementation is approximately sqrt(eps), so we'll allow
  // 10x roundoff error in this sanity check.
  const Vector6d K_expected = QuaternionSpatialForceFeedback::Kd();
  const Vector6d b_expected = Vector6d::Zero();
  const double tol = 10 * std::sqrt(std::numeric_limits<double>::epsilon());
  EXPECT_TRUE(CompareMatrices(K, K_expected, tol));
  EXPECT_TRUE(CompareMatrices(b, b_expected, tol));
}

// Test base for switching between actuation feedback and external feedback.
// When `choose_plant_port == 0`, MbP actuation input is used.
// When `choose_plant_port == 1`, MbP applied generalized force input is used.
class IcfExternalSystemsLinearizerChoosePortTest
    : public LinearizerTestFixture,
      public ::testing::TestWithParam<int /* choose_plant_port */> {};

// Checks linearization of the desired state input port.
TEST_F(IcfExternalSystemsLinearizerTest, DesiredStateInput) {
  // Add a pendulum.
  const ModelInstanceIndex model_instance =
      Parser(builder_.get())
          .AddModelsFromString(kActuatedPendulumXml, "xml")
          .at(0);
  plant_.Finalize();
  Build();

  // Configure its current state.
  const Vector2d q0{0.1, 0.2};
  const Vector2d v0{0.3, 0.4};
  plant_.SetPositions(plant_context_, q0);
  plant_.SetVelocities(plant_context_, v0);

  // Configure its desired state.
  const Vector2d qd{0.11, 0.22};
  const Vector2d vd{0.33, 0.44};
  plant_.get_desired_state_input_port(model_instance)
      .FixValue(plant_context_, (Vector4d() << qd, vd).finished());
  const double Kp = 0.5;
  const double Kd = 0.25;
  for (const JointActuatorIndex& actuator :
       plant_.GetJointActuatorIndices(model_instance)) {
    plant_.get_mutable_joint_actuator(actuator).set_controller_gains(
        {.p = Kp, .d = Kd});
  }

  // Linearize the implicit PD controller actuation around the current state.
  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  ASSERT_TRUE(result.actuation_feedback.has_value());
  EXPECT_FALSE(result.external_feedback.has_value());

  // Check that the desired state feedback came out as expected. Unfortunately
  // there's not much we can do here other than copy-paste the same formula as
  // is performed inside the implementation.
  Vector2d K_expected = Vector2d::Constant(Kp * h + Kd);
  Vector2d b_expected = -Kp * (q0 - qd) - Kd * vd;
  EXPECT_TRUE(CompareMatrices(result.actuation_feedback->K, K_expected, 1e-8));
  EXPECT_TRUE(CompareMatrices(result.actuation_feedback->b, b_expected, 1e-8));

  // Check that feedforward terms are linearized in combination with desired
  // state (i.e., that desired state doesn't accidentally overwrite them).
  // We'll set a constant actuation and confirm it shows up in the `b` term
  // (with `K` unchanged).
  const Vector2d u = Vector2d{22, 23};
  plant_.get_actuation_input_port(model_instance).FixValue(plant_context_, u);
  const auto with_ff = LinearizeExternalSystem(h);
  ASSERT_TRUE(with_ff.actuation_feedback.has_value());
  EXPECT_FALSE(with_ff.external_feedback.has_value());
  EXPECT_TRUE(CompareMatrices(with_ff.actuation_feedback->K, K_expected, 1e-8));
  EXPECT_TRUE(
      CompareMatrices(with_ff.actuation_feedback->b, b_expected + u, 1e-8));
}

// Run a test with a pendulum and an external (PID) controller.
TEST_P(IcfExternalSystemsLinearizerChoosePortTest, ActuationInput) {
  const int choose_plant_port = GetParam();

  // Build a diagram containing a double pendulum and PID controller.
  Parser(builder_.get()).AddModelsFromString(kActuatedPendulumXml, "xml");
  if (choose_plant_port == 1) {
    // We'll be using generalized force input, not actuation.
    plant_.RemoveJointActuator(
        plant_.get_joint_actuator(JointActuatorIndex(1)));
    plant_.RemoveJointActuator(
        plant_.get_joint_actuator(JointActuatorIndex(0)));
  }
  plant_.Finalize();
  auto target_source = builder_->AddSystem<ConstantVectorSource<double>>(
      Vector4d{M_PI_2, M_PI_2, 0.0, 0.0});
  auto pid_controller = builder_->AddSystem<PidController>(
      Vector2d{0.24, 0.19}, Vector2d{0.35, 0.3}, Vector2d::Zero());
  builder_->Connect(target_source->get_output_port(),
                    pid_controller->get_input_port_desired_state());
  builder_->Connect(plant_.get_state_output_port(),
                    pid_controller->get_input_port_estimated_state());
  if (choose_plant_port == 0) {
    builder_->Connect(pid_controller->get_output_port(),
                      plant_.get_actuation_input_port());
  } else {
    DRAKE_DEMAND(choose_plant_port == 1);
    builder_->Connect(pid_controller->get_output_port(),
                      plant_.get_applied_generalized_force_input_port());
  }

  Build();
  const Context<double>& pid_controller_context =
      pid_controller->GetMyContextFromRoot(*diagram_context_);

  // Set an interesting initial state.
  plant_.SetPositions(plant_context_, Vector2d{0.1, 0.2});
  plant_.SetVelocities(plant_context_, Vector2d{0.3, 0.4});

  // Linearize the non-plant dynamics around the current state.
  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  const std::optional<IcfLinearFeedbackGains<double>>& relevant_feedback =
      (choose_plant_port == 0) ? result.actuation_feedback
                               : result.external_feedback;
  const std::optional<IcfLinearFeedbackGains<double>>& empty_feedback =
      (choose_plant_port == 0) ? result.external_feedback
                               : result.actuation_feedback;
  ASSERT_TRUE(relevant_feedback.has_value());
  EXPECT_FALSE(empty_feedback.has_value());
  const VectorXd& K = relevant_feedback->K;
  const VectorXd& b = relevant_feedback->b;

  // Compute linearization τ̃ = D⋅x + y around x₀ via autodiff.
  auto expected_linearization = FirstOrderTaylorApproximation(
      *pid_controller, pid_controller_context,
      pid_controller->get_input_port_estimated_state().get_index(),
      pid_controller->get_output_port().get_index());
  const MatrixXd& D = expected_linearization->D();
  const VectorXd& y = expected_linearization->y0();
  const MatrixXd dtau_dq = D.leftCols(2);
  const MatrixXd dtau_dv = D.rightCols(2);
  const MatrixXd dtau_tilde_dv = dtau_dv + h * dtau_dq;  // N(q) = I

  // Since the linearization is exact for a PID controller, we can compute
  // τ(x̃₀) = D⋅x̃₀ + y.
  const int nq = plant_.num_positions();
  const int nv = plant_.num_velocities();
  const VectorXd x0 = plant_.GetPositionsAndVelocities(*plant_context_);
  const VectorXd v0 = x0.tail(nv);
  VectorXd x_tilde0 = x0;
  x_tilde0.head(nq) += h * v0;
  const VectorXd tau_tilde0 = y + D * x_tilde0;

  // And thus the expected linearization τ̃(v) = b - K⋅v is:
  const VectorXd K_ref = -dtau_tilde_dv.diagonal();
  const VectorXd b_ref = tau_tilde0 + K_ref.asDiagonal() * v0;

  // Confirm that our finite difference linearization is close to the reference.
  const double kTol = std::sqrt(std::numeric_limits<double>::epsilon());
  EXPECT_TRUE(CompareMatrices(K, K_ref, kTol, MatrixCompareType::relative));
  EXPECT_TRUE(CompareMatrices(b, b_ref, kTol, MatrixCompareType::relative));
}

// A bang-bang "controller" (with no hysteresis) that takes the plant's
// state_output_port as input and produces applied spatial forces as output. It
// assumes a 1-dof plant -- a ball with a prismatic joint to the world moving
// on the +x axis. This controller is designed to force the linearizer to use
// its explicit approximation.
class BangBang final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BangBang);

  static constexpr double kDesired = 22;
  static constexpr double kGain = 0.5;

  explicit BangBang(BodyIndex body_index) : body_index_(body_index) {
    const int nq = 1;
    const int nv = 1;
    DeclareVectorInputPort("in", nq + nv);
    DeclareVectorOutputPort("vector_out", 1, &BangBang::CalcVectorOutput);
    DeclareAbstractOutputPort("spatial_force_out",
                              &BangBang::CalcSpatialOutput);
  }

 private:
  double CalcDoubleOutput(const Context<double>& context) const {
    // Our input is connected to the the plant's state_output_port.
    auto q_v = get_input_port().Eval(context);
    const double q = q_v(0);

    // The feedback term is f = kGain * signum(kDesired - q).
    auto signum = [](double x) {
      return (0.0 < x) - (x < 0.0);
    };
    const double f_x = kGain * signum(kDesired - q);
    return f_x;
  }

  void CalcVectorOutput(const Context<double>& context,
                        BasicVector<double>* output) const {
    output->SetAtIndex(0, CalcDoubleOutput(context));
  }

  void CalcSpatialOutput(
      const Context<double>& context,
      std::vector<ExternallyAppliedSpatialForce<double>>* output) const {
    // Initialize the output.
    output->resize(1);
    ExternallyAppliedSpatialForce<double>& result = output->at(0);

    const double f_x = CalcDoubleOutput(context);
    result.body_index = body_index_;
    result.p_BoBq_B = Vector3d::Zero();
    result.F_Bq_W = SpatialForce<double>(/* tau = */ Vector3d::Zero(),
                                         /* f = */ f_x * Vector3d::UnitX());
  }

  const BodyIndex body_index_;
};

// Test linearization with a controller to force explicit approximations.  The
// explicit approximation is used when dτᵢ/dvᵢ(x̃₀) < 0.
TEST_P(IcfExternalSystemsLinearizerChoosePortTest, BangBang) {
  const int choose_plant_port = GetParam();

  const RigidBody<double>& ball =
      plant_.AddRigidBody("ball", default_model_instance(),
                          SpatialInertia<double>::SolidSphereWithMass(
                              /* mass = */ 1.0, /* radius = */ 0.01));
  const auto& joint = plant_.AddJoint<PrismaticJoint>(
      "prismatic", plant_.world_body(), {}, ball, {}, Vector3d::UnitX());
  if (choose_plant_port == 0) {
    plant_.AddJointActuator("actuator", joint);
  }

  plant_.Finalize();
  auto controller = builder_->AddSystem<BangBang>(ball.index());
  builder_->Connect(plant_.get_state_output_port(),
                    controller->get_input_port());
  if (choose_plant_port == 0) {
    builder_->Connect(controller->GetOutputPort("vector_out"),
                      plant_.get_actuation_input_port());
  } else {
    DRAKE_DEMAND(choose_plant_port == 1);
    builder_->Connect(controller->GetOutputPort("spatial_force_out"),
                      plant_.get_applied_spatial_force_input_port());
  }
  Build();

  // Set an interesting initial state.
  const double q = 21.9;
  const double v = 0.5;
  plant_.SetPositions(plant_context_, Vector1d{q});
  plant_.SetVelocities(plant_context_, Vector1d{v});

  // Linearize the non-plant dynamics around the current state. Note that the
  // result checks below are exact, rather than within a tolerance. Here's why:
  // * K: This case is testing the "explicit" codepath in the linearizer that
  //   sets K = 0.0. Therefore testing this and expecting anything other than
  //   exactly 0 would be confusing.
  // * b: The contract of the linearizer for the explicit codepath is
  //     b = tau(q0, x0) (notice tau and not tau_tilde).
  //   The controller that is used here computes (at this configuration)
  //     tau = kGain * signum(kDesired - q0)
  //   where kDesired = 22 and q0 = 21.9.
  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  if (choose_plant_port == 0) {
    ASSERT_TRUE(result.actuation_feedback.has_value());
    EXPECT_FALSE(result.external_feedback.has_value());
    const VectorXd& K = result.actuation_feedback->K;
    const VectorXd& b = result.actuation_feedback->b;

    ASSERT_EQ(K.size(), 1);
    ASSERT_EQ(b.size(), 1);

    // Check that the feedback was linearized as expected.
    const double K_expected{0.0};
    const double b_expected{BangBang::kGain};
    EXPECT_EQ(K(0), K_expected);
    EXPECT_EQ(b(0), b_expected);
  } else {
    ASSERT_TRUE(result.external_feedback.has_value());
    EXPECT_FALSE(result.actuation_feedback.has_value());
    const VectorXd& K = result.external_feedback->K;
    const VectorXd& b = result.external_feedback->b;

    ASSERT_EQ(K.size(), 1);
    ASSERT_EQ(b.size(), 1);

    // Check that the feedback was linearized as expected.
    const double K_expected{0.0};
    const double b_expected{BangBang::kGain};
    EXPECT_EQ(K(0), K_expected);
    EXPECT_EQ(b(0), b_expected);
  }
}

INSTANTIATE_TEST_SUITE_P(AllPorts, IcfExternalSystemsLinearizerChoosePortTest,
                         ::testing::Values(0, 1));

// MJCF model of an actuated double pendulum on a floating base. The floating
// base comes first in the tree. Therefore, the position index of each pendulum
// joint is not equal to the velocity index of the same joint.
constexpr char kFloatingBasePendulumXml[] = R"""(
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
      <body>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
        <geom type="capsule" size="0.01 0.1"/>
        <body>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 -0.1"/>
          <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1"/>
    <motor joint="joint2"/>
  </actuator>
</mujoco>
)""";

// A joint impedance "controller" that takes the plant's state_output_port as
// input and produces actuation for joint1 and joint2 of
// kFloatingBasePendulumXml. The stiffness Kp is a full matrix, so the actuation
// of each joint depends on the position of the other joint. Robot controllers
// with gravity compensation have the same coupling.
class CoupledPdController final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CoupledPdController);

  explicit CoupledPdController(const MultibodyPlant<double>& plant) {
    const Joint<double>& joint1 = plant.GetJointByName("joint1");
    const Joint<double>& joint2 = plant.GetJointByName("joint2");
    const int nq = plant.num_positions();
    q_indices_ = {joint1.position_start(), joint2.position_start()};
    v_indices_ = {nq + joint1.velocity_start(), nq + joint2.velocity_start()};
    DeclareVectorInputPort("in", plant.num_multibody_states());
    DeclareVectorOutputPort("out", 2, &CoupledPdController::CalcOutput);
  }

  static Matrix2d Kp() { return (Matrix2d() << 10, 5, 5, 8).finished(); }
  static Vector2d Kd() { return Vector2d{0.2, 0.3}; }

  // Returns the number of times that the output was calculated.
  int num_evaluations() const { return num_evaluations_; }

 private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* output) const {
    ++num_evaluations_;
    // Our input is connected to the plant's state_output_port.
    const VectorXd& x = get_input_port().Eval(context);
    const Vector2d q{x(q_indices_[0]), x(q_indices_[1])};
    const Vector2d v{x(v_indices_[0]), x(v_indices_[1])};
    // The feedback term is u = -(Kp q + Kd v).
    output->SetFromVector(-(Kp() * q + Kd().cwiseProduct(v)));
  }

  std::array<int, 2> q_indices_{};
  std::array<int, 2> v_indices_{};
  mutable int num_evaluations_{0};
};

// Tests kFloatingBasePendulumXml with a CoupledPdController on the actuation
// input port. When the parameter `add_damping` is true, the applied generalized
// force input port also has a viscous damping τₑ = -kDamping⋅v on all
// velocities.
class FloatingBasePendulumTest : public LinearizerTestFixture,
                                 public ::testing::TestWithParam<bool> {
 protected:
  static constexpr double kDamping = 0.7;

  void SetUp() override {
    const bool add_damping = GetParam();
    Parser(builder_.get()).AddModelsFromString(kFloatingBasePendulumXml, "xml");
    plant_.Finalize();
    controller_ = builder_->AddSystem<CoupledPdController>(plant_);
    builder_->Connect(plant_.get_state_output_port(),
                      controller_->get_input_port());
    builder_->Connect(controller_->get_output_port(),
                      plant_.get_actuation_input_port());
    if (add_damping) {
      const int nq = plant_.num_positions();
      const int nv = plant_.num_velocities();
      MatrixXd D = MatrixXd::Zero(nv, nq + nv);
      D.rightCols(nv) = -kDamping * MatrixXd::Identity(nv, nv);
      auto damping = builder_->AddSystem<MatrixGain<double>>(D);
      builder_->Connect(plant_.get_state_output_port(),
                        damping->get_input_port());
      builder_->Connect(damping->get_output_port(),
                        plant_.get_applied_generalized_force_input_port());
    }
    Build();

    // Set an arbitrary non-zero state.
    const RigidBody<double>& base = plant_.GetBodyByName("base");
    plant_.SetFloatingBaseBodyPoseInWorldFrame(
        plant_context_, base,
        RigidTransformd(RollPitchYawd(0.1, 0.2, 0.3), Vector3d(1, 2, 3)));
    plant_.SetFreeBodySpatialVelocity(
        plant_context_, base,
        SpatialVelocity<double>(Vector6d::LinSpaced(-0.6, 0.6)));
    joint1().set_angle(plant_context_, joint_q0_(0));
    joint2().set_angle(plant_context_, joint_q0_(1));
    joint1().set_angular_rate(plant_context_, joint_v0_(0));
    joint2().set_angular_rate(plant_context_, joint_v0_(1));
  }

  const RevoluteJoint<double>& joint1() const {
    return plant_.GetJointByName<RevoluteJoint>("joint1");
  }
  const RevoluteJoint<double>& joint2() const {
    return plant_.GetJointByName<RevoluteJoint>("joint2");
  }

  const Vector2d joint_q0_{0.1, 0.2};
  const Vector2d joint_v0_{0.3, 0.4};
  CoupledPdController* controller_{};
};

// Checks the gains against the analytic values. The joint positions and the
// joint velocities have different indices, so this test fails if the
// linearizer does not restore every position that it perturbed. In that case,
// the coupled stiffness leaks into the gain of the next joint.
TEST_P(FloatingBasePendulumTest, Gains) {
  const bool add_damping = GetParam();
  ASSERT_NE(joint1().position_start(), joint1().velocity_start());
  ASSERT_NE(joint2().position_start(), joint2().velocity_start());

  const double h = 0.01;
  const auto result = LinearizeExternalSystem(h);
  ASSERT_TRUE(result.actuation_feedback.has_value());
  ASSERT_EQ(result.external_feedback.has_value(), add_damping);

  // At x̃₀ = [q₀ + h⋅N₀⋅v₀; v₀], the joint positions are q₀ + h⋅v₀.
  // K = -du/dv = Kd + h⋅diag(Kp), and b = u(x̃₀) + K⋅v₀.
  const Matrix2d Kp = CoupledPdController::Kp();
  const Vector2d Kd = CoupledPdController::Kd();
  const Vector2d u_tilde0 =
      -(Kp * (joint_q0_ + h * joint_v0_) + Kd.cwiseProduct(joint_v0_));
  const Vector2d K_joints = Kd + h * Kp.diagonal();
  const Vector2d b_joints = u_tilde0 + K_joints.cwiseProduct(joint_v0_);

  // The base has no actuators, so K = 0 and b = 0 for the base velocities.
  const int nv = plant_.num_velocities();
  VectorXd K_expected = VectorXd::Zero(nv);
  VectorXd b_expected = VectorXd::Zero(nv);
  const std::array<int, 2> joint_v = {joint1().velocity_start(),
                                      joint2().velocity_start()};
  for (int j = 0; j < 2; ++j) {
    K_expected(joint_v[j]) = K_joints(j);
    b_expected(joint_v[j]) = b_joints(j);
  }

  // Forward differences of an affine function are exact, except for
  // round-off error.
  const double kTol = 1e-6;
  EXPECT_TRUE(CompareMatrices(result.actuation_feedback->K, K_expected, kTol));
  EXPECT_TRUE(CompareMatrices(result.actuation_feedback->b, b_expected, kTol));
  if (add_damping) {
    // τₑ = -kDamping⋅v, so K = kDamping and b = τₑ(x̃₀) + K⋅v₀ = 0.
    EXPECT_TRUE(CompareMatrices(result.external_feedback->K,
                                VectorXd::Constant(nv, kDamping), kTol));
    EXPECT_TRUE(
        CompareMatrices(result.external_feedback->b, VectorXd::Zero(nv), kTol));
  }
}

// Checks the cost of the actuation linearization. The linearizer evaluates the
// actuation at x₀, at x̃₀, and one time for each actuated velocity. The
// floating base has six velocities without actuators, and the linearizer does
// not evaluate the actuation for them.
TEST_P(FloatingBasePendulumTest, ActuationEvaluations) {
  const int num_before = controller_->num_evaluations();
  LinearizeExternalSystem(/* h = */ 0.01);
  EXPECT_EQ(controller_->num_evaluations() - num_before, 2 + 2);
}

INSTANTIATE_TEST_SUITE_P(WithAndWithoutDamping, FloatingBasePendulumTest,
                         ::testing::Bool());

}  // namespace internal
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
