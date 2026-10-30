#include <cmath>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/math/autodiff.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/multibody/contact_solvers/sap/sap_contact_problem.h"
#include "drake/multibody/contact_solvers/sap/sap_joint_friction_constraint.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/sap_driver.h"
#include "drake/multibody/plant/test/compliant_contact_manager_tester.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"

/* @file This file tests SapDriver's support for joint dry friction.

  Constraints are only supported by the SAP solver. Therefore, to exercise the
  relevant code paths, we arbitrarily choose one contact approximation that uses
  the SAP solver. More precisely, in the unit tests below we call
  set_discrete_contact_approximation(DiscreteContactApproximation::kSap) on the
  MultibodyPlant used for testing. */

using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::contact_solvers::internal::SapContactProblem;
using drake::multibody::contact_solvers::internal::SapJointFrictionConstraint;
using drake::systems::Context;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace drake {
namespace multibody {
namespace internal {

// Friend class used to provide access to a selection of private functions in
// SapDriver for testing purposes.
class SapDriverTest {
 public:
  static const ContactProblemCache<double>& EvalContactProblemCache(
      const SapDriver<double>& driver, const Context<double>& context) {
    return driver.EvalContactProblemCache(context);
  }
};

namespace {

// Relative tolerance for comparisons against analytical solutions. For the
// single DOF problems below the SAP cost is piecewise quadratic and Newton
// iterations converge to machine precision within a few iterations.
constexpr double kTolerance = 1.0e-10;

// Model parameters.
constexpr double kWheelInertia = 0.1;    // [kg⋅m²], about the wheel's axis.
constexpr double kWheelFriction = 0.5;   // [N⋅m]
constexpr double kSliderMass = 2.0;      // [kg]
constexpr double kSliderFriction = 1.0;  // [N]
constexpr double kElbowFriction = 0.3;   // [N⋅m]
constexpr double kLinkLength = 0.5;      // [m]

// Writes `dry_friction` into the parameter that stores the dry friction of
// `joint` in `context`, without the checks in Joint::SetDryFrictionVector().
void WriteDryFrictionParameter(const Joint<double>& joint,
                               const VectorXd& dry_friction,
                               Context<double>* context) {
  const VectorXd* storage = &joint.GetDryFrictionVector(*context);
  for (int i = 0; i < context->num_numeric_parameter_groups(); ++i) {
    if (&context->get_numeric_parameter(i).value() == storage) {
      context->get_mutable_numeric_parameter(i).SetFromVector(dry_friction);
      return;
    }
  }
  DRAKE_UNREACHABLE();
}

// Fixture with a model made of three kinematic trees, each with a joint that
// models dry friction:
//   1. A wheel free to spin about the world's z axis. A single revolute DOF.
//   2. A slider free to translate along the world's x axis. A single prismatic
//      DOF.
//   3. A two link arm in the x-y plane with a shoulder and an elbow revolute
//      joint about z. Only the elbow models dry friction. This tree is used to
//      verify the indexing of DOFs within a tree with more than one DOF.
// Gravity is set to zero so that actuation is the only force applied on the
// model besides friction. Therefore the wheel and slider dynamics have simple
// analytical solutions used to verify the implementation.
class SapDriverJointFrictionTest : public ::testing::Test {
 protected:
  void MakePlant(double time_step) {
    plant_ = std::make_unique<MultibodyPlant<double>>(time_step);
    // Only SAP supports the modeling of constraints.
    plant_->set_discrete_contact_approximation(
        DiscreteContactApproximation::kSap);
    plant_->mutable_gravity_field().set_gravity_vector(Vector3d::Zero());

    // Bodies with the same rotational inertia about every axis so that the
    // inertia about the joint's axis is known regardless of orientation.
    const SpatialInertia<double> M_wheel(
        1.0, Vector3d::Zero(),
        UnitInertia<double>::TriaxiallySymmetric(kWheelInertia));
    const SpatialInertia<double> M_slider(
        kSliderMass, Vector3d::Zero(),
        UnitInertia<double>::TriaxiallySymmetric(1.0));
    // The arm's inertial properties play no role in these tests.
    const SpatialInertia<double> M_link = SpatialInertia<double>::MakeUnitary();

    const RigidBody<double>& wheel = plant_->AddRigidBody("wheel", M_wheel);
    wheel_ = &plant_->AddJoint<RevoluteJoint>("wheel", plant_->world_body(),
                                              std::nullopt, wheel, std::nullopt,
                                              Vector3d::UnitZ());
    wheel_actuator_ = &plant_->AddJointActuator("wheel", *wheel_);
    plant_->get_mutable_joint(wheel_->index())
        .set_default_dry_friction_vector(Vector1d(kWheelFriction));

    const RigidBody<double>& slider = plant_->AddRigidBody("slider", M_slider);
    slider_ = &plant_->AddJoint<PrismaticJoint>(
        "slider", plant_->world_body(), std::nullopt, slider, std::nullopt,
        Vector3d::UnitX());
    slider_actuator_ = &plant_->AddJointActuator("slider", *slider_);
    plant_->get_mutable_joint(slider_->index())
        .set_default_dry_friction_vector(Vector1d(kSliderFriction));

    const RigidBody<double>& link1 = plant_->AddRigidBody("link1", M_link);
    const RigidBody<double>& link2 = plant_->AddRigidBody("link2", M_link);
    shoulder_ = &plant_->AddJoint<RevoluteJoint>(
        "shoulder", plant_->world_body(), std::nullopt, link1, std::nullopt,
        Vector3d::UnitZ());
    elbow_ = &plant_->AddJoint<RevoluteJoint>(
        "elbow", link1, math::RigidTransformd(Vector3d(kLinkLength, 0.0, 0.0)),
        link2, std::nullopt, Vector3d::UnitZ());
    plant_->get_mutable_joint(elbow_->index())
        .set_default_dry_friction_vector(Vector1d(kElbowFriction));

    plant_->Finalize();

    auto owned_contact_manager =
        std::make_unique<CompliantContactManager<double>>();
    manager_ = owned_contact_manager.get();
    plant_->SetDiscreteUpdateManager(std::move(owned_contact_manager));

    context_ = plant_->CreateDefaultContext();
    SetActuation(0.0, 0.0);
  }

  const SapDriver<double>& sap_driver() const {
    return CompliantContactManagerTester::sap_driver(*manager_);
  }

  // Returns the joint friction constraints of the problem for the state in
  // `context_`.
  std::vector<const SapJointFrictionConstraint<double>*> EvalConstraints()
      const {
    const SapContactProblem<double>& problem =
        *SapDriverTest::EvalContactProblemCache(sap_driver(), *context_)
             .sap_problem;
    std::vector<const SapJointFrictionConstraint<double>*> constraints;
    for (int i = 0; i < problem.num_constraints(); ++i) {
      const auto* constraint =
          dynamic_cast<const SapJointFrictionConstraint<double>*>(
              &problem.get_constraint(i));
      DRAKE_DEMAND(constraint != nullptr);
      constraints.push_back(constraint);
    }
    return constraints;
  }

  // Fixes the actuation input port so that `wheel_torque` is applied on the
  // wheel and `slider_force` is applied on the slider.
  void SetActuation(double wheel_torque, double slider_force) {
    VectorXd u = VectorXd::Zero(plant_->num_actuated_dofs());
    u(wheel_actuator_->input_start()) = wheel_torque;
    u(slider_actuator_->input_start()) = slider_force;
    plant_->get_actuation_input_port().FixValue(context_.get(), u);
  }

  // Advances the discrete dynamics by `num_steps` time steps, updating the
  // state stored in the context in place. The plant's forced update events
  // perform the same update as its periodic events. Time is not advanced,
  // which is irrelevant for these time invariant models.
  void Step(int num_steps) {
    for (int i = 0; i < num_steps; ++i) {
      plant_->ExecuteForcedEvents(context_.get(), /* publish = */ false);
    }
  }

  // Takes one step with the AutoDiffXd version of the plant from the state and
  // actuation in `context_`, and returns the resulting velocities. The
  // AutoDiffXd plant uses its own default parameters, and `locked_joints` are
  // locked in its context.
  VectorXd StepAutoDiffXd(
      const std::vector<const Joint<double>*>& locked_joints) const {
    std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_ad =
        systems::System<double>::ToAutoDiffXd(*plant_);
    std::unique_ptr<Context<AutoDiffXd>> context_ad =
        plant_ad->CreateDefaultContext();
    for (const Joint<double>* joint : locked_joints) {
      plant_ad->get_joint(joint->index()).Lock(context_ad.get());
    }
    context_ad->SetDiscreteState(VectorX<AutoDiffXd>(
        context_->get_discrete_state_vector().value().cast<AutoDiffXd>()));
    plant_ad->get_actuation_input_port().FixValue(
        context_ad.get(), VectorX<AutoDiffXd>(plant_->get_actuation_input_port()
                                                  .Eval(*context_)
                                                  .cast<AutoDiffXd>()));
    plant_ad->ExecuteForcedEvents(context_ad.get(), /* publish = */ false);
    return math::ExtractValue(plant_ad->GetVelocities(*context_ad));
  }

  // Analytical solution for the velocity of a single DOF with effective inertia
  // `inertia` after one step from rest under a constant applied generalized
  // force `applied_force` above the dry friction bound `friction`. Friction
  // saturates at its bound and opposes motion.
  double ExpectedSlidingVelocityFromRest(double applied_force, double inertia,
                                         double friction) const {
    const double dt = plant_->time_step();
    const double sign = applied_force > 0 ? 1.0 : -1.0;
    return dt * (applied_force - sign * friction) / inertia;
  }

  std::unique_ptr<MultibodyPlant<double>> plant_;
  CompliantContactManager<double>* manager_{nullptr};
  std::unique_ptr<Context<double>> context_;
  const RevoluteJoint<double>* wheel_{nullptr};
  const PrismaticJoint<double>* slider_{nullptr};
  const RevoluteJoint<double>* shoulder_{nullptr};
  const RevoluteJoint<double>* elbow_{nullptr};
  const JointActuator<double>* wheel_actuator_{nullptr};
  const JointActuator<double>* slider_actuator_{nullptr};
};

// Verifies the driver adds one SapJointFrictionConstraint per single DOF joint
// with non-zero dry friction, using the value stored in the context and the
// right indexing within each tree.
TEST_F(SapDriverJointFrictionTest, ConstraintsAreAdded) {
  MakePlant(1.0e-3);
  const double wheel_friction = 2.0 * kWheelFriction;
  wheel_->SetDryFrictionVector(context_.get(), Vector1d(wheel_friction));

  // Trees are numbered in the order their base bodies were added, and the
  // constraints are added in joint index order.
  struct ExpectedConstraint {
    int tree;
    int tree_nv;
    int tree_dof;
    double friction;
  };
  const std::vector<ExpectedConstraint> expected{
      {0, 1, 0, wheel_friction},   // Wheel.
      {1, 1, 0, kSliderFriction},  // Slider.
      {2, 2, 1, kElbowFriction},   // Elbow, the second DOF of the arm.
  };
  const std::vector<const SapJointFrictionConstraint<double>*> constraints =
      EvalConstraints();
  ASSERT_EQ(std::ssize(constraints), std::ssize(expected));
  for (int i = 0; i < std::ssize(expected); ++i) {
    EXPECT_EQ(constraints[i]->num_cliques(), 1);
    EXPECT_EQ(constraints[i]->first_clique(), expected[i].tree);
    EXPECT_EQ(constraints[i]->num_velocities(0), expected[i].tree_nv);
    EXPECT_EQ(constraints[i]->clique_dof(), expected[i].tree_dof);
    EXPECT_EQ(constraints[i]->parameters().tau_c, expected[i].friction);
  }
}

// Joints with zero dry friction do not lead to constraints.
TEST_F(SapDriverJointFrictionTest, ZeroFrictionJointsAreSkipped) {
  MakePlant(1.0e-3);
  wheel_->SetDryFrictionVector(context_.get(), Vector1d(0.0));
  const std::vector<const SapJointFrictionConstraint<double>*> constraints =
      EvalConstraints();
  ASSERT_EQ(constraints.size(), 2);
  EXPECT_EQ(constraints[0]->parameters().tau_c, kSliderFriction);
  EXPECT_EQ(constraints[1]->parameters().tau_c, kElbowFriction);
}

// Locked joints do not lead to constraints, both when the whole tree is locked
// (the slider) and when other DOFs of the tree are not locked (the elbow).
TEST_F(SapDriverJointFrictionTest, LockedJointsAreSkipped) {
  MakePlant(1.0e-3);
  slider_->Lock(context_.get());
  elbow_->Lock(context_.get());
  const std::vector<const SapJointFrictionConstraint<double>*> constraints =
      EvalConstraints();
  ASSERT_EQ(constraints.size(), 1);
  EXPECT_EQ(constraints[0]->parameters().tau_c, kWheelFriction);
}

// A joint with dry friction in a tree with a locked DOF behaves as a single DOF
// joint with dry friction.
TEST_F(SapDriverJointFrictionTest, FrictionJointInTreeWithLockedDof) {
  const double dt = 1.0e-3;
  MakePlant(dt);
  shoulder_->Lock(context_.get());
  const double initial_rate = 1.0;
  elbow_->set_angular_rate(context_.get(), initial_rate);

  // With the shoulder locked, link2 rotates about the elbow with its own
  // inertia about that axis (a unitary inertia at the elbow), decelerating
  // uniformly under the elbow's friction.
  const int num_steps = 3;
  Step(num_steps);
  EXPECT_EQ(shoulder_->get_angular_rate(*context_), 0.0);
  EXPECT_NEAR(elbow_->get_angular_rate(*context_),
              initial_rate - num_steps * dt * kElbowFriction, kTolerance);
}

// Under an applied force below the friction bound, joints stay in stiction:
// friction balances the applied force, and the joints move at most at a very
// small speed.
TEST_F(SapDriverJointFrictionTest, Stiction) {
  MakePlant(1.0e-3);
  const double wheel_torque = 0.5 * kWheelFriction;
  const double slider_force = -0.8 * kSliderFriction;
  SetActuation(wheel_torque, slider_force);

  // We compare against the velocities after one step without friction.
  const double dt = plant_->time_step();
  const double wheel_v_free = dt * wheel_torque / kWheelInertia;
  const double slider_v_free = dt * slider_force / kSliderMass;
  constexpr double kSmall = 0.01;

  const ContactSolverResults<double>& results =
      manager_->EvalContactSolverResults(*context_);
  EXPECT_LT(std::abs(results.v_next(wheel_->velocity_start())),
            kSmall * std::abs(wheel_v_free));
  EXPECT_LT(std::abs(results.v_next(slider_->velocity_start())),
            kSmall * std::abs(slider_v_free));
  // The arm is not actuated and remains at rest.
  EXPECT_EQ(results.v_next(shoulder_->velocity_start()), 0.0);
  EXPECT_EQ(results.v_next(elbow_->velocity_start()), 0.0);

  // The reported generalized forces include actuation and the friction force,
  // which balances the applied force.
  const VectorXd& tau = manager_->EvalDiscreteUpdateMultibodyForces(*context_)
                            .generalized_forces();
  EXPECT_LT(std::abs(tau(wheel_->velocity_start())),
            kSmall * std::abs(wheel_torque));
  EXPECT_LT(std::abs(tau(slider_->velocity_start())),
            kSmall * std::abs(slider_force));

  // Over many steps the joints do not break away.
  Step(100);
  EXPECT_LT(std::abs(wheel_->get_angular_rate(*context_)),
            kSmall * std::abs(wheel_v_free));
  EXPECT_LT(std::abs(slider_->get_translation_rate(*context_)),
            kSmall * std::abs(slider_v_free));
}

// Under an applied force above the friction bound, joints slide with friction
// saturated at its bound and opposing motion.
TEST_F(SapDriverJointFrictionTest, Sliding) {
  MakePlant(1.0e-3);
  const double wheel_torque = 3.0 * kWheelFriction;
  const double slider_force = -2.5 * kSliderFriction;
  SetActuation(wheel_torque, slider_force);

  const ContactSolverResults<double>& results =
      manager_->EvalContactSolverResults(*context_);
  const double wheel_v = results.v_next(wheel_->velocity_start());
  const double slider_v = results.v_next(slider_->velocity_start());
  const double wheel_v_expected = ExpectedSlidingVelocityFromRest(
      wheel_torque, kWheelInertia, kWheelFriction);
  const double slider_v_expected = ExpectedSlidingVelocityFromRest(
      slider_force, kSliderMass, kSliderFriction);
  EXPECT_NEAR(wheel_v, wheel_v_expected,
              kTolerance * std::abs(wheel_v_expected));
  EXPECT_NEAR(slider_v, slider_v_expected,
              kTolerance * std::abs(slider_v_expected));

  // The reported generalized force is the applied force minus the saturated
  // friction force opposing motion.
  const VectorXd& tau = manager_->EvalDiscreteUpdateMultibodyForces(*context_)
                            .generalized_forces();
  EXPECT_NEAR(tau(wheel_->velocity_start()), wheel_torque - kWheelFriction,
              kTolerance * wheel_torque);
  EXPECT_NEAR(tau(slider_->velocity_start()), slider_force + kSliderFriction,
              kTolerance * std::abs(slider_force));

  // With constant applied forces the joints accelerate uniformly.
  const int num_steps = 50;
  Step(num_steps);
  EXPECT_NEAR(wheel_->get_angular_rate(*context_), num_steps * wheel_v_expected,
              kTolerance * num_steps * std::abs(wheel_v_expected));
  EXPECT_NEAR(slider_->get_translation_rate(*context_),
              num_steps * slider_v_expected,
              kTolerance * num_steps * std::abs(slider_v_expected));
}

// A spinning wheel without actuation decelerates uniformly under friction until
// it comes to rest, dissipating its kinetic energy over an angle independent of
// the time step. The velocity never reverses sign.
TEST_F(SapDriverJointFrictionTest, StoppingUnderFriction) {
  constexpr double kInitialRate = 2.0;  // [rad/s]
  // Uniform deceleration and the resulting analytical stopping time and angle.
  const double deceleration = kWheelFriction / kWheelInertia;
  const double stopping_time = kInitialRate / deceleration;
  const double stopping_angle =
      0.5 * kInitialRate * kInitialRate / deceleration;

  for (const double dt : {1.0e-3, 4.0e-3}) {
    MakePlant(dt);
    wheel_->set_angular_rate(context_.get(), kInitialRate);

    // Sliding phase: the rate decreases by δt⋅τ_c/I each step.
    const int num_sliding_steps = static_cast<int>(stopping_time / dt) - 1;
    for (int step = 1; step <= num_sliding_steps; ++step) {
      Step(1);
      const double expected_rate = kInitialRate - step * dt * deceleration;
      EXPECT_NEAR(wheel_->get_angular_rate(*context_), expected_rate,
                  kTolerance * kInitialRate)
          << "dt = " << dt << ", step = " << step;
    }

    // Continue past the stopping time. The wheel must come to rest (up to a
    // very small speed) without reversing.
    const int extra_steps = static_cast<int>(0.1 * stopping_time / dt) + 5;
    for (int step = 0; step < extra_steps; ++step) {
      Step(1);
      EXPECT_GE(wheel_->get_angular_rate(*context_), 0.0);
    }
    EXPECT_LT(wheel_->get_angular_rate(*context_), 1.0e-12);

    // The kinetic energy ½⋅I⋅ω₀² is dissipated by friction over the stopping
    // angle, up to the first order discretization error of the symplectic
    // Euler scheme.
    EXPECT_NEAR(wheel_->get_angle(*context_), stopping_angle, kInitialRate * dt)
        << "dt = " << dt;
  }
}

// The scalar converted plant on AutoDiffXd, with its own default dry friction,
// produces the same velocities as the double plant.
TEST_F(SapDriverJointFrictionTest, AutoDiffXd) {
  MakePlant(1.0e-3);
  const double wheel_torque = 3.0 * kWheelFriction;   // Sliding.
  const double slider_force = 0.5 * kSliderFriction;  // Stiction.
  SetActuation(wheel_torque, slider_force);

  const VectorXd v_ad = StepAutoDiffXd({});
  Step(1);
  const VectorXd v = plant_->GetVelocities(*context_);
  EXPECT_TRUE(CompareMatrices(v_ad, v, kTolerance * v.norm()));
  // Sanity check that the model did move.
  EXPECT_GT(v(wheel_->velocity_start()), 0.0);
}

// Same as above, for a joint with dry friction in a tree with a locked DOF.
TEST_F(SapDriverJointFrictionTest, AutoDiffXdWithLockedDof) {
  MakePlant(1.0e-3);
  shoulder_->Lock(context_.get());
  elbow_->set_angular_rate(context_.get(), 1.0);

  const VectorXd v_ad = StepAutoDiffXd({shoulder_});
  Step(1);
  const VectorXd v = plant_->GetVelocities(*context_);
  EXPECT_TRUE(CompareMatrices(v_ad, v, kTolerance * v.norm()));
  // Sanity check that friction acted on the elbow.
  EXPECT_LT(elbow_->get_angular_rate(*context_), 1.0);
}

// The Joint setters reject a negative dry friction. Verify that the driver
// throws, and does not abort, when the parameter holds such a value because it
// was written by other means.
TEST_F(SapDriverJointFrictionTest, NegativeDryFrictionThrows) {
  MakePlant(1.0e-3);
  WriteDryFrictionParameter(*wheel_, Vector1d(-1.0), context_.get());
  DRAKE_EXPECT_THROWS_MESSAGE(EvalConstraints(),
                              "Joint 'wheel' has an invalid dry friction.*");
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
