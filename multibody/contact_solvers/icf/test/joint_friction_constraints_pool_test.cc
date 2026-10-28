#include "drake/multibody/contact_solvers/icf/joint_friction_constraints_pool.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include "drake/common/extract_double.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/contact_solvers/icf/icf_data.h"
#include "drake/multibody/contact_solvers/icf/icf_hessian.h"
#include "drake/multibody/contact_solvers/icf/icf_model.h"
#include "drake/multibody/contact_solvers/icf/icf_search_direction_data.h"
#include "drake/multibody/contact_solvers/icf/test_utilities/icf_model_test_helpers.h"
#include "drake/multibody/plant/slicing_and_indexing.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace internal {
namespace {

using multibody::internal::SelectRows;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kSigma = JointFrictionConstraintsPool<double>::kSigma;

// Velocities at which we evaluate the constraints, of size 18 to fit the model
// made by MakeUnconstrainedModel(). All are non-zero.
VectorXd MakeVelocities() {
  return VectorXd::LinSpaced(18, -1.0, 1.0) + VectorXd::Constant(18, 0.01);
}

// A friction constraint on a DoF, given by the clique and the clique-local DoF
// index, and its expected regime at the velocities of MakeVelocities().
struct FrictionSpec {
  int clique{};
  int dof{};
  bool sticks{};
};

// The constraints added by AddJointFrictionConstraints(). The constraints on
// clique 1 are not next to each other in the pool, to cover the accumulation of
// the Hessian for each run of constraints on the same clique.
const std::vector<FrictionSpec>& friction_specs() {
  static const std::vector<FrictionSpec> specs{{.clique = 1, .dof = 2},
                                               {.clique = 1, .dof = 4},
                                               {.clique = 2, .dof = 0},
                                               {.clique = 0, .dof = 5},
                                               {.clique = 1, .dof = 0}};
  return specs;
}

// Returns the regularization R = σ⋅w, with w = 1/Aᵢᵢ, for the given DoF.
template <typename T>
double CalcR(const IcfModel<T>& model, int clique, int dof) {
  return kSigma / ExtractDoubleOrThrow(model.A(clique)(dof, dof));
}

// Adds joint friction constraints to the given model, with a friction bound
// such that constraint k sticks at the velocities of MakeVelocities() if k is
// even, and slides if k is odd. Returns the friction bounds.
template <typename T>
std::vector<double> AddJointFrictionConstraints(IcfModel<T>* model) {
  const VectorXd v = MakeVelocities();
  const double dt = ExtractDoubleOrThrow(model->time_step());
  JointFrictionConstraintsPool<T>& pool =
      model->joint_friction_constraints_pool();
  pool.Resize(ssize(friction_specs()));
  std::vector<double> frictions;
  for (int k = 0; k < ssize(friction_specs()); ++k) {
    const FrictionSpec& spec = friction_specs()[k];
    const double vk = v(model->clique_starts()[spec.clique] + spec.dof);
    const double R = CalcR(*model, spec.clique, spec.dof);
    // The DoF sticks if |v| ≤ R⋅δt⋅τ_c.
    const double friction_at_transition = std::abs(vk) / (R * dt);
    const double friction = (k % 2 == 0 ? 2.0 : 0.5) * friction_at_transition;
    pool.Set(k, spec.clique, spec.dof, friction);
    frictions.push_back(friction);
  }
  return frictions;
}

// Undo the floating body annotation, so that the tests can lock some DoFs of a
// clique.
void UndoFloatingBodyAnnotation(IcfModel<double>* model) {
  std::unique_ptr<IcfParameters<double>> params = model->ReleaseParameters();
  params->body_is_floating = {0, 0, 0, 0};
  model->ResetParameters(std::move(params));
}

/* Checks that model.CalcData does not incur any heap allocations on a problem
with joint friction constraints. */
GTEST_TEST(JointFrictionConstraintsPool, LimitMallocOnCalcData) {
  IcfModel<double> model;
  MakeUnconstrainedModel(&model);
  AddJointFrictionConstraints(&model);
  EXPECT_EQ(model.num_joint_friction_constraints(), 5);
  EXPECT_EQ(model.num_constraints(), 5);

  IcfData<double> data;
  model.ResizeData(&data);
  EXPECT_EQ(data.joint_friction_constraints_data().num_constraints(), 5);

  const VectorXd v = MakeVelocities();
  {
    drake::test::LimitMalloc guard;
    model.CalcData(v, &data);
  }

  // Updating the Hessian should not cause any new allocations either.
  model.SetSparsityPattern();
  std::unique_ptr<IcfHessian<double>> hessian = model.MakeHessian(data);
  {
    drake::test::LimitMalloc guard;
    model.UpdateHessian(data, hessian.get());
  }
}

/* Checks that pool.ReduceInto does not incur any heap allocations. */
GTEST_TEST(JointFrictionConstraintsPool, LimitMallocOnReduceInto) {
  IcfModel<double> model;
  MakeUnconstrainedModel(&model);
  AddJointFrictionConstraints(&model);

  IcfModel<double> reduced_model;
  ReducedMapping mapping;

  // Do a not-smaller reduction to allocate memory in the reduced model.
  MakeModelReducible(&model, /* dofs_to_remove = */ {});
  model.ReduceInto(&reduced_model, &mapping);

  // Given prior allocation of a big enough model, the constraint pool
  // reduction does not allocate.
  {
    drake::test::LimitMalloc guard;
    model.joint_friction_constraints_pool().ReduceInto(
        mapping, &reduced_model.joint_friction_constraints_pool());
  }
}

/* Verifies that the joint friction constraints produce correct impulses, and
that the gradient, Hessian and cost along a line are consistent with the cost,
both in stiction and in sliding. */
GTEST_TEST(JointFrictionConstraintsPool, Data) {
  IcfModel<AutoDiffXd> model;
  MakeUnconstrainedModel(&model);
  model.SetSparsityPattern();
  EXPECT_EQ(model.num_joint_friction_constraints(), 0);

  const std::vector<double> frictions = AddJointFrictionConstraints(&model);
  EXPECT_EQ(model.num_joint_friction_constraints(), 5);
  EXPECT_EQ(model.num_constraints(), 5);

  IcfData<AutoDiffXd> data;
  model.ResizeData(&data);
  EXPECT_EQ(data.joint_friction_constraints_data().num_constraints(), 5);

  const int nv = model.num_velocities();
  const VectorXd v_value = MakeVelocities();
  VectorX<AutoDiffXd> v(nv);
  math::InitializeAutoDiff(v_value, &v);
  model.CalcData(v, &data);

  const double dt = model.time_step().value();
  const JointFrictionConstraintsDataPool<AutoDiffXd>& friction_data =
      data.joint_friction_constraints_data();
  ASSERT_EQ(friction_data.num_constraints(), 5);

  // Expected impulses γ = clamp(−v/R, −δt⋅τ_c, δt⋅τ_c), and Hessians.
  VectorXd tau_expected = VectorXd::Zero(nv);
  for (int k = 0; k < 5; ++k) {
    SCOPED_TRACE(fmt::format("constraint {}", k));
    const FrictionSpec& spec = friction_specs()[k];
    const int i = model.clique_starts()[spec.clique] + spec.dof;
    const double R = CalcR(model, spec.clique, spec.dof);
    const double gamma_max = dt * frictions[k];
    const bool sticks = (k % 2 == 0);
    const double gamma_expected =
        sticks ? -v_value(i) / R : -gamma_max * (v_value(i) > 0 ? 1.0 : -1.0);
    const double G_expected = sticks ? 1.0 / R : 0.0;
    EXPECT_NEAR(friction_data.gamma(k).value(), gamma_expected,
                4 * kEpsilon * std::abs(gamma_expected));
    EXPECT_NEAR(friction_data.G(k).value(), G_expected,
                4 * kEpsilon * G_expected);
    tau_expected(i) += gamma_expected;
  }

  // The impulse is the negative gradient of the cost.
  const VectorXd cost_gradient = friction_data.cost().derivatives();
  EXPECT_TRUE(CompareMatrices(-cost_gradient, tau_expected,
                              10 * kEpsilon * tau_expected.norm(),
                              MatrixCompareType::absolute));

  // The Hessian is the derivative of the gradient. This includes the
  // contributions of all constraints in each run of constraints on the same
  // clique.
  auto hessian = model.MakeHessian(data);
  const MatrixXd hessian_value = math::ExtractValue(hessian->MakeDenseMatrix());
  const MatrixXd gradient_derivatives = math::ExtractGradient(data.gradient());
  EXPECT_TRUE(CompareMatrices(hessian_value, gradient_derivatives,
                              10 * kEpsilon * hessian_value.norm(),
                              MatrixCompareType::absolute));

  // Check that CalcCostAlongLine works for joint friction constraints.
  const VectorX<AutoDiffXd> w =
      VectorX<AutoDiffXd>::LinSpaced(nv, 0.1, -0.2);  // Arbitrary direction.
  IcfSearchDirectionData<AutoDiffXd> search_data;
  const VectorX<AutoDiffXd> v_constant = v_value;
  model.CalcData(v_constant, &data);
  model.CalcSearchDirectionData(data, w, &search_data);

  // Small enough steps along the line, so that every constraint stays in the
  // same regime, and large ones that change it.
  for (const double alpha_value : {1.0e-7, 0.35, 2.0}) {
    SCOPED_TRACE(fmt::format("alpha = {}", alpha_value));
    const AutoDiffXd alpha = {alpha_value, VectorXd::Ones(1)};
    AutoDiffXd dcost, d2cost;
    const AutoDiffXd cost =
        model.CalcCostAlongLine(alpha, data, search_data, &dcost, &d2cost);
    const double scale = std::abs(dcost.value()) + std::abs(d2cost.value());
    EXPECT_NEAR(dcost.value(), cost.derivatives()[0], 10 * scale * kEpsilon);
    EXPECT_NEAR(d2cost.value(), dcost.derivatives()[0], 10 * scale * kEpsilon);
  }
}

/* Verifies that the impulse bound and the regularization follow the time step
of the model after IcfModel::UpdateTimeStep(). */
GTEST_TEST(JointFrictionConstraintsPool, UpdateTimeStep) {
  IcfModel<double> model;
  MakeUnconstrainedModel(&model);
  const std::vector<double> frictions = AddJointFrictionConstraints(&model);
  IcfData<double> data;
  model.ResizeData(&data);
  const VectorXd v = MakeVelocities();

  const double dt = 0.5 * model.time_step();
  model.UpdateTimeStep(dt);
  model.CalcData(v, &data);
  const JointFrictionConstraintsDataPool<double>& friction_data =
      data.joint_friction_constraints_data();
  for (int k = 0; k < 5; ++k) {
    SCOPED_TRACE(fmt::format("constraint {}", k));
    const FrictionSpec& spec = friction_specs()[k];
    const int i = model.clique_starts()[spec.clique] + spec.dof;
    const double vk = v(i);
    // The regularization uses A = M + δt⋅D with the new time step. The impulse
    // bound halves, so constraints that stuck can slide now.
    const double R = kSigma / (model.M0()(i, i) + dt * model.D0()(i));
    const double gamma_max = dt * frictions[k];
    const double y = -vk / R;
    const double gamma_expected = std::clamp(y, -gamma_max, gamma_max);
    EXPECT_NEAR(friction_data.gamma(k), gamma_expected,
                4 * kEpsilon * std::abs(gamma_expected));
    EXPECT_EQ(friction_data.G(k), std::abs(y) <= gamma_max ? 1.0 / R : 0.0);
  }
}

/* Verifies that reducing the joint friction constraints pool removes the
constraints on locked DoFs and keeps the others with the same data. */
GTEST_TEST(JointFrictionConstraintsPool, Reduce) {
  IcfModel<double> model;
  MakeUnconstrainedModel(&model);
  UndoFloatingBodyAnnotation(&model);
  AddJointFrictionConstraints(&model);

  IcfData<double> data;
  model.ResizeData(&data);
  const int nv = model.num_velocities();
  const VectorXd v = MakeVelocities();
  model.CalcData(v, &data);
  const auto& full_pool = model.joint_friction_constraints_pool();
  const auto& full_data_pool = data.joint_friction_constraints_data();

  auto check_reduced = [&](const std::vector<int>& locked_dofs) {
    SCOPED_TRACE(fmt::format("locked_dofs [{}]", fmt::join(locked_dofs, ", ")));
    IcfModel<double> reduced_model;
    ReducedMapping mapping;
    MakeModelReducible(&model, locked_dofs);
    model.ReduceInto(&reduced_model, &mapping);
    const auto& reduced_pool = reduced_model.joint_friction_constraints_pool();

    // The full constraint index of each reduced constraint.
    std::vector<int> kept;
    for (int k = 0; k < full_pool.num_constraints(); ++k) {
      const int i =
          model.clique_starts()[full_pool.clique()[k]] + full_pool.dof()[k];
      if (mapping.velocity_subsequence.participates(i)) kept.push_back(k);
    }
    ASSERT_EQ(reduced_pool.num_constraints(), ssize(kept));

    IcfData<double> reduced_data;
    reduced_model.ResizeData(&reduced_data);
    const VectorXd reduced_v =
        SelectRows(v, mapping.velocity_subsequence.inverse_permutation());
    reduced_model.CalcData(reduced_v, &reduced_data);
    const auto& reduced_data_pool =
        reduced_data.joint_friction_constraints_data();

    for (int r = 0; r < ssize(kept); ++r) {
      const int k = kept[r];
      const int c = full_pool.clique()[k];
      EXPECT_EQ(reduced_pool.clique()[r],
                mapping.clique_subsequence.permuted_index(c));
      EXPECT_EQ(reduced_pool.dof()[r],
                mapping.clique_dof_subsequences[c].permuted_index(
                    full_pool.dof()[k]));
      EXPECT_EQ(reduced_pool.friction()[r], full_pool.friction()[k]);
      // The constraint only depends on its own DoF, so its data does not
      // change.
      EXPECT_EQ(reduced_data_pool.gamma(r), full_data_pool.gamma(k));
      EXPECT_EQ(reduced_data_pool.G(r), full_data_pool.G(k));
    }
  };

  // Reduce by none; essentially, just copy.
  check_reduced({});

  // Lock some DoFs without friction.
  check_reduced({0, 17});

  // Lock DoFs with friction, in the middle of a run of constraints.
  check_reduced({8, 10});

  // Lock other DoFs in a clique with friction.
  check_reduced({7, 9, 11});

  // Lock an entire clique.
  check_reduced({6, 7, 8, 9, 10, 11});

  // Lock everything.
  std::vector<int> all_locked(nv);
  std::iota(all_locked.begin(), all_locked.end(), 0);
  check_reduced(all_locked);
}

}  // namespace
}  // namespace internal
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
