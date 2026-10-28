#pragma once

#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/icf/abstract_constraints_pool.h"
#include "drake/multibody/contact_solvers/icf/icf_data.h"
#include "drake/multibody/contact_solvers/icf/icf_hessian.h"
#include "drake/multibody/contact_solvers/icf/joint_friction_constraints_data_pool.h"
#include "drake/multibody/contact_solvers/icf/reduced_mapping.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace internal {

// Forward declaration to break circular dependencies.
template <typename T>
class IcfModel;

/* A pool of joint dry (Coulomb) friction constraints, each on a single DoF of
a clique.

The friction force τ on the DoF opposes its velocity v, and its magnitude is
bounded by a constant τ_c > 0 (see Joint::default_dry_friction_vector()). In
stiction, τ takes whatever value in [−τ_c, τ_c] is needed to hold the DoF at
rest. While the DoF slides, τ = −τ_c⋅sign(v).

Each constraint is associated with a convex cost ℓ(v) that is added to the
overall ICF cost. The gradient of this cost produces an impulse γ = -∇ℓ(v) =
δt⋅τ that enforces the constraint when the convex ICF problem is solved. We use
the same regularized cost as SapJointFrictionConstraint:

  ℓ(v) = v²/(2R)                if |v| ≤ R⋅γₘₐₓ  (stiction)
         γₘₐₓ⋅|v| − R⋅γₘₐₓ²/2   otherwise         (sliding)

with impulse bound γₘₐₓ = δt⋅τ_c. Thus the impulse is

  γ(v) = clamp(−v/R, −γₘₐₓ, γₘₐₓ),

and the Hessian is G = R⁻¹ in stiction and zero in sliding. Like SAP, we use
the regularization R = σ⋅w, with σ = kSigma and w = 1/Aᵢᵢ, where A = M + δt⋅D is
the linearized dynamics matrix of the clique. In stiction, under a constant load
τₐ with |τₐ| < τ_c, the DoF creeps at the small velocity v = σ⋅w⋅δt⋅τₐ.

Both γₘₐₓ and A depend on the time step δt. Therefore this pool computes them
from model().time_step() whenever it computes data, so that the constraint stays
consistent with SAP after IcfModel::UpdateTimeStep().

N.B. A gain constraint τ = clamp(−K⋅v, −τ_c, τ_c) with K = 1/(δt⋅R) has the same
cost, see GainConstraintsPool. We do not use it, because its gain K would keep
the value of δt at construction after IcfModel::UpdateTimeStep().

Joint friction constraints only apply to a DoF within a single clique, so they
do not change the sparsity structure of the problem.

@tparam_nonsymbolic_scalar */
template <typename T>
class JointFrictionConstraintsPool {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JointFrictionConstraintsPool);

  /* The dimensionless regularization parameter σ. This is the same value that
  SapDriver uses for its SapJointFrictionConstraint. */
  static constexpr double kSigma = 1.0e-3;

  /* Constructs an empty pool. */
  explicit JointFrictionConstraintsPool(const IcfModel<T>* parent_model);

  ~JointFrictionConstraintsPool();

  /* @see IsAbstractConstraintsPool. */
  const IcfModel<T>& model() const { return *model_; }
  int num_constraints() const { return clique_.size(); }
  void AccumulateGradient(const IcfData<T>& data, VectorX<T>* gradient) const;
  void AccumulateHessian(const IcfData<T>& data, IcfHessian<T>* hessian) const;
  void ReduceInto(const ReducedMapping& mapping,
                  JointFrictionConstraintsPool<T>* reduced_pool) const;

  /* Resizes the constraints pool to store the given number of constraints.

  @warning After resizing, constraints may hold invalid data until Set() is
  called for each constraint index in [0, num_constraints()). */
  void Resize(int num_constraints);

  /* Sets the given joint friction constraint.

  @param index The index of the constraint within the pool,
               must be in [0, num_constraints()).
  @param clique The clique index of the DoF.
  @param dof The clique-local index of the DoF.
  @param friction The friction force bound τ_c > 0.

  Calling this function several times with the same `index` overwrites the
  previous constraint for that index. */
  void Set(int index, int clique, int dof, const T& friction);

  /* Computes problem data as a function of the generalized velocities `v` for
  the full IcfModel. */
  void CalcData(const VectorX<T>& v,
                JointFrictionConstraintsDataPool<T>* friction_data) const;

  /* Computes the first and second derivatives of the constraint cost
  ℓ̃(α) = ℓ(v + α⋅w).

  @param friction_data Constraint data computed at v + α⋅w.
  @param w The search direction.
  @param[out] dcost The first derivative dℓ̃ /dα on output.
  @param[out] d2cost The second derivative d²ℓ̃ /dα² on output. */
  void CalcCostAlongLine(
      const JointFrictionConstraintsDataPool<T>& friction_data,
      const VectorX<T>& w, T* dcost, T* d2cost) const;

  /* Testing only access. */
  const std::vector<int>& clique() const { return clique_; }
  const std::vector<int>& dof() const { return dof_; }
  const std::vector<T>& friction() const { return friction_; }

 private:
  const IcfModel<T>* const model_;  // The parent model.

  // Each of the following has size num_constraints().
  std::vector<int> clique_;  // Clique of the k-th constraint.
  std::vector<int> dof_;     // Clique-local DoF of the k-th constraint.
  std::vector<T> friction_;  // Friction force bound τ_c.
};
static_assert(IsAbstractConstraintsPool<JointFrictionConstraintsPool>);

}  // namespace internal
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::icf::internal::
        JointFrictionConstraintsPool);
