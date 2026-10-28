#pragma once

#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace internal {

/* Stores data for joint dry friction constraints. This data is updated at each
solver iteration, as opposed to the JointFrictionConstraintsPool, which helps
define the optimization problem.

@tparam_nonsymbolic_scalar */
template <typename T>
class JointFrictionConstraintsDataPool {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JointFrictionConstraintsDataPool);

  /* Constructs an empty pool. */
  JointFrictionConstraintsDataPool() = default;

  ~JointFrictionConstraintsDataPool();

  /* Resizes the pool, allocating memory only as necessary. */
  void Resize(int num_constraints) {
    gamma_pool_.resize(num_constraints);
    G_pool_.resize(num_constraints);
  }

  /* Returns the number of joint friction constraints this data is for. */
  int num_constraints() const { return ssize(gamma_pool_); }

  /* Returns the friction impulse γ = -∇ℓ(v) for the k-th constraint in the
  pool. See JointFrictionConstraintsPool for details. */
  const T& gamma(int k) const { return gamma_pool_[k]; }
  T& mutable_gamma(int k) { return gamma_pool_[k]; }

  /* Returns the Hessian G = -∂γ/∂v of the k-th constraint in the pool. */
  const T& G(int k) const { return G_pool_[k]; }
  T& mutable_G(int k) { return G_pool_[k]; }

  /* Returns the total constraint cost ℓ(v) for all joint friction constraints
  in the pool. */
  const T& cost() const { return cost_; }
  T& mutable_cost() { return cost_; }

 private:
  T cost_{NAN};                // Total cost over all constraints.
  std::vector<T> gamma_pool_;  // Friction impulses γ.
  std::vector<T> G_pool_;      // Hessians G = -∂γ/∂v ≥ 0.
};

}  // namespace internal
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::icf::internal::
        JointFrictionConstraintsDataPool);
