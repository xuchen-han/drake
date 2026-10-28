#include "drake/multibody/contact_solvers/icf/joint_friction_constraints_pool.h"

#include "drake/multibody/contact_solvers/icf/icf_model.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace icf {
namespace internal {

template <typename T>
JointFrictionConstraintsPool<T>::JointFrictionConstraintsPool(
    const IcfModel<T>* parent_model)
    : model_(parent_model) {
  DRAKE_DEMAND(parent_model != nullptr);
}

template <typename T>
JointFrictionConstraintsPool<T>::~JointFrictionConstraintsPool() = default;

template <typename T>
void JointFrictionConstraintsPool<T>::Resize(int num_constraints) {
  clique_.resize(num_constraints);
  dof_.resize(num_constraints);
  friction_.resize(num_constraints);
}

template <typename T>
void JointFrictionConstraintsPool<T>::Set(int index, int clique, int dof,
                                          const T& friction) {
  DRAKE_ASSERT(0 <= index && index < num_constraints());
  DRAKE_ASSERT(0 <= dof && dof < model().clique_size(clique));
  DRAKE_ASSERT(friction > 0.0);

  clique_[index] = clique;
  dof_[index] = dof;
  friction_[index] = friction;
}

template <typename T>
void JointFrictionConstraintsPool<T>::CalcData(
    const VectorX<T>& v,
    JointFrictionConstraintsDataPool<T>* friction_data) const {
  DRAKE_ASSERT(friction_data != nullptr);
  using std::abs;

  const T& dt = model().time_step();
  T& cost = friction_data->mutable_cost();
  cost = 0;
  for (int k = 0; k < num_constraints(); ++k) {
    const int c = clique_[k];
    const int i = dof_[k];
    const auto v_c = model().clique_segment(c, v);
    const T& vk = v_c(i);

    // Regularization R = σ⋅w, with the Delassus operator estimate w = 1/Aᵢᵢ.
    // A = M + δt⋅D changes with the time step, so we compute R here.
    const T R = kSigma / model().A(c)(i, i);
    const T gamma_max = dt * friction_[k];
    const T y = -vk / R;

    T& gamma = friction_data->mutable_gamma(k);
    T& G = friction_data->mutable_G(k);
    // The stiction condition |v| ≤ R⋅γₘₐₓ is equivalent to |y| ≤ γₘₐₓ. At the
    // transition we use the stiction branch, like SapJointFrictionConstraint.
    if (abs(y) <= gamma_max) {
      gamma = y;
      G = 1.0 / R;
      cost += 0.5 * R * y * y;  // Equals v²/(2R).
    } else {
      gamma = y > 0 ? gamma_max : T(-gamma_max);
      G = 0.0;
      cost += gamma_max * abs(vk) - 0.5 * R * gamma_max * gamma_max;
    }
  }
}

template <typename T>
void JointFrictionConstraintsPool<T>::AccumulateGradient(
    const IcfData<T>& data, VectorX<T>* gradient) const {
  DRAKE_ASSERT(gradient != nullptr);

  const JointFrictionConstraintsDataPool<T>& friction_data =
      data.joint_friction_constraints_data();
  for (int k = 0; k < num_constraints(); ++k) {
    auto gradient_c = model().mutable_clique_segment(clique_[k], gradient);
    // The constraint Jacobian is J = eᵢᵀ, thus ∇ℓ = −Jᵀ⋅γ = −eᵢ⋅γ.
    gradient_c(dof_[k]) -= friction_data.gamma(k);
  }
}

template <typename T>
void JointFrictionConstraintsPool<T>::AccumulateHessian(
    const IcfData<T>& data, IcfHessian<T>* hessian) const {
  DRAKE_ASSERT(hessian != nullptr);

  const JointFrictionConstraintsDataPool<T>& friction_data =
      data.joint_friction_constraints_data();
  EigenPool<MatrixX<T>>& H_cc_pool = data.scratch().H_cc_pool;
  // Constraints on the same clique are usually next to each other in the pool.
  // We add each run of constraints on the same clique to the Hessian at once.
  int k = 0;
  while (k < num_constraints()) {
    const int c = clique_[k];
    const int nv = model().clique_size(c);
    H_cc_pool.Resize(1, nv, nv);
    typename EigenPool<MatrixX<T>>::MatrixView Hc = H_cc_pool[0];
    Hc.setZero();
    for (; k < num_constraints() && clique_[k] == c; ++k) {
      Hc(dof_[k], dof_[k]) += friction_data.G(k);
    }
    hessian->AddToBlock(c, c, Hc);
  }
}

template <typename T>
void JointFrictionConstraintsPool<T>::CalcCostAlongLine(
    const JointFrictionConstraintsDataPool<T>& friction_data,
    const VectorX<T>& w, T* dcost, T* d2cost) const {
  DRAKE_ASSERT(dcost != nullptr);
  DRAKE_ASSERT(d2cost != nullptr);

  *dcost = 0.0;
  *d2cost = 0.0;
  for (int k = 0; k < num_constraints(); ++k) {
    const auto w_c = model().clique_segment(clique_[k], w);
    const T& wk = w_c(dof_[k]);
    (*dcost) -= friction_data.gamma(k) * wk;
    (*d2cost) += friction_data.G(k) * wk * wk;
  }
}

template <typename T>
void JointFrictionConstraintsPool<T>::ReduceInto(
    const ReducedMapping& mapping,
    JointFrictionConstraintsPool<T>* reduced_pool) const {
  // Make sure the pool is (over) allocated.
  reduced_pool->Resize(num_constraints());
  // Remove old data.
  reduced_pool->Resize(0);

  for (int k = 0; k < num_constraints(); ++k) {
    const int c = clique_[k];
    if (!mapping.clique_subsequence.participates(c)) {
      // Entire clique is locked; remove the constraint.
      continue;
    }
    const auto& dof_subsequence = mapping.clique_dof_subsequences[c];
    if (!dof_subsequence.participates(dof_[k])) {
      // The DoF is locked; remove the constraint.
      continue;
    }

    // Fill in the reduced constraint.
    reduced_pool->clique_.push_back(
        mapping.clique_subsequence.permuted_index(c));
    reduced_pool->dof_.push_back(dof_subsequence.permuted_index(dof_[k]));
    reduced_pool->friction_.push_back(friction_[k]);
  }
}

}  // namespace internal
}  // namespace icf
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::icf::internal::
        JointFrictionConstraintsPool);
