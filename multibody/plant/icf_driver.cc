#include "drake/multibody/plant/icf_driver.h"

#include <algorithm>

#include "drake/multibody/contact_solvers/icf/icf_data.h"
#include "drake/multibody/contact_solvers/icf/icf_linear_feedback_gains.h"
#include "drake/multibody/contact_solvers/icf/icf_model.h"
#include "drake/multibody/contact_solvers/icf/icf_solver.h"
#include "drake/multibody/contact_solvers/icf/reduced_mapping.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/icf_builder.h"
#include "drake/multibody/plant/icf_external_systems_linearizer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/slicing_and_indexing.h"

namespace drake {
namespace multibody {
namespace internal {

using contact_solvers::icf::IcfSolverParameters;
using contact_solvers::icf::internal::EigenPool;
using contact_solvers::icf::internal::IcfBuilder;
using contact_solvers::icf::internal::IcfData;
using contact_solvers::icf::internal::IcfExternalSystemsLinearizer;
using contact_solvers::icf::internal::IcfLinearFeedbackGains;
using contact_solvers::icf::internal::IcfModel;
using contact_solvers::icf::internal::IcfSolver;
using contact_solvers::icf::internal::PatchConstraintsPool;
using contact_solvers::icf::internal::ReducedMapping;
using contact_solvers::internal::ContactSolverResults;
using systems::Context;

/* Scratch space for one ICF solve. See IcfSolverResultsCache. */
struct IcfWorkspace {
  explicit IcfWorkspace(const MultibodyPlant<double>& plant)
      : builder(&plant), linearizer(&plant) {
    const int nv = plant.num_velocities();
    actuation_feedback.Resize(nv);
    external_feedback.Resize(nv);
    v_guess.resize(nv);
    gradient.resize(nv);
  }

  IcfBuilder<double> builder;
  IcfExternalSystemsLinearizer<double> linearizer;
  IcfLinearFeedbackGains<double> actuation_feedback;
  IcfLinearFeedbackGains<double> external_feedback;
  // The full model, and its data at the solution.
  IcfModel<double> model;
  IcfData<double> data;
  // The model without the locked velocities, and its data.
  IcfModel<double> reduced_model;
  IcfData<double> reduced_data;
  ReducedMapping mapping;
  IcfSolver solver;
  VectorX<double> v_guess;
  VectorX<double> gradient;
  EigenPool<Vector3<double>> gamma_Bc_W;
  std::vector<Vector6<double>> Gamma_Bo_W;
};

IcfSolverResultsCache::IcfSolverResultsCache() = default;

IcfSolverResultsCache::IcfSolverResultsCache(const IcfSolverResultsCache& other)
    : results(other.results) {}

IcfSolverResultsCache& IcfSolverResultsCache::operator=(
    const IcfSolverResultsCache& other) {
  results = other.results;
  return *this;
}

IcfSolverResultsCache::~IcfSolverResultsCache() = default;

IcfDriver::IcfDriver(const CompliantContactManager<double>* manager)
    : manager_(manager) {
  DRAKE_DEMAND(manager != nullptr);
  const MultibodyPlant<double>& plant = this->plant();
  if (plant.deformable_model().num_bodies() > 0) {
    throw std::logic_error(
        "DiscreteContactSolver::kIcf does not support deformable bodies.");
  }
  if (plant.num_tendon_constraints() > 0) {
    throw std::logic_error(
        "DiscreteContactSolver::kIcf does not support tendon constraints.");
  }
}

IcfDriver::~IcfDriver() = default;

const MultibodyPlant<double>& IcfDriver::plant() const {
  return manager().plant();
}

void IcfDriver::set_icf_solver_parameters(
    const IcfSolverParameters& parameters) {
  parameters_ = parameters;
}

void IcfDriver::DeclareCacheEntries(
    CompliantContactManager<double>* mutable_manager) {
  DRAKE_DEMAND(mutable_manager == manager_);
  const auto& cache_entry = mutable_manager->DeclareCacheEntry(
      "ICF solver results",
      systems::ValueProducer(this, IcfSolverResultsCache(),
                             &IcfDriver::CalcIcfSolverResults),
      // The results include contributions from force elements, which could
      // involve user-injected dependencies. So we need to include all possible
      // tickets that users can choose to depend on.
      {systems::System<double>::xd_ticket(),
       systems::System<double>::all_input_ports_ticket(),
       systems::System<double>::all_parameters_ticket(),
       systems::System<double>::time_ticket(),
       systems::System<double>::accuracy_ticket()});
  icf_results_ = cache_entry.cache_index();
}

const IcfSolverResults& IcfDriver::EvalIcfSolverResults(
    const Context<double>& context) const {
  return plant()
      .get_cache_entry(icf_results_)
      .Eval<IcfSolverResultsCache>(context)
      .results;
}

void IcfDriver::CalcIcfSolverResults(const Context<double>& context,
                                     IcfSolverResultsCache* cache) const {
  const MultibodyPlant<double>& plant = this->plant();
  if (cache->workspace == nullptr) {
    cache->workspace = std::make_unique<IcfWorkspace>(plant);
  }
  IcfWorkspace& w = *cache->workspace;
  IcfSolverResults& results = cache->results;
  const double h = plant.time_step();
  const int nv = plant.num_velocities();

  // A discrete plant holds its inputs constant during the step. Thus we use
  // the inputs at x₀, and only the PD controllers are implicit.
  const auto [has_actuation, has_external_forces] =
      w.linearizer.CalcExplicitFeedbackGains(h, context, &w.actuation_feedback,
                                             &w.external_feedback);

  // Make the ICF model ℓ(v; q₀, v₀, δt). We use the geometry query results of
  // the manager, so that each step does one geometry query only.
  const NestedGeometryContactData<double>& geometry =
      manager().EvalGeometryContactData(context).get();
  w.builder.UpdateModel(context, h, geometry.point_pairs, geometry.surfaces,
                        has_actuation ? &w.actuation_feedback : nullptr,
                        has_external_forces ? &w.external_feedback : nullptr,
                        &w.model);

  // Solve for the next-step velocities. As CenicIntegrator does in fixed step
  // mode, we use the current velocities as the initial guess and the minimum
  // tolerance of the solver.
  w.solver.SetParameters(parameters_);
  const double tolerance = parameters_.min_tolerance;
  w.v_guess = plant.GetVelocities(context);
  bool solved{false};
  if (w.model.is_reducible()) {
    w.model.ReduceInto(&w.reduced_model, &w.mapping);
    w.reduced_model.ResizeData(&w.reduced_data);
    const std::vector<int>& indices =
        w.mapping.velocity_subsequence.inverse_permutation();
    w.reduced_data.set_v(w.v_guess(indices));
    w.reduced_model.SetSparsityPattern();
    solved =
        w.solver.SolveWithGuess(w.reduced_model, tolerance, &w.reduced_data);
    if (solved) {
      results.v_next = ExpandRows(w.reduced_data.v(), nv, indices);
      // Compute the data of the full model at the solution, to report forces.
      w.model.ResizeData(&w.data);
      w.model.CalcData(results.v_next, &w.data);
    }
  } else {
    w.model.ResizeData(&w.data);
    w.data.set_v(w.v_guess);
    solved = w.solver.SolveWithGuess(w.model, tolerance, &w.data);
    results.v_next = w.data.v();
  }
  if (!solved) {
    throw std::runtime_error(fmt::format(
        "The ICF solver failed to converge at simulation time = {}.",
        context.get_time()));
  }

  // From here on, w.data is the data of the full model at v_next. The
  // gradient of the cost of each constraint is ∇ℓ = −Jᵀ⋅γ, with γ the
  // impulse.
  const IcfModel<double>& model = w.model;
  const IcfData<double>& data = w.data;

  // Contact forces, for each contact pair.
  const PatchConstraintsPool<double>& patches = model.patch_constraints_pool();
  const int num_patches = patches.num_patches();
  const int num_pairs = patches.total_num_pairs();
  patches.CalcContactPairImpulses(data.patch_constraints_data(), &w.gamma_Bc_W);
  const EigenPool<Vector3<double>>& v_AcBc_W_pool =
      data.patch_constraints_data().v_AcBc_W_pool();
  results.patch_pair_start.resize(num_patches);
  results.patch_body_B.resize(num_patches);
  for (int p = 0; p < num_patches; ++p) {
    results.patch_pair_start[p] = patches.patch_pair_index(p, 0);
    results.patch_body_B[p] = patches.bodies()[p].first;
  }
  results.f_Bc_W.resize(num_pairs);
  results.v_AcBc_W.resize(num_pairs);
  for (int i = 0; i < num_pairs; ++i) {
    results.f_Bc_W[i] = w.gamma_Bc_W[i] / h;
    results.v_AcBc_W[i] = v_AcBc_W_pool[i];
  }

  // Generalized contact forces.
  w.gradient.setZero();
  patches.AccumulateGradient(data, &w.gradient);
  results.tau_contact = -w.gradient / h;

  // Generalized forces of joint limits and couplers.
  w.gradient.setZero();
  model.limit_constraints_pool().AccumulateGradient(data, &w.gradient);
  model.coupler_constraints_pool().AccumulateGradient(data, &w.gradient);
  results.tau_joint_constraints = -w.gradient / h;

  // Spatial forces on bodies due to contact, welds, balls and distances.
  w.Gamma_Bo_W.assign(plant.num_bodies(), Vector6<double>::Zero());
  patches.AccumulateBodySpatialImpulses(data.patch_constraints_data(),
                                        &w.Gamma_Bo_W);
  model.weld_constraints_pool().AccumulateBodySpatialImpulses(data,
                                                              &w.Gamma_Bo_W);
  model.ball_constraints_pool().AccumulateBodySpatialImpulses(data,
                                                              &w.Gamma_Bo_W);
  model.distance_constraints_pool().AccumulateBodySpatialImpulses(
      data, &w.Gamma_Bo_W);
  results.F_Bo_W.resize(plant.num_bodies());
  for (int b = 0; b < plant.num_bodies(); ++b) {
    results.F_Bo_W[b] = w.Gamma_Bo_W[b] / h;
  }

  // Net actuation τᵤ = clamp(−Kᵤ⋅v + bᵤ, e), the same as the gain constraints
  // of the model.
  results.has_actuation = has_actuation;
  results.tau_actuation.setZero(nv);
  if (has_actuation) {
    const VectorX<double>& Ku = w.actuation_feedback.K;
    const VectorX<double>& bu = w.actuation_feedback.b;
    for (JointActuatorIndex actuator_index : plant.GetJointActuatorIndices()) {
      const JointActuator<double>& actuator =
          plant.get_joint_actuator(actuator_index);
      const int dof = actuator.joint().velocity_start();
      const double e = actuator.effort_limit();
      const double y = -Ku[dof] * results.v_next[dof] + bu[dof];
      results.tau_actuation[dof] = std::clamp(y, -e, e);
    }
  }
}

void IcfDriver::CalcContactSolverResults(
    const Context<double>& context,
    ContactSolverResults<double>* results) const {
  DRAKE_DEMAND(results != nullptr);
  const IcfSolverResults& icf_results = EvalIcfSolverResults(context);
  const DiscreteContactData<DiscreteContactPair<double>>& contact_pairs =
      manager().EvalDiscreteContactPairs(context);
  const int num_point_pairs =
      manager().EvalGeometryContactData(context).get().point_pairs.size();
  const int num_contacts = contact_pairs.size();
  results->Resize(plant().num_velocities(), num_contacts);
  results->v_next = icf_results.v_next;
  results->tau_contact = icf_results.tau_contact;

  // The manager makes the contact pairs from the same geometry query results
  // as the ICF model (see CalcIcfSolverResults()). Each contact pair keeps the
  // index of its point pair, or of its surface and face. Thus we find the ICF
  // patch and pair from these indices. Note that the manager skips some
  // hydroelastic faces (e.g., faces with a very small area), but ICF does not.
  for (int i = 0; i < num_contacts; ++i) {
    const DiscreteContactPair<double>& pair = contact_pairs[i];
    int patch{};
    int k{};
    if (pair.point_pair_index.has_value()) {
      patch = *pair.point_pair_index;
      k = 0;
    } else {
      DRAKE_DEMAND(pair.surface_index.has_value());
      DRAKE_DEMAND(pair.face_index.has_value());
      patch = num_point_pairs + *pair.surface_index;
      k = *pair.face_index;
    }
    const int pk = icf_results.patch_pair_start[patch] + k;

    // The contact pair gives the force on its body B. When the body B of the
    // ICF patch is the body A of the contact pair, the force and the relative
    // velocity change sign.
    const double sign =
        (icf_results.patch_body_B[patch] == pair.object_B) ? 1.0 : -1.0;
    const math::RotationMatrix<double>& R_WC = pair.R_WC;
    const Vector3<double> f_Bc_C =
        R_WC.inverse() * (sign * icf_results.f_Bc_W[pk]);
    const Vector3<double> v_AcBc_C =
        R_WC.inverse() * (sign * icf_results.v_AcBc_W[pk]);
    results->fn(i) = f_Bc_C(2);
    results->ft.segment<2>(2 * i) = f_Bc_C.head<2>();
    results->vn(i) = v_AcBc_C(2);
    results->vt.segment<2>(2 * i) = v_AcBc_C.head<2>();
  }
}

void IcfDriver::CalcDiscreteUpdateMultibodyForces(
    const Context<double>& context, MultibodyForces<double>* forces) const {
  DRAKE_DEMAND(forces != nullptr);
  const MultibodyPlant<double>& plant = this->plant();
  const IcfSolverResults& icf_results = EvalIcfSolverResults(context);
  const double h = plant.time_step();
  const int nv = plant.num_velocities();
  VectorX<double>& generalized_forces = forces->mutable_generalized_forces();
  std::vector<SpatialForce<double>>& spatial_forces =
      forces->mutable_body_forces();

  // Forces at x₀ from force elements, actuation (without the PD controlled
  // actuators) and external inputs. ICF uses the same values for these forces.
  manager().CalcNonContactForces(context,
                                 /* include_joint_limit_penalty_forces */ false,
                                 /* include_pd_controlled_input */ false,
                                 forces);

  // ICF evaluates joint damping and reflected inertia implicitly. Therefore we
  // must subtract the explicit term evaluated above and include the implicit
  // term instead, the same as SapDriver.
  const VectorX<double>& x0 =
      context.get_discrete_state(manager().multibody_state_index()).value();
  const auto v0 = x0.tail(nv);
  const VectorX<double> a = (icf_results.v_next - v0) / h;
  generalized_forces -=
      manager().CalcEffectiveDamping(context).asDiagonal() * a;

  // Constraint forces.
  generalized_forces += icf_results.tau_joint_constraints;
  for (BodyIndex b(0); b < plant.num_bodies(); ++b) {
    // MultibodyForces indexes spatial body forces by MobodIndex.
    const MobodIndex mobod_index = plant.get_body(b).mobod_index();
    spatial_forces[mobod_index] += SpatialForce<double>(icf_results.F_Bo_W[b]);
  }

  // Actuation of PD controlled actuators, which CalcNonContactForces() above
  // left out.
  if (icf_results.has_actuation) {
    for (JointActuatorIndex actuator_index : plant.GetJointActuatorIndices()) {
      const JointActuator<double>& actuator =
          plant.get_joint_actuator(actuator_index);
      if (actuator.has_controller()) {
        const int dof = actuator.joint().velocity_start();
        generalized_forces[dof] += icf_results.tau_actuation[dof];
      }
    }
  }
}

void IcfDriver::CalcActuation(const Context<double>& context,
                              VectorX<double>* actuation) const {
  DRAKE_DEMAND(actuation != nullptr);
  // Without actuation in the ICF problem, the net actuation is the
  // (effort-limited) actuation input, which is zero.
  *actuation =
      manager().EvalActuationInput(context, /* apply_effort_limit = */ true);
  const IcfSolverResults& icf_results = EvalIcfSolverResults(context);
  if (!icf_results.has_actuation) {
    return;
  }
  const MultibodyPlant<double>& plant = this->plant();
  for (JointActuatorIndex actuator_index : plant.GetJointActuatorIndices()) {
    const JointActuator<double>& actuator =
        plant.get_joint_actuator(actuator_index);
    const int dof = actuator.joint().velocity_start();
    (*actuation)[actuator.input_start()] = icf_results.tau_actuation[dof];
  }
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake
