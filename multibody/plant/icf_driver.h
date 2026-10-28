#pragma once

#include <memory>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "drake/multibody/contact_solvers/icf/icf_solver_parameters.h"
#include "drake/multibody/tree/multibody_forces.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {

// Forward declaration.
template <typename>
class MultibodyPlant;

namespace internal {

// Forward declarations.
template <typename>
class CompliantContactManager;
struct IcfWorkspace;

/* The results of one ICF solve that IcfDriver needs to report the discrete
update. All forces are averages over the time step (impulse / δt). */
struct IcfSolverResults {
  /* Next-step generalized velocities, of size nv. */
  VectorX<double> v_next;

  /* True if the ICF problem has actuation (actuation input or PD control). */
  bool has_actuation{false};

  /* Net actuation for each velocity, of size nv, including PD control and
  effort limits. Only the actuated velocities have a meaning, and only if
  has_actuation is true. */
  VectorX<double> tau_actuation;

  /* Generalized contact forces, of size nv. */
  VectorX<double> tau_contact;

  /* Generalized forces of joint limit and coupler constraints, of size nv. */
  VectorX<double> tau_joint_constraints;

  /* Spatial force on each body B at its origin Bo, expressed in the world
  frame, due to contact, weld, ball and distance constraints. Indexed by
  BodyIndex. */
  std::vector<Vector6<double>> F_Bo_W;

  /* Index of the first contact pair of each ICF patch. Patch p is point pair p
  for p < number of point pairs. Patch p is hydroelastic surface p - (number of
  point pairs) for larger p, with one contact pair per face. See
  IcfBuilder::UpdateModel(). */
  std::vector<int> patch_pair_start;

  /* The dynamic body B of each ICF patch. ICF orders the bodies of a patch so
  that body B is never anchored. */
  std::vector<int> patch_body_B;

  /* Contact force on body B at the contact point C of each contact pair,
  expressed in the world frame. Indexed by patch_pair_start[p] + k, for pair k
  of patch p. */
  std::vector<Vector3<double>> f_Bc_W;

  /* Velocity of Bc relative to Ac for each contact pair, expressed in the world
  frame. Same indexing as f_Bc_W. */
  std::vector<Vector3<double>> v_AcBc_W;
};

/* The value of the cache entry for IcfDriver. The `results` have value
semantics. The `workspace` is scratch space for the solve, kept to reuse its
memory from step to step. A copy has a copy of the results, but not of the
workspace: the copy makes a new workspace when it is necessary. Thus each
context has its own workspace, and simulations in parallel can share a plant. */
struct IcfSolverResultsCache {
  IcfSolverResultsCache();
  IcfSolverResultsCache(const IcfSolverResultsCache& other);
  IcfSolverResultsCache& operator=(const IcfSolverResultsCache& other);
  ~IcfSolverResultsCache();

  IcfSolverResults results;
  std::unique_ptr<IcfWorkspace> workspace;
};

/* (Experimental) Performs the computations needed by CompliantContactManager
for discrete updates with the convex solver of Irrotational Contact Fields
(ICF). This is the solver that CenicIntegrator uses. See
DiscreteContactSolver::kIcf.

A discrete step with this driver is the same as a fixed step of CenicIntegrator
on a continuous plant, with one difference: a discrete plant holds its inputs
constant during the step. Thus:
  - Actuation is τᵤ = clamp(B⋅u(x₀), e), with no linearization in v.
  - External forces are τₑ = τₑ(x₀), with no linearization in v.
  - The PD controllers of the plant are implicit, with q = q₀ + δt⋅v:
      τᵤ = clamp(−Kp⋅(q₀ + δt⋅v − qd) − Kd⋅(v − vd) + u_ff, e).

The driver uses the geometry query results and the discrete contact pairs of
the manager. Thus a step does one geometry query only.

Only T = double is supported. */
class IcfDriver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IcfDriver);

  /* Constructs a driver for the given manager.
  @pre manager is not nullptr.
  @throws std::exception if the plant has a feature that ICF does not support
  (deformable bodies or tendon constraints). */
  explicit IcfDriver(const CompliantContactManager<double>* manager);

  ~IcfDriver();

  /* Sets the parameters of the ICF solver. The solver uses
  parameters.min_tolerance as its convergence tolerance, the same as
  CenicIntegrator in fixed step mode. */
  void set_icf_solver_parameters(
      const contact_solvers::icf::IcfSolverParameters& parameters);

  /* Declares the cache entries of the driver.
  @pre `mutable_manager` points to the manager given at construction. */
  void DeclareCacheEntries(CompliantContactManager<double>* mutable_manager);

  /* Computes the next-step velocities and the contact results, in the order
  of the discrete contact pairs of the manager. */
  void CalcContactSolverResults(
      const systems::Context<double>& context,
      contact_solvers::internal::ContactSolverResults<double>* results) const;

  /* Computes the multibody forces of the discrete update: force elements at x₀,
  implicit joint damping and reflected inertia, actuation, external forces and
  constraint forces (including contact). */
  void CalcDiscreteUpdateMultibodyForces(
      const systems::Context<double>& context,
      MultibodyForces<double>* forces) const;

  /* Computes the net actuation of the discrete update, including PD control
  and effort limits. */
  void CalcActuation(const systems::Context<double>& context,
                     VectorX<double>* actuation) const;

  /* Evaluates the cached results of the ICF solve for the state in
  `context`. */
  const IcfSolverResults& EvalIcfSolverResults(
      const systems::Context<double>& context) const;

 private:
  const CompliantContactManager<double>& manager() const { return *manager_; }

  const MultibodyPlant<double>& plant() const;

  /* Solves the ICF problem for the state in `context`. */
  void CalcIcfSolverResults(const systems::Context<double>& context,
                            IcfSolverResultsCache* cache) const;

  const CompliantContactManager<double>* const manager_;
  systems::CacheIndex icf_results_;
  contact_solvers::icf::IcfSolverParameters parameters_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
