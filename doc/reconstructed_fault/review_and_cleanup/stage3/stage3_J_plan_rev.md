  # Stage J — Atomic Constitutive History Feedback

  ## Summary

  Stage J will implement the accepted cycle

  [
  H_{k-1}\rightarrow\phi_k\rightarrow(u_k,p_k,V_k)
  \rightarrow{\Theta_k,T_k^{\rm coh},\tau_k,H_k,I_{h,k}}
  \rightarrow\phi_{k+1},
  ]

  with explicit timestep-zero semantics. All failure-capable history calculations
  and validation will occur before a terminal, non-throwing commit of histories,
  (V), and the bulk solution.

  The first pass will update and reread current_design.md and specification.tex.
  This will replace their obsolete ordering text and durably record the approved
  Stage-J decisions before production code is changed.

  Stage K benchmarks, fault propagation, 3-D faults, and overlapping faults remain
  deferred.

  ## Implementation Changes

  ### 1. Define bulk and fault-surface constitutive states

  Maintain two distinct coefficient evaluations:

  - Bulk Maxwell coefficients:
    [
    (\beta_b,\kappa_b)
    =\mathcal M(\mathbf f_b,T_b,\Delta t),
    ]
    using particle-local bulk composition and particle-local bulk temperature.

  - Surface cohesive coefficients:
    [
    (\beta_\Gamma,\kappa_\Gamma)
    =\mathcal M(\mathbf f_\Gamma,T_\Gamma,\Delta t),
    ]
    using projected Q1 surface composition and Q1 fault-surface temperature.

  At mechanical preparation, evaluate the frozen FE temperature at reconstructed-
  fault vertices and retain it as transient nodal fault-surface data. Q1
  interpolation defines (T_\Gamma(s)), ensuring every particle associated with the
  same fault coordinate uses the same surface temperature. This temperature is
  recomputed each mechanical timestep and is neither a particle history nor a
  checkpointed fault property.

  Correct the Stage-F/G constitutive evaluation accordingly:

  - Maxwell stress and the stress-derived (B/G) coefficients use (\kappa_b).
  - Cohesive traction, cohesive history correction, and
    (\partial T^{\rm coh}/\partial V=\kappa_\Gamma/I_h) use surface coefficients.

  - The (K_V) stress term remains
    (2\kappa_b\chi,S:S), while its cohesive term becomes
    (\kappa_\Gamma/I_h).

  - Friction, cohesive mechanics, (\Theta) evolution, and the RSF timestep
    restriction use the Q1 surface material mixture. Particle-local compositions
    never enter these surface operations.

  - Because the existing aging law has a global (D_c), the surface mixture does not
    currently alter the numeric update_state() result; no new (D_c) parameter will
    be introduced.

  ### 2. Preserve explicit timestep-zero history semantics

  The converged timestep-zero mechanical solve will:

  - solve and commit (V_0), making it the initial kinematic solution used by an
    explicitly selected RSF timestep model to choose the first real timestep;

  - retain the user-supplied (\Theta_0), without calling update_state() using
    Initial time step;

  - retain the initialized irreversible (H_0), without applying the finite-step
    cohesive-work update;

  - ensure current (I_{h,0}) is committed as the previous-(I_h) snapshot for
    timestep one;

  - retain the established initialization-specific (T_0^{\rm coh}) constructed from
    the initial phase field and (H_0);

  - retain the user-initialized Maxwell particle stress rather than applying a
    timestep-zero Maxwell evolution pass.

  For timestep (k>0), the actual simulator timestep drives all history updates.

  ### 3. Compute later-time history candidates without mutation

  For a converged mechanical state at (k>0):

  1. Evaluate accepted velocity gradients and local bulk temperatures at all locally
     owned phase-field particles.

  2. Read particle-local chemical compositions and old Maxwell stress for the bulk
     Maxwell coefficients.

  3. At every active particle/fault association, Q1-interpolate:
      - accepted (V_k);
      - surface chemical compositions and material fractions;
      - fault-surface temperature;
      - current and previous (I_h);
      - previous cohesive traction and, where applicable, (\Theta_{k-1}).

  4. Evaluate particle samples
     [
     T_k^{\rm coh}
     =\frac{\kappa_{\Gamma,k}V_k+
     \beta_{\Gamma,k}I_{h,k-1}T_{k-1}^{\rm coh}}
     {I_{h,k}}
     ]
     and consistently project them to the replicated Q1 fault.

  5. Validate the complete projected field and retain transverse projection
     diagnostics without imposing a new rejection tolerance.

  6. Compute nodal rate-and-state candidates with the accepted nodal (V_k):
     [
     \Theta_k=\operatorname{update_state}
     (V_k,\Theta_{k-1},\Delta t_k).
     ]
     The update is associated with the same nodal surface mixture used by surface
     friction. Rate-dependent friction creates no state candidate.

  7. Interpolate the projected Q1 (T_k^{\rm coh}) back to each associated particle
     before evaluating (H_k).

  8. Compute the history-corrected crack strain rate and candidate Maxwell stress:
     [
     \tau_k=2\kappa_b(\dot\epsilon_k-\upsilon_kS)
     +\beta_b\tau_{k-1}.
     ]
     Inactive particles use the accepted bulk strain rate without a fault correction
     and retain their previous (H).

  No transaction class and no particle copy of (T^{\rm coh}) will be added.
  Candidate data and interpolation helpers remain private or file-local.

  ### 4. Evaluate finite-step (H) stably

  For (g_k<1), deliberately evaluate the exact expression using

  [
  a=\frac{T_k^{\rm coh}}{g_k},
  \qquad
  b=\frac{\beta_{\Gamma,k}h_{k-1}T_{k-1}^{\rm coh}}
  {1-g_k},
  ]

  [
  \mathcal H_k=
  \frac{\Delta t_k}{2\kappa_{\Gamma,k}}(a-b)(a+b),
  \qquad
  H_k=\max(H_{k-1},\mathcal H_k).
  ]

  This is the finite-step cohesive-work expression, not its small-step
  approximation.

  - A negative (\mathcal H_k) remains a valid candidate. It is not independently
    clamped to zero; irreversibility is imposed only by max(H_old, candidate).

  - At (g_k=1), (h_k=h_{k-1}=0), use
    [
    \mathcal H_k=
    \frac{\Delta t_k}{2\kappa_{\Gamma,k}}
    (T_k^{\rm coh})^2.
    ]

  - If (g_k=1) but (h_{k-1}>0), report inadmissible healing.
  - Continue to reject non-finite degradation, coefficients, tractions, or resulting
    history candidates.

  ### 5. Make the accepted-state commit genuinely atomic

  Retain one narrow material operation:

  PhaseFieldFault<dim>::
  commit_reconstructed_fault_mechanical_history(
    const LinearAlgebra::BlockVector &accepted_bulk_state);

  It will branch internally between timestep-zero preservation and later-time
  evolution. For later timesteps it will compute, globally validate, and fully
  allocate all history candidates before making any persistent write.

  Before calling it, the solver will also:

  - prevalidate the manager’s active/current (V) state and its exact fault/vertex
    layout;

  - prevalidate accepted bulk-vector block sizes and ownership;
  - construct correctly laid-out staging vectors for bulk publication and the next
    linearization point;

  - ensure no further residual assembly, MPI reduction, allocation, or validation is
    needed after commit begins.

  The terminal mutation phase will contain only non-allocating operations:

  1. write prevalidated Q1 (T^{\rm coh}), previous (I_h), and optional (\Theta);
  2. write prevalidated locally owned particle Maxwell stress and (H);
  3. commit current (V) using a prevalidated, non-allocating manager operation;
  4. publish the accepted bulk solution and linearization point using preallocated
     vector swaps or equivalent non-allocating operations;

  5. close the nonlinear lifecycle.

  Normal particle migration/ghost synchronization will propagate the committed
  owned-particle histories through the existing particle lifecycle; Stage J will not
  introduce a failure-capable MPI exchange after the terminal commit begins.

  Any exception before this terminal phase follows the existing rollback path and
  leaves the production bulk solution, timestep-committed (V), all Q1 histories, and
  all particle histories unchanged. Post-commit notifications will not be treated as
  an opportunity to roll back an already accepted physical state.

  ### 6. Add an explicitly selected RSF timestep model

  Add:

  double
  PhaseFieldFault<dim>::
  compute_reconstructed_fault_time_step(double cfl_number) const;

  It will use timestep-committed Q1 (V), Q1 surface material fractions, and the
  existing

  FaultFriction::compute_time_step(..., use_operator_splitting=true)

  operation.

  Register an ASPECT time-stepping model named reconstructed fault time step. It
  will use ASPECT’s existing global CFL number and participate in the standard
  minimum over selected timestep models.

  It will not be activated automatically. In particular:

  - an empty Time stepping/List of model names retains ASPECT’s ordinary default
    selection;

  - an explicit model list is preserved exactly;
  - users who want the RSF restriction must explicitly include reconstructed fault
    time step;

  - Stage J adds no post-solve RSF cutback or repeat behavior.

  Rate-dependent friction and rate-strengthening rate-and-state mixtures retain the
  existing unrestricted return value.

  ## Public Interfaces

  The minimum new or adjusted interfaces are:

  - PhaseFieldFault::commit_reconstructed_fault_mechanical_history(accepted_bulk_sta
    te)

  - PhaseFieldFault::compute_reconstructed_fault_time_step(cfl_number) const
  - the registered reconstructed fault time step plugin
  - a manager-side prevalidation/non-allocating commit contract for the existing
    nonlinear (V) lifecycle

  Fault-surface temperature remains private transient PhaseFieldFault data. No new
  physical parameter, transaction class, constitutive base class, or public
  projection abstraction will be introduced.

  ## Test Plan

  - Add a timestep-zero lifecycle test proving:
      - (V_0) is solved and committed;
      - (\Theta_0) and (H_0) are unchanged by the artificial Initial time step;
      - (T_0^{\rm coh}), Maxwell stress, and (I_{h,0}) follow their approved
        initialization-specific rules;

      - an explicitly selected RSF timestep model can use committed (V_0) to choose
        the first real timestep.

  - Add the transverse-temperature regression: two particles at the same fault
    coordinate but with different bulk temperatures must obtain different bulk
    Maxwell coefficients and identical surface cohesive coefficients from their
    shared Q1 fault-surface temperature.

  - Extend constitutive tests to distinguish (\kappa_b) from (\kappa_\Gamma),
    including normally varying particle compositions with a fixed surface mixture.

  - Verify the exact finite-step (H) calculation against an independent reference
    for:
      - ordinary damaged states;
      - small Maxwell exponents;
      - a negative candidate, proving it is not clamped independently;
      - irreversible retention through max(H_old, candidate);
      - the exactly intact removable limit;
      - inadmissible healing at (g_k=1,\ h_{k-1}>0);
      - clear numerical distinction from the small-step approximation.

  - Verify Q1 cohesive projection and prove particle (H) uses the projected traction
    interpolated back to the particle rather than the raw particle sample.

  - Verify rate-and-state (\Theta) evolution uses the surface-state path and that
    rate-dependent friction performs no state update.

  - Test the opt-in timestep plugin for rate-weakening, rate-strengthening, and
    rate-dependent laws, plus its standard minimum interaction with another
    explicitly selected ASPECT timestep model.

  - Add a two-real-timestep integration test showing that (\phi_k) consumes (H_{k-
    1}), histories remain frozen during nonlinear trials, convergence commits all
    later-time histories, and (\phi_{k+1}) consumes (H_k).

  - Force nonlinear failure after at least one accepted Newton update and verify
    complete preservation of production bulk state, committed (V), Q1 histories,
    Maxwell stress, and (H).

  - Exercise history commit with one and two MPI ranks and add a focused checkpoint/
    restart comparison.

  - Build relevant targets with -j4 and run only Stage-J and directly affected
    reconstructed-fault tests; do not run the complete ASPECT suite.

  ## Assumptions

  - Fault-surface temperature means the current frozen FE temperature sampled at
    reconstructed-fault vertices and Q1-interpolated along the fault.

  - The existing global (D_c) remains authoritative for the aging law; Stage J does
    not invent compositional (D_c).

  - Fault geometry stays fixed after timestep-zero reconstruction.
  - (H) changes only on actively associated particles; inactive particles retain
    their existing irreversible history.

  - Existing fault-property and particle checkpoint mechanisms remain authoritative.
