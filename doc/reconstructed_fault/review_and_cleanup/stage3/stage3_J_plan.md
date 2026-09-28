  # Stage J — Constitutive History Commit and Phase-Field Feedback

  ## Summary

  Stage J will evolve and atomically commit the constitutive histories after a successful coupled mechanical solve, then expose the committed crack-driving force to the next phase-field solve:

  \[
  H_{k-1}\rightarrow\phi_k\rightarrow(u_k,p_k,V_k)
  \rightarrow{\Theta_k,T_k^{\rm coh},\tau_k,H_k,I_{h,k}}
  \rightarrow\phi_{k+1}.
  \]

  The current specification still contains an obsolete “bulk fields before phase field” ordering that conflicts with this approved indexed cycle and the existing simulator schedule. The first implementation pass will correct current_design.md and `specification.tex`, record all decisions from the Stage-J supplement, and reread that record before changing production code.

  Stage J will not implement fixed-fault benchmarks, propagation, 3-D faults, or overlapping faults.

  ## Implementation Changes

  ### 1. Correct and complete the authoritative Stage-J design

  Document these invariants and equations before production changes:

  - Phase field and fault geometry remain fixed throughout the mechanical nonlinear solve.
  - All histories remain frozen during residual evaluation, Jacobian assembly, Krylov iterations, and line-search trials.
  - Histories are computed and committed only after coupled convergence. Any rejected trial or nonlinear failure leaves every committed history unchanged.
  - Rate-and-state friction commits
    \[
    \Theta_k=\operatorname{update_state}(V_k,\Theta_{k-1},\Delta t_k)
    \]
    at Q1 fault vertices. Rate-dependent friction has no state field.

  - Cohesive history uses the projected surface mixture and
    \[
    T_k^{\rm coh}
    =\frac{\kappa_{\Gamma,k}V_k+
    \beta_{\Gamma,k}I_{h,k-1}T_{k-1}^{\rm coh}}{I_{h,k}}.
    \]

  - The particle crack-driving candidate uses the projected Q1 (T_k^{\rm coh}), interpolated back to the cached particle coordinate:
    \[
    \mathcal H_k=
    \frac{\Delta t_k}{2\kappa_{\Gamma,k}}
    \frac{(h_kT_k^{\rm coh})^2-
    (\beta_{\Gamma,k}h_{k-1}T_{k-1}^{\rm coh})^2}
    {(1-g_k)^2},
    \qquad
    H_k=\max(H_{k-1},\mathcal H_k).
    \]

  - Evaluate this expression in an algebraically stable form without replacing it by the small-step approximation.
  - At an exactly intact current point with
    \(g_k=1,\ h_k=h_{k-1}=0\), use the approved removable limit
    \[
    \mathcal H_k=
    \frac{\Delta t_k}{2\kappa_{\Gamma,k}}(T_k^{\rm coh})^2.
    \]
    If (g_k=1) but (h_{k-1}>0), report inadmissible healing.

  - Timestep zero uses the parsed positive Initial time step; subsequent commits use the actual simulator timestep.

  ### 2. Separate bulk and surface constitutive coefficients

  Correct the existing Stage-F point evaluation so its mechanics and Stage-J commit use the same coefficient ownership:

  - Compute \((\beta_b,\kappa_b)\) from the particle-local bulk material fractions. These coefficients govern the Maxwell stress and the bulk \(B/G\) terms.
  - Compute \((\beta_\Gamma,\kappa_\Gamma)\) from the projected Q1 surface material mixture at the cached fault coordinate. These coefficients govern \(T^{\rm coh}\), the cohesive history correction, and \(\partial T^{\rm coh}/\partial V=\kappa_\Gamma/I_h\).
  - Keep temperature local to the evaluation point; only the material/composition mixture is held to the projected surface value.
  - Preserve the existing profile-uniform surface composition rule. Particle-local chemical compositions must never enter the cohesive law.
  - In \(K_V\), retain the bulk stress derivative \(2\kappa_b\chi\bm S:\bm S\), but use \(\kappa_\Gamma/I_h\) for the cohesive derivative.
  - Keep the solver-facing bulk response’s kappa as \(\kappa_b\), so the established \(B/G\) semantics do not change.

  ### 3. Add one atomic post-convergence history operation

  Add the narrow public operation:
  ```
  PhaseFieldFault<dim>::
  commit_reconstructed_fault_mechanical_history(
    const LinearAlgebra::BlockVector &accepted_bulk_state);
  ```
  It will perform these mathematical phases:

  1. Evaluate accepted velocity gradients, temperature, current phase field, and previous phase field at all locally owned phase-field particles.
  2. For active particle/fault associations, interpolate accepted \(V_k\), old Q1 history, current \(I_h\), and the projected surface mixture at the cached fault coordinate.
  3. Evaluate particle samples of \(T_k^{\rm coh}\) with surface coefficients and consistently project them to the replicated Q1 fault.
  4. Validate the complete projected traction field and retain the existing transverse projection diagnostics without adding a rejection threshold.
  5. Compute candidate nodal \(\Theta_k\) for rate-and-state friction.
  6. Interpolate the newly projected Q1 \(T_k^{\rm coh}\) back to associated particles and compute:
      - the exact finite-step \(H_k\);
      - the exact history-corrected crack strain rate;
      - the new particle Maxwell stress
        \[
        \tau_k=2\kappa_b
        \left(\dot{\bm\epsilon}_k-\upsilon_k\bm S\right)
        +\beta_b\bm\tau_{k-1}.
        \]
  7. For particles outside the reconstructed-fault profile, update Maxwell stress with the accepted bulk strain rate and leave particle \(H\) unchanged.
  8. Perform MPI-wide validation of every candidate before the first persistent write.
  9. In one terminal, non-throwing mutation phase, commit:
      - Q1 \(T_k^{\rm coh}\);
      - current \(I_{h,k}\) as the next previous-\(I_h\) history;
      - Q1 \(\Theta_k\), when applicable;
      - particle Maxwell stress and irreversible \(H_k\).

  No transaction class or particle copy of \(T^{\rm coh}\) will be introduced. Candidate containers and interpolation helpers will remain private or file-local.

  The solver convergence branch will call this operation before committing manager-owned \(V\) and publishing the accepted bulk solution. All validation and other failure- capable work must precede the history writes. The timestep-zero-only guard will then be removed. The existing rollback path remains responsible for bulk state and trial/current \(V\); histories require no rollback because they remain untouched until terminal convergence.

  ### 4. Connect the RSF timestep restriction

  Add a narrow semantic operation:
  ```
  double
  PhaseFieldFault<dim>::
  compute_reconstructed_fault_time_step(double cfl_number) const;
  ```
  It will:

  - use timestep-committed Q1 \(V\) and the projected Q1 surface material mixture;
  - call the existing `FaultFriction::compute_time_step()` with use_operator_splitting=true;
  - take the minimum over all fault vertices;
  - return the largest finite double for rate-dependent friction or velocity-strengthening rate-and-state regions, as already defined by FaultFriction;
  - diagnose missing committed state instead of silently disabling the restriction.

  A dedicated reconstructed fault time step implementation of ASPECT’s existing time-stepping interface will forward the global ASPECT CFL number to this operation. The time-stepping manager will automatically activate it whenever reconstructed faults are enabled, including when the user supplies an explicit time-stepping plugin list, while avoiding duplicate activation.

  The manager’s existing global minimum combines this restriction with convection, conduction, maximum-step, and termination limits. Stage J will not add post-solve RSF cutback or timestep repetition.

  ## Public Interfaces

  New public operations are limited to:
  
  - `PhaseFieldFault::commit_reconstructed_fault_mechanical_history(accepted_bulk_state)`
  - `PhaseFieldFault::compute_reconstructed_fault_time_step(cfl_number) const`
  - the registered ASPECT time-stepping model reconstructed fault time step

  No new manager property type, history transaction, constitutive base class, or public particle-projection abstraction will be added.

  ## Test Plan

  - Extend pointwise constitutive tests to verify distinct bulk and surface Maxwell coefficients, including a normally varying particle composition whose cohesive response remains controlled by the fixed surface mixture while bulk Maxwell/(B/G) coefficients remain particle-local.

  - Test the finite-step \(\mathcal H_k\) equation against an independent direct calculation, including:
      - ordinary damaged states;
      - very small Maxwell exponents;
      - distinction from the small-step approximation;
      - irreversible max(old, candidate) behavior;
      - the approved exactly intact removable limit;
      - rejection of \(g_k=1,\ h_{k-1}>0\).

  - Test consistent Q1 traction commit and prove that particle (H) uses the projected traction interpolated back to the particle, not the pre-projection sample.

  - Test exact nodal \(\Theta\) evolution for rate-and-state friction and absence of state updates for rate-dependent friction.

  - Test the automatic timestep restriction for:
      - rate-weakening rate-and-state friction;
      - velocity-strengthening rate-and-state friction;
      - rate-dependent friction;
      - interaction with another ASPECT timestep restriction through the minimum rule.

  - Add a two-timestep integration test demonstrating:
      - \(\phi_k\) reads \(H_{k-1}\);
      - histories remain fixed throughout nonlinear trials;
      - convergence commits \(T^{\rm coh}\), \(I_h\), \(\Theta\), Maxwell stress, \(H\), and \(V\);
      - the following phase-field solve reads the newly committed \(H_k\).

  - Extend forced nonlinear-failure coverage so failure after at least one accepted Newton update preserves the production bulk solution, timestep-committed \(V\), Q1 constitutive histories, particle Maxwell stress, and particle \(H\).

  - Run the history-commit and timestep tests with one and two MPI ranks, and add a focused checkpoint/restart comparison for the newly evolving persistent histories.

  - Build relevant targets with -j4 and run only the focused Stage-J and affected reconstructed-fault tests. Do not run the complete ASPECT test suite.

  ## Assumptions and Deferred Work

  - Fault geometry is fixed after timestep-zero reconstruction; Stage J does not reconstruct or extend faults at later timesteps.
  - Particle-local temperature is used when evaluating both bulk and surface-mixture viscosities.
  - \(H\) is updated only for particles actively associated with a reconstructed-fault normal profile; other particles retain their previous crack-driving history.
  - Existing particle and generic fault-property checkpointing remains the persistence mechanism.
  - Stage K fixed-fault benchmarks, propagation, 3-D reconstructed surfaces, and multi-fault overlap remain deferred.
