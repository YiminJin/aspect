  # Stage C — Distributed adaptive (I_h) with separated phase-field thresholds

  ## Summary

  Implement Stage C only: compute transient (I_h) from the distributed Q1 phase field, using one fault-surface material mixture (\mathbf f_\Gamma) per segment quadrature
  point and holding it fixed over the complete (+\mathbf n/-\mathbf n) profile. Refactor phase-field range semantics without changing reconstruction, refinement,
  prescribed-fault, or slip-rate-normalization behavior.

  No cohesive state, friction coupling, (K_V), or Stage D lifecycle work will be added.

  ## Interface and semantic changes

  - Redefine MaterialModel::PhaseFieldModel::get_phase_field_range() as the physical invariant range and return exactly ([0,1]).
  - Add distinct virtual accessors:
      - get_phase_field_activation_threshold()
      - get_phase_field_upper_admissibility_threshold()

  - Preserve historical defaults of 0.01 and 0.99 in the generic base class. PhaseFieldFault overrides the activation accessor with its configured threshold and
    explicitly returns 0.99 as its current fault-model upper admissibility threshold.

  - Migrate every existing caller according to its invariant:
      - mesh refinement and reconstruction weights/support use the activation threshold;
      - prescribed fault-core validation uses activation and upper-admissibility thresholds;
      - SlipRateNormalizer uses activation and upper-admissibility thresholds.

  - Add a thin PhaseFieldHandler::get_length_scale() accessor forwarding the already configured geometric-function length scale.
  - Add a geometry-only ReconstructedFaultManager::project_to_normal_profiles(point) forwarding accessor so the private (I_h) kernel can detect other-fault encounters
    without exposing projection widths.

  - Keep the (I_h) evaluator and its transient nodal/profile data private to PhaseFieldFault; expose only a narrow friend test accessor. Add no public constitutive (I_h)
    API or new simulator lifecycle hook.

  - Add numerical parameters under PhaseFieldFault for quadrature and tail tolerances, both defaulting to (10^{-8}). Refinement-depth, boundary-bisection, and outward-
    extension guards remain fixed internal safeguards.

  ## Stage C implementation

  - During PhaseFieldFault::initialize(), register one generic fault property containing the raw chemical compositional-field components, unless there are no chemical
    fields.

  - Before evaluating (I_h), project each mapped particle compositional-field component to that property through the existing generic particle-to-fault projection. Reuse
    the configured particle-property mappings and existing coverage diagnostics; introduce no duplicate composition parameters.

  - Enumerate QGauss<1>(3) profiles in fault-major, segment-major, quadrature-point order and assign balanced contiguous ID ranges to MPI ranks.
  - At each profile origin:
      - Q1-interpolate the projected chemical components;
      - convert them once with ASPECT’s existing composition-fraction utility to obtain (\mathbf f_\Gamma), including the background fraction;
      - validate that the resulting fractions are finite, nonnegative, and normalized;
      - retain this same (\mathbf f_\Gamma) for both normal directions and every sample on that profile.

  - At every normal-profile sample evaluate
    \[
    \bar g(\phi,\mathbf f_\Gamma)=\sum_m f_{\Gamma,m}g_m(\phi),
    \qquad
    h=1/\bar g-1.
    \]
    Never sample or recompute composition away from the profile origin.

  - Batch all owner-generated phase-field requests collectively with deal.II distributed point evaluation. Owners alone retain adaptive state and make refinement/tail
    decisions; there is no rank-zero coordinator.

  - Use initial panel width
    \[
    \Delta\zeta_0=\tfrac12\min(\ell,h_{\rm local}),
    \]
    compare 4- and 8-point Gauss estimates, bisect until
    \[
    |I_8-I_4|\le\epsilon_{\rm quad}\max(|I_8|,\ell),
    \]
    and regrow accepted widths up to (\min(2\Delta\zeta,\ell/2,h_{\rm local}/2)).

  - Integrate (+\mathbf n) and (-\mathbf n) independently. Terminate only after two successive non-overlapping outer windows, each spanning at least (\ell), satisfy
    \[
    I_{\rm window}\le\epsilon_{\rm tail}\max(I_{\rm accumulated},\ell).
    \]
    The activation threshold is never an (I_h) integration boundary, and no monotonic-tail condition is imposed.

  - Permit profile tails to attain (\phi=0). For every phase-field sample:
      - require finite (0\le\phi\le1) as the physical internal invariant, without clamping;
      - then require finite (\bar g>0);
      - report (\bar g=0), including the (\phi=1) case, as an explicit I_h constitutive singularity with fault/profile/sample metadata—not as a phase-field range error;
      - reject negative, reciprocal-overflowing, or non-finite (h) with the same singularity diagnostics.

  - Detect a different reconstructed-fault influence region before tail completion and issue an explicit unsupported-overlap error.
  - On leaving the domain, collectively bracket and bisect the first boundary crossing, integrate the remaining connected in-domain interval, terminate that side, and
    prohibit disconnected re-entry.

  - Assemble owner-unique Q1 fault mass/RHS contributions, MPI-sum them, solve the replicated tridiagonal mass systems, and verify finite positive nodal (I_h) and rank
    consistency. Store current (I_h) only in private transient PhaseFieldFault state.

  ## Tests and documentation

  - Add private-kernel tests for:
      - smooth analytical and stationary phase-field profiles;
      - compact support whose tail reaches exactly (\phi=0);
      - weak integrable and non-monotonic tails;
      - panel bisection and width regrowth;
      - physical-range violations below 0 and above 1;
      - (\phi=1,\bar g=0) producing an explicit (I_h)-singularity diagnostic;
      - a separate second-fault encounter;
      - boundary clipping without re-entry;
      - a heterogeneous surface mixture proving that one (\mathbf f_\Gamma) is fixed across both profile sides.

  - Add integration tests for one-rank/two-rank equivalence, bulk-mesh convergence, and Q1 fault-surface projection convergence.
  - Add regression tests demonstrating that reconstruction and mesh refinement still use the activation threshold and that prescribed-core/normalizer behavior still uses
    the former admissibility bounds.

  - Run the focused unit-test executable, relevant reconstructed-fault regression tests, and the MPI Stage C test with one and two ranks; report exact commands and
    results.

  - Update current_design.md, specification.tex, and the Stage C architectural document to:
      - distinguish physical range, activation threshold, and fault-model upper admissibility;
      - specify the fixed-per-profile (\mathbf f_\Gamma) rule;
      - state the applicability condition (L_{\rm mat}\gg\ell);
      - explain that models with material transitions on the diffuse-fault scale or narrower are outside this formulation’s validity.

  - Do not update superseded redesign documents.

  ## Assumptions

  - Stage C remains two-dimensional, matching current reconstructed-fault geometry support.
  - Chemical composition is obtained from the existing mapped particle properties, projected component-wise, then converted to material fractions at the surface quadrature point. With no chemical fields, \(\mathbf f_\Gamma=(1)\) represents the background material.

  - The projected mixture—not bulk composition sampled along the normal—is authoritative for \(I_h\).
  - The current fault-model upper admissibility threshold remains 0.99; this refactor does not introduce a new user parameter for it.
  - \(I_h\) recomputation is invoked only through the private Stage C/test seam for now; mechanical-solver lifecycle integration belongs to a later explicitly requested stage.
