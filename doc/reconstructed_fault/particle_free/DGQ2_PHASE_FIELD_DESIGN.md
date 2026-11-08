> **Bulk-tip use, 7 November 2026:** the new guarded `bulk tip sections` route
> reuses the existing authoritative DGQ2 fracture/SIPG operators and Bernstein
> constraint coordinates for bulk energetic growth. It adds no shared-advection
> refactor or transport integrator change. Native quadrature tau/A histories,
> nonlocal RSF closure and qualification limits are documented in
> [TIP_TREATMENT.md](TIP_TREATMENT.md) and its implementation report. The current
> phase architecture remains the restored native implementation.

# Native phase in ASPECT: current implementation contract

> **Completed numerical validation:** the four restored-build A/B/D/C cases
> reached T=.8; see [report](DGQ2_COUPLED_VALIDATION_REPORT.md). Both directions
> are stable, but coupled spatial stress/state sensitivity and localized-work
> integration remain unresolved. The architecture freeze remains in force;
> no follow-up run, optimization, server pilot or cohesive extension is selected.

> **Frozen active architecture (6 November 2026):** this contract describes the
> restored native-integration source (`84f57503c`, restored by `319b10b94`).
> Keep the registered phase cell assembler and private mapped phase face loops,
> SSPRK3/BP, quadrature and obstacle/acceptance rules fixed during the next
> numerical-validation stage. Full ordinary-face sharing is a reverted review
> experiment, not the active implementation. See
> [restoration](PHASE_ARCHITECTURE_RESTORATION.md). Long experiments remain deferred
> until explicitly selected.

The selected [integration task](CODEX_NATIVE_PHASE_ASPECT_INTEGRATION.txt)
supersedes the separate native phase-block architecture. The
[integration report](NATIVE_PHASE_INTEGRATION_REPORT.md) owns current verification;
the [original DGQ2 report](DGQ2_PHASE_FIELD_REPORT.md) retains the completed long
comparisons. Neither report enables moving cohesive growth or production use.

## One field, one attempted transport

Declare exactly one composition named `phase_field`, with semantic type and
method `phase field`. Like porosity, it has special physics in ordinary
composition storage. It is not a chemical fraction: chemical material lists
use chemical indices, unit-sum normalization rejects it, and its method excludes
particle mapping and generic reaction/implicit-advection solves. The phase
handler resolves its component/block through `Introspection::phase_field()`;
it adds no second variable. Legacy CPDI still declares its separate Q1 field.

Initialization uses `Initial composition model`. Native phase no longer reads
the material's private initial-profile function/file. The retained fixed-fault
seed is supplied by a benchmark initial-composition plugin. Composition output
includes phase once. All mechanics, normalization and fracture derivatives read
this same field. Accepted and candidate histories retain their existing snapshots;
these are time levels, not competing authoritative phase discretizations.

The ordinary simulator timestep prepares the accepted material parent and bulk
spin, transports stress, then dispatches `solve_native_phase_transport` for the
phase composition. Full accepted velocity is frozen through that one attempt.
`stage_material_geometry` explicitly receives the transported phase, binds it
with transported fault labels/surface state, and rebuilds normalization. Rotation
and coupled mechanics follow. The generic composition loop skips this already
handled field. Retries prepare from the same accepted parent with the new dt.
There is no new mature timestep driver. Q1 comparison uses the retained remapper.

Supported fixed-fault growth uses accepted phase as its lower obstacle. Moving
mature phase is transported with growth disabled. The existing endpoint/KKT and
energy acceptance, final slip guess, accepted-gradient spin lag, full plane-strain
stress and RSF laws are unchanged. Joint acceptance publishes FE/surface history
once; rollback restores the parent and staged geometry. Native publication does
not require particles. Optional particles sample the native candidate and do not
supply its constitutive parent or physical acceptance criteria.

## Material advection and mapped bounds

On a fixed bilinear quadrilateral mesh, the material equation and weak rate are

    phi_dot + v.grad(phi) = 0
    (w, phi_dot) = (grad(w).v, phi) + (w div(v), phi)
                  - <w, (v.n) phi_upwind>.

`Assemblers::PhaseTransport` occupies the phase slot in the ordinary assembler
manager and uses ASPECT scratch/copy bases. Its spatial cell kernel is also used
by the standalone operator. `DGOperator` owns distributed spatial data and local
mass inverses, and applies SSPRK3/subcycling without a global implicit solve.
The existing BE/BDF2 composition cell assembler is deliberately not used.

`FEValues`, `FEFaceValues` and `FESubfaceValues` supply physical gradients,
normals, points and JxW. Neighbor traces are evaluated at the same physical points
through inverse mapping, including static coarse/fine faces. There are no physical
axis/coordinate-overlap assumptions in phase flux or SIPG assembly. Each rank
assembles its owned rows. Periodic topology, curved higher-order mappings, dynamic
AMR transfer, ALE and 3-D native phase remain outside this admission boundary.

Let Jmin be the minimum cell Jacobian determinant at the reference corners.
For a valid 2-D bilinear map the determinant is affine, so this is a positive
lower bound throughout the cell. For a nonnegative reference Q2 polynomial,
Lobatto integration gives `integral_face p <= 6 integral_cell p`. A sufficient
mean-loss rate is

    r_K = sum_faces 6 max_face((v.n)+ J_face) / Jmin
          + max(0, -min_volume div(v)).

Subface Jacobians are converted to full reference-face scale before taking the
maximum over face quadrature nodes. Choose substep `ds * max_K r_K <= .8`. Positive quadrature weights,
bounded incoming traces and the constant-state identity then bound both the
phase mean and its complementary upper gap. Bernstein enclosure certifies
whole-cell bounds; scaling about the physical J-weighted mean preserves that
mean. An inadmissible mean is rejected, never clipped or globally corrected.
The physical singularity at phi=1 is not replaced by an epsilon cap.

Four-point Lobatto quadrature integrates Q2 mass times the bilinear Jacobian
exactly. Five-point volume/face rules integrate the material weak form with
mapped Q2 velocity: the Jacobian cancels inverse-Jacobian factors in the gradient
terms. This is not an assertion that arbitrary nonpolynomial velocity or mapped
SIPG stiffness is integrated exactly. Both the nonzero-divergence identity and
manufactured spatial errors are checked separately.

Boundary values come from the phase composition's fixed boundary indicators and
boundary model. Values are requested only on locally incoming faces, at SSP stage
times t, t+ds and t+ds/2. Function and box models implement that explicit-time
interface; other models must implement it or reject with a diagnostic. The
compact-profile fixture explicitly selects zero data on all four boundaries.
A nonzero manufactured inflow tests time dependence and flux accounting.

## Fracture energy, constraints and caches

Let A=Gc/(c0 ell), k=2 A ell^2. Add the existing local A alpha(phi) and
recoverable mechanical energy to

    1/2 sum_K (k grad(phi), grad(phi))
    - sum_F <{k grad(phi).n}, [phi]> + 1/2 sum_F <sigma [phi], [phi]>.

`dg_sipg.cc` differentiates this functional into one distributed sparse operator.
Both consistency terms and symmetric Nitsche boundary terms are retained; native
natural zero normal derivative adds no terms. The normal trace scale is bounded
by full-face J divided by cell Jmin; sigma uses the maximum neighboring scale,
coefficient and (p+1)^2. Static subface scaling is included on both sides.
`scalar_symmetric_penalty` is shared with the existing ordinary advection
boundary assembler, without changing its thermal or time prefactors. Native
fracture coefficients remain homogeneous under the existing material admission.

The constrained solve uses Bernstein coefficients of phi-phi_obstacle and of
the physical upper gap. Their nonnegativity is sufficient for whole-cell bounds,
and is generally stronger than pointwise irreversibility. Growth can change cell
means. No transport limiter or physical-time mass term is applied to equilibrium.
Normalization derivatives and their mechanical work adjoint remain in the coupled
functional and final nonlinear/KKT/energy acceptance.

The handler caches cell mass inverses and constant SIPG sparsity/matrix. Transport
velocity-dependent blocks are rebuilt per attempt. Fracture coefficient/penalty
changes rebuild its matrix; refinement/mesh-deformation signals clear spatial
caches. This invalidation does not qualify dynamic material-history transfer.
Existing endpoint/nonlocal derivative reductions remain replicated; no new global
dense phase matrix or universal advection framework is introduced.

## Observation, migration and limits

On a DG face, observations choose the lexicographically greatest persistent
`CellId::to_string()` among containing cells before evaluating nonlinear h(phi).
This deterministic rule is independent of MPI partitioning and physical vertex
ordering. Cartesian normal-ray clipping resolves overlapping face intervals with
the same CellId rule; derivatives reuse exactly those intervals and quadrature.
The growing path retains Gauss12 rays, checked against Gauss32.

Minimal native declaration (in addition to supported material/solver inputs):

```text
subsection Compositional fields
  set Number of fields = 1
  set Names of fields = phase_field
  set Types of fields = phase field
  set Compositional field methods = phase field
end
subsection Discretization
  set Composition polynomial degree = 2
  set Use discontinuous composition discretization = true
end
```

For Q1 comparison set composition degree 1 and family false. Remove obsolete
phase-specific degree/family settings: their defaults 0/auto derive from the
composition; explicit conflicting settings reject. Migrate initial data to
`Initial composition model`, boundary data to `Boundary composition model`, and
remove mandatory particle postprocessing/mappings from native inputs. There is
no native DG separate-block compatibility adapter. Legacy CPDI remains Q1 with
its original particle/domain requirements.

Checkpoint `phase-space.txt` identifies native composition v2, phase index,
degree and family before bulk deserialization. A separate-block archive cannot
resume as this layout; no production conversion is provided. Legacy Q1 absence
metadata retains its historical interpretation. Existing stress representation,
constitutive law, exterior-provider and accepted-history identities still apply.

Mapped scalar support does **not** enable the entire coupled model on mapped
meshes. `phase_field_fault/fe_strain.cc` still guards Cartesian Box history and
stress quadrature; `material_model/history/tensor_advection.cc` retains its
Cartesian tensor-face implementation. Reworking these independent stress
operators is outside this phase integration. Normalization has a mapped remote
sampling fallback, but it does not remove the stress admission guard. Mature
moving growth, cohesive-tip failures, endpoint energy accuracy and production
qualification remain unresolved. The selected restored-build forward/reverse and fine-mesh T=.8 matrix is now
complete; its measured coupled accuracy limits are recorded in
[DGQ2_COUPLED_VALIDATION_REPORT.md](DGQ2_COUPLED_VALIDATION_REPORT.md). Further
trajectories require explicit selection.

## Entry points

| Responsibility | Owner |
|---|---|
| Registration, degree/family migration | `simulator/parameters.cc`, `introspection.h`, `simulator/phase_field.cc` |
| Initial and boundary data | ordinary composition managers; explicit-time boundary interface |
| Once-per-attempt dispatch | `simulator/core.cc`, `solver_schemes.cc` |
| Owned cell assembly and stages | `phase_field/assemblers.h`, `dg_transport.cc`, `native_phase.cc` |
| Mapped faces, SIPG, shared scalar kernel | `mapped_face.h`, `dg_sipg.cc`, `simulator/assemblers/scalar_dg.h` |
| Bounds and obstacle solve | `dg_bounds.h`, `obstacle.cc`, `endpoint_phase_constraints.h` |
| Native energy and coupled growth | `fracture.cc`, `solver/reconstructed_fault_endpoint_phase.cc` |
| Geometry/normalization/transaction | existing material and reconstructed-fault owners |
| Publication, output, restart | `phase_field_fault/history.cc`, `visualization.cc`, `checkpoint_restart.cc` |
