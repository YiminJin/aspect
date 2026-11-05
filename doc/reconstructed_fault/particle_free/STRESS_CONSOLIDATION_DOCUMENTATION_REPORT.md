# Stress consolidation and linked developer guide

Worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, starting
revision `d7e4adbe99126f8c9faf9f35c657e287f2c29cea`; tracked work was clean.
The six working binaries and relevant source matched the preceding readability
manifest. Those binaries were copied before rebuilding, and all **63** protected
artifacts remain unchanged. [Entry and final identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/stress_consolidation_guide)
record hashes rather than assuming binaries match HEAD. No checkout/reset,
production continuation, checkpoint conversion or server job was used.

## Implemented

1. `localize_surface_values` replaces the misleading memory-only name. Its
   contract states linear interpolation of supplied nodal values and I,
   `2*h*a*S/I`, spatial frame, support, factor two and resulting units. Shared
   reaction sampling uses `anchor_values`/`residual_values`; direct mechanics
   reads physical stress, strain forcing and localized slip forcing.
2. `TensorField::project` has one multiple-field mass/RHS implementation and a
   single-field overload. Material code samples complete endpoint fields; one
   cell matrix/inverse serves both separated tensors. Quadrature, component/RHS
   accumulation, owned cells, ghost construction and global participation are
   retained. Pre/post projection energy stays material-side, with reductions
   outside callbacks. This removes duplicated algebra, not physical projection
   error.
3. `solve_reconstructed_fault_stokes` explicitly controls projection-consistency
   iteration and calls `solve_reconstructed_fault_stokes_inner`. The inner body
   is reused once; `material_reaction_solving` is removed. Real correction data
   stays in the material. Energy labels use an explicit candidate-purpose input;
   outer updates pass false and accepted publication passes true. Checker-only
   scratch calls explicitly pass false. No hidden solver-mode state replaces
   the flag. Fixed anchor, 24-iteration limit, tolerance, terminal private rates,
   residual/KKT gates and ordinary acceptance callback remain unchanged.

The short [landing page](MATURE_MOVING_FAULT.md) links seven shallow chapters:
model, discrete operators, geometry/localization, mechanics, lifecycle,
configuration/code map and optional alternatives. It includes a coherent
once-per-attempt versus repeated-solve trace, weak-form projection derivation,
SSPRK3 stages, ownership/copy/MPI semantics and complete generated examples.
Current names and ownership are reflected in status/source-map references.
Historical reports remain evidence at their revisions.

## Verification

[Commands, identities and numerical comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/stress_consolidation_guide)
contain the detailed record. Native cases ran sequentially with two ranks,
except the two four-rank restart continuations; builds used at most four workers.

| Check | Result |
|---|---|
| Build and full unit suite | Core, four matching native DSOs and BP3 support build; **183170 assertions / 152 cases pass**, 7.99 s. |
| Seven matched moving controls | Direct/separated × constant/layered viscosity, defaults, forced rejection and existing-checkpoint continuation: **zero** differences in exported bulk/surface state, physical tensor L2 and work traction. Accepted clocks, linear/outer iterations, line-search decisions/alpha and publication counts match. |
| Fresh lifecycle | Two-rank midpoint restarted on four ranks: maximum bulk difference **8.88e−16**, surface **1.07e−14** against uninterrupted layered direct run at both resumed endpoints. Fresh prefix and rejection/schedule control pass retained thresholds. |
| Shared projection / analytical source | Single-field explicit-input check and direct/separated multi-field decomposition pass independent references. Constant and layered sources test whole-product projection and velocity/slip directional derivatives. Layered within-DG remainder ≤**1.03e−16**, velocity derivative error ≤**3.63e−10**, slip derivative error ≤**9.88e−12**; retained tolerances unchanged. |
| Energy observations | All eight constant and sixteen layered physical-step source-energy observations are byte-identical, including publication labels, local/projected energy and signed loss. Initialization-only scratch probes are explicitly nonpublishing. |
| Stationary and non-fault dispatch | Six stationary CSV comparisons have zero difference and unchanged linear iterations; ordinary non-fault statistics are byte-identical. |
| Guide examples | The documented complete-generator commands are the tested `dc` and `dl` arguments, differing only in fresh output labels. Inputs, commands and summaries are retained. No hidden old-output template is used. |
| Documentation | Relative links, referenced entry symbols and selectors checked. Existing Doxygen options/filter plus local LaTeX produced representative equations/tables, inspected in headless Chromium. An unescaped underscore found by LaTeX was corrected. |

The seven matched case times range **10.84–26.18 s**, versus **10.04–22.46 s**
for the retained baseline runs. These single samples were taken under different
local load; they are not a controlled performance comparison and support no
speedup claim. Iteration counts and numerical work are unchanged; the shared
kernel still builds only one mass matrix per cell in the two-field reaction.

Doxygen does not directly consume this Markdown's `$$` display delimiters through
the repository's inline-dollar filter. Rendering used temporary delimiter-adapted
copies and the existing filter/LaTeX backend; no second maintained manual or new
renderer dependency was added. The scoped render reports links to out-of-scope
historical documents and existing obsolete-option warnings. Repository-relative
links are validated against the actual Markdown/source tree separately. Generated
HTML is a local inspection artifact, not a published standalone documentation site.

## Intentional distinctions and remaining work

Native Gauss3 strain-source projection retains differentiation-based roundoff
bounds and inverse caching. Stationary separated rotation retains its exact
zero-spin copy branch; legacy geometry-only shear correction and cohesive
formed-side/native/lifting spaces retain their different contracts. None is
silently replaced by an uncut evaluated-field projector.

This completes implementation, documentation and bounded behavioral verification;
it adds no physical-accuracy or production-readiness claim. Direct remains
opt-in, separated/default behavior and checkpoint meanings are preserved.
Finite-end tensor approximation, coupled energy/flux accuracy, moving AMR/birth
and independent cohesive tip/energy failures remain open. The inherited loss of
39/45 native-pilot frames and temporary logs remains explicit and unrepaired.
Scientific campaigns remain paused. The next justified scientific experiment is
the previously proposed frozen endpoint-aware tensor-space comparison with
matching work integration, only if selected by the user.
