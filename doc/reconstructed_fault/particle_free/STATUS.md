# Evolving-phase continuation

19 October 2026. Worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, entry revision `612b75d3570d81763407875f3a12c12999b9fa8e`.
The consolidation requested in `../introspection.md` and the subsequently
approved **bounded serial fixed-mesh hanging-node phase task are complete**.
Continue from **cohesive, domain-free PF6a: independent DGQ2 history, ordinary
FE fracture, consistent-endpoint reduced Newton, fixed reference fault**.
Phase growth on that fixed fault is distinct from geometric extension or
physical tip propagation. The subsequently selected bounded P2 response and
original P3 synthetic geometry-transfer diagnostic are recorded below. No
physical propagation task is selected.

## Bounded P3 synthetic straight append

[P3 report](P3_SYNTHETIC_EXTENSION_REPORT.md) defines the transfer contract and
qualification boundary. The imposed finite seed has an interior tip at x=.5;
one collinear append reaches x=.75 on a fixed 32×64 mesh. The existing compact
stationary cap is sampled directly; phase and its physical obstacle stay fixed.
Old material/vertex identities, Z=G s_elas, accumulated slip and rates survive
exactly; prescribed new surface rows are zero and the old tip histories are
zero. No bulk material is born. This is new surface representation within
already initialized cap damage, not demonstrated virgin-material activation.

The joint stress-coordinate map is `T+=T-+Pi_Q2[2(Z+chi+-Z-chi-)S]`. Measured
full-tensor stress transfer error is 2.11e-7 relative L2 (localized 3.10e-7),
maximum .01544 Pa; retaining T unchanged gives 3.73e-7. Bulk energy transfer
defect is 3.12e-11 J/m; the separately predicted cohesive storage change is
4.96e-14 J/m. Weak-load transfer defect is 3.11e-8 relative. The imposed new
surface contributes 203.732 Pa m shear and 250 Pa m friction load, distinguished
from overlap-normalization changes and transfer error. Native Gauss8/12
coefficients agree to 5.74e-15 relative; independent Legendre projection,
normalization and weak loads pass their predeclared 1e-11 screens. All nine
DGQ2 modes, the full plane-strain metric and unlimited history are retained.

Native-component tests cover identity, rejected candidates, deterministic
retry, single geometry publication and archives on both sides of the append.
Serial/two-rank owned and ghost DG data are checked; a separate two-rank process
continues the serial pre-append archive to the uninterrupted accepted record.
The final native suite passes 131 cases /165436 assertions; portable replay
verifies all 31 archived files and the independent mathematical checks.
The neutral manager append stages properties/widths/rates together and
invalidates projection/search caches. Configured boundary-completion paths
remain guarded; this horizontal boundary contact needs no completion wedge.

**Scope limit:** this is a synthetic component transaction using native
manager/archive, FEValues and MPI vectors. It is not a simulator outer-loop
finite-tip geometry transaction or a native material checkpoint conversion.
Existing finite-tip mechanical/evolution guards remain; no live Stokes cache
is exercised by this fixture. Coupled owner/cache integration, a general
nonzero-tip birth rule and phase activation remain unqualified. The next
native extension task must close those integration gaps before physical
propagation. P2's strict branch failure and all BP3 evidence remain preserved.

## Completed bounded P2 physical response

[P2_PHYSICAL_RESPONSE_REPORT.md](P2_PHYSICAL_RESPONSE_REPORT.md) and
[replayable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/README.md)
record the interval/criteria frozen before trajectories and the **passing bounded
physical-response qualification**. The same continuum pre-existing-damage seed
is sampled directly at 2×512 native cell vertices, with unloaded T/Z/slip and
the physical seed as obstacle. Initial relative phase/gradient L2 errors are
0.002769%/0.67223%; independent unloaded KKT and native unloaded phase drift
are zero. No independently relaxed seed, coarse-state continuation or physical
history reset supplies the growth reference.

The native ramp/hold and its single dt halving complete 76/152 accepted steps.
Measurable onset is .53125 s, peak .654399 at .5625 s and irreversible growth
.0543973, followed by no further phase/slip growth through .59375 s. Nonzero
stress and cohesive history decay consistently with the retained Maxwell law.
Maximum common-time temporal differences are phase Linf 6.31e-8 and full-tensor
stress 0.001187 Pa. The targeted 512-projected/1024-continuum spatial reference
gives whole/localized stress differences 0.294%/0.609%, phase Linf 2.52e-4 and
common phase-driving difference 0.0876%. Physical quadrature differences are
below these differences. All predeclared practical screens pass.

Absolute loading external-work remainders fall 0.006033→0.001461 J/m
(0.01287%→0.003116% of boundary work); separate mechanical/contact/phase path
works also decrease. Hold remainders are at roundoff. Closure alone is not used
to qualify work. Native gates 14/318/622, exactly 2/76/152 publications, 2666 raw
CSV identities and all 34 protected BP3 identities pass. All 858 production
source/header identities match entry; core, laws, normalization, acceptance,
limiter/default settings and prior evidence are unchanged. Same-core
rejection/retry/restart and native-unit evidence is reused within its prior scope.

**strict_reference_match_passed=false remains.** Native and matched reference
mirror centroids differ by about 1.7e-5 m; maximum unregistered phase difference
is 5.91e-5. These branch differences remain in every comparison and are not
counted as convergence. Original Gauss6, spatial and endpoint temporal/BP3
findings remain. This passes the selected early physical-response interval,
not general propagation or full PF5/temporal/spatial qualification.

This P2 result motivated the original **bounded P3 synthetic straight geometry
extension**, subsequently selected and tested above. Unchanged T does not in
general preserve tau when localization changes. Native finite-tip integration
and physical activation are not inferred from the synthetic transfer result;
no next implementation is automatically selected.

`../current_design.md` and `../specification.tex` retain scientific authority.
The derivation is at `../particle_free_history.tex`; local PLAN/PROMPTS files
are proposals, not required tracked instructions. Detailed changes, commands,
cache audit and the implemented master-space formulation are in
[SOURCE_AUDIT](SOURCE_AUDIT.md) and [PF6_HANGING_REPORT.md](PF6_HANGING_REPORT.md).
The complete preceding status is preserved byte-for-byte in
[STATUS_THROUGH_20261019.md](STATUS_THROUGH_20261019.md); its dated next actions
are historical, not concurrent development instructions.

## Development line and ownership

| Route | Purpose and accepted owner | Maintained entry/control | Evidence and limit |
|---|---|---|---|
| Older P2 native FE / CPDI | Recoverable total-energy cohesive endpoint experiment. Material owns accepted displacement history d_R; surface Z and physical phase are fixed incoming anchors during trials. Fracture uses CPDI. | `fe_strain.cc`, `fe_phase_energy.cc`, `reconstructed_fault_endpoint_phase.cc`; retained `propagation/run_native_endpoint.py` and native operator/growth controls. | [P2 closeout](../propagation/P2_CLOSEOUT.md): native onset/growth/saddle and bounded MPI checks; full trajectories chiefly prototype evidence. Not the starting backend for new phase development. |
| **PF6a domain-free DG / cell FE** | Independent material-owned DG T (27 coefficients/cell), cohesive Z=G s_elas, I and accepted physical phase. Native FE velocity supplies strain. Mesh/fault geometry and configured homogeneous materials replace particle domains. | Same endpoint solver; `dg_history.cc`, `fe_normalization.cc`, `fe_phase_energy.cc`, `simulator/phase_field.cc`, `dg_accepted.cc`; maintained `run_pf6_domain_free.py` / `pf6_transfer_checks.cc`. | [PF6a](PF6A_REPORT.md), [publication](PF6A_PUBLICATION_REPORT.md), [domain-free](PF6_DOMAIN_FREE_REPORT.md): operators, short evolution, rollback, transfer and same-mode restart. The bounded serial homogeneous domain-free hanging-node phase extension is verified below; unsupported routes stay guarded. Spatial/temporal accuracy and mirror-branch reference matching remain open. |
| Stationary mature DG / BP3 | Frozen prescribed phase and geometry; T and auxiliary Z=G s_mem advance. Cohesive resistance/storage stay zero. This Z is not cohesive elastic slip. | Mature branch in `reconstructed_fault_stokes.cc`, shared material/coupling, `core.cc` outer publication; maintained BP3 `coupled_checks.cc` and server `work_audit.cc`. | [Mature pilot](PF6_MATURE_PILOT_REPORT.md), [optimization](BP3_DG_OPTIMIZATION_REPORT.md), [observer repair](BP3_DG_WORK_OBSERVER_FIX.md). Stationary evidence does not qualify phase evolution or full-event accuracy. Running/server artifacts are protected. |

The shared type name `FEStrainState` does not make FE displacement authoritative
in DG mode: its velocity-history block stays zero, its phase block holds the
accepted anchor, and `dg_history` owns T. Material owns constitutive computation;
the manager stores surface properties/rates; the simulator owns acceptance.

For every cohesive trial, retain the same accepted T_n, Z_n and phase obstacle
phi_n. Form T_e=beta*T_n+2*kappa*D3(symgrad u), Z_e=beta*Z_n+kappa*V;
reconstruct tau=T_e-2*Z_e*h(phi)/I(phi)*S. The phase row varies the selected
bulk/cohesive energy and FE fracture at fixed endpoint T_e,Z_e. Its reduced
tangent includes mechanical response and both normalization derivatives/adjoints.
Keep the full embedded 3D plane-strain norm, all nine Q2 modes, physical
irreversibility and the existing energy/KKT/stability acceptance. A predictor
or converged inner trial is not history. DG publication stays pending through
the actual outer repeat decision; acceptance publishes once, rejection restores
the incoming anchor. No one-pass phase-first combination is adopted.

## Fresh bounded verification

Separate build: `build-particle-free/evolving-phase/native` and matching
`plugin`; no live BP3 build/package was rebuilt. Candidate executable SHA256
`4b36efad2177deea24bd1aeae81b113644a2e730cca6bd623b04e8600cec6e5b`.
[Replayable compact checks and identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/evolving_phase_consolidation/verification.json)
use the existing physical seed and fixture, Gauss12 mechanics / Gauss3 source,
two accepted dt=.125 s steps through .25 s, unlimited DG and frozen diagnostic
particles. Physical steps use 64 cells; the retained manufactured/identity
remaps do not advance time or histories. Surface geometry stays fixed.

- Entry and consolidated runs pass **100 native gates each**. All **162**
  compared CSVs excluding the timing CSV are byte-identical, including every
  exported phase/history/time level, stress, normalization, energy and weak load.
  Mechanical residual and phase-search traces agree exactly. Runtime is
  14.49 / 14.24 s; this small comparison is not a speedup claim.
- Phase grows by **.0189269852885 Linf**, with **zero healing**. Its peak remains
  .600001937763: this short control measures flank growth, not the later core
  onset or mirror departure. Final Z_max=6870.63404561 Pa m. The final native
  scaled phase KKT is 9.28e-17 against its unchanged 6.04e-14 target;
  integrated fault residual Linf is 4.12e-13 Pa m.
- Independent Gauss24/32 physical checks give maximum scaled energy/load errors
  **6.66e-16 / 3.15e-15**. Independent I/DI/D²I scaled errors are at most
  **9.62e-16**. Initial nodal seed error is zero, but the original coarse seed
  representation error remains **8.01966% relative L2**, separately reported.
- Existing actual rejection/retry and terminal inconsistent-source controls
  retain accepted histories/RNG/bookkeeping and once-only publication.
  A native step-1 checkpoint continued to step 2 matches **24** common
  pre-step/final exports, including RNG, byte-for-byte; final publications=2.
- Fresh native suite: **127 cases / 165,094 assertions pass**. Protection hashes
  verify **34** existing BP3/input/executable/plugin files unchanged. No server
  job was launched, submitted, stopped or otherwise controlled here.

Consolidation shares the existing fracture-Hessian addition and fixed-layout
profile derivative, retaining separate stability/search masks. It corrects
FE/CPDI-only comments/log labels and adds explicit binary/plugin paths to the
existing fixture runner. It adds no history owner, numerical operator, physical
parameter, limiter, acceptance rule or production default.

## Remaining limits and next task

**`strict_reference_match_passed=false` remains.** The opposite mirror branches
in [the 512-normal-cell confirmation](PF6A_PUBLICATION_REPORT.md#comparison-throughout-onsetgrowth-strict-target-miss-retained)
still miss the original strict unregistered phase/stress/driving targets. No
alignment, branch preference, tolerance change or replacement reference is used.
The original spatial/temporal findings and Gauss6 quadrature failure remain;
the new checks establish bounded functionality and entry equivalence, not
scientific convergence. Previous MPI evidence is reused within its scope;
this pass adds serial evidence only. Diagnostic particles still own fixed
points, matching-cell publication checks and native persistence/RNG; they do
not supply selected mechanics, material coefficients or fracture integration.

## Completed fixed local phase qualification

[PF6_HANGING_REPORT.md](PF6_HANGING_REPORT.md) and
[replayable verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verification.json)
record the approved fixed 67-cell serial extension. Native closed homogeneous
convex Q1 rows define physical phi=C*psi on 86 masters (90 physical nodes).
Use native `AffineConstraints` expansion and complete raw C^T g / C^T H C
reductions once, including fracture, normalization and mechanical response.
Canonical physical obstacles, projected KKT/linear accuracy, positive roundoff
majorants, search metric and stability variations all use admissible masters;
artificial physical constrained-row diagonals are excluded. Physical bounds
and once-only acceptance/rollback/checkpoint ownership are unchanged.

The single refined patch now intersects the growing flank at
`0.5<x<0.75, 0.1875<y<0.25`. Hanging-node growth is 0.01917 in step one and
0.000638 in step two, while one hanging edge stays obstacle-active; no healing.
Cumulative nodal growth is 0.0198283. Independent gradient/fixed-action errors
are <=1.27e-14, re-equilibrated FD action errors <=2.33e-9. The full ten-free-master
Hessian has minimum eigenvalue 7.3572 J/m and symmetry defect 9.29e-17;
76 masters are strictly active. Native KKT <=2.405e-13 J/m. Gauss12 physical
energy/load controls pass against Gauss24/32 without changing normalization.

The final uniform control preserves all 162 non-timing CSVs exactly. Native
failure/retry, pending discard, terminal sampling rejection and single accepted
publication pass. Actual step-two controller cutback keeps accepted histories
unchanged and matches all three accepted states of the no-rejection sequence
exactly. Same-count serial restart and its parent prefix match 20 complete
flat/reference exports exactly. The final full native unit suite passes
129 cases / 165144 assertions. Core SHA256:
`686288cbbda7cb9793249a5d9dc31cd6f5ea8414737896c376bf71aefa090f80`.
All 34 protected BP3/package/build identities still pass; no server action or
package regeneration occurred.

A single dt halving gives final full-tensor stress relative L2 difference
4.13e-5 and physical phase Linf difference 1.43e-7. This is sensitivity evidence,
not temporal qualification. The seeded coarse field initially has a nonzero
projected phase gradient; early growth includes relaxation. Independent work
accounting closes algebraically, but external interval remainders 0.4563 and
4.05e-5 J/m remain and do not establish converged physical work. Keep
**strict_reference_match_passed=false**, all original spatial/endpoint temporal,
Gauss6 and mirror-branch findings. The report retains two failed diagnostic
attempts and their guard-preserving fixes.

The hanging-node and subsequent P2 physical-response tasks are closed. The
P3 recommendation above awaits selection. Retain representable-phase coarsening only.
MPI local-phase evolution, runtime AMR, lossy obstacle transfer, transport,
surface birth/remeshing, moving geometry, physical tips and coupled propagation
need later authorization and qualification. Unlimited DG, all limiter/filter
settings, constitutive laws, physical inputs and production defaults remain
unchanged. Stop after this bounded task.
