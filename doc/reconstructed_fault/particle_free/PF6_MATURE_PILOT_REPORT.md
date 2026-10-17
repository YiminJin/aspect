# Bounded mature DG pilot qualification

17 October 2026; branch `pf-rsf-particle-free`, base
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Scope](PF6_MATURE_PILOT_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_pilot/README.md),
[server preparation](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_pilot/README.md).

**Unlimited domain-free DGQ2 remains the experimental candidate.** The bounded
ownership/rejection/restart and intended surface-initialization checks pass.
Spatial accuracy and absolute endpoint temporal accuracy remain unresolved.
Particle controls remain comparison evidence; neither backend is an accuracy truth.
No server mechanical pilot or first event was launched. Production defaults,
physical laws, limiters and Helmholtz20 are unchanged.

## Fixed-profile mechanics and time controls

Retain the 60° fault, physical initialization/loading, G=32038120320 Pa,
eta=1e26 Pa s, mature zero cohesive resistance/storage and 59-node fixed surface.
The retained FF mesh has 11528 cells, finest 3.90625 m and 108621 mechanical DoFs.
The new mesh has 35342 cells, finest 1.953125 m, 328001 mechanical DoFs and
954234 independent tensor DoFs (27 per cell), besides 1238854 native FE DoFs.
Use exactly prolonged FF Q1 phase and its unchanged 3.90625 m exterior lattice;
T/Z remain freshly unloaded. All 36575 overlay vertices agree **exactly**;
I differs by at most 1.0914e-11 m (8.52e-16 relative). No I/history is imported.
The fixed initial phase has .6466% relative L2 representation error against the
physical seed on the finer common quadrature, separately from evolution.

Gauss6 mechanics and Gauss3 native-source projection remain fixed. For affine
Cartesian Q2 velocity, the complete deviatoric gradient is representable in
DGQ2 and Gauss3 integrates its mass/source products exactly. All nine tensor
coefficients are retained. Positive steps use dt=200000 s to T=600000 s. The FF
mesh also uses dt=100000/50000 s; a paired finer-mechanics halving is below.
The separate initial constitutive interval stays **200000 s** at every level.
Ten step-zero field files are byte identical across the FF timestep controls;
temporal differences exclude a changed initialization.

The retained fixed-chi native law is

```
beta = exp(-G*dt/eta), kappa = -eta*expm1(-G*dt/eta)
T_new = beta*T_old + 2*kappa*D3(symgrad u_new)
Z_new = beta*Z_old + kappa*V_new
 tau   = T - 2*chi*Z*S
```

Z=G*s_mem is auxiliary memory, independent of exactly zero cohesive traction.
The usual native aging update is applied once from accepted Theta. No transport,
spin, phase step or changing-profile transformation is introduced.

Compare complete tensors with |A|²=Axx²+Ayy²+Azz²+2*Axy² and
Azz=-(Axx+Ayy); Cauchy differences also include 3*(delta p)². Use common physical
fine-cell points and fixed physical-profile h weights, including both endpoint
windows s<125 m or s>L-125 m. Independent saved-coefficient reconstruction
matches native stress within 6.76e-9 Pa. It retains the actual normal-support
selector and constant endpoint continuation; h/I is not applied outside its
native association. The following are differences, not accuracy errors.

| Final fixed-FF mechanical change | Measured value |
|---|---:|
| Whole/local tensor RMS difference | .793601 / 6.597651 kPa |
| Local Cauchy RMS difference | 13.5082 kPa |
| Local tensor signal, FF → finer mechanics | 8.1851 → 9.9227 kPa |
| Largest point normal/shear difference | 25.772 / 12.896 kPa |
| Largest shear/raw/filter weak projection difference | 10.952 / 23.198 / 18.972 Pa |
| Largest rate difference / Vp | 1.670e-5 |

The stress signal increases with this mechanical refinement; a smaller jump or
small rate change would not establish accuracy. Independent Gauss6 diagnostic
integration on 1272312 common points changes the local difference by only
.00483 Pa (7.32e-7 relative). This check changes no native mechanical quadrature.

For loads, subtract each case's own M*1*background and also evaluate the common
geometric functional M_geom*(M_native^-1*b-background). Retain the actual native
bulk-work mass. At the last time the mechanical contrast's native perturbation
shear/raw/filter load fractions are 1.469/8.341/8.163% in the interior and
.763/2.305/6.498% at endpoints. Background-measure effects remain separate.

| FF temporal change at T=600000 s | dt→dt/2 | dt/2→dt/4 |
|---|---:|---:|
| Local tensor RMS, Pa | 8.9841 | 4.7211 |
| Shear projection, Pa | 40.6955 | 21.4355 |
| Raw normal projection, Pa | 24.4544 | 12.8505 |
| Filtered normal projection, Pa | 10.0368 | 5.2864 |

Observed orders are .925–.928, consistent with approximately first-order native
endpoint forcing over this interval. The finest Richardson estimates are
5.23 Pa local tensor and 23.86/14.23/5.88 Pa shear/raw/filter projections; these
are estimates from three levels, not rigorous bounds. The last halving changes
interior perturbation loads by .212/.072/.096%, but endpoints by
**1.150/.523/1.047%**. The 1% endpoint screen still fails for shear/filter.
Absolute endpoint temporal uncertainty remains comparable to the mechanical
load change. No threshold or WENO timestep tuning occurred.

The paired finer-mechanics run also reaches T=600000 s at dt=100000 s.
Its same-mesh temporal change is 8.9990 Pa local tensor and
40.754/24.074/9.907 Pa shear/raw/filter projection. Step-zero fields are byte
identical to its dt=200000 s run. At dt=100000 s the mechanical contrast is
6.597684 kPa local tensor and 11.001/23.248/18.941 Pa projections, compared
with 6.597651 kPa and 10.952/23.198/18.972 Pa at dt=200000 s.

Compute the **difference of signed spatial differences**, retaining tensor
components before taking norms. At T=200000/400000/600000 s its localized RMS
is .0750/.1389/.1935 Pa. The final largest shear/raw/filter projection change
is .247/.380/.130 Pa (interior .111/.0286/.0285 Pa; endpoints carry the
largest values). Own-mass perturbation-load contrast changes by at most
3.916/2.583/1.079 Pa m. Temporal bias largely cancels in this specific spatial
contrast: its change is small relative to the measured mechanical differences.
This controls that contrast without establishing absolute endpoint accuracy.
No further mesh or timestep sweep is included.

## Acceptance, MPI and restart

Reuse the existing manager, native outer timestep decision and CellId checkpoint
infrastructure. Source corrections are confined to the experimental route:

- A converged mature DG solve now stages its mechanical result until the outer
  accept/repeat decision. Rejection discards it; acceptance performs one native
  history commit, manager rate commit and publication notification.
- The MPI phase guard compared owned and ghosted layouts. With the guard intact,
  the failing reproduction measured **zero changed owned DoFs and zero physical
  error**. It now compares all owned physical values exactly and reduces that
  result; no tolerance was added.
- Version-4 mature checkpoints retain all 27 coefficients/cell, physical phase
  anchors, Z/I, geometry/material metadata and accepted versus next-step clocks.
  Z is not reconstructed from I*cohesive traction. Mature cohesive traction must
  be exactly zero. Cohesive versions 1–3 retain their earlier meanings.

The first diagnostic attempts also exposed step-zero slip double counting,
NaN provenance-marker equality, and a restart read before transient boundary
association preparation. The observer now respects native time levels, checks
all property bytes including NaN sentinels, and prepares associations before
point diagnostics. Original failed inputs/logs remain. The independent native
Maxwell publication check and its operand-derived allowance are unchanged.

Serial/2/4-rank controls use 4136 graded cells and 66176 stationary points.
Comparisons include every independent DG coefficient, phase, reconstructed stress,
pressure, I/Z, Theta, committed V, cumulative slip and all diagnostic point
properties. Fresh rank-dependent particle IDs are compared by physical points;
IDs are identical on restart. No source/history conversion occurs.

| Comparison | Local tensor RMS difference, Pa | Result |
|---|---:|---|
| Fresh 2 / 4 ranks versus serial | 5.56e-9 / 7.14e-9 | Rounding scale |
| Serial split restart versus uninterrupted continuation | 0 | All compared histories/fields bit identical |
| Same checkpoint, 1→4 ranks | 4.84e-9 | All histories retained |
| Same checkpoint, 4→2 versus 4→4 continuation | 2.09e-9 | All histories retained |
| Completed-trial rejection versus matching unrejected clocks | 0 | Bit identical |
| Four-rank nonlinear cutback versus matching clocks | 5.82e-9 | Rounding scale |

Both actual outer tests reject step two at (t,dt)=(400000,200000) s and retry at
(300000,100000) s. Before retry all DG coefficients, Z/I/phase, surface properties,
committed/current rates, publication count and point properties remain unchanged.
Publication count is one before retry and exactly n after accepted step n.
The matching accepted sequence is dt=200000,100000,191000,200000 s (the existing
91% growth cap); the four-step endpoint is 691000 s. A separate one-owner registry
inconsistency is rejected collectively after native candidate construction;
an unchanged-dt retry succeeds. Lifecycle continuations end by step six,
T=1200000 s, within the retained local strengthening fixture.

Across ownership/restart comparisons the largest DG component differences are
below 3.6e-8 Pa and weak projection differences below 2e-7 Pa. All six point
properties are bit identical for serial split restart and completed-trial retry;
In the audited restart/cutback pairs, MPI stress-sample differences are below
1.86e-8 Pa, with inert initial fields
exact. Twenty retained serial field files reproduce byte for byte. The final
ASPECT suite passes **124 cases / 118034 assertions**; the existing standalone
reference suite passes 55/2625. No solver or publication tolerance was relaxed.

The 32 recorded local native attempts, including original failures/timeouts,
sum to 5527.06 elapsed seconds; the 13 functional cases sum to 711.31 seconds.
The finer dt/dt2 runs take 546.78/778.57 seconds; FF dt2/dt4 take
252.39/483.68 seconds. The finer halving peaks at 2874184 KiB reported RSS.
These overlapping, differently contended local timings are costs, not a scaling
benchmark. The three full-geometry initialization attempts add 143.42 seconds.
Exact commands, pinned artifacts, inputs and per-attempt costs are archived.

## Intended full-geometry surface initialization

Homogeneous bulk data now samples configured initial chemical fractions directly
at fixed surface nodes, retaining them in the native manager/checkpoint. This
removes the former uniform-background assignment that would erase BP3's physical
a=.010→.025 transition. Bulk/profile coefficients and all friction laws remain.

Initialization-only full-geometry controls use 41226 cells (31.25 m finest,
1000 m coarsest), 659616 points and requested surface spacings 40/10 m. The
intended 10 m case has **5776 nodes**, preserving the 15/18 km anchors. Independent
regularized-RSF evaluation agrees within 2.23e-16; nodal initial Theta agrees
within 1.16e-14 relative. Nodal prestress balance is within 1.87e-8 Pa. Surface
fraction interpolation is exact to 3.78e-15; Q1 interpolation of the exponential
state has a 6.63e-5 relative error and a **49.704 Pa** interior balance defect
(40 m: .001061 and 795.004 Pa). It is retained as representation error, without
prestress compensation. This control qualifies material/state assignment, not
full-geometry mechanical resolution or normalization accuracy on the coarse mesh.

## Remaining dependencies and bounded pilot

The RK2 freeze environment is unset; effective motion is disabled by
`BP3 stationary samples`. Every checked ID/coordinate is unchanged, maximum
motion is zero, and births/deletions are zero. Selected DG inputs disable domains,
CPDI and face data. Legacy CPDI sources and old particle controls are preserved.

Remaining particle reads are initial diagnostic H/initial compositions; fixed
ID/position/CellId ownership checks; old point stress/H for the independent native
publication comparison; and native diagnostic/checkpoint bookkeeping. They do
not supply accepted DG mechanics, surface loads, bulk coefficients, physical
phase or fault geometry. A later removal can use deterministic mesh-cell
validation points and native FE gradients, retaining independent source and
reconstruction checks before deleting point history/property requirements.

The separate prepared full-geometry pilot retains the physical phase/geometry,
Gauss6, unlimited DGQ2, Helmholtz20, native RSF controller and initial constitutive
interval **4e6 s**. It caps physical dt at 50000 s, positive accepted steps at eight,
end time at 400000 s, graceful wall time at 7200 s and supervisor time at 7500 s.
It checkpoints each accepted step and exports authoritative tensors, all nine
coefficients and raw/filtered **perturbation** weak loads. The material controller
can shorten dt further. Dense output and fixed diagnostic points remain costs.

The fresh and resume inputs both pass the final native `--validate` check
(exit 0, 3.47/3.42 seconds). This checks declarations/basic parameter syntax;
it does not build the full mechanical operator or qualify cross-parameter
runtime guards. The prepared directory is
`build-particle-free/mature-pilot/server-pilot-final`, containing pinned compatible
core/plugins, input hashes, source patch/untracked source archive, qualification
summary, explicit bounded launcher and offline diagnostics. Regenerate it at the
server destination and rebuild if its ABI differs. No full-geometry mechanical
solve or server job was launched.

The recommendation is the selected **bounded stationary-history pilot**, with
explicit accuracy limits and retained particle comparisons. Full-geometry
conditioning/cost and weakening evolution remain its questions. Do not infer
spatial convergence, accuracy superiority, full PF5 completion or propagation
qualification from these functional checks. Phase evolution, transport, AMR,
moving geometry, limiter changes and the full first event remain subsequent work.
