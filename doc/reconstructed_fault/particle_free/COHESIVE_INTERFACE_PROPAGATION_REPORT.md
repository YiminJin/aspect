# Integrated adaptive propagation and mature sliding

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved work.
[Declared scope](COHESIVE_INTERFACE_PROPAGATION_SELECTION.md) ·
[figures, indexed CSV/JSON and provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation/INDEX.md).

The native evolution reaches fully decohered material whose direction was
selected during the physical loading turn, and continues RSF sliding there
during displacement hold. This demonstrates adaptive propagation followed by
early mature sliding on the selected curve, within the declared finite specimen.
It is not merely a newly drawn tied extension. No unchanged-hold extension
beyond the original .460 s endpoint was needed.

## Experiment and physical result

Resume the verified coarse .440 s checkpoint, preserving all 150 material
records, formed geometry, signed slip, age and Maxwell memory. Continue the
original affine boundary loading to .455 s, then hold displacement to .460 s.
The 12×12 m finite-wall specimen, confinement 10 kPa, homogeneous plane-strain
Maxwell law, current-pressure RSF/radiation, strength 3000 Pa and Gc=1 J/m²
are unchanged. Bulk Q2/P1-discontinuous 64×64, mechanics Gauss6, surface
Gauss3, selection Gauss8, physical R=.75 m and numerical support .75 m stay fixed.
Future surface spacing is .09375 m and Δt=.000125 s. The support and R have
different roles despite sharing a numerical value. Full parameters/units are
in `parameters.json`; the loading is in `boundary-loading.csv`.

| Quantity | Measured main trajectory |
|---|---:|
| Positive accepted steps / new registry commits | 160 / 27 |
| Connected a=1 front, arclength beyond original seed | .497439 → 1.593763 m |
| Final developed front, physical (x,y) | (7.570198, .258004) m |
| Final developed-front tangent | 8.278° |
| Final a=.5 / a=.05 fronts beyond seed | 2.653722 / 3.843750 m |
| Completely decohered late-turn material | 5 cells, .46875 m, 15 Gauss records |
| First .1875 m late-turn band's additional slip after decohesion | .141799 Dc |
| Same band's mean absolute normalized-age change after decohesion | 13.1714% |
| Same band's final signed slip rate / normal pressure | .0370884 m/s / 11.1344 kPa |
| Same band's cohesive / RSF / radiation resistance | 0 / 1519.69 / 958.22 Pa |

The decisive cohort was admitted during .425–.440 s, on the actual turned
geometry, and was partially damaged at .440 s. Its first whole coarse cell
becomes fully decohered in (.456000,.456125] s; the second in
(.457000,.457125] s; the fifth in (.459750,.459875] s. These are brackets,
not exact onset times. The initial seed ends at x=6 m. The formed length grows
4.6875→7.21875 m, with .09375 m of additional zero-history candidate geometry.
The actual developed front traverses the curved, persistent material; further
partially damaged and tied material ahead remains a separate quantity.

All averages above use physical surface Gauss weights. Each point's post-full
slip/age reference is captured when its own damage reaches exactly one;
histories are never reset. At a=1, cohesive storage and q vanish exactly;
friction, radiation and aging continue. .142 Dc establishes evolving mature-law
sliding, **not steady-state RSF or sliding over several Dc**. Continued growth
during hold draws on stored mechanical energy; holding displacement does not
remove stress. The finite walls are part of the specimen's physics.

## Numerical and lifecycle evidence

Focused common-time comparisons, final energy windows and fitted-current-pressure
response measurements are tabulated in the indexed evidence. The spacing
control continues its own verified fine .440 s branch with .046875 m future
cells to .457375 s; it does not project coarse accepted histories. Its comparison
uses growth relative to each branch's own initial damage/slip. The temporal
controls use half Δt on .45375–.45625 s and .45875–.460 s. The second directly
checks the reported mature-slip/age endpoint. Later hold response is outside
the fine-spacing window; two spacings do not establish a convergence order or
a complete bulk-space error bound.

| Focused comparison | Added damage difference | Added signed-slip difference | Developed/process front difference |
|---|---:|---:|---:|
| Half Δt, .45375–.45625 s | −.13335% (−.40123 mm integral) | −.05489% | 0 / <1e-9 m |
| Half Δt, .45875–.460 s | −.12773% (−.21613 mm integral) | −.04538% | 0 / 0 m |
| Future spacing halved, .440–.457375 s | +.17947% (+2.54852 mm integral) | +.07463% | −14.383 / +14.360 mm |

Both front differences are smaller than one .09375 m surface cell; spatial
Gauss brackets and absolute differences remain exported. The fine-spacing
control takes 1403.3 s on two ranks. It tests future geometry/jump resolution,
not a separate bulk-mesh refinement. The earlier early-growth spatial failure
is retained as historical evidence rather than overwritten by these results.
The late replay changes the first .1875 m band's post-full slip and age-change
metrics by +.01078% and +.04578%. Its twenty steps take 55.9 s on two ranks.

The fitted comparison independently re-equilibrates the same actual event
jump/Maxwell-memory field. Separate normal-operator perturbation solves retain
the full current-pressure tangent and all incoming histories. They are
diagnostics, never published evolution. A full fitted one-step result also
contains a changed incoming bulk approximation and is explicitly labelled as
such. R-disk full-tensor/angle and common physical weak-shear tests accompany
the pressure comparison. Raw sided tractions remain exported and unqualified.
The previous borderline bend-normal target and ahead-region/point-traction
failures are not erased or redefined using the large background confinement.
At .456125→.456250 s the fitted normal discrepancy is 52.40 Pa RMS over the
late-turn cohort. Its fully coupled response changes mean velocity −.42164%
and damage increment −.08074%; the strict weak residual is 4.27e-9 Pa.
The selection tensor changes .0004609% and preferred angle .0001256°.
Regional work-conjugate shear differences are at most .578%, but normal
perturbations differ 2.545% in the bend and 3.255% in late-turn material.
Thus the former 2% normal target remains failed; the measured response is
less sensitive than that normal-reaction diagnostic. Raw normal-jump RMS
in the bend is 422.8 versus 18.0 Pa fitted. The coupled FD discrepancy is
7.96e-10 relative, with no pressure lag or pressure-reference adjustment.

An initial counterfactual probe reused the old operator's predictor/active set
and failed the unchanged weak certificate at 651.57 Pa. Recomputing the
existing qualified admission predictor with the changed operator succeeds;
no native equation or tolerance was changed. Its failed input/log remains in
`pressure-impact-main/base-predictor-failure.json` and `focused-final.log`.
A verification bookkeeping error (NumPy boolean JSON encoding) was fixed and
resumed from the atomic accepted state; trajectories and failed logs are
preserved. Diagnostic settings are explicitly restored after the fine branch
to prevent its process-local spacing leaking into a coarse probe.

Each physical step publishes once; each new material admission adds its
fracture/storage changes in that same physical solve. Metadata commits add
zero work and zero fracture energy. Retained physical columns and fields are
bitwise unchanged during candidate replacement; measured transfer stress and
storage defects are zero. Temporary records carry zero slip/damage/Maxwell
memory and the virgin clock, without erasing formed material. Independent
Maxwell, trace, RSF and aging recurrence checks cover every accepted step.
Actual non-root worker failure leaves the accepted state unchanged; retry is
deterministic. Interrupted/resumed and uninterrupted transaction routes agree
exactly; loaded rebuilt-context JSON restart and duplicate-publication rejection
also pass. These are experimental owner/native-context tests, not yet Simulator
checkpoint integration or power-loss durability qualification.

Endpoint work, trapezoidal reaction work, physical dissipation, backward-Euler
bulk remainder and surface endpoint remainder are separately exported.
The main sum of absolute physical remainders is .00908822 J/m, .1931% of
sliding loss and .5126% of added fracture work. The loading/hold values are
.1719/.2369% of sliding and .4847/.5611% of fracture. No cancellation between
timesteps conceals this error. Added fracture is 1.77303 J/m; sliding loss
4.70706 J/m; Maxwell loss .209965 J/m. Bulk/cohesive storage changes are
23.41585/.217936 J/m. The hold releases 2.26018 J/m of bulk storage while
fracture and sliding continue. The exact BE ledger's maximum error is
1.27e-12 J/m, a separate algebraic check. Maximum weak residual is
9.98e-9 Pa against the unchanged 1e-8 Pa allowance; minimum variational
normal pressure is 9734.17 Pa.
The beyond-seed damage increment is 1.770248 m; the difference from total
Gc-weighted fracture increment comes from .002782 m of retained seed-region
damage growth. Initial seed storage is retained, not charged again.

## Parallel implementation and cost

Parallel execution was implemented and qualified **before** the main run.
The opt-in native `column-mpi` path assembles/broadcasts the invariant bulk
matrix once, owns response columns round-robin, owns expensive quadrature by
whole cell and gathers basis fields without summation. R-disk contributions
are folded in cell order, independent of rank count. The small reduced solve
and accepted-state owner remain coordinated. Mesh/matrix storage is replicated;
this is real parallel native work, not ASPECT distributed-mesh integration.

1/2/4-rank operators, basis fields and advanced histories agree bitwise.
Selector regrouping differs from the old serial path by 4.04e-15 relative;
energy diagnostic MPI sums by at most 2.34e-14. The default serial path retains
19 output tables byte-for-byte. Rank-2 injected failure aborts collectively
within the bounded timeout and recovers by retry before publication.

The two-rank main trajectory took 1759.3 s (29.3 min). It used the qualified
second implementation, whose two-rank export cost was 21.94 s. Subsequent
shape-value caching reduced that representative export to 10.11 s with
unchanged arithmetic; the actual grown endpoint's operators/bases also match
byte-for-byte. Final measured export costs are 8.93/10.11/12.01 s on 1/2/4
ranks, so **no robust MPI speedup is established on this host**. Two ranks are
the fastest measured multi-rank choice. Do not extrapolate the caching speedup
to an unmeasured complete rerun. Per-phase timings, per-rank RSS and superseded
performance results are retained; the main and verification runs did not overlap.
Peak recorded native RSS is 627.25 MiB per rank across the new cached exports;
the .440 s two-rank kernel used about 374 MiB/rank. This is a measured local
prototype footprint, not a server memory/scaling guarantee.

## Remaining work to make this usable in ASPECT

The maintained driver computes native accepted states; it does not replay a
reference trajectory. It nevertheless remains a homogeneous experimental
response-basis solver with JSON ownership. Its current bulk histories lie in
the retained affine/lifting response span. General native/DG tensor history
and production restart cannot be replaced by those coefficients silently.

1. Add a default-off exact-jump/cohesive–RSF path to the existing native fault
   owners: signed jump unknowns, persistent Gauss identities, damage/reference
   jump/age, physical bulk Maxwell tensors and temporary/formed transactions.
   Preserve general physical history with an explicit stress/energy-checked
   representation map, including any native-gradient information DG lacks.
2. Integrate the work-adjoint mixed mechanical/pressure blocks, weak intact
   admission constraints and complete nonsymmetric current-pressure tangent
   into the native solver. Reuse matrices/factorizations in process rather than
   rebuilding a separate MPI executable for each candidate geometry.
3. Connect geometry/admission and every history to actual outer rejection,
   cutback, checkpoint/restart and MPI ownership. Invalidate association,
   quadrature, constraint and solver caches together. The replicated prototype
   does not qualify distributed bulk ghosts or native restart across ranks.
4. Keep one fracture inventory Gc∫a dΓ. Geometric φ remains provisional here;
   an ASPECT option must not also charge a bulk fracture functional for the
   same formed surface. New checkpoint semantics require explicit versioning,
   not reinterpretation of existing production/BP3 histories.
5. Retain the open bulk/point-traction accuracy findings and the declared
   finite-wall interpretation. The causal maximum-shear sampler/persistence
   hypothesis and R are not calibrated physical path laws. This smooth graph
   test does not qualify intersections, secondary fracture, opening, general
   bends, inertia or physical rupture speed.

Production/BP3/defaults and prior evidence are preserved. No further calibration
or verification campaign is started automatically.

## Reproduction and changed files

With the retained native ASPECT/deal.II build and source checkpoints present:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-propagation/build -DAspect_DIR="$PWD/build-particle-free/native" -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/cohesive-interface-propagation/build --target cohesive_interface_path cohesive_interface_turn_solver -j2
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_propagation.py demo
```

The maintained driver qualifies MPI before evolution; `run --resume --ranks 2`
resumes an unfinished main transaction. `report` and `verify` do not advance
histories. Native commands, every executed source/binary version and accepted
state remain in the build directory, with event checkpoints outside rolling
files. See `execution-provenance.json` and `manifest.json` for hashes.

Changed implementation: `cohesive_interface_path.cc` (opt-in MPI and shape
cache), new `cohesive_interface_path_parallel.h` and
`cohesive_interface_propagation.py`. Added this report, the selection and
indexed evidence; prepended `STATUS.md` and `SOURCE_AUDIT.md`. Existing CMake
targets are reused. Production source/CMake, defaults, BP3 and manual changes
are untouched by this task.
