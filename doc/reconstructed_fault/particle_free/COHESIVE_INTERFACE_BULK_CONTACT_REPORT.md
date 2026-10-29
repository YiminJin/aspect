# Fixed-path bulk/contact investigation

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved work.
[PNG/data index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_bulk_contact/INDEX.md) · [selection, gates and justified follow-ups](COHESIVE_INTERFACE_BULK_CONTACT_SELECTION.md).

**The dominant spread is numerical lifting stiffness.** A resolved fixed
lifting support reduces the fresh causal Q2 growth spread from **9.75% to
1.01%**. An independently solved fitted reference supports the correction.
Raw cut-interface traction/incompressibility defects remain separate limitations.
The cohesive–RSF law, confinement, R, persistence rule, physical loading and
single fracture inventory are unchanged. Production/BP3 are untouched.

## Cause and independent check

The analytical lifting imposes the jump; its arbitrary outer support is not a
physical band. The continuous bulk correction must cancel its gradient kink.
When a support edge cuts an element, the FE correction cannot cancel that kink
exactly. Accurate split quadrature integrates the resulting artificial spring;
it cannot supply the missing approximation freedom. Shrinking support as 12/n
does not ensure a resolved approximation and also changes cut placement.
Two liftings of the same jump differ by a continuous, zero-jump interior
displacement, which the resolved bulk space can absorb. Fixing the numerical
support while refining is therefore consistent with the same continuum law;
it adds no physical gradient penalty or slip hardening.

In a pressure-free straight strip of height H, constant jump m, shear strain γ
and half-support w, the physical shear is `G(γ−m/H)`. Independent Qk FE assembly
and polynomial projection give the same extra energy per fault area:

`ΔEΓ = G m²/(8w²) ||(Id−Π[k−1,h]) 1[|y|<w]||²`.

It contributes artificial resistance `2ΔEΓ/m`. For the actual old-region
physical test, the prediction is **−126.811 Pa**, versus **−126.858 Pa** measured
in 2D: error **.0468 Pa / .0369%**. This control has identically zero divergence
and no pressure/friction law, identifying the stiffness mechanism independently.
The 1D assembled energies match the formula within 1e−10 J/m².

The frozen comparison uses the **same saved evolved work96 geometry, Q1
relaxed-jump memory field and Maxwell affine coefficient** at .445 s. Each
approximation independently reconstructs its bulk response. All Γ coordinates,
weights and physical tests match bitwise. Mean p′ is zero and p0 remains 10000 Pa;
no pressure shift, lag or prescribed friction pressure is used.

| Approximation | Jump-induced storage, J/m | Developed-region tangential load, Pa |
|---|---:|---:|
| Q2 64², w=.1875 m | 1.049917 | 2078.063 |
| Q2 96², w=.125 m | 1.062371 | 1985.776 |
| Q3 64², w=.1875 m | 1.041683 | 2079.370 |
| Q2 96², w=.1875 m | 1.214708 | 2060.610 |
| Q2 64², w=.75 m | 1.035835 | 2158.809 |
| Q2 96², w=.75 m | 1.034906 | 2136.459 |
| Q3 64², w=.75 m | 1.034584 | 2128.698 |
| Fitted Q2 64 / 96 | 1.037427 / 1.034005 | 2103.589 / 2103.890 |
| Fitted Q3 64 | 1.032451 | 2104.037 |

Tangential loads are work-conjugate common Q1 tests, not raw nodal stress.
The fitted grid respects every surface knot, exact quartic graph and lifting
support boundary; both resolutions and the Q3 alternative are checked. Its
storage spread is .331% and regional load differences below .02%. Higher degree
alone is not treated as truth. Whole bulk storage is about 511 J/m, so its large
background would obscure the approximately 1 J/m jump-induced response.

The selected correction holds **numerical** lifting and normal-test vertical half-widths
at .75 m (four/six bulk cells), independently of n. Equality to physical R is
coincidental; no material length or energy is added. One .6875 m off-grid
support control changes storage by 1.19% and short-window growth by .897%,
showing the improvement is not solely fortuitous support alignment. There is
no width tuning campaign. Default experimental mesh-scaled support remains
available; the corrected fresh route selects fixed support explicitly.

## Contact, pressure and remaining approximation error

Common .75 m normal virtual bubbles isolate bulk response from changes in the
contact test itself. All mixed residuals are below 5e−15; independent Gauss6/8
storage errors below 3e−12. Integration is not the dominant error. Global
`∫p′ div(r) dx` is near roundoff, while pointwise incompressibility is not:
tip-region RMS relaxed-strain divergence is 3.45e−5 for narrow Q2/96,
3.32e−5 for wide Q2/96 and 3.49e−6 for fitted Q2/96. The hard-clipped tip
region has ≤.1% quadrature-measure uncertainty, recorded separately.

Raw sided normal-traction jump RMS in the developed-region test is **409.64 Pa**
on all Cartesian Q2/Q3 controls, versus **17.27 Pa** fitted Q2/96 and 12.35 Pa
fitted Q3. Inside a cut cell the continuous bulk gradient and polynomial
pressure have common traces; raising their degree cannot create the missing
internal sided trace. Wider lifting does not repair this. Raw normal sides
and raw sided-mean Cauchy shear are retained unfiltered. The work multiplier
does not establish point-traction accuracy.

Wide Q2/96 developed shear differs 1.55% from the fitted reference. Its weak
normal perturbation differs .295% there. The predeclared 2% normal-perturbation
target **still fails in the tied-ahead region**: 2.85%, or 2.29 Pa against
10000 Pa confinement. That failure remains in the assessment; it is not
renormalized away. The fitted approximation passes the bounded reference checks.

## Evolution verification and initialization distinction

The .42025–.445 s common-history replay retains all surface histories, current
pressure coupling and the .44 s load-to-hold transition. Its bulk reconstruction
is a **new diagnostic initial state**, not an accepted-history conversion.
For wide96, reconstruction changes storage by .002476 J/m and leaves a
598 Pa mass-normalized free surface-force defect. Missing new tied reactions
are distinguished from that free defect; unconstrained least-squares multiplier
peaks are not admissibility certificates. First-step growth is .335% of the
window total. These imported controls therefore remain diagnostic.

Four additional **fresh, equilibrated fixed-path controls** remove that
initialization ambiguity, reusing the existing operators. They start with the
same prescribed old seed, uniform physical stress, zero Maxwell jump memory,
retained prepared seed slip/age and virgin tied records ahead. Initial true
weak errors are below 4.4e−12 Pa; tied peaks 391.61 Pa; initial deviator errors
below 3.3e−11 Pa. No old run or history is reset.

| Comparison | Growth difference | Developed-front difference |
|---|---:|---:|
| Imported window, original lifting Q2 64/96 | 9.83% | .07477 m |
| Imported window, original lifting Q3/Q2 64 | 8.37% | .03107 m |
| Imported window, fixed lifting Q2 64/96 | 1.53% | .00913 m |
| Imported window, fixed lifting Q3/Q2 64 | 2.25% | .00228 m |
| Fresh fixed path, wide Q2 64/96 | **1.01%** | .01213 m |
| Fresh fixed path, fitted Q2 64/96 | **.104%** | .00106 m |
| Fresh fixed path, wide96 / fitted96 | **1.54%** | .01101 m |

All corrected evolution comparisons meet the retained 5% integral and one
.09375 m surface-cell front targets; process-front differences are also below
one cell. The imported wide/fitted difference is 4.03%, larger than the fresh
1.54%, reinforcing the initialization distinction. Separate late replays from
each fresh run's **own accepted .42025 s checkpoint**, without reconstruction,
give .409%/.433% growth changes (wide/fitted) on timestep halving and front
changes .00286/.00335 m. The imported diagnostic halvings give .393%/.422%.
Temporal uncertainty is below the wide/fitted gap and corrected Q2 mesh spread; the
smaller fitted-to-fitted evolution differences lie below that uncertainty. No convergence
order or precise first-activation timing is inferred.

The complete fresh causal workflow also passes: four zero-time insertions,
1780 physical publications, unchanged old stress/pressure columns and six
histories, zero measured transfer stress/storage defect. Fine new damage
integral is 1.194255 m; fully decohered advance **.43508 m along Γ**, tangent
8.841°, versus .44721 m on the coarse mesh. Process-front position differs
.03266 m and full-front position .01213 m. This materially changes the earlier
fine .24147 m advance; the original failed spatial evidence is preserved.

Actual native contact-loss rejection, unchanged accepted state, correct retry,
rebuilt-context JSON restart and duplicate-publication rejection pass for every
short control. Coupled FD errors stay below 4.38e−7 against the unchanged 1e−6
gate; true residuals below 1e−8 Pa and gaps below 1.1e−18 m. The restricted
tangent diagnostic is not a general RSF stability theorem.

Fine causal absolute physical-work remainder is .010993 J/m: **.2495% of
sliding dissipation / .6047% of added fracture**, below the unchanged 1% screen.
The fresh fitted result is .2542%/.6073%. Short-window halving approximately
halves the physical remainder. Exact BE ledger closure is reported separately
(below 1e−12 J/m in the short controls).

## Cost, cleanup and conclusion

Native frozen kernels take about 27–91 s each; Q2/96 has 100609 bulk masters,
the fitted Q2 controls 67645/118493 (see exact counts in `native-cost.csv`). Short
99-step controls take about 3 s, fresh 1780-step fixed-path controls 27–28 s.
Capped 198-step own-checkpoint time replays take about 4 s each.
Causal controls take 103/238 s including on-demand extension construction,
excluding initial-kernel setup. Timings are sequential, with one BLAS thread;
native peak RSS was not independently measured. New raw working data are about
.58 GiB. Unique kernel times and controller times overlap and must not be added.

Cleanup freed **2,669,065,640 bytes (2.49 GiB)** before new work: 125 tables
losslessly compressed and verified, two rebuildable precompiled headers removed,
three byte-verified redundant ZIPs cleared. Unique historical packages, accepted
checkpoints, failures and BP3 remain. Restore mappings are in the maintenance
report; the new evidence stays as indexed PNG/CSV/JSON without another ZIP.

**Conclusion:** unresolved counter-lifting caused the dominant numerical spring
and much of the growth spread. Fixed, resolved support is a justified numerical
correction for this bounded experiment; a separately resolved fitted response
checks its accuracy. Contact point traces and a small ahead-region weak-normal
target remain unresolved. Keep the fitted reference and explicit fixed-support
route for subsequent numerical work. R calibration, quantitative free-path
adoption, native production integration and another long run remain deferred.
