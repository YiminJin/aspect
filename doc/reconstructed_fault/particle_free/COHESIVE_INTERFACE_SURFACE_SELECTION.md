# Focused surface/jump resolution comparison

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus retained local work.
Compare only the surface approximation of the completed finite-strip experiment.
No native bulk rebuild, new physical trajectory, constitutive change, BP3 change,
free path, bend, or broad mesh campaign.

Use the existing exact incompressible mixed-strip DtN, projected onto periodic
Q1 jumps with the sharp Galerkin alias sum. Keep L=16 m, H=1 m, fault y=.4 m,
physical x=[−4,12], Γ=[−3.5,11.5], mature seed [2.5,5.5]. Retain physical p0=10000
Pa, all cohesive/RSF/Maxwell parameters, current-pressure derivatives, Gauss3
surface histories, signed weak admission, and the single Gc integral of damage.
The zero mode stays distinct from the nonzero long-wave limit.

Run two freshly initialized analytical-bulk evolutions with surface spacing
.0625 and .03125 m (256/512 Q1 cells). Seed endpoints and candidate endpoints are
aligned on both meshes. Start both at zero jump/Maxwell memory, unit age and the
same sharp mature seed; do not transfer nonlinear Gauss histories from the native
or coarse solution. Use the completed native half-step trajectory's actual time
sequence: original early schedule, .000625 s in the pre-event window, .0003125 s
after 2.45, load to 2.55, hold to 2.58. Preserve its pending-step exceptions and
clipped endpoint as recorded, without silently changing clocks.

The existing monotone predictor is a search strategy. Acceptance requires the
complete signed Gauss multiplier box certificate, exact Q1 tied-cell nullspace,
positive retained admissible symmetric tangent, contact and physical histories.
Extend the existing independent front reference with a cached physical layout;
retain its default unit-domain behavior. Verify the extension against its entry
version, native frozen one-step tests and coupled derivative checks.

Separate (a) native bulk FE versus analytical bulk at the same 256 surface
resolution/time schedule, and (b) 256 versus 512 surface evolution with identical
analytical bulk/time sequence. Inspect common physical times 2.30, 2.45, 2.50,
2.55 and 2.58 s. Compare both flanks, integrated new damage/fracture, signed and
absolute slip, reaction, normal pressure, full-decohesion and partial-damage fronts,
cohesive-zone widths and work. Use native Gauss averages over identical .125 m
physical boxes for nonlinearly stored damage/age/tractions; do not interpolate
histories into new constitutive states. Continuous full fronts may additionally
be reconstructed from monotone Q1 jump and the unchanged critical opening.

Save the fine analytical state at 2.42375 s and halve subsequent intervals through
2.58 to measure fine-grid temporal uncertainty. This is one focused temporal
control, not another mesh campaign. Declare practical spatial interpretation
targets of 5% integrated growth/slip and .03125 m full-front change at common
developed-growth times; 1% physical-work error relative to new fracture and sliding
in the event and hold separately. These are comparison targets, not changes to
solver acceptance. Report failures and compare spatial differences with measured
fine-grid temporal differences. Two meshes cannot establish a convergence order;
the original 65.4% early spatial failure and missing bulk-inertia qualification
remain historical limitations.

Execute sequentially with one BLAS thread. Initial computational bound: 900 s per
reference run, 3 GiB observed RSS, roughly 2–5 minutes expected per trajectory
without native matrix rebuilding. Preserve last accepted state and offending
trial on a solver/physical stop. Keep all existing source/point-law tolerances,
including 1e−8 Pa complete weak residual and 1e−6 derivative comparison.
