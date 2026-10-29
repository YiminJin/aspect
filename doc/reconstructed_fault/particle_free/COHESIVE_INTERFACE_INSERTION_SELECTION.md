# Finite tied-record insertion: declared contract and gates

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work;
entry-source.zip and entry.json retain the actual fixture/source evidence.
No production/BP3 change is selected.

The physical surface is a finite subset of the fixed straight candidate mesh.
Only interior nodal jumps of installed elements are represented; endpoint and
uninstalled jumps are exactly zero. Insertion adds jump unknowns and actual
native-manager surface records, with a=0, jump=reference=relaxed memory=absolute
slip=0 and Theta=1+(V0/Dc)t from the virgin material aging clock. Existing material
Gauss histories are copied by lattice identity, without interpolation/reset.
New elements initially remain tied by the complete weak strength certificate.
Insertion alone has zero physical elapsed time and zero fracture work.

The mixed bulk FE/pressure space is fixed across this comparison, including the
qualified independent pressure traces on the candidate mesh line. Pressure-space
enrichment is a numerical choice and is **not** coupled to surface-record birth.
The actual mechanical change is enlargement of the jump trace subspace and its
constraint map. This avoids a separate bulk-history projection while testing real
new jump degrees of freedom, native geometry ownership and Gauss-record birth.
It does not qualify changing the bulk pressure approximation at insertion.

Before any commit, compare complete physical stress, bulk storage, generalized
loads, accepted histories and clocks; rejected geometry must restore the old
manager and subspace, and retry must be deterministic. Accept transfer only if
stress/load changes <=1e-8 Pa, storage change <=1e-10 J/m and old histories are
bitwise identical. New jump/a/fracture inventory must be exactly zero. Preserve
the existing 1e-8 Pa weak residual, full pressure derivatives, positive contact,
irreversibility and curvature checks. Restart must match the uninterrupted
accepted sequence bitwise. Compare earlier/later append and larger/smaller tied
buffers against an installed-beforehand control; differences after the process
zone reaches a finite endpoint are boundary-condition effects, not transfer loss.

Only after those gates pass, select one localized specimen using native elastic
stress and energy estimates at unchanged calibration. Use one spatial and one
temporal refinement, not a parameter sweep. Report a>=.01/.05/.5 and a=1 extents,
initial-seed-connected versus remote growth, and process-zone resolution. Declare
the chosen loading interval and stop criteria before that trajectory. Target
relative temporal errors <2% in integrated growth/slip and physical-work residual
<1% of both added fracture and sliding loss, with absolute errors reported near
zero. Spatial errors <=5% in integrated growth/slip are a practical target, not a
license to hide event-bracket or front-position differences. Negative curvature
stops publication and requires diagnosis of constrained mechanics versus physical
softening. Phi remains geometric, not another fracture-energy inventory.

## Conditional localized specimen selected after the transfer gate

The independent transfer/restart/weak-certificate gate passed. Use a fresh 8 m ×
1 m periodic strip, Gamma at y=.4 m, installed interval [1,7] m and mature seed
[2.5,5.5] m. Upper/lower affine relative displacement rate .001 m/s, zero
fluctuations on those walls; p0=10000 Pa, zero-mean solved pressure perturbation.
Load through 2.7 s, then hold to 2.75 s. No artificial remote damage mask.
Prescribed finite endpoints are computational candidate limits, not extensions of
the initial mature seed. Keep at least .25 m of tied buffer before either endpoint;
if growth reaches that buffer or detached a>=.01 appears, stop and diagnose.

The native elastic estimate with an approximate 1300 Pa seed sliding traction
predicts a tip stress above 3000 Pa, remote stress below 3000 Pa, and seed slip
above delta_f=0.0006667 m. The strip residual-stress energy estimate
(2700−1300)^2 H/(2G)=.98 J/m² is comparable to Gc. These are selection estimates,
not a replacement for the actual pressure-dependent RSF solution. This changes
both height and seed length; merely lengthening the old shallow specimen would
not supply the same release. A failure to advance fully is a valid outcome.

Use 64×40 then 128×80 fixed cells, independent fresh histories, Gauss3 operators
and Gauss6 independent work checks; geometric lift width .2 m. dt=.005 s and one
dt=.0025 s replay on the finer mesh, with load/hold boundaries exact. Per-case
wall bound 900 s, sequential. Preserve rejected trials at negative curvature,
contact loss, or unresolved weak admission rather than altering calibration.
The fully decohered a=1 front and process-zone levels .01/.05/.5 are separate.

The first 64×40 trajectory reached the declared .25 m buffer at trial t=2.44 s;
last accepted t=2.435 s. It already showed new a=1 material in [2.380,2.385] s,
with no detached activation. Preserve that stopped run. Select t=2.42 s as the
common endpoint for the two finer controls, retaining the same loading history
(no hold before that endpoint). This focused endpoint precedes the observed
buffer limit; do not extend the interface or increase the load to evade it.
Compare common physical times through 2.42 s, including 2.38/.40/.42 around
full-decohesion onset. The planned hold to 2.75 s is not exercised in this specimen.
