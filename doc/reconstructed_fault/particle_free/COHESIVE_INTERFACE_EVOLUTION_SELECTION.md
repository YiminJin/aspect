# Native signed admission and conditional straight propagation

Selected 2026-10-26 on f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved
local work. Reuse the native mixed Schur operators and unchanged C++ point law.
Implement a default-off signed semismooth solve with complete Gauss tie reduction
and full current-pressure derivatives. Native C++ computes trial and accepted
states; no input trajectory supplies a solution. Immutable incoming history,
bounded weak multipliers, contact, irreversible damage, positive admissible
symmetric tangent, BE ledger and original tolerances remain acceptance gates.
Geometry birth remains zero-time tied admission; no new bend/contact rule.

First compare the oblique 16×8 operator reference at tied insertion, activation,
established damage and hold. Use selected saved states as incoming data and
physical targets only. Require weak error <=1e-8 Pa, derivative relative error
<=1e-6, ledger <=1e-10 scaled, history/jump differences <=1e-10 in natural units,
physical stress difference <=1e-7 Pa (or explicitly diagnose arithmetic scale).
Exercise rejection/retry, duplicate publication and actual native checkpoint
restart. Preserve reference archives. A native matrix component evolution solver
is distinct from production Stokes/topology integration; keep its scope explicit.

After this passes, reuse the localized 8×1 m strip, mature seed [2.5,5.5] m,
Gamma y=.4, p0=10 kPa, relative wall speed .001 m/s, G=1 MPa, eta=100 MPa s,
Gc=1 J/m², peak=3000 Pa and existing RSF/radiation constants. Full Γ is straight;
closed corner constraints do not enter. Start with the earlier finite candidate
and reproduce a bounded established-growth window. Select the longer specimen
and horizon only after inspecting candidate-buffer influence and cost. A changed
finite candidate must be compared while its extra material remains tied; do not
move endpoints silently to evade a guard. Prefer additional tied surface in the
same bulk domain, if it is independently certified and remote strength holds.

Target >=.5 m new fully decohered seed-connected advance, with a cohesive zone
spanning >=8 axial cells, then a .05 s hold if admissible. Track .01/.05/.5/a=1
fronts separately. Stop before publication on existing solver/curvature/contact
failure, detached a>=.01 growth, a>=.01 front within .25 m of a candidate endpoint,
or remote bulk maximum shear >=peak. Keep all failed trials and last checkpoint.
Maximum physical load horizon 2.7 s plus .05 s hold; per-run wall bound 900 s,
sequential. Select one temporal control around full decohesion and one targeted
axial refinement where front interpretation requires it, not a general sweep.
Use <2% temporal integrated growth/slip, <.05 m full-front uncertainty and <1%
absolute physical-work remainder/new-fracture and /sliding as practical gates.
Report early 65.4% spatial failure unchanged and any later spatial discrepancy.
No new direction selection, fracture reservoir, production/BP3 change or MPI
expansion. Deliver sketch, PNG/GIF, indexed CSV/JSON, report and STATUS.

## Event-selected longer interval (after the native pilot)

The 64×40 pilot matches the retained state/storage through 2.42 s to roundoff.
An additional tied buffer, Γ=[.5,7.5] m on the same 8×1 bulk domain, advances the
fully decohered left front by at least .514 m at 2.45 s. Forward loading reaches
the existing .25 m buffer guard at trial 2.48 s; last accepted 2.475 s. Preserve
this stop, do not move Γ again. Select loading end 2.45 s from that measured
physical target, followed by a requested .05 s hold. A fresh 64×40, dt=.0025
control reaches the buffer during the hold at trial 2.4825 s (last accepted2.48).
This is a boundary-of-qualified-candidate stop, not lost contact/curvature.

Use 128×80 with the SAME physical initialization/loading and dt=.0025 through
common t=2.47 s, safely before the observed buffer. Predicted native cost 2–4
minutes, under 900 s per segment. Preserve a native ASPECT checkpoint near 2.3 s
before full-decohesion advance, then restart to the common endpoint. Use one
half-step replay of this focused window from exactly that checkpoint. The
64×40 half-step control isolates spatial change at equal dt; the 128×80 replay
isolates temporal change. No new specimen or parameter sweep is selected.

The per-step remote stop uses the native projected weak shear on uninstalled
candidate records; evaluate physical bulk maximum shear separately on saved
Gauss6 fields. Do not call the projected surface measure a complete all-time
bulk nucleation proof. Keep positive contact and the candidate buffer unchanged.
