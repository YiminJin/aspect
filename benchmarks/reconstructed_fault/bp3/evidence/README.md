# BP3 initialization04 checkpoint

`initialization04-bp3.cc` is the plugin source used for the measured attempt.
Its SHA256 must match initialization04.resources.json. The current plugin
adds two mass-matrix output columns only; that extension builds but has not
been exercised in ASPECT.

`core-tested.patch` applies to fb4411915. It contains the generic capabilities
used in initialization04 plus the subsequent opt-in condensed-linear timer.
For the exact initialization04 source remove the five-line linear_timer block
from source/simulator/solver.cc after applying the patch. Both versions build;
the newer version passes the recorded focused one-/two-rank unit selections.

The mesh fix is already present in bp3_smoke.prm/bp3_pilot.prm. The only changes
to those files since initialization04 are the explicit DRAFT warning comments.
The kernel, fault and station inputs are unchanged from the run. Resolved
parameters and raw outputs are retained in initialization04/. The executable
and plugin SHA256 are in initialization04.resources.json; build binaries are
not intended for source control.

Prior failures retain their own resolved parameters and output directories.
Do not interpret their old mesh choices as alternate BP3 physical cases.
