# Bounded committing junction-trace replay

This user-authorized experiment extends the frozen independent-rate comparison
to initialization and seven real steps, using the original accepted clock.
It does not change the default continuous-Q1 formulation.

Select `ASPECT_BP3_SPLIT_TRACE_REPLAY=795` only with the fresh, mature,
committing work-measure BP3 replay. Node 795 represents the free trace at
40 km. Segment 794, wholly on the deep side, uses the constant prescribed
rate at node 794; every deeper rate remains prescribed. The free-side state
and accumulated slip live at node 795 and are updated once after acceptance.
The deep trace uses node 794's independently updated state and slip. Initially
both traces have the original supplied values. No history is transferred from
a previously notched run.

Velocity/source/test weights on segment 794 are (1,0), as in the qualified
frozen experiment. The state interpolation on that segment is now also
(1,0); geometric/material/background/I_h interpolation is unchanged.
Thus mechanics uses lagged, side-consistent Theta, with the ordinary fixed-state
V derivative. No extra physical integration measure or deep free segment is
introduced. Bulk source, B, residual, K, G, mass and accepted raw-stress
observations use the same kinematics. Slip is dt times the accepted rate,
accumulated separately on each side. The existing exact nodal aging audit
checks both traces independently.

Compare against saved shared-trace steps 0–7 at identical physical times.
Report the within-free notch separately from V(40-)-Vp and slip(40-)-Vp*t,
and use matched bulk QPs for current constitutive stress (not published new
particle history). Fixed geometry, full I_h, both boundary corrections,
initial background, mesh, tolerances, and split history timing are unchanged.
Stop after seven accepted real steps, or on a numerical/invariant failure;
this is not qualification of a general discontinuous fault representation.
