# Authorized noncommitting BP3 free-trace comparison

The user authorizes the next noncommitting BP3 test recommended by the
gradient-kink report. The separately requested particle-density task was
subsequently withdrawn. Defaults and committing histories are unchanged.

Copy the accepted revised-work step-9 checkpoint and solve mechanics 10 over
the unchanged interval 15.30625726--29.24190894 yr. Incoming represented FE
fields, particle data, all nodal histories, background, phase and I_h must
match the saved lagged-state control. No field is committed.

At 40 km, node 795 becomes the independent **free-side** rate. Segment 795
and shallower retain their original Q1 rate basis. On the adjacent wholly
prescribed segment 794, evaluate the kinematic field as V794=Vp, with
mechanical weights (1,0), instead of mixing in the now-free V795. All deeper
rates remain prescribed Vp. This is an exact elimination of the prescribed
side's constant rate trace, not a relocation of the 40-km junction. It adds
one free unknown without changing vector size or geometric connectivity.

Only kinematic/test/trial weights change. Geometric xi still interpolates
Theta, composition, I_h and fixed background at their original positions.
The same modified mechanical weights enter bulk source, B, surface residual,
K, G, mass/norms and reported weak loads. Each physical QP retains its full
existing work measure once. Bulk material/stress evaluation uses the same
current rate. No domain admission/support changes or extra friction terms.

This is a bounded diagnostic exception to continuous Q1 V across a
prescribed/free interface, explicitly selected by
ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC=795 together with the existing disposable
step-10 guards. It is not a general committing/restart representation or a
change to Theta interpolation. The selector cannot be used with candidate
state coupling or committing history publication.

Before the solve, check K columns including 795, B finite differences and
bulk/surface virtual work for that column, plus G velocity and pressure derivatives.
Compare sparse/reference actions during the solve. Export actual raw QP
stress and mechanical/geometric weights separately, including segment 794.
Require unchanged solver criteria and fresh linear checks. After intentional
completion, verify ordinary rollback of bulk, V, all histories and geometry.

Use the saved A solution as baseline and a single four-rank trace solve
(600-s cap; past identical-mesh probes took 50--137 s). Its purpose is to
discriminate a forced continuous trace from an intrinsic free-side notch,
not to advance the BP3 trajectory. If it fails numerically, preserve the
failure without automatic solver retuning.

## Authorized early-onset follow-up

The next bounded comparison uses mechanics step 2, the first clear notch
growth in the saved trajectory. Reconstruct only initialization and accepted
step 1 if necessary, verify that prefix against the original outputs, and
resume its copied checkpoint for one noncommitting independent-trace solve.
The existing shared-trace step 2 is the control. The unchanged kinematic
representation, exact lagged Theta, source/work measure and rollback rules
above apply. `ASPECT_BP3_EARLY_TRACE_STEP=2` changes only the benchmark's
selected comparison clock and CSV inputs, not equations or history timing.
Export the actual constrained FE old stress for comparison with the retained
stress reconstructed from the saved control's current stress and strain.
No later continuation or repeated alternative solve is authorized here.
