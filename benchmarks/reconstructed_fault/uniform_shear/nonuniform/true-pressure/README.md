# Approved K2.3 feasibility pair

Current accepted status: K2.3 feasibility verification is complete, with a
spatial-resolution limitation. Endpoint amplitude is approximately constant
and width O(h_Gamma), with strong normalized-coordinate collapse. After three
coarse-element exclusions per end, interior Delta sigma_n and -Delta(tau:N)
differences remain 19.6%/21.9% of the fine interior signals. The references
remain provisional; this is not fully converged reference data or Gate K2.
Do not launch 128x512 now. The accepted 64x256 pair is the baseline for K2.4
preparation; see `doc/reconstructed_fault/benchmarking/stage_K2_4_preparation.md`.
The historical review recommendation below is superseded by this disposition.

The single authorized 64x256 refinement pair is also complete: 150.524/139.018
s, all existing checks pass. `spatial-comparison.json`, `spatial-common-*.csv`
and `spatial-normal-feedback.png` reuse the coarse evidence. At .5 s the
normal-feedback spatial change is 87.93% of the fine anomaly signal, so stop
for review before K2.4 or any 128x512 confirmation. Details are in
`doc/reconstructed_fault/benchmarking/stage_K2_3_spatial_review.md`. Runtime
and comparison options do not alter any mathematical acceptance criterion.

Completed outcome: both runs genuinely converge at 0/.5/1 s, in 39.184 and
36.589 s. Six final nonlinear acceptances and 34 fresh linear checks pass;
all 17 vertices are free. Omitted fraction 5.83027e-5 and actual normalization
error <=5.23649e-5 pass their separate pilot-only allowances. At 1 s the Q1
bump-minus-control sigma_n RMS is .00218997 Pa; its continuum resolution is
not established by this coarse pair. See the full report at
`doc/reconstructed_fault/benchmarking/stage_K2_3_pilot_report.md`, relative to
repository root. Run `verify_pilot.py pilot`, `verify_pilot.py homogeneous`
and `verify_pilot.py compare` for the bounded saved-output checks. No further
simulation follows automatically.

Baseline: 8ace2d1b8, continuous Q2 ADD/count. The user approved replacing top
normal velocity by total normal traction -1000 Pa, retaining top tangential
velocity, bottom full velocity and x periodicity. Physical FE pressure is in
Pa, compression-positive sigma_n=p-tau:N. Pressure normalization is `no`;
there is no +1000-Pa offset, nullspace projection or post-solve pressure shift.
This changes the physical boundary condition, not merely the pressure gauge.

Both fixtures use 32x128 cells, maximum dt=.5 s and end time 1 s, with the
same initial phase/H, support, full I_h, domain quadrature and solver settings.
The pilot retains the 5% compact Theta bump; the control removes only that
bump. Both retain supplied initial stress at timestep zero. H and the converged
Q1 phase profile remain frozen by the existing benchmark mechanism.

Run pilot first; only if it genuinely converges without a new issue run the
control at the same accepted times. Hard cap 120 s per run, no automatic retry,
600 s aggregate exploratory budget. The analogous prescribed-pressure replay
took 38.01 s; reserve up to 120 s for each true-pressure run. This is not a
convergence study and K2.2's reference remains provisional/Gate K2 unmet.

Enable `ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC=1`. The optional diagnostic
records rank-local Q1 load moments and raw extrema at the exact production
surface-domain points on each linearization, before commit. It obtains the
actual sigma_n from the response's mu*sigma_n divided by its positive mu,
then tau:N=p-sigma_n. No second constitutive update is performed. The last
snapshot is valid only with final nonlinear/fresh-linear acceptance. Combine
rank moments with SUM and extrema with MIN/MAX; solve with the recorded
consistent mass matrix for along-fault fields, not averaged particle weights.
Raw bulk columns and post-commit particle stresses are distinct diagnostics.

Report constitutive p/sigma_n/tau:N mean/range and Q1 variation; actual weak
q/C/friction/damping/balance; V/slip/Theta and active/free-node status. Compare
bump minus matched homogeneous at common physical coordinates/times. Small
normal feedback is a valid result, not permission to change the formulation.
Check final criteria, fixed profiles, retained t0 histories and split Theta
updates. Particle VTU is committed stress; bulk stress composition is old input.

This pilot-only omitted-fraction allowance is provisionally 1e-4 (original
1e-6 target). Independently require actual slip-normalization error <=1e-4
at measured locations/times. No support extension or I_h renormalization.
Stop for review on a new issue; do not run the control after a failed pilot.

Saved-data endpoint localization at t=.5 s is reproducible with
`OPENBLAS_NUM_THREADS=1 python3 localize_endpoints.py`. It reads only the
accepted 32/64 surface exports and saved weak-system data, with no simulation.
See `endpoint-localization.json`, `endpoint-cuts.csv`, and the two
`endpoint-*.png` plots. The report is
`doc/reconstructed_fault/benchmarking/stage_K2_3_endpoint_localization.md`.
The grid-scale endpoint lobe dominates global squared error, but an interior
stress-normal discrepancy remains. No 128 case or production change follows
automatically.
