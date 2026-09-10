# K2.2 controlled refinement record

**Historical point-rule sequence.** The user has now accepted the domain-rule
and global-accumulator baseline and authorized its complete new 32/64/128
sequence, followed conditionally by time refinement. Current inputs, estimates
and results are under `../domain-convergence/`; none of the old runs below
enters that new convergence sequence. The historical diagnoses remain valid
evidence of why the revision was made, not an active solver-development task.

The focused manufactured-moment follow-up is recorded in
`doc/reconstructed_fault/benchmarking/stage_K2_endpoint_moment_regression.md`.
It supports a surface-quadrature correction for review, leaving domain
construction unchanged. No production correction or additional mechanical
trajectory was run. Approval and focused verification must precede the
64/128 through-1-s comparison; the expensive temporal sequence stays held.

Follow-up diagnosis: `doc/reconstructed_fault/benchmarking/stage_K2_endpoint_diagnosis.md`
records the saved-data moment split and one unchanged 64x256 replay. The
dominant endpoint signature is moving wall-clipped particle quadrature;
no history-transfer or production correction was made. The review hold remains.

**Review stop:** all three spatial runs completed through 2 s. Allowances
pass, but actual particle/Q1 traction shows a late-time endpoint plateau.
The temporal inputs below are prepared, **not run**. See
`doc/reconstructed_fault/benchmarking/stage_K2_2_spatial_review.md` and the
local `spatial_audit.json` / `spatial_surface_audit.png`. K2.2 is incomplete.

The user accepted K2.1 as a feasibility pilot, **not Gate K2**, and explicitly
revised the omitted-profile-fraction target from 1e-6 to a provisional 1e-4
for this fixed-profile, prescribed-normal-stress K2.1/K2.2 fixture family
only. The independent 1e-4 **actual slip-normalization** requirement remains
unchanged and is checked at measured columns and accepted times. This is not
an approval for true normal stress, evolving profiles or later benchmarks.

Physics, H/Theta/stress initialization, association support width, full I_h,
boundary loading and all solver settings are unchanged. The physical initial
state is the same compact 5% C3 bump used by K2.1. Same-support mesh convergence
does not prove full-profile equivalence; omitted tails remain a separately
reported approximation.

## Sequence and prospective resource budget

1. Reuse K2.1 64x256, ds=1/128 m, dt=.5 s through 2 s. Run only its t=0
   initialization again to add the new initial evaluated-particle traction
   diagnostic (initial retained particle stress is not that traction).
2. Run 32x128 / ds=1/64 / dt=.5, then 128x512 / ds=1/256 / dt=.5. Inspect
   support and normalization before interpreting errors. Compare total fields,
   initialized fields and zero-mean nonuniform components separately.
3. If spatial results are interpretable without an unexplained plateau, run
   dt=.25 and .125 on the resolved mesh. These are the only temporal runs;
   do not form the full parameter product. Use the finest fully coupled
   same-support run as numerical reference, not local scalar K1 solves.

Measured pilot cost: 414 s, 0.89 GiB RSS. Before new runs, conservative local
estimates are: 64x256 t=0 diagnostic 100--250 s / 1 GiB; 32x128 full run
100--250 s / .5 GiB; 128x512 full .5 s run 1700--3500 s / 4--6 GiB. If the
128 mesh is needed for time refinement, estimate .25 at 2500--5000 s and .125
at 4500--8500 s, retaining 4--6 GiB. Update these estimates from measured cost
before launching temporal runs. Run sequentially; current local available
memory is about 24 GiB. No server job is requested. Build plugin with -j4.

Measured: coarse full run 72.69 s / 401224 KiB, 64 initialization 104.47 s /
927272 KiB, fine full run 2412.76 s / 3071536 KiB. The 414.14 s pilot is reused.
No temporal cost was incurred after the spatial review gate was triggered.

Stop for review on an allowance failure, unexplained convergence plateau,
or need for a production correction. No support extension, tail
renormalization, raw-stress smoothing or solver change is authorized here.

## Actual surface traction diagnostic

For k>0 use the accepted **particle** Maxwell stress published by production
history commit at the current associated particle location. Contract it with
the actual segment slip tensor. Use old Q1 Theta and old Q1 cohesive history,
not post-update friction/state, to reconstruct F=t-C-mu*sigma-eta_d*V.
At k=0 use the new one-time production-evaluated particle point export;
retain the initialized histories as before. The benchmark's homogeneous
fixed-profile cohesive relation and friction formula are also evaluated
independently in Python. Their particle-weighted weak residual must reproduce
the production solver's final strong-residual RMS diagnostic.

Export raw particle t/F plus the same consistent-Q1 mass projection used to
interpret the weak surface equation. The latter is labeled as a surface weak-
form diagnostic, never substituted for raw stress or fed back into mechanics.
Normal-column bulk averages remain distinct and are not used for this balance.
