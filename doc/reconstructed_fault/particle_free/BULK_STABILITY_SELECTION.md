# Bounded publication stability remedy — selection before verification

Entry: f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Use four ranks, the retained 96x128 mesh/Gauss8/source3, dt=.00125 s,
unchanged laws/seed/loading and native tolerances. No full trajectory.

The incremental potential and symmetric row factors are those independently
verified in BULK_CONTINUATION_REPORT: Pi=Eb+Ec+Ef+beta*Df_step,
mechanical/pressure factor beta*G, phase factor Kf. Eliminate the mixed
mechanical block of the already condensed tangent; no double condensation.
The numerical critical superset contains every positive phase increment and
obstacle masters with normalized gradient <= the existing 1e-7 KKT tolerance.
Use Kf times lumped master mass. Require positive minimum relaxed curvature
outside the exported floating-point margin, or a strictly active phase obstacle.
A negative mode must admit a nonzero physical bound-feasible amplitude to be
called an instability. Inconclusive checks also prevent publication. Limit the
dense critical space to 1024 masters; one accepted-state retry, then useful failure.
No global minimum or general nonsmooth critical-cone theorem is claimed.

Supported continuation uses accepted phase, zero interior displacement and
beta-scaled pressure when prescribed velocities change or are all zero, then
applies current boundary increments and solves all variables. Unchanged nonzero
loading retains its established predictor. Test-only forced predictor cannot
disable the stability gate.

Required short checkpoint checks, sequential: loaded 139 -> 140 must retain
growth and positive relaxed curvature; hold 140 -> 143 with the supported
strategy; forced old predictor at 141 must reproduce the recorded negative
curvature/max increment and recover before publication; split/restart at 141
must match the unsplit hold. Exercise the existing actual outer rejection/retry
fixture on this short hold, preserving every accepted owner bitwise. Check all
existing mechanical 1e-9, phase 1e-7, independent source 1e-12 and BE-ledger
1e-10-scaled gates. Publication increases exactly once per accepted interval.
Compare complete persistent exports, reconstructed stress/weak residuals,
stationary particles and byte-identical checkpoint inputs. Report wall times,
extra factorization/backsolve/eigensolve cost and peak rank memory separately.

No new temporal/spatial sweep. Earlier P4 failures and boundary dependence remain.
Only after these checks pass close this local instability investigation and
recommend, but do not implement, a single P5 constitutive extension.
