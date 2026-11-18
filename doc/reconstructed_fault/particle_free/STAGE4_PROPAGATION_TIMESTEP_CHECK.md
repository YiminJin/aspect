# Propagation pilot: bounded 80/40-microsecond comparison

The user selected four40-microsecond intervals from accepted68 at5.44 ms,
with comparison to the retained80-microsecond trajectory at5.52 and5.60 ms.
This is a single local timestep check, capped at one hour of numerical work.
The long propagation continuation remains stopped.

## Reproducible configuration

Root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`.
HEAD `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`; incoming manual source preserved.
The executable is independently identified by SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`, banner5f1fc6f3c.
Checker SHA256 is
`8b582573e7b199d4f0fd31ab0ec2eb91e418a3eed52fd3d5a013a42cffb46796`.
No rebuild, constitutive change, loading change or tolerance relaxation.

The common parent is original slot03 of `stage4-propagation-pilot-resume51`.
Its exact hashes are in `build-particle-free/stage4-propagation-pilot/cost-review-parent68.json`.
The pilot driver now forwards the existing native runner's optional
`--snapshot-id` selector. This selects68 without changing the original marker
or creating a redundant intermediate checkpoint copy; defaults are unchanged.
The native runner copies the selected snapshot into a new output directory.

```
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  stage4-propagation-half68 3720 \
  python3 benchmarks/reconstructed_fault/particle_free/stage4_propagation_pilot/run.py \
  half68 --amplitude .00024 --input input-candidate \
  --resume-from build-particle-free/bulk-tip/runs/stage4-propagation-pilot-resume51/output \
  --snapshot-id 3 --steps 72 --end-time .0056 --dt .00004 --seconds 3600 \
  --dense-stride 32 --checkpoint-steps 4 --search-order adaptive-exact-first
```

Four MPI ranks, one thread/rank. Numerical hard timeout3600 s; native graceful
wall criterion2400 s leaves checkpoint finalization reserve. The outer3720 s
wrapper includes final file handling, not an extra selected simulation window.
Checkpointing every four intervals reduces storage; accepted phase/section/work
diagnostics remain per interval. A native wall stop also requests a checkpoint.
The exact generated-input diff changes only dt, stopping/output settings and
checkpoint frequency (`half68-input.diff`). Material, midpoint law, native
Bernstein representation, integration rules and solver gates are unchanged.

Closed redundant checkpoint slots were archived with verified member hashes;
separate saved parents remain native. No scientific data was discarded. The
checkpoint-archive manifest and restoration notes retain old failed/incomplete
snapshots as such. Compression manifests preserve diagnostic hashes. Derived
analysis-view links were reconciled with compressed filenames. Earlier cleanup
availability losses remain unchanged.

## Common-parent check and observation conventions

`audit_restore.py half68 --parent pilot-resume51 --step 68` checks exact section,
plane and native phase histories, phase evaluation at native quadrature,
full3D stress energy and signed section integration measures. The audit's clock
contains the persisted80-microsecond pending interval before the restart-dt
signal. The actual native header and accepted-parent log confirm40 microseconds
and first target5.48 ms. The accepted parent remains at5.44 ms; it is not
reinitialized or aged again.

| Physical endpoint | 80-us baseline step | 40-us branch step |
| --- | ---: | ---: |
| 5.52 ms | 69 | 70 |
| 5.60 ms | 70 | 72 |

Phase, accumulated slip and age are endpoint quantities. Stored slip rate and
section resistance are midpoint-stage quantities: comparing the last stored
stage directly would mix different physical times. The reducer averages the
two fine stages over the same preceding80-microsecond interval as each coarse
stage and verifies the slip-increment/rate identity. It also retains the actual
stage times. No endpoint-rate reconstruction is invented.

Phase L2 uses6x6 Gauss on the actual Q2 polynomials; this is exact for squared
polynomial differences on this affine mesh. Coefficient maximum errors are
labelled separately. Phase/slip error is normalized by evolution since the
common parent, not the much larger common history. Section RMS errors use
common parent `reference_dx`; physical integrated slip uses each state's `ds`.
All-domain, slipping-patch and left/right-taper errors are separated. Physical
work errors are accumulated signed and absolute over5.44 ms to each endpoint,
with native bulk/cohesive/fracture inventories. Full endpoint stress fields were
not exported; bulk energy and stage section resistance do not constitute a
full stress-field error norm. Missing matched endpoints must remain missing.

The reducer is `benchmarks/reconstructed_fault/particle_free/stage4_propagation_pilot/compare_half68.py`.
Native output is `build-particle-free/bulk-tip/runs/stage4-propagation-half68/`.
The parent, input diff, launch record and operational checks are under
`build-particle-free/stage4-propagation-pilot/`.

## Outcome

The one-hour attempt ended with timeout exit **124** after 3600.366 s of native
process wall time (3600.993 s including the outer wrapper). **Zero of four new
intervals were accepted.** The last accepted state of this branch is still
step 68 at 5.44 ms. All four MPI ranks and launch wrappers have exited. This is
a budget stop during the first interval, not a reported native iteration-limit
failure. No continuation or tolerance change was made.

| Comparison time | Retained 80-us result | Requested 40-us result | Comparison |
| --- | --- | --- | --- |
| 5.52 ms | Step 69 available | Step 70 not reached | Unavailable |
| 5.60 ms | Step 70 available | Step 72 not reached | Unavailable |

The restart audit passed: section/plane histories and native phase were exact;
quadrature phase differed by at most 1.11e-15 and independently reconstructed
full-3D stress energy by 6.57e-14 J/m. Source and copied step-68 checkpoint
payload hashes remained unchanged. Final verification also checked 23 protected
identities and the separately saved older parents. Incoming source/manual work
was preserved. `half68-preservation.json` records this verification.

### What consumed the budget

All numerical work was in trial step 69, targeting 5.48 ms. The last complete
nonlinear observation was Newton 12: phase KKT **1.04771e-6**, mechanics
**7.58859e-15**. The phase gate remains **1e-7**. The last five phase KKT values
were 1.67453e-5, 8.37747e-6, 4.18994e-6, 2.09527e-6 and 1.04771e-6.
Their local decrease is not a reliable completion estimate because earlier
constraint solves repeatedly became unstable and the full trial continued to
return a larger residual.

In the first constrained solve sequence, a completed linear solve reported zero
residual, followed by phase residual 35913.7 on the next constrained pass;
subsequent passes reached 1.68267e9. A later sequence similarly reported zero
linear residual before a phase residual of 398775. These are different logged
residual evaluations across constrained passes: they identify where to inspect,
but do not by themselves prove an incorrect linear solve or physical instability.
Later, a feasible linearized phase residual of 1.8963e-13 was followed by a full
nonlinear trial with merit 9520.38 (phase KKT approximately 9.52038e-4); the
half trial reduced the current residual and was accepted by the search. These
are **unaccepted trial states**, not published physical stress or phase fields.

The fine attempt logged 36 completed linear solves, 2730 iterations and
1992.493 s in those solves. That timer excludes unfinished solves and all other
work; it does not partition the full hour. Retained coarse step 69 logged
4596.571 s in 79 completed linear solves, versus 551.396 s in nine for step 70.
Thus halving the timestep did not produce an affordable accepted comparison
within the selected budget. No temporal accuracy or convergence order can be
inferred from this incomplete branch.

### Evidence and decision

[Solver trace figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/half68-solver.png)
shows nonlinear KKT and constrained-pass residuals. The
[trajectory/comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/half68-comparison.png)
explicitly marks both missing matched endpoints; it does not invent a fine
trajectory. PDF counterparts are retained. The
[machine-readable reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/half68-comparison.json)
records `complete: false`, empty comparisons and the trial traces. No matched
comparison CSV exists. The accepted coarse trajectory and its previously
reported subcell contour motion remain evidence, without new temporal
qualification.

**Next justified diagnostic, not launched:** use saved parent 68 to identify the
first phase rows and active-set transitions associated with the large residual
jumps. Recompute the true residual using the same assembled operator and active
set before and after each change, including the apparently zero linear-residual
cases. Inspect the rows responsible for the repeated full-step KKT near
9.52e-4, their bounds/admission flags and derivative consistency. Establish that
cause before another expensive physical continuation. This task has changed no
constitutive law, solver tolerance or acceptance criterion.

The [subsequent row audit](STAGE4_PROPAGATION_PHASE_ROW_AUDIT.md) completed the proposed diagnostic.

The subsequent [guarded replay](STAGE4_PROPAGATION_LINEAR_RESIDUAL.md) implements
the explicit residual test, checks the full coupled tangent, and records the
accepted-state comparisons using the original physical tolerances.
