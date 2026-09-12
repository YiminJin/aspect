# K3 coupled fault-resolution study

## Outcome

**The coupled 16-to-32-element fault refinement resolves the demonstrated
normalization limitation.** On both bulk meshes, the maximum full-profile
identity defect falls from O(1e-4) to below 4.45e-9. Complete supported
normalization and containment pass at every exported state, with no change to
the equations, projection, full I_h, support or acceptance thresholds.

The cases expose a separate remaining limitation. The 32x128/fault32 run
reaches 3 s, then fails the existing homogeneity gate because its endpoint
I_h variation exceeds the intended reference feedback signal. The
32x256/fault32 run reaches the same gate at 2.625 s. Both are mechanically
converged at their final exported states, but neither is a full K3 pass.

No further 64-element fault or 32x512 case was run: I_h representation error
is now negligible, while the remaining physical-profile/endpoint variation
does not decrease under normal refinement. Preserve the 32-element coupled
normalization evidence and review the endpoint/phase-homogeneity limitation
before expanding the convergence study. No separate I_h representation or
direct column normalization is proposed or implemented.

## Approved discrimination and preflight

The accepted diagnosis is insufficient tangential resolution of the surface
Q1 representation of I_h. This action changes only the existing structural
point spacing, first from .015625 to .0078125 m on the same 32x128 bulk mesh.
The fault has 32 elements with h_Gamma=h_x, unchanged physical endpoints and
support. All surface properties are initialized normally from the original
physical initial data; there is no transfer of old nodal state. The common
dt=.375-s fixture and 3-s/eight-real-step interval are retained.

Question: does coupled fault refinement suppress the previously predicted
full-profile identity defect while retaining the existing complete supported
normalization criterion? Only after this succeeds does the study advance to
the corresponding 32x256 bulk mesh with the same fault resolution. Another
normal or fault level is useful only if the resulting trend requires it.

The specification's full normal integral and consistent Q1 projection,
constitutive equations, pressure treatment, support policy, physical ramp,
solver tolerances, history timing and acceptance criteria are unchanged.
The approved projection is not replaced with direct column normalization.
The diagnostic retains J_k, interpolated hat I_k, e_k=J_k/hat I_k-1 and

    (integral_full(v_k)-V_k)/V_k
      = e_k + beta C_(k-1) hat I_(k-1)/(kappa V_k) * (e_k-e_(k-1)).

This positive-sign prediction is explicitly compared with measured full
integration; the existing signed supported deficit uses the opposite sign,
(V-integral_supported(v))/V. Tail and bulk-quadrature terms remain separate.

Pre-execution estimate for 32x128/fault32: 3--5 minutes, approximately .7 GiB,
600-s process-group cap. The test begins directly with the coupled case; no
additional normal-integration or support-tail diagnosis is performed first.
Existing executable/plugin are reused without production edits or rebuild.

## Executed cases and initialization

| Quantity | 32x128/fault32 | 32x256/fault32 |
|---|---:|---:|
| Normal cell size (m) | .0078125 | .00390625 |
| Tangential and fault cell size (m) | .0078125 | .0078125 |
| Fault elements/nodes | 32/33 | 32/33 |
| Last mechanically converged time (s) | 3 | 2.625 |
| Last fully passing benchmark time (s) | 2.625 | 2.25 |
| Real steps computed | 8 | 7 |
| Wall time (s) | 114.044 | 219.007 |
| Peak RSS | 724112 KiB (707 MiB) | 1185336 KiB (1.13 GiB) |
| Exit status | 1, homogeneity guard | 1, homogeneity guard |

The first run satisfied the user-approved normalization condition for
proceeding to the fine bulk mesh. Its separate final homogeneity failure was
reported and retained, not treated as a passing K3 result. The fine run tests
whether that remaining variation decreases. It stops at its guard without
retry, criterion adjustment or a forced continuation to 3 s. Aggregate ASPECT
execution is **333.051 s**. Caps were 600/900 s, not convergence budgets.

Both runs use a new output directory and fresh initialization. The resolved
parameters differ from the common-dt/fault16 case only in output directory,
structural spacing and (for the second case) normal Box repetitions. There is
no restart or nodal state import. Exported geometry verifies 33 evenly spaced
vertices on the same physical curve, independent endpoints and unchanged
support; coordinates remain unchanged during each trajectory.

Initial particle IDs and H are exactly identical to the corresponding
fault16/bulk-mesh initial data. Independently initialized surface properties
agree with the fault16 fields interpolated to the new nodes up to roundoff:
max differences are V<=5.56e-17 m/s, Theta<=1.53e-11 s, C<=1.40e-11 Pa,
I_h<=7.96e-13 m. This agreement is measured after normal initialization, not
enforced by copying old properties. Across the two bulk meshes the existing
initialization differences remain: C0=318.7476558/317.5137627 Pa and
I_h0=108.0980951/108.1448338 m, versus independent continuum
317.7426503 Pa and 108.1349225 m. These are not reset away.

## Representation and complete normalization

At the previous failure time, t=1.125 s:

| Metric | 32x128, fault16 | 32x128, fault32 | 32x256, fault16 | 32x256, fault32 |
|---|---:|---:|---:|---:|
| Max full-profile identity defect | 1.350854e-4 | **8.015397e-10** | 1.658528e-4 | **8.009636e-10** |
| Actual supported normalization | 1.887058e-4 | **5.362689e-5** | 2.273543e-4 | **6.151077e-5** |

Over all new states, max full-profile defects are **4.443483e-9/4.365624e-9**.
The exported predicted positive-sign identity agrees with direct full-profile
integration within **3.23e-14/3.71e-14** at every sampled column. This is the
coupled result, not the earlier frozen-data substitution experiment.

Both current/previous J and interpolated I_h, their relative errors and the
history amplification factor are preserved in `identity_audit_*.csv`. At
x=.249119518239 m and t=1.125 s:

| Quantity | 32x128/fault32 | 32x256/fault32 |
|---|---:|---:|
| e_k | -7.259127e-11 | +4.460299e-11 |
| e_(k-1) | -8.111378e-11 | +3.468070e-11 |
| beta C_old I_old/(kappa V) | 75.86696 | 75.60945 |
| Predicted (full integral - V)/V | +5.739862e-10 | +7.948215e-10 |
| Measured (full integral - V)/V | +5.739826e-10 | +7.948250e-10 |

The following signed contributions sum to the actual supported **deficit**
(V-integral_supported)/V at that same column. They must not be confused with
the opposite-sign full-identity prediction:

| Contribution | 32x128/fault32 | 32x256/fault32 |
|---|---:|---:|
| Full-profile deficit | -5.739826e-10 | -7.948250e-10 |
| Omitted instantaneous + history tail | +5.968764e-5 | +6.056730e-5 |
| Ordinary bulk quadrature contribution | -6.060183e-6 | +9.442608e-7 |
| Actual supported deficit | +5.362689e-5 | +6.151077e-5 |

Normal refinement reduces the ordinary bulk quadrature error; its negative
coarse contribution partially cancels the tail, so a smaller coarse total is
not evidence of greater accuracy. No tail renormalization or width adjustment
is used. All actual instantaneous/history/total integrals remain separately
exported for every sampled column at every state.

### Every-state gates

| t (s) | 128 omitted h | 128 actual normalization | 256 omitted h | 256 actual normalization |
|---:|---:|---:|---:|---:|
| 0 | 5.830274e-5 | 5.236487e-5 | 5.913959e-5 | 6.006423e-5 |
| .375 | 5.831816e-5 | 5.335760e-5 | 5.915475e-5 | 6.113454e-5 |
| .75 | 5.832140e-5 | 5.261844e-5 | 5.915810e-5 | 6.035463e-5 |
| 1.125 | 5.834184e-5 | 5.362689e-5 | 5.917923e-5 | 6.151077e-5 |
| 1.5 | 5.836087e-5 | 5.330031e-5 | 5.919866e-5 | 6.112260e-5 |
| 1.875 | 5.839147e-5 | 5.360747e-5 | 5.922783e-5 | 6.136950e-5 |
| 2.25 | 5.844775e-5 | 5.454994e-5 | 5.928009e-5 | 6.230620e-5 |
| 2.625 | 5.847444e-5 | 5.353071e-5 | 5.930440e-5 | 6.124849e-5 |
| 3 | 5.843796e-5 | 5.306752e-5 | not run after guard | not run |

All containment and complete-normalization values pass 1e-4. Full-precision
omissions and every other gate, including the **failed homogeneity gates**,
are in `coupled-fault-resolution.csv` and the original `guard_*.json`. Do not
interpret this table of support measures as an all-gates pass.

## Common-time primary reference comparison and remaining limitation

The independent continuum reference was rerun from its original initial
histories using each exact exported timestep/loading prefix. Histories are
never reset from subsequent ASPECT output. Compare the two bulk meshes at
common times through **2.625 s**, not their differing terminal times.

| Quantity at 2.625 s | 32x128/fault32 | 32x256/fault32 | Independent reference |
|---|---:|---:|---:|
| Mean V (m/s) | .002233530807 | .002233440735 | .002233265314 |
| Mean Theta (s) | 3.085461347 | 3.082730069 | 3.084237735 |
| Mean C (Pa) | 350.1517134 | 348.9457822 | 349.1711732 |
| Mean I_h (m) | 108.1031737 | 108.1487939 | 108.1376737 |
| Slip (m) | .004337535816 | .004338586316 | .004338024536 |
| Raw q RMS error (Pa) | 1.329949 | .349812 | — |
| Max absolute phi profile error | 2.157802e-4 | 7.740628e-5 | — |
| Max absolute H profile error (Pa) | .0230956 | .0258857 | — |

Initial phi errors were 2.074680e-4/7.839128e-5 and initial raw q RMS errors
1.089703/.277851 Pa. The phase/raw-stress improvement is real, but initialization
differences continue to influence mechanics and history. The H metric includes
the previously documented sampled-reference interpolation/initial row
representation effect; it does not demonstrate monotone H convergence.
Full transverse profiles and feedback increments are retained, not smoothed.

At common t=2.625 s the along-fault I_h ranges are
**.003286013/.004042486 m**, against the predeclared full-interval reference
feedback scale **.003304695 m**. The coarse case is just below the unchanged
comparability limit; the fine case exceeds it and stops. At t=3 s the coarse
range grows to .004756832 m and also stops. Those are homogeneity failures,
not normalization or nonlinear failures.

The range is endpoint-dominated. On the fixed interior
x in [.046875,.203125], I_h ranges at 2.625 s are only
.000198337/.000211071 m. The center I_h increments from their own initial
values are .004973727/.003862403 m, while endpoint increments are
.002490798/.000700244 m. The reference increment is .002751196 m.
Showing these increments keeps initial errors visible; it does not remove
their physical influence. Normal refinement improves the mean transverse
phase increment but does not reduce the endpoint I_h range. The finer Q1
surface field now accurately resolves that variation instead of filtering it.

At the common time, full along-fault phase ranges are
1.262836e-5/1.314257e-5, H ranges .000547392/.000567022 Pa. The history and
geometry checks remain valid. There were 350 periodic crossings through the
coarse final time and 678 through the fine final time; counts are not compared
as equal-density/equal-time quantities. Wrapping is not itself failure. The
pre-wrap prefix already contained endpoint phase variation, so these results
do not establish wrapping as its cause. No particle-domain, history-transfer
or endpoint-topology correction is inferred or implemented.

![Coupled fault refinement and retained endpoint variation](../../../benchmarks/reconstructed_fault/uniform_shear/evolving/coupled-fault-resolution.png)

## Verification, next decision and artifacts

- All **37/34 fresh-linear checks pass**, max fresh/target .962008/.981745.
  Nonlinear phase and mechanical checks pass at all computed states. Maximum
  weak surface RMS residuals are 5.786973e-6/5.784571e-6 Pa, satisfying their
  configured scaled convergence criteria. All 33 surface nodes remain free.
- Stable-ID preceding-H handoff, normal/exceptional phase-probe restoration,
  nondecreasing H, and exact Theta updates pass; maximum Theta update error is
  1.78e-15 s. Initial Theta/H/Maxwell/cohesive history semantics are unchanged.
- Maximum phi is .599926283/.599770450, inside the .8 envelope. Geometry and
  support are unchanged. Full-identity prediction versus measurement and
  signed tail/quadrature accounting are asserted column by column.
- Two existing offline support tests pass (.023 s), and seven independent
  reference tests pass (.867 s). Python compilation and `git diff --check`
  pass. No MPI campaign, production rebuild or broad integration suite.

**Next review decision:** retain the coupled 32-element normalization result
as verified, but do not declare K3 convergence. The next discriminating task
is the endpoint/seam phase-homogeneity limitation, not another I_h representation,
normal-integration diagnosis or support change. A 64-element fault is not
justified to reduce a full-identity error already below 4.45e-9. A 32x512
normal-only case would leave the fixed tangential/endpoint issue in place;
the present two-level trend does not justify it before reviewing that issue.
No additional level was launched.

Recoverable files under
`benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `spatial0375_n128_f32.prm`, `spatial0375_n256_f32.prm`: fresh-initialization
  overlays, with only the approved structural/bulk spacing changes.
- Matching output directories, `.log` and `.resources.json`: separate runs,
  exit reasons, full exports and executable/plugin/parameter hashes.
- `identity_audit_*.csv`/`.json`: J, hat I, e, previous values, amplification,
  predicted and measured positive full-identity defect, and separate signed
  supported-deficit accounting. No prior failed output was overwritten.
- Matching `-reference/` and `-comparison.json`: independent exact-time
  comparisons, explicitly `complete_smoke=false` for both final outcomes.
- `coupled-fault-resolution.csv`/`.json` and `.png`: all states, common-time
  comparison, initialization differences and resource summary. Generated by
  `assess_coupled_fault_resolution.py` and `plot_coupled_fault_resolution.py`.

Executed ASPECT commands from the repository root:

```
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/run_smoke.py --case spatial0375_n128_f32 --wall-cap 600
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/run_smoke.py --case spatial0375_n256_f32 --wall-cap 900
```

For each output directory `CASE`, the existing reference was run with
`--parameters CASE/parameters.prm --ramp-peak .00225 --accepted-times CASE
--output CASE-reference`; the comparison used `--expected-steps 8` and the
corresponding case name. `audit_profile_identity.py CASE` retained the required
diagnostics. These read-only analyses used 120-s process caps and completed
without retries. Resource JSONs record HEAD
`dc1a96d3f72dce123393c90ad025e729885e8ee9` and the reused executable/plugin
hashes. This action changed benchmark fixtures, case/geometry guards, diagnostic
exports and documentation only. All production equations and implementations,
older failed evidence and unrelated working-tree changes are preserved.
