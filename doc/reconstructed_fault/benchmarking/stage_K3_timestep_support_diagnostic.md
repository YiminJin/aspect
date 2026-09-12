# K3 reference-only timestep and frozen-profile support diagnostic

## Decision

No ASPECT execution occurred. The 32x256 case remains a **nonpassing support-
gate result**, with its spatial phase/reference evidence intact. Its output,
parameters, histories and comparison files were not changed.

All three independent reference sequences through 3 s satisfy the existing
1e-4 containment and complete supported-normalization criteria with the current
half-width **0.308821593907 m**. Normalization approaches about 5.9100e-5 under
the requested timestep refinement. No support extension or criterion change
is justified by this diagnostic.

There is an important limitation: the H/phi/I_h feedback decreases strongly
with timestep refinement and is zero at dt=.125 s, where the specified H
maximum retains H0 at every step. Thus this is positive support evidence, not
proof of convergence of the large feedback signal in the original two-step
smoke. A common-dt=.5-s spatial pair is proposed below as a **fixed-discrete-
cycle comparison**, not a temporally converged evolving reference. No pair is
prepared or launched by this action.

## Equations, sequence and unchanged inputs

The existing `evolving/reference.py` was reused without modification: independent
2048-cell P1 normal phase solve with six-point Gauss integration, splits at the
actual .1 initialization activation transition and stationary support boundary,
natural normal phase boundary conditions, configured rational admissibility,
bounded-negative degradation evaluation and the existing H maximum rule.
These follow `current_design.md` sections 21/25, the cohesive and Stage-J parts
of `specification.tex`, and the indexed K3 cycle in `stage_K3_preparation.md`.

Each sequence initializes the continuum histories once. The artificial
initialization interval remains **2 s**, with initial mechanical evaluation
but retained supplied q0/Theta0/H0, initialized C0 and full I_h0. Real steps are
exactly .5/.25/.125 s (6/12/24 steps respectively), ending at the common time
3 s. The unchanged physical ramp is
U(t)=1e-4+(.00225-1e-4) min(t/2,1) m/s. These are reference-only prescribed
sequences, **not accepted ASPECT timesteps** or an ASPECT CFL prediction.

At real step k, H_(k-1) drives phi_k; current full I_k is integrated; mechanics
uses the old q/C/Theta/I; the approved finite-step candidate updates H only
inside the original admitted strip, followed by the exact aging update. Initial
C0 projection, loading, full I_h and support admission are unchanged. No
conditional production-initialized reference replaces this primary reference.

For a diagnostic half-width a, write

    J_k(a) = integral_{-a}^{a} h_k dy,
    v_inst = h_k V_k / I_k,
    v_hist = beta_k C_(k-1)/kappa_k * (h_k I_(k-1)/I_k - h_(k-1)),
    D_k = max(|V_k|, V_ref),
    E_k(a) = |V_k J_k(a)/I_k
              + beta_k C_(k-1)/kappa_k
                * (J_k(a) I_(k-1)/I_k - J_(k-1)(a)) - V_k| / D_k.

The signed omitted instantaneous/history integrals are computed separately
outside the original strip. History integrates to zero on the full profile;
therefore the signed omitted sum measures the complete supported defect, not
merely the omitted fraction of h. Here V>V_ref, so the
normalized instantaneous omission equals the h fraction. Both dimensional
terms and their normalized signed sum are exported at **every state**.

## Support results

| Maximum real dt (s) | Real steps | Maximum h omission | Maximum normalized signed history omission | Maximum E_k | E at 3 s |
|---|---:|---:|---:|---:|---:|
| .5 | 6 | 5.923045e-5 | +2.404815e-6 | 6.160459e-5 | 6.025844e-5 |
| .25 | 12 | 5.910148e-5 | +4.104065e-8 | 5.914151e-5 | 5.913174e-5 |
| .125 | 24 | 5.909980e-5 | +9.153128e-20 | 5.909980e-5 | 5.909980e-5 |

The .25-to-.125 change in the worst normalization error is 4.172e-8, far
smaller than the remaining approximately 4.09e-5 allowance. This is an
observed refinement trend, not a theorem about every smaller timestep.
It does not guarantee the next production pair will pass: the prior production
normalization differed from its own reference, and the unchanged per-state
production guard must remain decisive.

### Per-step history and feedback

Here `f` is omitted instantaneous localization (h fraction), `e_hist` is the
signed omitted history divided by D_k, and `E` is the total supported error.
H and phi columns are maximum pointwise absolute step increments; I is the
signed step increment in full I_h. All entries pass phi admissibility and have
an interior mechanical root. Full dimensional values, residuals and minima/
maxima for each individual state are in `timestep-support/all-steps.csv`.

| dt | t (s) | f | e_hist | E | dH (Pa) | dphi | dI (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0 | 5.909980e-5 | +1.320e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .5 | .5 | 5.909980e-5 | +1.324e-20 | 5.909980e-5 | .0805845 | 0 | 0 |
| .5 | 1 | 5.910275e-5 | +1.724527e-7 | 5.927520e-5 | 0 | 1.436403e-6 | .000232116 |
| .5 | 1.5 | 5.910275e-5 | +2.836e-21 | 5.910275e-5 | .176140 | 0 | 0 |
| .5 | 2 | 5.912679e-5 | +8.098009e-7 | 5.993660e-5 | .270072 | 1.129008e-5 | .001816517 |
| .5 | 2.5 | 5.919977e-5 | +2.404815e-6 | 6.160459e-5 | .0847869 | 3.270025e-5 | .005234952 |
| .5 | 3 | 5.923045e-5 | +1.027998e-6 | 6.025844e-5 | .0290259 | 1.334502e-5 | .002129737 |
| .25 | .25 | 5.909980e-5 | +2.202e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | .5 | 5.909980e-5 | +3.648e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | .75 | 5.909980e-5 | +3.735e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | 1 | 5.909980e-5 | +3.154e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | 1.25 | 5.909980e-5 | +2.598e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | 1.5 | 5.909980e-5 | +2.183e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | 1.75 | 5.909980e-5 | +1.882e-20 | 5.909980e-5 | 0 | 0 | 0 |
| .25 | 2 | 5.909980e-5 | +1.659e-20 | 5.909980e-5 | .0107317 | 0 | 0 |
| .25 | 2.25 | 5.909985e-5 | +3.687215e-9 | 5.910354e-5 | .0272657 | 2.758350e-8 | 4.461792e-6 |
| .25 | 2.5 | 5.910047e-5 | +4.104065e-8 | 5.914151e-5 | .0135022 | 3.048283e-7 | 4.927870e-5 |
| .25 | 2.75 | 5.910103e-5 | +3.693632e-8 | 5.913797e-5 | .00885506 | 2.703311e-7 | 4.366596e-5 |
| .25 | 3 | 5.910148e-5 | +3.026304e-8 | 5.913174e-5 | .00728931 | 2.182319e-7 | 3.522586e-5 |

For dt=.125, **all 24 real states** at .125, .25, .375, .5, .625, .75,
.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.125,
2.25, 2.375, 2.5, 2.625, 2.75, 2.875 and 3 s have f=E=5.909980e-5 to the
shown precision, dH=dphi=dI=0, and phi_max=.5997028553. Their individually
recorded signed history omissions are roundoff, 3.22e-20 to 9.15e-20 when
normalized. No physical feedback is inferred from those roundoff values.

| dt (s) | Max H(3)-H0 (Pa) | Max absolute phi(3)-phi0 | I(3)-I0 (m) | V(3) (m/s) | Theta(3) (s) | C(3) (Pa) | q(3) (Pa) |
|---|---:|---:|---:|---:|---:|---:|---:|
| .5 | .640608787 | 5.862147e-5 | .009413323 | .002242596908 | 1.425741116 | 356.951211 | 1272.773399 |
| .25 | .067643978 | 8.207041e-7 | .000132632 | .002244749393 | 1.721141768 | 354.598687 | 1267.372905 |
| .125 | 0 | 0 | 0 | .002244854108 | 1.900565782 | 353.407461 | 1264.599032 |

The saved-candidate audit verifies the existing maximum rule, not a missing
update: at dt=.125 the largest admitted candidate/H_old over all steps is
**.56772778**, and the largest candidate-H_old is **-.21613611 Pa**.
Every candidate is below retained H. At .5/.25 the maximum positive candidate
increments are .27007179/.02726566 Pa. The implemented split finite-step
equations were not altered to preserve the two-step smoke's feedback.

All phase residuals are <=9.998e-12, each bracketed mechanical root passes the
existing residual check, and all profiles remain within the phi<.8 envelope
(maximum across sequences .59970796). Full-profile identity checks and the H
maximum reconstruction pass. No production lifecycle/MPI test is claimed by
this independent calculation.

## Offline width diagnostic, not a support-policy change

For every frozen state, integrate current and previous P1 h with ten-point
Gauss quadrature, including the partial cells at +/-a. Do not use a
particle-center cutoff or change the full I_k. Search for the last failing
sampled width interval, solve its threshold crossing, and check a wider-width
envelope because signed history need not imply monotone error. This is a
numerical required-width diagnostic, not a proof for arbitrary profiles.

| dt (s) | Required half-width for 1e-4 over all states (m) | Required half-width for 5e-5 over all states (m) | Extension beyond current width for 5e-5 (m) |
|---|---:|---:|---:|
| .5 | .3016226994 | .3118723225 | .0030507286 |
| .25 | .3010243083 | .3112738385 | .0024522446 |
| .125 | .3010139804 | .3112634954 | .0024419014 |

Thus the existing .3088215939-m half-width is sufficient for 1e-4 in every
reference sequence. Across all three, a diagnostic half-width just above
.311872323 m would meet the stricter 5e-5 integration target on these **frozen
trajectories** (about .988% wider than current). Equality widths in the table
are threshold crossings, not guaranteed strict bounds to all printed digits.
Neither these wider widths nor the smaller 1e-4 widths are applied to production,
C0 projection, H admission or any new trajectory. Widening those operations
would change the histories and requires separate review.

Ten-versus-twenty-point quadrature at the current, required and full widths
changes normalized errors by <=4.90e-14. The independent partial-cell
integral agrees with the existing six-point fixed-strip computation within
2.71e-13. These checks resolve the width-integration error, not a new spatial
convergence claim for the reference phase solver.

## Proposed next action, awaiting review

Compare **32x128 and 32x256 at common dt=.5 s through 3 s** (initialization
plus six real steps), keeping ell, tangential/fault resolution, particle
density, physical ramp, support, full I_h, initialization interval and solver
tolerances unchanged. Set the maximum real/first timestep to .5 s and end at
3 s/six steps; retain the artificial `Initial time step=2 s`. Confirm actual
accepted sequences match before interpreting a spatial difference, then rerun
the independent reference with those exact accepted sequences.

This choice retains a measurable finite-step phase/history response while
removing the mesh-dependent second timestep that confounded the prior pair.
It is not selected as a converged temporal reference: .25/.125 demonstrate
substantial remaining state/feedback dependence. Retain initial profile errors,
compare full profiles and cumulative feedback at common physical times, and
keep all existing lifecycle, geometry, seam, containment and normalization
gates. If feedback is not distinguishable from production initialization/seam
error, stop rather than reinterpret a stationary profile as a K3 feedback pass.

Planning estimate from the saved 43.4/88.0-s two-step cases: about **120--180 s
for 32x128 and 240--360 s for 32x256**, with peak memory approximately
.6/.9--1.1 GiB. These are conservative step-count extrapolations, not new
performance measurements; the fine case requires an explicitly approved
longer budget. No ASPECT case or execution is authorized by this proposal.

## Reproducibility and checks

All new files are in `benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `diagnose_timestep_support.py`: explicit reference sequences and original-
  support diagnostics; `dt050`, `dt025`, `dt0125` beneath `timestep-support/`
  retain full reports, every-step CSVs and all phi/H snapshots.
- `summarize_timestep_support.py`: read-only saved-state reintegration,
  candidate audit and required-width table. `timestep-support/all-steps.csv`
  contains all 45 initialization/real-state rows with dimensional and normalized
  integrals, feedback, admissibility, residuals and per-state widths;
  `summary.json` retains aggregate values. No ASPECT output is modified.
- `test_timestep_support.py`: constant-field/partial-cell reproduction and a
  nonmonotone signed-width example; **2 tests pass, .017 s**.
- Existing `test_reference.py`: **7 tests pass, .480 s**. No equations or
  reference solver implementation changed.

Commands from the repository root (prefix each with `OPENBLAS_NUM_THREADS=1
timeout 120 python3`):

```
benchmarks/reconstructed_fault/uniform_shear/evolving/diagnose_timestep_support.py --dt 0.5 --output benchmarks/reconstructed_fault/uniform_shear/evolving/timestep-support/dt050
benchmarks/reconstructed_fault/uniform_shear/evolving/diagnose_timestep_support.py --dt 0.25 --output benchmarks/reconstructed_fault/uniform_shear/evolving/timestep-support/dt025
benchmarks/reconstructed_fault/uniform_shear/evolving/diagnose_timestep_support.py --dt 0.125 --output benchmarks/reconstructed_fault/uniform_shear/evolving/timestep-support/dt0125
benchmarks/reconstructed_fault/uniform_shear/evolving/summarize_timestep_support.py
benchmarks/reconstructed_fault/uniform_shear/evolving/test_timestep_support.py -v
benchmarks/reconstructed_fault/uniform_shear/evolving/test_reference.py -v
```

Reference trajectory/width passes took .584/.826/1.413 s respectively; the
saved-state summary took only a few seconds. All invocations remained below
120 s and aggregate execution far below ten minutes. The reference and resolved
parameter hashes are saved in each report. Unrelated working-tree changes and
all previous benchmark evidence were preserved. K3 is not declared converged;
K2 references remain provisional and Gate K2 remains unmet.
