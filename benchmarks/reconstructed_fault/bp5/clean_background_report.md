# Clean-background initialization comparison

## Decision

The inherited effective background is a major source of the original initial
velocity pulses, but not the sole source. Replacing it with uniform nominal
traction reduces the native-weighted RMS velocity departure from `Vp` over
13–20 km by **67.5%**, from 0.05793 to 0.01882. The sharp remaining pulses reverse
sign. They coincide with an independently measured initialization mismatch
between the projected surface composition and the state initialized from the
unprojected physical transition, plus nonlinear interpolation of that state.

Exactly one four-rank initialization was run. No real timestep, additional solve,
production code change, tolerance change, or state correction was made. The old
`dc010-ell100/candidate-six` results and prestress input are preserved. New audit
CSVs supplement, but do not replace, their saved data.

## Controlled replacement

The current production background evaluator uses

`tau_bg(xi) = Q1(stored_shear) - Q1(a) - Q1(b)/Q1(d)`.

The new input retains all vertex coordinates and sets **all four** shear-related
inputs together: stored shear = **26,546,122.365139291 Pa**, `a=b=0`, `d=1`.
Normal background remains **50,000,000 Pa**. This nominal value is the configured
deep steady-state friction at `Vinit=Vp=1e-9 m/s`, plus radiation damping
`eta_d*Vinit=0.00462444 Pa`; it is read from the verified friction-configuration
export, not copied from a historical BP3 friction law.

The only PRM differences are the prestress path, output path, stopping at accepted
step zero, and disabling checkpoints. Mesh, regularization, material/friction,
Theta initialization, artificial initialization interval `4e6 s`, endpoint
completion/source, work measure, loading, phase, solvers and tolerances are
unchanged. Surface Theta, projected strengthening and I_h are **bitwise identical**
between the two initial outputs; fault coordinates and sampled row measures also
match. No physical time elapsed and retained particle stress remains zero.

## Initial velocity

| Quantity, 13–20 km | Inherited background | Uniform effective background |
|---|---:|---:|
| Minimum V/Vp | 0.760663 at 14.90 km | 0.957455 at 18.00 km |
| Maximum V/Vp | 1.052898 at 18.10 km | 1.142298 at 15.00 km |
| Native row-weighted RMS of V/Vp−1 | 0.0579315 | 0.0188247 |
| Largest absolute V/Vp−1 | 0.239337 | 0.142298 |
| V/Vp at 10 km (control) | 0.997548 | 0.999252 |
| V/Vp at 22 km (control) | 0.999907 | 0.999945 |

Thus the broad original 15-km depression is largely removed, but a +14.2% spike
remains at the transition entrance. The original 18-km positive pulse becomes
a −4.25% dip. It would be incorrect to call this a uniform-sliding initialization.

## Native weak force budget

All entries below are production weak rows divided by their own
`sum(JxW*chi*N_i)` measure, in **kPa**. They are not point tractions or averages
over normal columns. Let `q_nom=26,546.122365 kPa`. Friction includes actual
constitutive normal stress and incoming state. Initial incoming and outgoing
Theta coincide.

| Location / case | background−q_nom | bulk delta shear | total shear−q_nom | friction−q_nom |
|---|---:|---:|---:|---:|
| 15 km, inherited | −167.755 | +55.384 | −112.372 | −112.372 |
| 15 km, uniform | 0 | −40.522 | −40.522 | −40.523 |
| 18 km, inherited | +172.494 | −15.001 | +157.493 | +157.493 |
| 18 km, uniform | 0 | +10.998 | +10.998 | +10.998 |

The contribution `mu*(sigma_n−50 MPa)` at these rows is only 6.5–10.5 **Pa**;
damping is 0.0036–0.0051 **Pa**. Neither accounts for the kPa-scale pulses.
The uniform run's unreplaced residual/row measure is +0.25369 Pa at 15 km and
+0.18827 Pa at 18 km: small relative to these force differences, but not zero.
These are accepted under the unchanged global convergence criterion; the
inherited run happened to converge more tightly. Current bulk stress here uses
the frozen working history and accepted mechanics, not committed particle stress.

## Quadrature initialization check

The offline audit reads **every exported production source QP** in the transition,
its segment/xi and `JxW*chi*N_i` weights, the saved projected Q1 strengthening field,
and the actual retained Q1 Theta. Only fully covered rows (5–27 km) are used for
reproduction checks. Chemical fractions are clipped exactly as in production.
The implemented regularized law is evaluated after mixing its parameters.

Reconstructed accepted friction loads agree with production to at most
**1.24e-7 Pa** after row normalization. Shear and mass also agree; duplicate
MPI cell/QP identities are excluded. This check prevents replacing the weak
balance by a nodal friction approximation.

At the intended `Vinit`, using uniform background and zero perturbation traction,
the initial force mismatch is

`R_i^nom = sum JxW*chi*N_i [q_nom - mu(Vinit,Theta_Q1,f_Gamma)*50MPa - eta_d*Vinit]`.

It is **+59.444 kPa at 15 km** and **−68.927 kPa at 18 km** after row normalization.
Pointwise friction excess ranges from −99.112 to +99.494 kPa. These values are
identical for both cases, because their state and material projections are
identical. This nominal-traction mismatch is not the converged residual.

An exact diagnostic decomposition is obtained by inserting the physical
unprojected fraction `f(s)=clamp((s−15km)/3km,0,1)`:

1. **State-interpolation excess:**
   `50MPa*[mu(Vinit,Theta_Q1,f(s))−mu_target]`.
   Its weak average reaches **9.536 kPa** through the transition. Theta grows
   approximately exponentially there; linear interpolation and logarithmic
   friction do not commute, even though the nodal inverse is accurate.
2. **Composition-projection excess:**
   `50MPa*[mu(Vinit,Theta_Q1,f_Gamma)−mu(Vinit,Theta_Q1,f(s))]`.
   Weak averages are **−64.297 kPa at 15 km**, **+64.248 kPa at 18 km**.
   The projected nodal fractions are respectively **0.007989744** and
   **0.991998122**, whereas Theta was initialized for fractions 0 and 1.

The two contributions sum to the actual friction excess. At the endpoint rows
the state-interpolation terms are +4.853 and +4.678 kPa respectively. The projected
composition mismatch dominates the residual sharp pulses, while state
interpolation explains the broader small deficit inside the transition.

As an offline identity check only, invert the law independently at each actual
QP mixture. The resulting diagnostic Theta makes the nominal local friction
balance accurate to **1.11e-8 Pa**. This field was **not committed, projected, or
used in another solve**; it is not a qualified new state representation.
The CSV also records the full Vinit residual holding the accepted bulk iterate
fixed, including the change `delta q = -kappa*chi*(Vinit−Vaccepted)`; this is
distinct from the nominal zero-perturbation budget above.

## Convergence, cost and provenance

- Release, four MPI ranks, same binary/plugins as the saved baseline.
- Accepted states: **[0] only**; 1156 free nodes, zero lower-active nodes.
- Final normalized residuals: bulk **2.017753e-14**, surface **4.274880e-9**.
- Surface strong RMS: **0.01150896 Pa**; fresh linear checks passed on all three
  returned directions: 0.8383744/1.015397, 2.218793e-5/4.541690e-5,
  5.953856e-9/1.186481e-8 (achieved/requested absolute residual).
- 94 total Krylov iterations; minimum alpha=1; two accepted Newton updates.
- Theta audit error=0; retained initial particle stress=0.
- Wall time **151.49 s**; launcher peak child RSS **2,689,960 KiB** (~2.57 GiB).
  This is the launcher resource metric, not summed simultaneous MPI memory.

Commands, from repository root:

```sh
python3 benchmarks/reconstructed_fault/bp5/clean_background.py prepare
python3 benchmarks/reconstructed_fault/bp5/clean_background.py run
python3 benchmarks/reconstructed_fault/bp5/analyze_clean_background.py
```

The launcher is deliberately non-overwriting. Exact input/binary/plugin hashes,
MPI command and parameter changes are in
`dc010-ell100/clean-background-init/launch.json`; execution result and complete log
are alongside it. Results:

- `dc010-ell100/clean_background_comparison.png`: comparison and mismatch split.
- `dc010-ell100/clean_background_comparison.json`: extrema and verification.
- Each compared case's `initial_friction_weak_audit.csv` and
  `initial_friction_qp_audit.csv`: labeled row and raw QP diagnostics.
- The new `run.prm`, `uniform_effective_prestress.txt`, ordinary graphical output,
  `state_work_0.csv`, and `work_weak_0.csv` preserve the complete new initial state.

**Recommended next decision:** retain uniform effective background for this new
friction law, then review how the initial state should be made consistent with
the actual projected surface material and the chosen Q1 state representation.
Do not compensate by reinstating inherited prestress pulses. This experiment
identifies initialization effects; it neither establishes later-time stability
nor qualifies a changed state representation. The previously recorded real-step
bound failure is separate and was not modified or retested.
