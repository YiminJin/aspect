# Experimental K-gradient qualification — fixed phase

The bounded task is complete. The added gradient energy removes the demonstrated
short-wave defect on the tested homogeneous branches and gives resolved,
well-conditioned responses in the small native block. It also changes physical
backstress/storage. **Phase-active and large-specimen progression remain blocked.**
This is an uncalibrated constitutive option, not a numerical filter or a general
well-posedness/stability certificate.

## Published formulation and assumptions

Before implementation, [the literature mapping](COULOMB_GRADIENT_LITERATURE.md)
identified Gurtin (2003), *On a framework for small-deformation viscoplasticity*,
particularly its separable plastic-strain-gradient defect-energy theory.
Its plastic strain maps to signed tensor K, its elastic stress to physical
sigma, and its polar microstress to M below. Our auxiliary T is not the paper's
elastic stress. Q remains conjugate to relaxing A, not to K. The retained
Maxwell/cohesive/phase energies have no claimed identity with that paper's
complete constitutive law.

Additional assumptions are explicit: energetic signed K; full rather than
Curl-only gradient; constant isotropic coefficient; an independent physical
length; recoverable non-relaxing gradient memory; physical normal traction;
zero dilation/non-association; explicit microfree boundaries; small strain and
fixed material coordinates; fresh histories with an authoritative spatial
representation. No loaded local-law checkpoint is converted. Details and
alternatives are in the mapping; [the derivation](COULOMB_GRADIENT_FORMULATION.md)
gives weak equations and the complete nonsymmetric pressure/direction tangent.

## Implemented equations

```
psi_grad = G ell_s² grad K : grad K
M = 2G ell_s² grad K;  X = -div M;  t_K = M n
tau = T - 2G A;  sigma = tau - (p0+p') Id
F = P_S tau - Q - X;  pn_i = p0+p' - nu_i.tau.nu_i
f_i = F:m_i - c_r - mu pn_i <= 0;  W = sum_i w_i m_i
T+ = beta(Tn+2G de);  A+ = beta(An+Delta K);  Q+ = 2G A+/h
beta = (1+dt G/eta)^-1;  W = Delta K/dt;  div du = 0
```

Both conjugates and their changing directions remain. In shear coordinates,
the exact joint normal map is `F=Proj_C(F+aW)`, where C is the intersection of
the two disks with centers `±mu J dev2(tau)` and radius
`c_r+mu[p0+p'-(tau_xx+tau_yy)/2]`. Set
`a=2G beta dt(1+1/h)` solely as residual scaling. No viscosity is added.
An independently evaluated prescribed-X point inverse has **three admissible
roots** at the saved phi=.99 counterexample. Its original guard/failure remain;
the joint native solve does not select one of those local roots.

The native K space is continuous bilinear on the actual Cartesian Gauss6
history-point axes, linearly extrapolated to the boundary. Native JxW is its
constitutive lumped mass; exact piecewise gradient integration supplies X.
Natural zero microtraction replaces no mechanical boundary condition. Gauss6
mechanics, Gauss3 source, all nine DGQ2 modes and native FE remainder are retained.
The existing K/A/Q/W/L owners serialize unchanged; nonzero ell_s extends the
checkpoint material tuple, and derived spatial caches rebuild on restart.
Default ell_s=0 follows the old law and tuple. The option is serial/fixed-phase/
Cartesian only, selected by the documented benchmark environment.

The work balance adds `Delta Egrad + Egrad(Delta K)` once; the second term is
BE numerical loss. Physical friction remains `F:W`; boundary microtraction
power is `t_K:W`. Existing source, mechanical, contact, dissipation and energy
allowances are unchanged. No symmetric phase Hessian is reused.

## Verification and measured response

Retained constants: G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m,
Kf=8 Pa, m=4, strength-drop=2000 Pa, mu=.1, c_r=0, p0=10000 Pa.
Strength-drop is not residual cohesion. Native phase is uniformly .8 and fixed.
The .25 m square starts freshly unloaded, uses the existing nonuniform shear
then normal-deviator loading, dt=.00625 s and endpoint .125 s. A separate hold
ends at .15 s; a separate prescribed-strain strip includes reversal.

| Check | Measured result |
|---|---:|
| Zero-X point identity, 384 states | exactly zero |
| Complete unique-branch point derivative, 76 controls | 1.45e-8, bound 1e-6 |
| Convex projection derivative / basis covariance | 1.27e-10 / 2.59e-16 |
| Constant K force/energy; analytic affine and xy energy | exact zero; relative 2.23e-16 |
| Native joint point action FD | 7.84e-12 |
| Forward strip wavelength response / combined K–strain–pressure FD | 8.35e-10 / 3.32e-10 |
| Native homogeneous mean / maximum point defect at retained physical scales | 4.52e-14 / 4.05e-13 |

The full incompressible mixed symbol, including pressure and direction changes,
is regularized in 28 homogeneous branch states, phi=.2–.99, including single
and double slip. At `k ell_s=1000`, its smallest sampled coefficient is
.999874 G. The single-slip continuous-rate threshold is
`k_c ell_s=sqrt(1/100.4981374-1/h)` when real. For phi=.8, exploratory
ell_s=1/3/6 mm gives critical wavelengths .1146/.3438/.6876 m; at phi=.99,
.0630/.1891/.3782 m. Shorter wavelengths are stabilized on those branches.
Long-wave modes, switches and global uniqueness are not certified. The native
3/6 mm comparison is physical length sensitivity, not calibration.

At .125 s, ell_s=6 mm:

| Bulk cells | K components | Peak tensor slip rate [s⁻¹] | Participation width [m] | Mixed Riesz condition |
|---|---:|---:|---:|---:|
| 4×4 | 1152 | .071741 | .156485 | 7.02 |
| 8×8 | 4608 | .073108 | .155395 | 8.37 |
| 16×16 | 18432 | .073033 | .155121 | 8.89 |

Width is `(integral s)^2/[.25 m integral s²]`, `s=sqrt(2 W:W)`, using actual
native weights. Final 8→16 differences are .10% peak, .18% width and .11% in
the common Gaussian-profile norm (sigma=.0234375 m). Raw peak and width do not
use that averaging. No shrinking narrow band is observed. The retained local
law also had broad slip in this short window; this is not evidence that a
collapsed local band has been repaired. Its earlier matched-schedule condition
numbers were 13.77/31.04/73.17 on its own trajectories. Here the actual full
u/p/K matrix is reduced by sparse K elimination and measured in the same
mechanical H1/pressure L2 norm. Singular values establish neither energetic
stability nor uniqueness.

One native dt halving changes final full 3D deviator stress by 1.24% and tensor
slip rate by 4.03%; total sigma changes .107% and p' differs by 4.41 Pa RMS.
The total-stress denominator includes p0; it must not hide the larger deviatoric
uncertainty. Temporal error exceeds the last spatial difference. No temporal
or continuum convergence order is claimed.

Total accepted-solve time (assembly, direct solve, history and overhead) is
about 2.37/16.46/164.69 s for
the three accepted trajectories; peak rank memory is about 201/270/480 MiB.
The fine fresh confirmation takes 179.1 s wall time. These are prototype costs,
with diagnostic matrix output and serial pressure condensation, not performance
predictions. Native gradient storage reaches .0020293 J/m. It stays constant in
the hold as K sticks while Maxwell stress relaxes. In the separate strip,
reversal releases gradient storage: the option introduces recoverable backstress,
not just less oscillatory slip. The manufactured microtraction-work check and
native BE ledger retain their separate storage, physical and numerical terms.

Across 187 selected accepted native intervals, maximum mechanics/source errors
are 1.41e-10/5.30e-15 (bounds 1e-9/1e-12); maximum microforce error is
6.01e-13 (bound 1e-12), and maximum BE ledger defect is 6.50e-17 J/m.
The fixed initial fracture storage is .4 J/m, retained separately from the
new gradient storage. Actual-point contact and dissipation checks pass.

The native load/hold checks pass actual rejection, deterministic retry and
single publication. Same-rank restart is bitwise for every persistent record,
physical output, phase and cell moments; changed-length restart is rejected.
Default local-law histories are bitwise identical to the prior serial binary.
Two original fine-run failures remain: initial boundary lifting, then premature
sticking cleanup outside convergence. A fresh-guess elastic solve and cleanup
only after convergence correct those demonstrated implementation issues;
neither changes final equations or allowances. The saved-parent retry and
fresh final fine trajectory both pass. All commands ended.

## Evidence and next decision

![Native fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_gradient/native-fields.png)

![Physical response and recoverable storage](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_gradient/gradient-response.png)

Maps show calculated Gauss-point slip/stress on identical scales. Loading,
uniform fixed phase, microfree conditions and uncalibrated lengths were imposed.
The strip is a separate prescribed-strain problem. No phase advance is shown.

Compact CSV/JSON, useful PNGs, inputs, commands, checksums and source changes:
`benchmarks/reconstructed_fault/particle_free/evidence/coulomb_gradient.zip`.
Full matrices, checkpoints, failed trials and logs remain separately under
`build-particle-free/coulomb-gradient/` and the existing bounded-run log directory.

The tested fixed-phase remedy supports a subsequent **small nonsymmetric
coupled phase–microforce formulation/derivative and stability-selection task**,
not a specimen. It must handle the exact intact constraint, irreversible
phase, varying h, critical directions, physical energy and branch selection;
the old associated-potential Hessian is inapplicable. Length calibration,
microboundary choice, MPI/non-Cartesian spaces and long-wave/global behavior
remain open. Preserve the earlier P4/P5c accuracy/energy/boundary findings and
`strict_reference_match_passed=false`. BP3 and production defaults are unchanged.
