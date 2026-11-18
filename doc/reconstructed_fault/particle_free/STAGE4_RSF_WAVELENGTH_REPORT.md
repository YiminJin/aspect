# Stage 4: the candidate does not yet select a resolved wavelength

12 November 2026. Task: `CODEX_STAGE4_RSF_WAVELENGTH.txt`.

**Do not launch the smaller-Dc native instability pilot yet.** On the unchanged
.075 s specimen, refinement admits a faster, shorter slip pattern. The leading
coupled carrier moves from **.0939 m to .0625 m**, remaining at only 1.5–2
along-fault cells per wavelength. Neither its spacing nor a short-wave growth
plateau is qualified. The matched b=a and original-Dc controls remain
non-growing in every tested space. No free-growth, native background preparation
or reduced pulse replay was launched.

The task explicitly permits stopping when the leading wavelength migrates
toward the new mesh limit. That condition is met on 128×128 and 256×128;
512×128 was not needed for this immediate decision and was not run. This is a
negative qualification decision, not proof of the full continuum dispersion
curve. The established qualitative steering control remains intact.

## Results

Fixed normal spacing is hy=.03125 m. Force patterns are supported on x=[2.5,5.5].
Six smooth sines remain in the cumulative spaces. Additional sine/cosine pairs
have n=24,32,48,64 and, on the finer mesh, n=96, with wavelength 6/n metres.
Every responsive section age is retained: **444/768 sections on the coarse
mesh, 888/1536 on the finer mesh**. There is no rate cutoff.

The table gives **pair-space** eigenvalues, not scalar spring estimates.
Kmin is the minimum elastic stiffness in that pair's actual equilibrated slip
response space; it need not be the stiffness of the fastest RSF eigenvector.
All units are SI; stiffness is displayed in MPa/m, growth in 1/s. “0” means no
positive eigenvalue above numerical roundoff: the largest control values over
all spaces are 3.33e-13/s for b=a and 2.76e-16/s for original Dc. Very slow
outside-patch ages explain the near-zero stable spectral edge.

| hx (m) | Wavelength (m) | Cells/wavelength | Kmin (MPa/m) | Candidate growth | Matched b=a | Original Dc |
|---:|---:|---:|---:|---:|---:|---:|
| .0625 | .25000 | 4 | 1.70238 | 10.0667 | 0 | 0 |
| .0625 | .18750 | 3 | 1.39404 | 18.6710 | 0 | 0 |
| .0625 | .12500 | 2 | .796109 | 32.6498 | 0 | 0 |
| .0625 | .09375 | 1.5 | .741721 | 36.1062 | 0 | 0 |
| .03125 | .25000 | 8 | 1.71489 | 9.8414 | 0 | 0 |
| .03125 | .18750 | 6 | 1.44796 | 17.1998 | 0 | 0 |
| .03125 | .12500 | 4 | 1.13908 | 25.3077 | 0 | 0 |
| .03125 | .09375 | 3 | .960467 | 30.0323 | 0 | 0 |
| .03125 | .06250 | 2 | .649774 | 37.6198 | 0 | 0 |

For **coupled spaces**, including the six smooth modes and all longer tested
pairs, the original .125 m cutoff gives 27.2235/s coarse and 24.2917/s fine.
Adding neighboring shorter patterns changes the answer substantially:

| Mesh | Shortest included trial | Kmin (MPa/m) | Leading growth | Signed carrier | Central zero-crossing wavelength |
|---|---:|---:|---:|---:|---:|
| 128×128 | .09375 m | .740271 | 37.7061/s | .09389 m | .09900 m |
| 256×128 | .06250 m | .649753 | 36.7932/s | .06250 m | .06262 m |

These leading eigenvalues are oscillatory, with imaginary parts 118.669 and
104.789/s, respectively. Growth rates are their real parts.

The similar final growth numbers do **not** establish convergence: they belong
to different physical wavelengths and different coupled spaces. On the fine
mesh, adding the .0625 m pair increases the coupled growth from 28.5250 to
36.7932/s. The fastest single sine at .0625 m gives 34.0690/s, already above
the .09375 m pair's 30.0323/s; the migration survives removing the favorable
cosine alignment.

Alignment matters despite a nondegenerate sampled force basis. At .125 m the
single sine gives K=1.14279→1.14812 MPa/m and growth 25.8135→25.3077/s;
the cosine gives K=.796109→1.13908 and growth 32.6498→22.2845/s. Six Gauss
labels per cell distinguish these patterns, but two mechanical cells do not
resolve their responses equally. Weighted Gram eigenvalues stay within
[.984029,1.015971]; the problem is not duplicate sampled force columns.

The six-smooth result remains 15.12399→15.11810/s. Thus the transfer preserves
the earlier smooth diagnostic while exposing its missing short wavelengths.

## Same specimen, native mechanics, explicit limits

Native DG Bernstein coefficients from accepted step15/.075 are subdivided
exactly in x; no new growth or seed equilibration is performed. The retained
diagnostic planes are frozen, including its converged virgin-plane selection
from the fixed-phase .075–.0775 interval. All formed angles agree exactly with
the .075 parent. Between original labels, angle and candidate V are defined by
piecewise-linear interpolation in physical x. New measures are dx/cos(angle)
and JxW/ds. Normal resolution, quadrature, mechanics and confinement are unchanged.

The operating point is the original smooth V, peak 2e-4 m/s, with uniform
theta=.001610556306315 s. Candidate Dc=3.2211126126299625e-7 m; matched b=a
uses mu0=.6080471895621705. The original-Dc control uses Dc=5e-4 at this **same
operating point**, not the previously observed physical trajectory. This is a
frozen-coefficient variational analysis of a generally transient state.

Transfer and operator checks:

- Subdivided phase values agree at 10,000 physical points to 2.23e-16; the
  x=4.01 m cross-section is identical. Native initialization limits zero cells.
- Represented V differs from the common physical interpolant by at most
  1.95e-21 m/s; theta is identical. Integrated operating rate changes by 7.01e-6
  relatively and total physical section measure by 3.19e-6.
- The first smooth restoring action changes by .0884% relative to the
  piecewise-linear coarse physical interpolant. Across all six this comparison
  ranges from .0884% to 1.2366%, including coarse interpolation error.
- All seven previously retained native actions are reproduced on the coarse
  mesh within 2.25e-15 weighted relative error.
- Across 30 new native actions: true mixed residual <1.00e-11, independent
  energy/work error <2.56e-13, symmetric point-update error <3.12e-16. Existing
  gates remain 2e-9, 2e-8 and 2e-8, respectively. Persistent diagnostic state
  is unchanged by each measurement.

The benchmark uses the existing bulk-tip mixed operator, native material
derivatives, section adjoints and ASPECT Stokes inverse. At fixed phase and
planes, stress/cohesive-strain derivatives are independent of parent tau/A;
zero histories therefore suffice for this derivative fixture. Coarse action
reproduction tests that construction. **Its t=0 field is not a prepared native
slipping state**, and no checkpoint is converted or continued with altered laws.
The complete derivation, transfer choices and entry points are in the
[benchmark guide](../../../benchmarks/reconstructed_fault/particle_free/stage4_wavelength/README.md).

## Finite growth bound and next experiment

Rates continue to increase over the finer mesh's short-wave pairs: 25.31,
30.03 and 37.62/s. This is not a measured plateau. Nor does it imply unbounded
growth under refinement. Positive elastic energy and the signed regularized
RSF derivatives give, for any growing eigenvalue with real part s,

    s <= max_sections [ftheta*theta*sign(V)/(Dc*fV) - abs(V)/Dc]
      = 155.226 /s.

The proof, including the common Maxwell relaxation, is in the benchmark guide.
It applies to the frozen positive-energy law, not to the nonlinear evolving
fault. A cohesive-only, uncoupled reference gives 61.672/s; that calculation
omits finite-body compatibility and transverse FE restrictions and is **not**
a measured native short-wave limit. A finite amplification timescale could
eventually converge while slip-patch spacing remains undetermined.

**Next justified experiment:** derive that short-wave limit for the actual
transverse profile and perform a targeted frozen 512×128 comparison of the
.0625 m pair (four cells) with the newly accessible .03125 m pair (two cells),
retaining the six smooth patterns and all responsive ages. Ten mechanical
actions would cost roughly 10–15 minutes on four cores, extrapolating the
measured scaling, plus spectrum analysis. This would test the finite-rate
limit; a long free-growth history is not justified by the present evidence.
Changing Dc again would shift the instability margin without establishing a
physical tangential length scale. If a plateau with no selected wavelength is
confirmed, any tangential regularization must be an explicit constitutive
choice with its own energy/work derivation, not a mesh-dependent filter.

## Reproduction, costs and preservation

Root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`;
HEAD `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with incoming manual/source work.
The protected executable is independently identified by SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`;
its old source banner is not equated with HEAD. Only a separate benchmark
checker was built. Production sources, protected runtimes and old evidence
were unchanged; entry/exit audits retain hashes.

| Work | Native actions | Native elapsed | Action timers | Final spectrum analysis |
|---|---:|---:|---:|---:|
| 128×128 reproduction + family | 14 | 351.30 s | 333.68 s | 54.68 s |
| 256×128 family | 16 | 599.52 s | 563.77 s | 243.55 s |

Native runs used four bound cores sequentially: **15.85 minutes total**.
Final analyses used two single-thread workers concurrently; coarse analysis
also had an earlier 27.80 s pass before the signed-carrier observer was tightened
to avoid underintegrated high-frequency Fourier aliases. The observed modes
are independently checked by signed zero crossings. No native action was
repeated for that observer correction. All jobs have exited.

[Exact commands and code](../../../benchmarks/reconstructed_fault/particle_free/stage4_wavelength/README.md)
and [complete comparison CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/comparison.csv)
are retained with per-run inputs, hashes, logs, compact native responses and
compressed exact phase coefficients. Full run directories also remain intact.
The existing cleanup record's historical losses are unchanged.

- [Growth/stiffness versus wavelength, PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/wavelength-spectrum.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/wavelength-spectrum.pdf)).
- [Leading signed rate profiles, PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/leading-signed-modes.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/leading-signed-modes.pdf)).
  Real parts use unit maximum complex modulus and a common cosine phase
  reference at x=4 m; real/imaginary samples are both retained in CSV.
- [Phase, operating point and smooth-action transfer checks, PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/transfer-checks.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_wavelength/transfer-checks.pdf)).
