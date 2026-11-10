# Propagation specimen selection — 9 November 2026

**Decision: reject the current 4×2 m specimen, 0.8 m seed half-length and
Gc=26.6667 J/m² for the next growth test.** Reconsider toughness relative to
strength and specimen dimensions before selecting a new loading. A uniform
background is mechanically preferable to the closed compact return flow, but
neither existing case establishes a tip-focused, energetically suitable specimen.
The scaling gate does not support another baseline/perturbation pair on these
unchanged scales; **no new coupled runs, parameter sweep or growth test were
launched**. The comparisons below reuse clearly identified earlier outputs.

Entry: `5f1fc6f3c656dfb7e1ea03e2a6507258929dbca2`, branch
`pf-rsf-particle-free`, existing worktree, initially clean tracked files.
The retained core is `5d8f9ab1…c18e3e`; no source, executable, checkpoint,
constitutive law, timestep convention or acceptance criterion was changed.

## 1. Actual energy, driving and scales

The [current bulk-tip model](tip_treatment/model.md), not the old surface-cohesive
pilot, has

\[
 E_f=\int K(\phi+\ell^2|\nabla\phi|^2)\,dx+\text{SIPG faces},\quad
 K=\frac{G_c}{(8/3)\ell}=40\ {\rm Pa},\quad
 h=\frac{m\phi(1+\phi)}{(1-\phi)^2},\quad
 m=\frac{2GK}{c^2}=80.
\]

With A=eS, S:S=1/2, q=Ge/h and Y=h'q²/(2G), the exact implemented BE
elimination gives

\[
 \frac{Y}{K}=F(\phi)(X/c)^2,\qquad
 F=\frac{(1+3\phi)(1-\phi)}{[m\phi(1+\phi)+(1-\phi)^2]^2},\qquad
 X=\beta[\tau_n:S+2G\Delta D:S+Ge_n]-f_s,
 \quad\beta=(1+\Delta tG/\eta)^{-1}.
\]

F at φ=0, .01, .1, .5, .8 is 1, .319, .0127, .000344, .0000512.
A high seeded phase does **not** imply a large local driving force at the same
predictor X. The actual phase force includes fracture-gradient/SIPG terms and
the coefficient obstacle; Y/K alone is not its propagation criterion.
The retained BE point exports satisfy Y=h'q²/(2G) within 5.69e-14 Pa.

The 1000 Pa `Cohesions` input calibrates m; it is not an extra friction yield
stress. In an entirely intact section V=f=0, Y/K=(tau_xy/c)². In intact points
of an active section the reaction instead uses q=tau_xy−f_s. Adding 600 Pa
friction to c and treating 1600 Pa as the remote intact threshold would be wrong.
At the initial age 10 s and V=V0=.001 m/s, the configured smooth RSF plus
radiation gives f=601 Pa. This is a reference value, not constant resistance;
friction remains rate/state dependent and nonlocal across each section.

| Quantity from the current input | Value | Interpretation |
|---|---:|---|
| G, eta, Maxwell time eta/G | 1e6 Pa, 1e8 Pa s, 100 s | Loading to .08 s is nearly elastic. |
| Uniform nominal shear at .08 s | 799.680 Pa | The earlier ≈800 Pa estimate was correct for homogeneous remote loading. It did not assess propagation energy. |
| Seed half-length a / ell | .8/.25 = 3.2 | Diffuse finite seed; no well-separated sharp-crack scale. |
| Side ligament; center-to-top distance | 1.2 m = 4.8 ell; 1 m = 4 ell | Nearby walls; not an infinite-body specimen. |
| Energy/strength length L=G Gc/c²=(4/3)m ell | 26.667 m; a/L=.030 | Fracture cost is large compared with elastic release on the seed scale. |
| Illustrative length G Gc/(c−601)² | 167.50 m; a/L=.00478 | Positive sliding resistance further reduces the usable shear contrast. Not a universal RSF criterion. |
| Homogeneous elastic energy at tau_xy=c in the whole box | 4 J/m | Reference storage scale, not a bound under arbitrary heterogeneous loading. |
| Fully developed extension .25 m at each end | Gc×.5 = 13.333 J/m | Already exceeds that whole-box reference storage. Not the cost of infinitesimal diffuse damage. |

For an isolated transverse AT1 profile constrained to peak alpha,
phi(n)=(sqrt(alpha)−|n|/(2ell))² on its support. Direct integration gives
half-width 2ell sqrt(alpha) and energy per axial length Gc alpha^(3/2).
At alpha=.8 these are .4472 m and 19.081 J/m². Thus the equilibrated profile is
wider than the input bump's .2 m half-width. The actual 2-D discrete seed has
20.987724 J/m fracture energy; its zero-clock minimization includes DG faces.
That pre-existing fracture inventory is not recoverable elastic fuel for growth.

These lengths and storage comparisons are screening indicators, **not** a
Griffith formula or proof that every possible loading must fail. Boundary work
can continue to supply energy, and small diffuse adjustments cost much less
than a fully developed extension. They do show why merely increasing loading
until something grows is not a suitable selection strategy here.

## 2. Retained field check at .08 s

Same nominal seed and material values, native Bernstein phase. The uniform BE
case allowed phase to evolve (dt=.005); the two pure compact cases held phase
fixed (dt=.02). They are **not** a newly controlled baseline/perturbation pair.
Tip masks are |x−1.2| or |x−2.8|≤.375 with |y−1|≤.375; remote is |x−2|>1.3;
boundary is distance≤.25 to any wall. Masks overlap, as in the original reports.

| Retained case | Tip max Y/K | Remote max Y/K | Boundary max Y/K | Tip / boundary max abs(tau_xy), Pa |
|---|---:|---:|---:|---:|
| `bernstein-localized01`, uniform shear | .07310 | .73209 | .73209 | 847.44 / 855.62 |
| `tip-time-screen-wide-seed01`, compact width2 | .01889 | .56264 | 1.10400 | 326.99 / 1333.11 |
| `tip-time-screen-seed01`, compact width1.2 | .00872 | .15186 | 4.48818 | 444.86 / 2054.31 |

Uniform loading has one same-sign active band at this endpoint: integrated
positive/negative fault deformation is +.01785895 / 0 m²/s. Compact width2 has
+.001303042142 / −.001303042142 m²/s, while integral(ds abs(V)) is only
4.495e-13 m²/s: **99.999999983% cancellation**. These are fault-only w values,
not total elastic/viscous strain. The signed local friction/radiation work through
.08 s is +.017086170503 / −.017086170500 J/m, leaving only 2.51e-12 J/m net.
Opposite signs can coexist inside one connected sampled band; counting bands
alone would miss this issue. No clipping, rate floor or extra localization was
introduced. Actual discrete local/section power identities agree within
2.78e-17 J/m over these steps.

The uniform run adds only 1.2712e-5 J/m fracture, with max phase change .0003494;
its physical work remainder is .151266 J/m. Compact width2 changes phase exactly
zero and has .429680 J/m remainder. Signed and accumulated absolute remainders
coincide here; BE stress loss dominates. Small ledger errors are not an accuracy
certificate. Existing no-resolved-advance and compact boundary-damage evidence
remain in the [time-update report](TIP_TIME_UPDATE_AND_GROWTH_REPORT.md).

## 3. Required scale change and next decision

**Do not retain this loading/specimen for a growth experiment.** The parameter
to reconsider first is Gc relative to G, c and a—not a larger loading amplitude
or a weaker solver tolerance. At fixed G, c and a, merely making a/L order one
would put Gc at .8 J/m² without friction, or .1274 J/m² using the illustrative
601 Pa resistance. These are scale diagnostics, **not selected new toughness
values or predicted thresholds**. Changing Gc also changes K and m, hence the
cohesive compliance law; it is a material change requiring explicit calibration.
If Gc must remain 26.6667, the seed and body dimensions instead need to be much
larger relative to the 26.7–167.5 m energy/strength scales, with walls moved away
from the enlarged seed. Extending only the seed inside the present box is not
that change. Raising c alone would not remove the energy cost or enlarge the box.

The next justified action is to select a physically intended Gc/c² and a
compatible seed/body scale, retaining the stated RSF closure limitations. Only
then select one uniform-background and one perturbed fixed-phase comparison.
No particular perturbation or next growth run is authorized by this assessment.
The BE accuracy limitation and experimental midpoint failure remain independent
constraints on any subsequent growth interpretation.

## Reproduction and availability

Run `python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/assess_specimen.py`.
It performs algebra, independent AT1-profile quadrature and reductions of existing
CSV files; it launches no solver. The local
[assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/propagation_specimen/assessment.json)
records per-region stresses/rates, signed work, original executable/input/data
hashes and fixed-phase checks. [Entry identity](../../../benchmarks/reconstructed_fault/particle_free/evidence/propagation_specimen/entry.json)
distinguishes the current binary from the earlier binaries that produced these
fields. Source equations are `bulk_tip.cc::update`,
`bulk_cohesive_history.cc::initialize_bulk_cohesive_history`, and
`phase_field/fracture.cc::assemble_dg_fracture`; no mechanics was reimplemented.
Inherited losses remain 39/45 pilot frames, early temporary logs and unavailable
old `bulk_continuation.zip`. All retained outputs/checkpoints are untouched.
