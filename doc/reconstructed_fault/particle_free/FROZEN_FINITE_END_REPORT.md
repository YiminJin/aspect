# Frozen finite-end spatial/quadrature comparison — 4 November 2026

**Complete: the dominant frozen error is the uncut DGQ2 approximation of a
nonzero source jump at finite mature curve ends. Increasing ordinary quadrature
alone is not an effective remedy.** Keep the direct history as an opt-in
independent reference and separated history as default. This comparison changes
neither native source nor constitutive, endpoint, transport or checkpoint rules.

[Protocol](FROZEN_FINITE_END_PROTOCOL.md) ·
[Evidence and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_finite_end/README.md) ·
[Figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_finite_end/frozen-finite-end.png)

## Common frozen problem and reference

Entry verified the existing worktree, branch `pf-rsf-particle-free` and HEAD
`69e8a10a93a08718d31af961197899e320657c04`. Incoming source/manual work remains
dirty and preserved. Actual five current binary and 33 protected-control hashes
match the preceding provenance; HEAD is not used as a binary identifier. No
rebuild, MPI run, checkpoint continuation or server job was needed. All new
analysis runs used one thread. `entry.json` and `verification.json` record
source/input/control identities and preservation.

The initial state of retained `true-c-p-ref` supplies **one fixed 64² Q1 phase**,
18 labels from (-.8,-.24) to (.8,.24), their I and full-normal support. These
data are identical on every candidate tensor grid; neither reinitialization nor
profile remapping participates. The compact fixture is saved separately. The
prescribed phase underlying that Q1 field was
`.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4` on [-2,2]².
G=10, localization m=50, dt=.1; eta is either 10 or
10(1+.5 tanh(4x)). Surface rates are prescribed as
V_j=.002(1+.2 cos X_j.x), interpolated on the same labels. Incoming physical
stress is zero. The common frozen reaction is

```
D = (.03, -.03, .01), S = sym(t ⊗ n), chi = h(phi_Q1)/I
kappa(x) = -eta(x) expm1(-G dt/eta(x))
f(x) = 2 kappa(x) [D - chi(x) V(x) S]
```

The source has no tangential extension beyond the finite curve. Its normal
support covers the whole box; there is no computational-boundary source defect
in this fixture. At both curve ends the one-sided constant-eta shear jump is
−.00486642 Pa; the layered left/right jumps are −.00463635/−.00494688 Pa. These
are mature localization ends, **not** the unresolved cohesive physical-tip
admission problem. No new tip law or smoothing is inferred.

Candidate DGQ2 grids are 16²,32²,64²,128² with ordinary cell Gauss orders
8,12,16,24. An independent reference P* partitions at every fixed Q1/tensor
cell, surface-label normal line, finite-end line and norm-window boundary.
41,600 triangles on a common 128² integration partition use Duffy-mapped
Gauss6 and Gauss10. This is an integration partition, not additional history
degrees of freedom. Full 3D plane strain uses
|a|²=a_xx²+a_yy²+(a_xx+a_yy)²+2a_xy².

The source evaluator agrees with the previous independent native-source
reconstruction to 1.39e−17 Pa. Recomputed native q8 candidate norm bridges on
the retained constant/layered fixtures differ by 1.89e−15/5.00e−16 Pa m.
Analytic triangle moments and all finest-cell 9×9 DGQ2 mass matrices pass,
with maximum errors 4.17e−17 and 1.28e−17. Reference6→10 coefficient L2
changes are at most 2.68e−14 Pa m and work-traction changes below 9.55e−16 Pa.
All predeclared screens pass without relaxation. The independently integrated
orthogonal decomposition

```
||Pq f - f||² = ||P* f - f||² + ||Pq f - P* f||²
```

closes within 5.03e−22 Pa² m² for q8. `comparison.json` includes every grid,
quadrature and viscosity, component/window errors and timing. `audit.json`
adds regional work, in-rule energy and native/analytic checks.

An initial endpoint-jump diagnostic evaluated ownership exactly at the last
vertex; floating-point projection put it outside the strip and reported zero.
It was corrected to the **one-sided limit** using endpoint phi/I/V. The original
audit is retained as `audit-original-endpoint-equality.json`. Integrated tensor,
traction, quadrature and energy results were unaffected. No native condition or
tolerance changed.

## Spatial error versus quadrature error

Constant eta; tensor L2 errors in Pa m, using the common converged measure:

| Tensor grid | Best-space error ||P*f−f|| | Gauss8 quadrature error ||P8f−P*f|| | Gauss8 total error | Quadrature fraction of squared total |
|---|---:|---:|---:|---:|
| 16² | 5.40920e−4 | 6.28476e−5 | 5.44559e−4 | 1.332% |
| 32² | 4.10764e−4 | 5.95825e−5 | 4.15062e−4 | 2.061% |
| 64² | 2.88812e−4 | 3.21099e−5 | 2.90591e−4 | 1.221% |
| 128² | 2.04254e−4 | 3.15815e−5 | 2.06681e−4 | 2.335% |

Best-space one-halving orders are .397, .508, .500. The two finest values are
consistent with a jump crossing uncut tensor cells: its error has finite
amplitude in a strip of area proportional to h, giving L2 proportional to
sqrt(h). This interpretation applies to this frozen source; it is not a proof
of the order of coupled evolution. At 64², component L2 errors
(xx,yy,xy,zz) are (1.12415e−4,1.12415e−4,1.70496e−4,0) Pa m.

The complete source has norm .340851 Pa m, dominated by the smooth D forcing;
the localized slip part has norm .00554305 Pa m. Thus the constant-eta best
64² tensor error is .08473% of the complete source but **5.210% of the localized
source**. Both denominators are reported to avoid hiding end error behind the
smooth background.

The layered best errors are 5.33316e−4,4.04688e−4,2.84528e−4,2.01224e−4 Pa m.
At 64², the Gauss8 quadrature error is 3.16337e−5; total relative error is
.08509% of the complete source, or 5.214% of its localized part. Spatially
varying kappa products are fully integrated; they do not change the principal
finite-end conclusion.

At 64², constant-eta quadrature errors for q8/12/16/24 are
3.21099e−5 / 2.25222e−5 / 1.57656e−5 / 6.66243e−6 Pa m.
Increasing q8→24 reduces the **total** tensor error by only .59%, approaching
the unchanged 2.88812e−4 approximation floor. Results on 32² and128² agree
qualitatively. The q8 error need not decrease smoothly as endpoints change
position relative to quadrature nodes.

Fixed end windows extend .3 m tangentially to each side of each endpoint and
have |normal|<.6 m. They contain 97.891%,99.574%,99.9953%,99.9989% of the
constant-eta best squared error on the four grids. The central window excludes
.3 m from both ends and uses |normal|<.4 m; its errors are
7.76983e−5,2.66600e−5,1.98866e−6,6.83626e−7 Pa m. Layered localization is
equivalent (99.9951% at64²). These fixed windows differ from the previous
evolved-state 2h windows; their percentages are not interchangeable.

## Point traction, work traction and energy

Point shear traction is integrated along the finite curve with splits at all
cell faces and labels. Work traction solves M t=b with
M_ij=integral chi N_i N_j dA and b_i=integral chi N_i(tau:S) dA. All projections
use the **same converged mass** for the comparison below. This separates tensor
representation error from quadrature in the traction observation itself.

| Grid | Point error P* / P8 | Work error P* / P8 | Best work error, central / ends |
|---|---:|---:|---:|
| 16² | 1.466% / 1.671% | 1.031% / 1.007% | .02078% / 1.749% |
| 32² | 1.642% / 1.897% | .6372% / .6128% | .01385% / 1.081% |
| 64² | .9363% / .9233% | .3172% / .3264% | .006789% / .5381% |
| 128² | .5794% / .6282% | .1588% / .1632% | .003404% / .2694% |

These are constant-eta relative L2 errors. Regional work norms partition the
surface coordinate at .3 m from each end and integrate the full normal profile;
they are not pointwise or restricted-normal bulk norms. Layered best point/work
errors are .9314%/.3156% at64² and .5763%/.1580% at128². Constant best point
errors at64² are .01450% in the central interval and1.609% at the ends. Point
error is not monotone on the coarser grids; a bulk optimal projection is not a
point-traction optimal projection. Small accidental improvements with q8 are
not evidence against the converged reference.

The fixed localization integral is 1.67048942 m versus curve length1.67044904 m,
a +.002417% discrepancy. I is retained, not renormalized to erase that error.
Uncut q8 observation gives +.13231%,+.02847%,−.008161%,−.005477% on the four
grids. `audit.json` records these observation errors and work traction/power
computed entirely in q8 as a separate diagnostic.

For constant eta, signed weak slip-power error of P* is −.1987%,−.1146%,
−.05665%,−.02833%; q8 tensor error measured by converged work is −.1823%,
−.1184%,−.05842%,−.02914%. The prescribed source's slip power is negative
(−7.73678e−5 W/m); these are signed relative errors, not frictional dissipation
claims. Absolute signed differences are also retained.

Projection energy uses E=integral |tau|²/(4G) dA, in J/m:

| Grid | E(f)−E(P*f), converged | E(f)−E(P8f), converged | E(f)−E(P8f), q8 rule |
|---|---:|---:|---:|
| 16² | 7.31486e−9 | **−1.36635e−9** | 7.14348e−9 |
| 32² | 4.21817e−9 | 2.99854e−9 | 4.04409e−9 |
| 64² | 2.08531e−9 | 2.78571e−9 | 2.08661e−9 |
| 128² | 1.04299e−9 | 1.52189e−9 | 1.02605e−9 |

P* satisfies E(f)−E(P*f)=||f−P*f||²/(4G). P8 satisfies the analogous identity
in its **own** quadrature measure, verified independently. It need not lose
energy in the converged measure, as16² demonstrates for both viscosity laws.
Energy differences and point traction can be nonmonotone with q even while
the tensor quadrature error decreases. This frozen reaction has no transport,
flux, endpoint time-integration balance or cumulative trajectory; it neither
closes nor replaces the earlier coupled energy remainder diagnosis.

## Cost, decision and next experiment

The full comparison took18.93 s on one Python thread; the final additional
audit is separately timed in `audit.json`. At64², ordinary q8/q24 source
projections took .0559/.5299 s; at128², .223/2.163 s. The Gauss10 cut reference
took4.22 s **for all four grids and both laws, including work masses/loads**;
partition construction plus fixture load took1.52 s. These are offline timings
with different workloads, not native solver speedups. Quadrature points per
cell grow ninefold from q8 to q24, for little total-error improvement here.

Adopt the cut-integrated fixture as an accuracy reference; **do not adopt a
blanket q24 increase as the finite-end remedy**. The next justified experiment
is a bounded **frozen endpoint-aware tensor-space prototype**, preserving the
same discontinuous source and physical inputs, with matching work integration.
Compare its additional history cost and end/central tensor and traction errors
against this uncut DGQ2 baseline before selecting any coupled implementation.
Accurate integration without additional representation freedom has now been
tested and is insufficient for this source. No such prototype was enabled here.

This isolates a genuine local mechanism consistent with the earlier evolved
end-localized differences; it does not prove that mechanism accounts for every
coupled error. No friction solve, slip/age evolution, translation-direction,
rejection or restart experiment was added: unchanged native lifecycle evidence
is reused. The layered comparison is of the same local continuum forcing,
not an assertion that the separated variable-eta discrete candidate equals f.
Separated defaults, experimental direct adoption status and production limits
remain. Broader Stage C transport/AMR/birth qualification, boundary traction,
cohesive energy/tip/shape work and server runs remain deferred. The unavailable
39 of45 native-pilot frames and earlier temporary logs remain unavailable;
BP3 accepted4237/spatial4182 and the RSF marker2/step748 evidence are preserved.
