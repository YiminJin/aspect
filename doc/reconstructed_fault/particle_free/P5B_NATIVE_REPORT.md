# P5b — bounded serial native tensor integration qualified

Cleanup and the selected native-integration task are complete. The experimental
fixed-tensor and radial-tensor bulk modes use the existing native owners,
constraints, coupled solve, acceptance callback and checkpoint infrastructure.
The scalar default, P4 inputs/archives, P5a package and protected BP3 artifacts
remain. **No directional specimen, MPI expansion or propagation run was begun.**

Entry: f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
[Selection](P5B_NATIVE_SELECTION.md),
[verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/verification.json),
[changed files](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/changed-files.json),
[commands/provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/provenance.json),
and [compact package](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native.zip)
provide the inputs and measured evidence. Full native runs/checkpoints remain
separate under `build-particle-free/p5b-native/`.

## Storage, equations and acceptance

`Bulk cohesive flow rule` defaults to `fixed scalar`; `fixed tensor` and
`in-plane radial tensor` are explicit experimental bulk-only, serial selections.
They reuse the P5a stateless return. DGQ2 T plus its native FE source complement
remain authoritative, with Gauss3 projection and all nine coefficients. Cohesive
A/K/Q/W/L are physical histories at the existing CellId/mechanical-quadrature
points. The FE displacement memory is a separate owner. Physical phase and its
accepted obstacle retain their single native FE owner; no quadrature phase copy
or evolving scalar director is stored.

With beta=1/(1+dt G/eta), trial T=beta(Tn+2G dev3 eps(du)), tau=T-2GA and
W=(A-beta An)/(beta dt), the return minimizes

    F = ||T-2GA||_3D²/(4G) + G A:A/h + R sqrt(2)||A-beta An||_3D.

Use the P5a intact limit, stick/slip rule and full direction derivative.
K advances by dt W and L by dt sqrt(2)||W||_3D. Old tensors are not rotated.
The mixed unknowns are displacement increment, p/G and physical phase. The
mechanical weak form uses tau/G and scaled pressure; the phase residual is
(fracture derivative minus Y)/Kf. The full constitutive action enters both
cross blocks and the phase block, including direction changes. Full 3-D
plane-strain contractions include zz and both symmetric xy contributions.

The reduced incremental potential is the quadrature sum of F plus native
fracture storage minus beta G times the p/G incompressibility work. Therefore
row factors beta G and Kf recover its mechanical/phase Hessian. Native
AffineConstraints apply C^T g and C^T H C once. Stability eliminates the actual
mixed mechanics/pressure block; it does not freeze mechanics or include
artificial constrained-row diagonals. Existing continuation, KKT, source,
energy and stability acceptance remain before the once-only pending commit.
Rejected trials keep T, native memory/projection, A/K/Q/W/L and phase unchanged.

Tensor modes have separate `bulk cohesive fixed/radial tensor cell history v1`
archive tags. Every independent tensor component, L, all DG/native coefficients,
physical phase anchors, cell geometry, coefficients, quadrature, clock and
publication count are retained. Derived native projection is rebuilt. Scalar
v2/v3 layouts remain unchanged. Incompatible mode/layout restart fails; loaded
scalar-to-tensor conversion is not supplied. This bulk fixture has no fault
surface or Z: reported weak loads are actual bulk stress/mechanical and phase
functionals, not a claim of surface-history qualification.

Rebuild bulk observer plugins that include `bulk_cohesive_internal.h` with the
new executable; preserved P4 binaries/plugins remain a matched pair. Existing
scalar observers and generic composition/particle stress output are not promoted
to tensor authority. This fixture exports stress directly from the new owner.
Generic particle infrastructure remains (684 diagnostic points, freeze setting
1); no particle stress feedback or domain read is added to the bulk solve.

## Controls and measured results

The control is a .25 m square material block, initially unloaded with constant
physical phi=.2 as the pre-existing damage obstacle. Representation error of
that constant is zero; unloaded KKT is obstacle-active. It is **not** a diffuse
seed or propagation specimen. An 8x8 base with four central cells refined once
has 76 cells, 2665 native DoFs, 2052 DG coefficients, 707 independent coupled
unknowns and 64 active hanging-node constraint rows. Mesh and geometry stay fixed.

Coefficients reproduce the selected P5a tuple: G=1e6 Pa, eta=1e8 Pa s,
R=1000 Pa, m=4, Gc=1 J/m2, ell=.046875 m and Kf=8 Pa. Native input derives
m=Kf/(cohesion²/(2G)), so this point-integration control needs cohesion=2000 Pa.
Correction (P5c): the selected P4 fracture.prm overrides cohesion to 2000 Pa, so its actual m=4 matches this integration fixture. The earlier attribution to m=16 missed that override; original archived evidence is retained.
These are distinct parameter tuples; this report does not relabel the control
as a parameter-matched P4 specimen.

Identical boundary histories impose engineering xy rate .015/s through .3 s,
normal-difference rate .015/s during (.3,.6] s, then hold to .8 s. The qualified
sequence has 64 intervals, dt=.0125 s, Gauss8 mechanics and unlimited DGQ2.
All changed layouts start fresh. Gauss8 is retained; this homogeneous affine
control does not qualify quadrature on a general evolving seed.

| Verification | Measured defect/result | Retained bound |
|---|---:|---:|
| Scalar versus fixed-tensor stress/history/storage/driving identity | 1.10e-13 | 1e-12 |
| Stress/total weak-load identity | 3.41e-14 / 3.98e-14 | 1e-12 |
| Radial/fixed identity before turning | 2.89e-14 | 1e-12 |
| Complete coupled action FD, scaled / relative | 1.88e-10 / 5.73e-8 | 1e-6 |
| Independent C^T g / C^T H C | 5.42e-21 / 1.18e-16 | 1e-12 |
| Independent/native relaxed minimum (89 critical masters) | 5.940868186672286 / 5.940868186672295 | defect 1.50e-15 |
| Cross-block symmetry | 1.45e-15 | 1e-11 |
| Uniform C identity | exactly zero defect | 1e-12 |
| Uniform/hanging physical mean comparison | 6.33e-14 | 1e-12 |
| Main-mode source / mechanical residual / phase KKT maxima | 3.26e-15 / 5.26e-14 / 8.70e-11 | 1e-12 / 1e-9 / 1e-7 |
| BE ledger maximum absolute remainder | 2.83e-15 J/m | 1e-10 scaled |

The minimum mode happens to be uniform, so frozen and relaxed minima coincide.
That does not mean relaxation was omitted: the normalized Schur correction has
norm 5.21574 and reduces a nonuniform admissible direction's curvature by
.392237. An independent inverse includes pressure and constitutive response.
The unchanged stability kernel also detects a synthetic relaxed negative mode
(-.25), a positive mode (.5) and an unresolved zero mode. The gate remains a
bounded local smooth-branch/critical-space test, capped at 1024 masters; it is
not a global minimum or general nonsmooth-stability theorem.

Independent history recurrences check once-only T forcing, Maxwell plus flow
in A, K/L accumulation, physical stress, A/Q identity, Y, storage and all three
dissipations to <=1.65e-13. These use actual volume-averaged exports; no history
is remapped to another constitutive state. The original scalar suite still
passes 240 assertions/five cases; the full P5a point fixture also passes again.

![Native response](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/native-response.png)

At .6 s, scalar/fixed phi=.292522, tauxx=4492.98 Pa and tauxy=1862.97 Pa.
Radial phi=.483120, tauxx=1287.36 Pa and tauxy=520.00 Pa. Its nonzero old Axy
is retained while Axx grows; active tensor-flow angle reaches about 81 degrees.
Directions are absent during zero flow. These are different selected
constitutive responses, not accuracy errors against either law. The hold
relaxes stress/memory and has a positive obstacle multiplier; phase was never
blanket-frozen during it.

![Native work ledger](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/native-energy.png)

The initial uniform AT1 storage is .1 J/m (per out-of-plane thickness), subtracted
from plotted storage changes. CSVs retain bulk/cohesive/fracture storage, three
physical dissipations, each signed BE remainder and mechanical/phase residual
work separately. Exact BE closure is not physical-energy/time qualification.

## Lifecycle, costs and preserved failures

Restart at publication 40 (.5 s), two converged outer rejections, deterministic
retry and advancement to publication 64 reproduce all 24 subsequent states
bitwise: 120 history/solution/stress/weak-load/phase-force files plus every
accepted ledger row. The fixed-tensor mode separately replays publication
63 to 64 bitwise. Mode and quadrature mismatches both stop before publication
and leave copied checkpoint bytes unchanged.

One-run solver totals for scalar/fixed/radial are 7.066/7.468/13.596 s; process
walls including startup/output/checkpoints are 14.190/15.288/28.371 s. Radial
assembly/linear/history/stability costs are 9.887/.390/.313/.517 s; peak process
memory is about 216 MiB. Radial has 28 growing steps versus four in the fixed
control, so these are measured costs, not an intrinsic algorithm speed ratio.
Timing runs did not overlap. The final source was built with the recorded GCC
12.4/deal.II 9.6 incremental dependency/link chain; all protected artifacts
retain their hashes. No server package/job or fresh BP3 trajectory is involved.

Preserved failures and corrections:

- Initial include validation referenced old fixture plugin names. Registering
  the established aliases fixed only fixture attachment. An unmatched scalar
  m=16 run is retained; it is excluded from the m=4 comparisons.
- Radial dt=.1 failed Newton globalization from the existing zero-interior
  continuation guess. Boundary-cell trial gradients, unlike the equilibrated
  affine field, activate phase. One smaller interval was selected from that
  amplification, without a tolerance change. The .1 failure remains; large-step
  radial robustness/temporal accuracy is not qualified.
- The smaller-step check exposed artificial hanging-row diagonals in the old
  FD diagnostic: .0227692 error despite independent-row errors around 1e-11.
  Excluding constrained rows fixes the mathematical action comparison; native
  Newton/acceptance equations were not changed by this diagnostic repair.
- An analysis dependency was replaced with existing NumPy/csv infrastructure.
  A fixed-restart hold probe initially assumed configured dt; the native saved
  interval was only 8.88e-16 s owing to the end-time cap. No clock rule was
  changed: the preceding valid checkpoint was replayed instead.

Cleanup losslessly indexed nine completed dense exports: 2,669,563,692 bytes
became 616,477,233 bytes; measured free space rose 2,044,260,352 bytes (1.90 GiB).
108,619 retained regular files and 173 links passed preservation checks, and
an actual restore passed. See the
[cleanup index](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_22/README.md).
New P5b raw outputs are stored separately from the compact review package.

## Readiness and next task

P5b passes within this serial homogeneous fixed-mesh/Gauss8 bulk scope. Native
history, coupled tensor actions, constrained local stability and lifecycle are
connected and tested. Virgin/intact native phase activation, general nonuniform
seed quadrature, MPI, transfer, finite rotations and propagation are not covered.
P4's failed temporal/spatial criteria (8.302%/7.442% against 5%), 57.26% boundary
sensitivity, original energy/branch findings and strict_reference_match_passed=false
remain. No new physical propagation accuracy claim follows from this uniform block.

Next, review a finite-specimen directional control with explicit boundaries,
initial seed, material tuple and loading interval. The selected P4 fracture specimen and this integration fixture share m=4; the original m=16 attribution was incorrect (see P5c baseline correction).
Then select one short physical response/timestep check with the existing relaxed
stability gate. That specimen task is proposed only and has not begun.
