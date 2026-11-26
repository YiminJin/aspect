# Historical vertical-Gaussian mature fault fixture

These are the measurements at `04aaf6db0`. The current [public example](README.md) uses finite-segment distance; the retained fixture below prescribes the original vertical profile.

Start at `68f6d1510` on `pf-rsf`, after the four local Assignment D repair
commits. The production commits are `69342d99a` (prescribed segment geometry),
`540c4ff5b` (physical slip sense) and `0af424ddf` (accepted boundary restart).
The follow-up example and verification files run against those production
sources. All writer runs, including the legacy controls, start at time zero
on this branch or its earlier installation. The BP3 donor and distributed
server checkpoints are separate experiments.

![Accepted FE phase, contacts and local up-dip motion](vertical-gaussian.png)

**Figure.** The prescribed five-vertex fault crosses a 1 m square box in a
nonuniform static DGQ2 phase band. Squares mark the left and right contacts.
At 25 ms the four points on the right show the hanging-wall velocity relative
to the footwall, projected up dip at each segment midpoint. The velocity
samples lie 0.065 m to either side along the local normal. The measured
relative velocities are 17.384, 36.699, 53.510 and 73.443 mm/s. Persistent
vertex shear drives the motion with stationary prescribed wall components
and zero applied wall traction. The labels carry positive RSF speeds ranging
from 0.112328 to 0.170746 m/s at this clock. The graph shows a difference of
two bulk velocities at finite separation.

## Run the historical input

Use the preserved historical installation (`build-mature-release/install`).
Its executable reports production revision `0af424ddf`, and its example
files match `04aaf6db0`. Only the installed executable and data are needed.
The parameter file extends the existing homogeneous mature example.
Edit a coordinate in the table to change the prescribed curve, its
`background_shear_Pa` column to change persistent driving, or
`Mature slip sense` in the parameter file to reverse the physical motion.
These commands were run from a fresh external directory, with launch settings
equivalent to the `public` case in the replay below.

```bash
W=$PWD
D=$W/build-mature-release
I=$D/install
export ASPECT_SOURCE_DIR="$I" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
E=$(mktemp -d /tmp/aspect-mature-release.XXXXXXXX)
C=$I/benchmarks/fixed_fault/curved_mature
(cd "$E" && "$I/bin/aspect-release" "$C/curved_mature.prm")
```

The standard label tables give accepted position, age, slip distance and
speed. The named visualization fields give the accepted DG stress in Pa.
The independent verification adds an existing postprocessor to sample
native fields. Adding it reproduced every standard label table byte for byte.

## What each experiment measures

| Check | Measured result |
| --- | --- |
| [Ordered geometry and old controls](geometry-checks.md) | Incoming IDs and unequal lengths survive initialization. The unit probe places a point on the global chord whose correct normal offset from the middle segment is −1/16 m. The public curved fit, crossed trace and repeated vertex produce distinct input errors. The homogeneous and frozen three-label controls retain their original references. |
| In-box normal integrals | The four segment lengths sum to 1.26223969 m. The independent projected integrals at IDs 0–4 are 0.21158940, 0.25189317, 0.22521693, 0.18324906 and 0.11352145 m. Maximum error 1.12×10⁻¹⁶ m. |
| Normal support at the bend and contacts | Each segment admits points whose perpendicular foot lies on it. On the inside of a bend multiple segments may admit a point. The closest wins, with the first ID winning a tie. Outside the end strips or an outer bend a point can have no admitted foot. The bulk samples included 401 with multiple eligible feet and 129 with none. The independent station and slip-weight errors over all 2,304 Q3 points were 2.23×10⁻¹⁶ m and 6.67×10⁻¹⁵ m⁻¹. |
| Resolved, nonuniform curved work | In the three-ID work probe using the same phase and two contacts, the actual unconstrained Stokes action was −0.00080620588233 m in the left contact cell and −0.00443265883749 m in the bend cell. All 256 cell actions agreed with independent local-frame Q8 integration within 2.0×10⁻¹⁶ m. The total Stokes action was −0.387448270023805 m and the reverse production row was −0.387448270023790 m. |
| [Physical sense and solved thrust](sign-checks.md) | Actual Stokes and reverse actions agree with ±0.589018955762740 m on the one-cell right-dipping trace; remapping the ordered properties gives the same answers. At fixed stress every measured compression is invariant. Under thrust loading the solved up-dip difference at 25 ms is +0.0499211 m/s; switching the sense gives −0.0384916 m/s. A shortened **fixed interior** seed exercises the candidate section and projected stress contract separately; the paired velocities there are +0.0445088 and −0.0359863 m/s. |
| Old stress and spin | On the 12.5–25 ms interval, the test reconstructs the old accepted DG tensor, rate and bulk spin at Q8, rotates, then applies Maxwell relaxation. It compares cell-mean shear to the observed Q3 constitutive offset less the independently computed current signed slip stress. The maximum difference was 7.12×10⁻¹⁷ Pa. Changing just the sign of slip in the old spin gives **0.00132274 Pa**. |
| [Publication and continuation](restart-checks.md) | Clean versus retried and serial checkpoint-only continuation match exactly. After restoring a serial writer on two ranks, the 25 ms stress differs from the clean two-rank result by at most 6.39×10⁻¹⁶ Pa. A straight seed whose fit displaces the first contact by 0.1692946363 m resumes exactly. Incompatible sense, positions, persistent shear and law coefficients are diagnosed. Fresh earlier interior RSF and age-free histories continue on the current binary. |

The phase oracle reconstructs a separate polynomial on each physical cell
from its nine native Gauss values. It intersects each normal with the box,
splits it at every Cartesian cell face, uses independent Gauss(16) and
Gauss(24) panel rules and solves the consistent Q1 mass system on the four
unequal edges. It retains the three production along-segment stations. Near
the left contact the normal runs from −0.143666998 to +0.189311382 m and
integrates to 0.210140888975 m. On either side of the middle bend the last
station of segment 1 gives 0.237544478347 m and the first station of segment
2 gives 0.215396560084 m. The tolerance for the projected integral is
2×10⁻¹³ m. On this mesh the active point (0.53125, 0.71875) m has foot
(0.50189946, 0.68817652) m; projecting onto the endpoint chord would give
(0.41035284, 0.54096007) m. Full-ray normalization followed by nearest-foot
admission does not form a partition of unity around a sharp bend.

For the work probe only, keep the two contacts and middle vertex. The two
external perturbations change the middle speed by ±0.001 m/s at zero
physical duration. From the same analytic virtual shear and the same Q8
measure the independent test computes each cell's action using its own
in-box integrals and reconstructed DG phase. The 2×10⁻¹³ m absolute budget
applies to the Stokes cells and the global reverse row. The complete
five-ID public trajectory solves for every rate through the coupled Picard
iteration. [Weak form, sign and physical normalization](sign-checks.md)

## Replay the two installations

The historical installation is read-only. Its vertical Gaussian is named
`curved_mature.prm`. In the current installation, that name selects the
finite-segment Gaussian and the identical historical input is available as
`vertical_gaussian.prm`. The [current build recipe](README.md#reproduce-the-runs)
installs the candidate and its observer into fresh destinations.

For the retained local installations the following reproduces the historical
and candidate label checks, native fields, figures and mechanical work.
Each driver runs a bare public case followed by its matching observed case
and three invalid-geometry controls. For the comparison, the historical
drivers are read from the pinned Git revision, the candidate uses the
current drivers, and each loads its own observer DSOs. ASPECT resolves
included files and vertex tables from the corresponding install prefix.

```sh
W=$PWD
R=$W/refactoring/phase-field-fault/results/mature-release
old=$W/build-mature-release
current=$W/build-curve-phase
E=$(mktemp -d /tmp/aspect-vertical-pair.XXXXXXXX)
bash "$R/replay_vertical.sh" "$old/install" "$old/layered-plugin/liblayered_moving_fault.release.so" \
  "$current/install" "$current/layered-plugin/liblayered_moving_fault.release.so" "$E/replay"
```

At all three clocks (0, 12.5 and 25 ms) the two runs agreed exactly on
phase at 2,304 native points. Across those frames the largest differences
were 1.67×10⁻¹⁶ m/s in speed, 5.38×10⁻¹⁶ Pa in stress, and 1.23×10⁻¹⁶ J/m
in integrated virtual work. Regenerating the historical figure produced
the recorded PNG byte for byte. Both work probes matched independent cell
integration within 2.01×10⁻¹⁶ m.

Exchanging the layered libraries stops the replay during setup. The error
identifies the configured and expected SDK paths. This command also checks
that no run directory was created.

```sh
if bash "$R/replay_vertical.sh" "$old/install" "$current/layered-plugin/liblayered_moving_fault.release.so" \
  "$current/install" "$old/layered-plugin/liblayered_moving_fault.release.so" "$E/swapped" \
  > "$E/swapped.log" 2>&1; then exit 1; fi
grep 'Layered observer .*expected' "$E/swapped.log"
test ! -e "$E/swapped"
```

Before and after the replay, SHA-256 for the historical executable was
`4029757b271ef8c016442877529ef8db70cfaf08e20025e468e5696187ce1de5`,
for its `curved_mature.prm` was
`2c2581c77785cc1d4bcecf5094d9f0ba3be83994f3df8e9a3f602d9bc7d4abc0`,
and for its layered observer was
`2c0959881436db71e2d6671431c288172b5c9844bf47c3ff022c515093adcadc`.
The inherited parameter file, vertex table and installed work probe
(`93d0927c438683b6dc569c78f76072c6da2b6beec9af4893eaba06c80b65dbae`)
were also verified unchanged.
The candidate used for this particular *vertical* replay reports
`91cd47e39`, executable
`1513bf4c64038f729a628cda639d7a4fd6c26dd4876f724ee4d3a079a5d26eb4`
and observer `6f4ace515f0cb03dc6074c284602234b02da8a9788f9007a533dd933c6563821`.
The [finite-segment reference](public-baseline.md) has its own capture at
`ae676e240`. The [geometry](geometry-checks.md#commands),
[signed transfer](sign-checks.md#reproduce) and
[restart](restart-checks.md#commands-for-the-original-vertical-gaussian-fixture)
records preserve the wider historical qualification.

The replay-repair checks used `/tmp/aspect-vertical-pair.X6sWa4gi`. The
paired run is in `verified`, with its before/after identities and the
separate swapped-observer diagnostic in `swapped.log`. Both installed
restart suites, the legacy RSF and stateless continuations, the two straight
lifecycle suites and the public material-interface suite also passed in
fresh directories. The signed-work replay reproduced the opposite actions
of ±0.58901895576274 m. Its strict free-32 comparison covered 38 labels at
all three clocks, with a maximum speed difference of 1.39×10⁻¹⁵ m/s.
The historical 12.5 ms checkpoint cost replay measured 417,550 bytes,
7.06 s elapsed and 209,952 KiB peak RSS for the two-step run.

In the original qualification, the scoped CTests all passed and the full Release unit suite passed 2,605
assertions in 64 test cases. The earlier D review prefix can supply the old
layered writer, but lacks the external dashpot DSO. The complete two-law
replay used the pre-change Assignment D prefix in the linked commands.

The original qualification ran in `/tmp/aspect-mature-release.ApY6B8Sb`.
The executable reported production revision `0af424ddf`, SHA-256
`4029757b271ef8c016442877529ef8db70cfaf08e20025e468e5696187ce1de5`.
The measured one-rank public run with per-step checkpoints took 7.26 s
and peaked at 229,788 KiB process RSS while other local checks ran. Its
12.5 ms checkpoint, including the distributed FE data, occupied 417,551
bytes across five files. The extra replicated input geometry costs two
doubles per prescribed vertex. The normal integrals and stress projection
are rebuilt from accepted state and the distributed FE fields. The fixed
mesh and five IDs here establish no full-size BP3 throughput.

To repeat the historical checkpoint cost measurement, use the variables
from “Run the historical input” above and set that example's
output to a fresh directory and use one numerical thread as above. The time
includes initialization, two physical steps, checkpoint writes and
visualization. File sizes vary with recorded run metadata and paths.

```bash
cat > "$E/resource.prm" <<EOF
include $C/curved_mature.prm
set Output directory = $E/resources
subsection Checkpointing
  set Steps between checkpoint = 1
  set Number of checkpoints to keep = 4
end
EOF
python3 - "$I/bin/aspect-release" "$E/resource.prm" <<'PYTHON'
from pathlib import Path
import resource, subprocess, sys, time
run = Path(sys.argv[2]).parent
start = time.monotonic()
with (run/'resource.log').open('w') as log:
    subprocess.run(sys.argv[1:], cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
print(time.monotonic()-start, 's', resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'KiB')
print(sum(f.stat().st_size for f in (run/'resources/restart/02').iterdir()), 'checkpoint bytes')
PYTHON
```
