# Portable fresh workflow and domain protection

Baseline `f2f730529289eff2c5e8fbba09a224c9bd9a7247`, branch
`pf-rsf-particle-free`, with local cleanup changes. All prior data, failures,
native anchors and the qualified pilot are preserved. No constitutive law,
physical calibration, quadrature, solver tolerance or production/BP3 default
changed. No longer propagation experiment was launched.

[Active workflow and direct commands](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/ACTIVE_WORKFLOW.md)
· [fresh input](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/active.prm)
· [verification record](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_portable/README.md).

## Active inputs and lifecycle

The standalone `active.prm` selects fresh **prescribed fields** at t=0. It reads
no prototype JSON, response kernel, checkpoint, or archived evidence. Standard
ASPECT Box parameters supply domain origin/extents/repetitions. Seed x/y,
projected x length, angle and cells are configurable. Six functions of normalized
seed coordinate prescribe jump, damage, normalized age, cohesive reference,
relaxed jump and accumulated absolute slip. Separate functions supply all three
DG tensor components and the two native displacement-memory components.

The established representation remains
`T_b=T_DG+2G dev3 eps(d_R)−Pi3[2G dev3 eps(d_R)]`. Initial jump is retained in
the existing Gauss offset. Relaxed jump must have zero endpoint values and match
the Q1 trace at Gauss points within the existing 1e−13 m identity allowance;
inconsistent histories are rejected, not silently projected/reset. The initial
normal diagnostic uses the actual incoming tensor and native normal-work measure.
Initial energy is inventory; no extra fracture charge is introduced.

The boundary plugin now accepts total strain and angle functions of physical
time, retaining the existing affine strain basis. Its interval-average velocity
uses prescribed displacement at t and t−dt, so a loading-to-hold transition is
included consistently. Virgin age, candidate epoch, completion retry/extent
budgets and diagnostic warning distance are explicit. The formed-length default
remains **7.5 m**. Physical R and numerical lifting width remain separate.

Native prescribed setup, Box dimensions and loading schedule are checked on
restart. Changed loading and changed domain controls fail before publication.
The unchanged retained-pilot mode keeps the old arithmetic and v2 native-anchor
signature; it does not require conversion. Extending only End time is distinct
from changing a loading history or mesh under a checkpoint.

Build/compiler/deal.II paths are ordinary CMake inputs. Runner binary, plugin,
work directory, launcher, MPI flags and rank-count flag are configurable through
CLI/environment. The runner verifies the actual MPI rank count. Active analysis
uses only native outputs, supports direct launches and relocated directories,
and does not read the historical reference CSV. Archive packaging/version-query
paths are configurable too; archived regression analysis remains explicitly
separate from the active workflow.

## Geometric certificate

For the formed endpoint x_tip and its tangent t, protect the complete disk
`D_R = B_R(x_tip + R t)`. Protect the actual lifting support
`S_w = {(x,y): x0<=x<=x_end, |y−g(x)|<=w}` over **every formed and temporary
piece**. This is the implemented vertical graph strip, not a substituted normal
tube. Actual bulk-mesh rectangular bounds are used.

Disk bounds are analytic. Quartic graph bounds use outward-rounded interval
Horner evaluation on 32 fixed reference-coordinate subintervals. This includes
extrema between Gauss points, and is a conservative safety certificate rather
than an exact contact location. It changes no integration rule or physical
length. Strict positive certified clearance is required. The unsupported
nonrectangular, moving, periodic and general-mesh routes remain guarded.

Checks occur before initial publication, direction sampling, candidate mechanical
assembly, formed admission and restart. Accepted diagnostics export sampling and
lifting clearances. Approach within the warning distance is written to the log
and `cohesive-domain.csv`, with stage, time, accepted step and limiting wall.
Negative clearance rejects before publication; the last accepted checkpoint
remains the recovery anchor. No truncated-disk renormalization is allowed.

| Control | Sampling clearance | Complete lifting clearance | Result |
|---|---:|---:|---|
| Fresh default seed | 4.500 m | 3.000 m | Pass |
| Sampling disk outside right wall | −.100 m | +1.400 m | Initial publication rejected, even with formed budget=100 m |
| Descending seed near top wall | +1.19843 m | −.150 m | Complete strip rejected although formed-tip disk fits |
| Oversized temporary reserve | +.100 m | −.224462 m | Rejected at candidate assembly; accepted step remains 0 |
| Near-boundary valid seed | +.200 m | positive | Explicit approach warning; advancement passes |
| Retained final pilot geometry | **+.0277772 m** | +1.34027 m | Pass with explicit right-wall approach warning |

The original formed-length budget and this geometric condition are independent.
Increasing that budget cannot extend a path with missing sampling/lifting support.
Positive clearance does **not** establish absence of physical wall influence.

## Bounded verification

- Three geometric unit controls pass: analytic turned-disk bounds, a lifting
  maximum outside the box while all Gauss3 points fit, and independently sampled
  curved-support enclosure (16023 assertions).
- Twelve fixed native controls on two ranks pass their expected outcomes:
  fresh initialization/loading/hold; nonzero Q1 relaxed jump and nonzero native
  memory on an oblique seed; uninterrupted/split/restarted advancement; incompatible
  loading/domain and nonrepresentable-memory rejection; three support failures; checkpoint recovery; approach
  reporting. Existing six-history, ownership and once-only publication checks remain.
- Custom restarted bulk records are bitwise identical; all six Gauss histories,
  identities, geometry and publication counts match. Recovery from the failed
  temporary-tail attempt reproduces the accepted initialization unchanged.
- Two optional archived-pilot controls pass: actual step 175→176 at .462 s and
  completed .462375 s checkpoint readback. Surface/geometry and reconstructed
  bulk exports are **bitwise identical** to the qualified reference. The original
  179-step pilot is not rerun or relabeled as a new qualification.
- All ten retained/new parameter files pass ASPECT validation. The active suite
  launches from a new empty directory outside the source tree. Direct MPI launch
  using the input's relative plugin path produces bitwise-identical physical
  exports at all five accepted states. Relocated-output analysis agrees exactly.

The four-step fresh control has no measurable growth: this tests the API and
lifecycle, not propagation. Maximum scaled source discrepancy is 4.321e−16,
mechanical weak error 1.419e−12 Pa, surface weak error 2.360e−12 Pa, and BE ledger
error 3.564e−14 J/m. In the nonzero-memory custom control these maxima are
3.218e−15, 8.805e−11 Pa, 9.954e−12 Pa and 1.166e−13 J/m. Existing allowances
are unchanged. The tiny sliding/fracture denominator in a sticking smoke run
does not provide a useful relative physical-energy qualification.

Fresh native run: 9.36 s; custom control/split/restart: 7.39/7.49/6.84 s; archived
step/readback: 53.58/2.23 s. These are local control costs, not scaling results.
A direct OpenMPI launch without idle yielding was much slower on this laptop;
it was interrupted at trial step1, retaining its accepted initialization checkpoint.
With the runner's `OMPI_MCA_mpi_yield_when_idle=1`, the direct run's native timer
is about 6 s and its physical results match bitwise. This launcher setting is
documented/configurable; it adds no model damping. Original interrupted artifacts
remain separate from completed evidence.

## Changed files and remaining scope

Core changes are confined to `cohesive_geometry`, private `cohesive_system`
parameters/configuration, initialization, state, advance and preparation guards.
The fixture loading plugin, runner, active analysis and archive tool path options
are updated. New files are the active input/workflow, geometric unit checks,
this report and compact evidence. Existing CMake source/test discovery is reused.

The law and pilot reference remain intact. Point/normal-reaction accuracy,
spatial/branch findings, finite-wall sensitivity and the uncalibrated path/R
hypothesis remain unresolved. A fresh prescribed state is explicit data, not an
automatic equilibrated preparation or proof of reachability for arbitrary
histories. No general mesh/AMR, topology, transport, opening, inertia or longer
propagation qualification follows from this cleanup.
