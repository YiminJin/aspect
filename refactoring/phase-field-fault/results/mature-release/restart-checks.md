# Mature boundary checkpoint verification

ASPECT stores the mesh, clock and three FE solution levels. Its normal material
checkpoint hook writes one accepted row per material ID, in input order:
position and support radius (m), persistent background shear (Pa), optional
law-owned state (RSF age in s), positive speed (m/s), accumulated slip distance
(m), and the interior surface Maxwell memory (Pa m). The endpoints are the
first and last positions in this ordered trace. Their box faces follow from
the incident segments of the prescribed seed. The full-ray support convention
is zero radius at every ID. Trial rate iterates, quadrature projections and
normal integrals are rebuilt.

The new metadata holds the physical box, prescribed vertex positions *before*
fitting or motion, geometry mode, fit choice, initial support and slip sense.
This costs one extra replicated point per initial ID (16 bytes of coordinates
in 2-D). It allows the same vertex table to be moved to a different directory.
The checkpoint reader also checks persistent shear, law identity and
parameters, and the existing adiabatic profile signature where applicable.
ASPECT restores the label records before loading the distributed mesh and
solution. The material's `post_resume_load_user_data` callback binds the normal
integrals to the restored accepted trace and old FE phase. Only a fresh
boundary seed enters the ridge fit. At the next attempt, `prepare_parent`
uses the accepted rate and the restored FE stress and velocity. Acceptance
installs the returned rates and law state once, after the repeat decision.

Earlier branch writers could archive interior traces with the original slip
sense. Their law-specific label keys and state-free wire layout remain valid.
These records omit the virgin coordinates and fixed/advected mode. The old
reader checks the label count, available law signature and radius convention,
and restores accepted positions and persistent loading from the records.
Only the resolved interior geometry and original sense can read this format.
A branch-generated archive with physical seed metadata checks the full identity.

## Commands for the original vertical-Gaussian fixture

From the worktree root use the installed Release executable and rebuild all
observers against that SDK. Each simulator rank uses one numerical thread.
The retry plugin is the one compiled by the existing fixed-fault integration
test. The compatibility control uses the retained pre-change installation
at `build-assignment-d/install` (Release executable SHA-256
`d75d72aec4714266631244fbe621fb2a007388c0c85abc5197e6c26517792edc`).
It writes fresh legacy checkpoints from the branch inputs and inspects the
wire keys to ensure that this is the legacy reader.

```bash
W=$PWD
D=$W/build-mature-release
I=$D/install
M=/opt/openmpi/5.0.6/bin/mpirun
R=$I/benchmarks/fixed_fault/curved_mature
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cmake -S "$I/benchmarks/growing_fault" -B "$D/law-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/law-plugin" -j1
# See geometry-checks.md for the fixed_fault test build and README.md for the layered DSO.
python3 -BO "$R/check_restart.py" "$I" "$D/layered-plugin/liblayered_moving_fault.release.so" \
  "$D/tests/libfixed_fault.release.so" "$D/restart-qualification" --mpiexec "$M" --vertical-gaussian
python3 -BO "$R/check_legacy_restart.py" "$W/build-assignment-d/install" "$I" \
  "$D/layered-plugin/liblayered_moving_fault.release.so" "$D/law-plugin/libgrowing_fault_checks.release.so" \
  "$D/legacy-qualification"
```

Each destination must be fresh. The implementation run executed both scripts from the checkout. The final
replay ran the copies from the installed prefix in a fresh external directory
and included the persistent-shear negative. The older D review prefix lacked
the external dashpot DSO. The documented pre-change Assignment D installation
contains both plugins and successfully wrote both legacy checkpoints.

## First physical endpoint after the checkpoint

The serial curved writer terminated after accepting 12.5 ms. The two new
reader directories received only its generation 02 and an identical vertex
table at a different path. Each reader first published step 2 at 25 ms. The
oracle checks clocks within 10⁻¹⁵ s, field values and identities of all five
IDs on every rank and of 2,304 Q3 samples keyed by (CellId, q). The serial
writer owned all 256 cells. Each rank in the continuation owns 128 cells
and 1,152 Q3 samples, including 1,123 and 1,052 points with nonzero slip weight.
Every rank must own active fault support. Identical-layout runs require exact numerical
agreement. Across partitions the existing mature-field comparison uses
10⁻¹⁰ absolute and 10⁻⁸ relative. Maxima below are across labels or native
samples in their respective units.

| 25 ms comparison | Position, I (m) | Age (s) | Slip (m) | Speed (m/s) | Accepted stress (Pa) | Internal virtual work difference (J/m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One rank, uninterrupted vs resumed | 0, 0 | 0 | 0 | 0 | 0 | 0 |
| Two ranks uninterrupted vs one-to-two-rank continuation | 0, 0 | 2.67e−15 | 1.31e−18 | 8.33e−17 | 6.39e−16 | 5.28e−18 |
| One rank uninterrupted vs one-to-two-rank continuation | 0, 0 | 1.78e−15 | 2.17e−18 | 1.81e−16 | 1.00e−15 | 2.31e−17 |

The work is the integral of the accepted DG shear stress against the unit
engineering shear of the virtual displacement (y,0), per unit out-of-plane
thickness. For each rectangular cell the checker recovers its area from the
outer physical Q3 coordinates and uses the tensor Gauss weights. The serial
work was −6.86656059214564e−6 J/m at 12.5 ms and −7.2018216454774e−6 J/m
at 25 ms. This comparison follows accepted stress publication. The separate
transfer-work probe exercises the pre-constraint Stokes RHS. The 25 ms
forced attempt was rejected once, and the accepted 0, 12.5 and 25 ms frames
of the retried run agreed exactly with the clean solution.

On the same FE phase the fitted control started with a straight seed whose
contacts were (0,0.65) and (1,0.28) m. The first fit placed them at
(0,0.8192946363) and (1,0.1399707555) m. The maximum displacement of any
vertex was 0.1692946363 m. At 25 ms the fitted continuation matched all
five IDs, all 2,304 bulk points and the work of −2.36886643069755e−6 J/m
exactly. Reversing the sense, moving ID 2 by 0.01 m, changing its persistent
shear by 0.1 Pa, and changing RSF radiation damping each stopped a separate
restore before it could publish an endpoint.

The fresh legacy layered RSF parent resumed with zero differences in the
accepted quantities across all 20 moving vertices and 1,728 Q3 samples,
including internal virtual work (5.65475496073642 J/m). The external linear
dashpot's legacy interior checkpoint produced the same 25 ms label table
byte for byte as the current uninterrupted calculation. Its two rates were
0.190216421301 and 0.190216429887 m/s and its law history was empty.
