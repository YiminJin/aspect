# Physical slip sense on a mature trace

Set the sense under the public material. Increasing vertex ID defines the
geometric tangent **t**. Rotating it counterclockwise gives the geometric
normal **n**. In this example the IDs increase down dip, so the default
along-tangent speed moves the hanging wall down dip. With opposite tangent
the same positive speed moves the positive-normal side up dip relative to
the negative-normal side. Positive background shear drives the selected
motion. Slip distance and the speed used by the aging law remain positive.

```text
subsection Material model
  subsection Phase field fault
    set Mature slip sense = opposite tangent
  end
end
```

## Work with prescribed rates at a common parent

The benchmark runs two ordinary initial Stokes solves per measurement. It
changes the middle speed by ±0.001 m/s about 0.005 m/s. Physical time is
zero. The FE phase is 0.35, the incoming xx, yy and xy stresses are 0.13,
−0.08 and 0.04 Pa, and the Maxwell interval is 0.1 s. Rate-and-state ages
and per-ID background shear follow the same material locations when the
vertex order is reversed. The current middle rate enters the Stokes RHS
through the projected constitutive slip stress
−2κ sym(χ V ε **t**⊗**n**). The negative sign in the Stokes weak load
therefore yields a positive work for ε = +1 on the straight trace.

The probe pairs the actual unconstrained local Q3 RHS with a Q2-representable
virtual velocity **v** = **t**₀(**n**₀·**x**) in m/s. Its engineering
shear on the first segment is 1 s⁻¹. Divide the difference of the two
RHS works by 0.002 m/s, κ = 10(1−exp(−0.1)) Pa s and the unit shear rate.
In the reverse direction difference the production middle section row for
this virtual velocity and for zero velocity with the same frozen parent,
phase, rate and pressure. Both rows use zero friction traction and slope.
The identical slip stiffness and persistent background tractions cancel.
Dividing by the same κ and shear rate gives the reverse action in metres.

| Ordered geometry | Slip sense | Stokes (m) | Reverse (m) | Independent Q8 (m) |
| --- | --- | ---: | ---: | ---: |
| Down 60° | Along | +0.589018955762748 | +0.589018955762740 | +0.589018955762740 |
| Down 60° | Opposite | −0.589018955762722 | −0.589018955762739 | −0.589018955762740 |
| Same physical trace with reversed IDs | Along | +0.589018955762751 | +0.589018955762740 | +0.589018955762740 |
| Same physical trace with reversed IDs | Opposite | −0.589018955762722 | −0.589018955762740 | −0.589018955762740 |
| Bend | Along | +0.368882182497310 | +0.368882182497313 | +0.368882182497312 |
| Bend | Opposite | −0.368882182497313 | −0.368882182497313 | −0.368882182497312 |

The independent Q8 calculation constructs the local tangent and normal from
each edge, chooses the nearest eligible foot in physical space and integrates
χ times the middle Q1 hat times the virtual engineering shear. This sign
experiment uses the measured three Q1 normal integrals. The separate public
curved example checks those integrals independently from the FE phase.
The work tolerance is 2×10⁻¹³ m absolute. Straight reversal leaves both
vectors reversed and their dyad invariant.

The bent vertices are (0.12, 1), (0.33, 0.41), (0.85, 0) m. The same
affine virtual velocity resolves 1 and 0.43355536 s⁻¹ on its two segments.
Substituting the endpoint chord in the shear contraction gives 0.448314120344111 m
on the unchanged local measures. This exceeds the correct positive action by
0.079431937846799 m.

At each active Q8 point the test also compares the measured compression
with 5 Pa minus the analytical normal component of the relaxed parent and
trial viscous stress. This is an independent contraction against the geometric
normal at both zero and nonzero virtual velocity. The pressure tolerance is
2×10⁻¹³ Pa absolute. Switching ε produced identical samples at all 58 straight
points (4.89853061249 Pa) and all 56 bent points (4.88086922551 to
5.82188485781 Pa under virtual shear).

## Solved right-dipping thrust

The public material and stock point-values postprocessor run on an 8 × 8
unit square. Both runs have stationary prescribed velocity components, zero
applied wall traction, 3.71 Pa of persistent driving shear in the selected
sense and a smooth transverse phase band. Thus there is no imposed motion
between the two observation points. The hanging-wall station is
(0.5649519053, 0.5375) m and the footwall station (0.4350480947, 0.4625) m.
The positive up-dip direction is (−0.5, √3/2).

At 25 ms the projection of the hanging-wall velocity minus the footwall
velocity onto up dip is +0.0499211 m/s with opposite tangent and
−0.0384916 m/s with along tangent. The corresponding midpoint RSF speeds
are +0.104587370108 and +0.0918137333858 m/s. The two relative velocities
use the Cartesian velocities from each accepted Stokes solution. The stock
point-values file prints six significant digits. Restoring the thrust's own
12.5 ms checkpoint in a fresh directory reproduces the endpoint label file
and both point histories byte for byte at 25 ms.

Shortening the same seed to 80% of its length about the center puts both
endpoints strictly inside the box. Switching to `fixed interior` now
exercises the separate candidate trace and the Q8 projected stress response.
The same two velocity probes gave +0.0445088 m/s for opposite tangent and
−0.0359863 m/s for along tangent. Accepted midpoint speeds were positive
(0.102148542189 and 0.086952263041 m/s).

`FaultPhaseProfile::slip_dyad` supplies ε **t**⊗**n** to the shear
contraction in `FaultStokesCoupling::sample_bulk` and to
`FaultPhaseProfile::slip_gradient`. `prepare_parent` uses the latter in the
accepted bulk-minus-slip spin. Boundary-spanning trials and acceptance use
`slip_stress`; the Q8 trial moments reach the ordinary Stokes Q3 assembly
through `FaultStressLoad`. Interior mechanics uses the same gradient in
`project_interior_response`. It publishes the DG moments that passed coupled
convergence. The positive speed is paired with law-owned history after the
repeat decision. This assignment changes no ASPECT core Stokes assembler.

## Reproduce

Run from the worktree root with the destination-configured Release install.
These checks used one MPI rank and one numerical thread. Set the scratch
directory to a fresh path. The integrated BP3 probe runs the signed tests,
the clean checkpoint continuation, the strict free-32 reference and the
remaining admission diagnostics.

```bash
W=$PWD
I=$W/build-mature-release/install
export ASPECT_SOURCE_DIR=$I OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
# This retained installation already contains the two matching benchmark DSOs.
E=$(mktemp -d)
R=$W/refactoring/phase-field-fault/results/bp3-preparation
python3 -B "$R/check_branch.py" "$I" "$E/bp3-release"
cat "$E/bp3-release/reference.log"
```

For just the signed-work and solved-motion checks, the checkpoint is optional:

```bash
T=$I/benchmarks/fixed_fault/transfer_work
python3 -B "$T/check_sense.py" "$I/bin/aspect-release" "$T/libtransfer_work.release.so" "$E/sense-only"
```
