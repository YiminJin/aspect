# Running the bounded control

[Guide](../TIP_TREATMENT.md)

The input is
[`benchmarks/reconstructed_fault/particle_free/bulk_tip/finite_tip.prm`](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/finite_tip.prm).
It selects `Fault constitutive mode = bulk cohesive` and
`Bulk cohesive flow rule = bulk tip sections`, with an authoritative DGQ2
`phase_field` composition. The inherited `Bulk history representation = DG trial
stress` admits the existing bulk owner; the **bulk tip flow selector overrides
that owner's physical history meaning** to direct native-quadrature τ and A.
It does not enable the mature direct/separated history option. Checkpoint v1 and
point exports explicitly identify this distinction. General stress-composition
fields are inert compatibility fields in this fixture; use the bulk-tip physical
point exports for stress observations.

The specimen is [0,2]×[0,1] m, initially unloaded, with prescribed boundary
velocity (0.02(y−0.5),0) m/s. The initial damaged patch has exact support
x∈[0.5,1.5], y∈[0.25,0.75]; all other material is intact. The compact biquadratic
profile has maximum 0.2. It is declared pre-existing damage, not a tip admission
seed. The coarse grid is 8×4, with ell=0.25 m. Refinement holds ell and inputs
fixed. This is a bounded implementation control, not a spatially qualified
fracture experiment.

G=1 MPa, η=100 MPa·s, Gc=26.6666666667 J/m², calibration c=1000 Pa,
p_N=1000 Pa, radiation=1000 Pa·s/m, μ0=0.6, a=0.015, b=0.02,
V0=0.001 m/s, Dc=0.01 m, initial age=10 s. Thus b>a: the law is
velocity weakening. Mechanical boundary velocities determine work and a pressure
gauge; p_N is an independently prescribed frictional confinement assumption.
Fracture boundaries are natural, with no transport inflow in this fixed-reference
calculation. No particles or reconstructed surface are required.

Use a separate candidate build. Do not overwrite the protected mature build:

```sh
cmake -S . -B build-particle-free/bulk-tip/native \
  -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/opt/dealii/9.6-local \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DASPECT_UNITY_BUILD=ON -DASPECT_PRECOMPILE_HEADERS=ON -DASPECT_WITH_PYTHON=OFF
cmake --build build-particle-free/bulk-tip/native -j4
cmake -S benchmarks/reconstructed_fault/particle_free/bulk_tip \
  -B build-particle-free/bulk-tip/points -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DAspect_DIR="$PWD/build-particle-free/bulk-tip/native"
cmake --build build-particle-free/bulk-tip/points -j4
build-particle-free/bulk-tip/points/bulk_tip_points
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bulk-tip-example 620 python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py \
  example --seconds 600
```

Run one heavy job at a time, ≤4 build workers/MPI ranks and one numerical thread
per rank. The runner creates a fresh label directory and records executable,
checker, inputs, source hashes, command, runtime and status. It never resumes an
old formulation. `--resume-from` copies only a new-law checkpoint into a fresh
output directory; its source remains untouched. `--snapshot-id` selects a retained
regular checkpoint within the copy when a terminal checkpoint carries a tiny
end-time-clipped timestep. `--unloaded`, `--intact`,
`--rejection`, `--dt`, and `--refinement` select the bounded checks. See the report
for actually executed commands and failures before selecting a longer run.
