# 60° BP3-QD thrust on a fixed Cartesian box

`production.prm` starts the historical 150 × 50 km experiment from its own equilibrium ages and zero perturbation stress. Both ends of the 57.735 km fault slide by the mature RSF law. The 900 s regression uses `pilot.prm`, its original 40 km trace, broader phase profile and separate `output/`. A pilot checkpoint cannot seed production.

## Stampede3

Use a Release installation of **this pf-rsf source revision**, including the headers and CMake package. Load the compiler, MPI and deal.II modules used for that installation. The ZIP contains source for the benchmark plugin and all input data. The following commands assume the ZIP is in the current directory. Set the installation prefix and billing account for your allocation.

```sh
export ASPECT_PREFIX=/absolute/path/to/pf-rsf-release
export ACCOUNT=your_allocation
export BP3_RUN="$SCRATCH/bp3-historical-first-event"
unzip bp3-first-event.zip
bash bp3-first-event/prepare.sh "$ASPECT_PREFIX" "$BP3_RUN"
cd "$BP3_RUN"

# Fresh full grid. End after the first accepted physical step, then checkpoint.
sbatch -A "$ACCOUNT" --export=ALL,BP3_END_STEP=1 stampede3.sh production start
# After inspecting that job and its checkpoint, resume through accepted step 3.
sbatch -A "$ACCOUNT" --export=ALL,BP3_END_STEP=3 stampede3.sh production resume
# After checking both contacts, rates, limits and restart, continue from step 3.
unset BP3_END_STEP
sbatch -A "$ACCOUNT" --export=ALL stampede3.sh production resume
```

The launcher uses 80 MPI ranks on one ICX node, `ibrun` and one numerical thread per rank. These are qualification allocations. Measure peak memory, solve time and checkpoint time before choosing a production allocation. Production uses `historical-output/`. The logfile records the exact solver, SDK, DSO, table and expanded input. `inputs.sha256` is verified before every launch. Each startup requires a fresh output directory. Each resume requires the native checkpoint pointer from that case. Do not re-run the preparation command in a used directory.

The simulation checkpoints every 1,800 wall seconds and on termination. The launcher creates `historical-output/terminate-aspect` 30 minutes before the Slurm deadline, allowing the current step and terminal write to finish. Increase the margin in a new staged input set if the full-grid measurements require it. To request an early stop, create the same marker. After a successful terminal checkpoint, move the marker aside before the next resume. The continuation stops after the first accepted peak at or above 0.001 m/s followed by five consecutive accepted peaks strictly below it, or at 1,500 Julian years. Equality resets the streak. Event and output state survive native restart. `history.py` exports the accepted statistics, and `history_events.py` retains the separate recovery criterion for analysing cycles.

## Physical input and measurements

* The shared file `bp3-common.prm` sets the 32.03812032 GPa shear modulus, 10²⁶ Pa s Maxwell viscosity, 2,670 kg/m³ density, zero thermal expansion and gravity, 50 MPa background compression and 26.546122365 MPa persistent thrust shear. The free, geometrically fixed top anchors the pressure. Sides and base prescribe both velocity components. Static temperature is 293 K in the historical experiment. ASPECT's original elasticity is disabled.
* Edit `production.prm` for the historical mesh and controls. With 2 km root squares the ninth refinement gives 3.90625 m cells. A band and two contact footprints stay fine. Far cells reach 1 km. Persistent surface-to-base IDs in `historical-vertices.txt` preserve all five donor anchors and both transition knots. Positive slip moves the hanging wall up dip. Friction continues below 40 km. This finite-box, incompressible experiment differs from the SEAS elastic half-space with prescribed deep creep. Opening and reversed slip are outside the mature law.
* The stationary AT1 profile has peak 0.6, length 20 m, curvature one and compact radius 39.529164 m. The historical plugin builds the 5,000-point donor table at startup. The FE phase is its Q1 vertex interpolant embedded in static DGQ2. The wall ramp integrates the same profile's excess compliance. Mechanics integrates the FE excess compliance over station normals, including exterior Q1 completion, and extends endpoint speed into each physical contact wedge. The donor's compliance magnitude is 120.1429512. Its common scale cancels in normalized slip and loading. The 45 m support encloses the radius and an entire finest cell's projected width.
* The 15–18 km friction transition uses true distance down dip, log interpolation of positive nodal ages at friction quadrature, and exact nodal aging in seconds. The 20 m along-fault Helmholtz filter acts only on the resolved perturbation. Set `Normal traction filter length = 0` for the raw pointwise control. The branch uses Q8 for the paired section and Stokes slip work.
* The first and subsequent ceilings are 4 million and 40 million seconds. The accepted-state slip/Dc and absolute log-age budgets are each 0.1. Growth starts at 25%, with block AMG, linear tolerance 1e−9, nonlinear tolerance 1e−8 and 60 iterations. The supported initial Maxwell interval is 100 s. On a resolved contact mesh the archived 4 million second choice exhausted 60 Picard iterations at time zero. Each attempted-step log records the material proposal, material ID, distance, realized increments, retry and next-step restriction.
* Statistics contain the fault-wide peak and distance plus material samples every 2.5 km and at the base. Samples at intermediate coordinates use linear slip and rate and log-linear age. Accepted full profiles include every ID, signed physical position, speed, age, slip, positive-updip shear **including radiation damping**, raw and friction compression (positive in closure), and the projected normal integral. They are written at 0.1 m accumulated slip at any ID, one year, onset, and completion. The initial frame is included. Native bulk graphics use 1 m or 100 years. The DG stress fields contain accepted perturbation stress in Pa. ASPECT pressure is the Stokes pressure perturbation.

## Local controls

```sh
cd "$BP3_RUN"
BP3_LOCAL_RANKS=1 bash stampede3.sh pilot start
python3 check_pilot.py output --read-vtu
BP3_LOCAL_RANKS=2 bash stampede3.sh contact start
OPENBLAS_NUM_THREADS=1 python3 check_historical.py contact-output
BP3_LOCAL_RANKS=1 BP3_END_STEP=1 bash stampede3.sh intermediate start
BP3_LOCAL_RANKS=1 bash stampede3.sh intermediate resume
python3 make_historical_vertices.py > regenerated.txt
sha256sum historical-vertices.txt regenerated.txt
python3 make_historical_vertices.py --verify-shipped historical-vertices.txt
```

Staging verifies the published vertex SHA-256 and independently evaluates every equilibrium age (relative tolerance 1e−13) and coordinate (1e−8 m). The 2 × 1 km contact case resolves both 20 m filtering and the transverse profile at production spacing. The intermediate calculation spans the entire historical fault at 31.25 m finest cell size and approximately 100 m station spacing. Its contact loads are resolution measurements. Use the independent cut and traction checks to interpret their error. The historical donor logged onset at 163.893 years and 10.85 km down dip. The branch has passed startup controls only. Event timing, completed-event throughput and spatial/temporal convergence require server runs.
