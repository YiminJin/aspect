# 60° thrust BP3-QD on a fixed mature fault

Run `pilot.prm` for a 900 s startup and restart check. Run `production.prm` for the 1,500 Julian year experiment. Both include `bp3.prm` and use exactly the same vertex table, mesh, constitutive law, loading and timestep control. Both can start from zero. The production input can also continue a pilot checkpoint. The small public curved-mature fault uses the same material, signed transfer and accepted-label restart.

## Stampede3

Start with a **pf-rsf Release installation containing the fault RSF timestep and fault slip output plugins** containing the executable, headers and CMake SDK. Load the compiler, MPI and deal.II environment used to build it. Use an absolute run directory on shared scratch. Substitute your installation prefix and allocation below. Preparing the case validates both parameter files and leaves the allocation unsubmitted.

```bash
unzip bp3_qd_thrust.zip
case_dir=$PWD/bp3_qd_thrust
export BP3_RUN="$SCRATCH/bp3-thrust-60"
bash "$case_dir/prepare.sh" /path/to/pf-rsf/install "$BP3_RUN"
cd "$BP3_RUN"
sbatch -A YOUR_ALLOCATION -t 00:30:00 stampede3.sh pilot start
```

After a successful pilot, submit `sbatch -A YOUR_ALLOCATION stampede3.sh production resume` from the same directory. To start production at time zero in a separate experiment, prepare a new directory, set `BP3_RUN` to it and use `production start`. Use `production resume` for subsequent allocations. The default allocation is one ICX node with 80 MPI ranks and one numerical thread per rank. Supply time and account to suit your allocation. The launcher uses TACC's `ibrun`. [Stampede3 user guide](https://docs.tacc.utexas.edu/hpc/stampede3/)

`prepare.sh` builds the benchmark DSO against the specified prefix, checks the SDK's executable link, and stages both inputs, their shared include, the vertex file and the Release plugin. All command-line runs execute **in the staged directory**. `inputs.sha256`, `executable.path`, `sdk.path`, the expanded ASPECT output parameters and per-launch input/log identify the calculation. Each launch verifies the manifest. A fresh launch requires an unused output path. Resume requires the native ASPECT checkpoint pointer. Production rotates three slots, writes every 1,800 wall seconds and checkpoints on normal termination. To stop gracefully during a run, create `output/terminate-aspect`. Move that marker out of `output` before continuing.

For a local pilot use `BP3_RUN="$PWD/run" bash run/stampede3.sh pilot start` after preparing `run`. Set `BP3_LOCAL_RANKS=2` for two MPI ranks. The default checker uses standard-library Python. Add --read-vtu to open the actual pieces with VTK and NumPy. The figure option uses Matplotlib.

## Physical input and coordinates

ASPECT uses metres, seconds and upward-positive y. The first vertex is on the traction-free surface at (0, 34641.0161514) m. The last is on the base at (20000, 0) m. Material IDs increase down dip. The positive normal points into the hanging wall. The input selects `Mature slip sense = opposite tangent`, so positive accumulated slip moves the hanging wall **up dip** relative to the footwall. The side and bottom velocity is −(Vp/2) tanh(r/w) t, where t is the unit down-dip tangent and r is signed normal distance. Far from the trace the hanging wall moves left at 0.25 nm/s and up at 0.433013 nm/s. The footwall moves oppositely.

| Input | Value |
| --- | --- |
| Density, shear wave speed, shear modulus | 2,670 kg/m³, 3,464 m/s, 32.03812032 GPa |
| Effective background normal compression | 50 MPa plus the resolved Stokes normal-stress perturbation |
| Persistent thrust shear traction | 26.546122365133367 MPa |
| Plate speed and target initial speed | 1 nm/s |
| Direct effect a | 0.010 through 15 km down dip, linear to 0.025 at 18 km, constant thereafter |
| Evolution effect b, reference coefficient, reference speed | 0.015, 0.6, 1 µm/s |
| Characteristic slip distance, radiation damping | 8 mm, 4,624,440 Pa s/m |
| Initial perturbation stress and accumulated slip | Zero |

The public material's built-in RSF selection takes a scalar direct effect. The benchmark registers a resistance law that reads the actual station coordinates and supplies Eq. 18 at the section quadrature point. It calls the production `RateAndStateFriction::resistance` for both the traction and its fixed-age derivative. `RateStateHistory` owns age initialization, aging at acceptance and restart. The material still solves each label rate and returns its stress to ordinary ASPECT Stokes assembly. The benchmark also reports accepted station observations. At section quadrature it interpolates the logarithm of the two positive nodal ages. This reproduces the prescribed transition at the target speed. Each nodal age still evolves and restarts in seconds.

`vertices.txt` gives each ordered ID, x, y and initial age. The generator solves the SEAS initial equilibrium at the plate speed with radiation damping. Ages are 8,000 s in the shallow zone, 252,982.213 s at 16.5 km and 8,000,000 s in the deep zone. A change to prestress, rate, compression or friction requires consistent updates to the shared physical input and initial ages. Reproduce the shipped table with `python3 make_vertices.py > /tmp/bp3-vertices.txt && diff vertices.txt /tmp/bp3-vertices.txt`. [SEAS BP3 specification, Eqs. 14–19 and 24–26, Table 1](https://strike.scec.org/cvws/seas/download/SEAS_BP3.pdf)

## What the long run measures

* The branch solves incompressible plane-strain Stokes with a fault-owned Maxwell update. It cannot impose SEAS's elastic Poisson ratio of 0.25. The chosen bulk viscosity of 10²⁶ Pa s gives a Maxwell time of approximately 99 million years. The instantaneous normal stress entering friction is background compression plus pressure minus the resolved normal deviatoric stress. The free surface anchors pressure. The older research donor additionally filtered the normal perturbation along the trace.
* The 80 × 34.641 km box truncates the SEAS half-space. The branch has no prescribed-rate continuation of a mature fault past 40 km. Side and basal loading impose the plate contrast at finite distances. The bottom material rate is still solved. The free surface is fixed geometrically. The mature route admits a chosen physical sense and positive speeds, so backslip and opening lie outside its scope.
* The slip return is diffuse. Its fixed profile has peak phase 0.6 and excess compliance proportional to sech²(r/w), with w = 250 m. The wall ramp is the full-line integral of this shape. Mechanics normalizes the **FE** profile on clipped in-box normal rays and admits a 2,250 m association radius. The largest measured omitted fraction was 3.21 × 10⁻⁸. The finest box cells measure 125 × 120.28 m. Material spacing is 100 m in the transition and 250 m elsewhere. SEAS suggests 25 m resolution. Spatial, width, domain and timestep convergence of ruptures remain open.
* The standard box takes four global startup refinement passes. The maximum-refinement function tests the center against the 1,500 m band plus the projected cell half-extent at each level. Startup coarsening is disabled to preserve cells inserted for 2:1 mesh balance. Phase and geometry stay fixed afterwards.

## Timestep resolution

Edit the two dimensionless budgets in the shared input. Each defaults to 0.1 when the timestep plugin is selected. Zero disables that particular limit. Omitting the plugin leaves the previous ASPECT timestep selection in effect.

```text
subsection Time stepping
  set List of model names = fault RSF
  subsection Fault RSF
    set Maximum slip fraction = 0.1
    set Maximum logarithmic state change = 0.1
  end
end
```

Every evolved material ID participates. The first budget bounds |V| Δt / Dc using the returned rate. The second bounds |ln(Θ₁/Θ₀)| using the same exact age update as publication. At rest age increases by Δt. At steady sliding only the slip cap applies. A completed trial exceeding a budget repeats from its accepted parent. The manager combines the prospective bound from the endpoint with other selected timestep models and ASPECT's startup and growth ceilings. Signed strip rates use their magnitude for aging over the full interval. A law without age or a characteristic slip distance contributes no corresponding limit.

The donor used a log-age budget of 0.2 for BP3 and a separate operator-split stability criterion involving CFL × (a/b) Dc/V. This branch's all-station slip budget controls accuracy. Resolving rupture requires comparison on its accelerating trajectory.

## Observations and events

The pilot writes all 179 accepted material labels and five FE wall probes. Production writes the whole-fault maximum speed and its down-dip location plus rate, age and cumulative slip at 17 stations every 2.5 km, on every accepted clock. These use round-trip precision in ASPECT's checkpointed statistics table. This compact output supports event timing across allocations. Full SEAS traction and off-fault displacement submission files are outside the supplied output set.

Both inputs enable native graphical output. Open `output/solution.pvd` for velocity (m/s), pressure (Pa), phase and accepted `fault_tau_xx`, `fault_tau_yy`, `fault_tau_xy` (Pa, spatial deviatoric stress). The first frame establishes a slip reference at each ordered material ID. A frame is due when any ID accumulates the configured net slip since the last actual frame. Production starts with 0.1 m and a five-Julian-year fallback. The pilot uses 0.7 μm to exercise accumulation. All triggers together write at most once per accepted step. They do not restrict the timestep. A checkpoint with no previous slip reference starts its graphical series at the first following accepted state. The historical donor used 1 m for heavy output and 0.1 m for profiles, with bulk graphics disabled. Reassess the bulk cadence and storage after observing the first acceleration.

From the staged run directory, check a pilot and plot its accepted profiles with

```bash
python3 check_pilot.py output --figure bp3-pilot.png
```

For a production catalogue, use a new export directory and the existing event definition. The exporter reads only accepted statistics and preserves material station distances and the full-precision interval clock.

```bash
python3 history.py output/statistics bp3-event-input
python3 history_events.py bp3-event-input
```

An onset crosses 0.001 m/s upward in the fault-wide maximum. At least one fixed weakening-zone station must gain 8 mm of excess slip during fast sliding. Recovery requires five consecutive accepted observations at or below 10⁻⁸ m/s spanning at least 8 million seconds. The third recovery establishes three distinct events. The 1,500 year horizon covers the requested benchmark interval. Time to a third event on this branch requires measured loading, rupture and recovery histories.
