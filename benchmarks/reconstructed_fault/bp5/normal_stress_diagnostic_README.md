# Bounded BP5 normal-traction restart diagnostic

This observer does not change equations, quadrature, the eight-panel normalization,
support, history updates, mesh, pressure treatment, timestep criteria, or solver
tolerances. It does not diagnose the historical origin of the existing noise.

## Exact restart and lifecycle

The requested source is **`directory/of/output/restart/01`**, accepted step
**5612**, at **5310111071.5634108 s**. The staging program reads and verifies
`bp3_accepted_state.txt`; it never edits `resume.z`. ASPECT's checkpoint contains
the pending next time/dt as well as the retained accepted histories. Thus the
first new state is step 5613, and the five-state run ends at 5617 if all converge.
No fixed short physical end time is imposed. An accepted-state 600-second wall
limit can end the run earlier; it cannot interrupt an individual nonlinear solve.

`normal_restored_fault.csv` and `normal_restored_particles_rank*.csv` are
**restored-history inventories**, not reconstructions of step 5612 traction.
The first newly accepted solve is the baseline for mechanical increments.
The surface system captures the actual pointwise constitutive response in its
linearization with incoming FE Maxwell history, physical pressure, and Q1 state.
Rejected residual trials do not overwrite that capture. A failed nonlinear
solve never publishes it. The observer verifies time, step, convergence, and
exact accepted V before output. It does not reevaluate stress after commit.

## Build on the server

Apply the accompanying source changes to the actual production source checkout.
Rebuild ASPECT **and all loaded plugins** against the same headers/build; the
point-response type gained output fields. Do not load an old plugin against the
new response layout. No persistent/checkpoint layout was changed.

```bash
# Replace these paths with the server's existing build/source locations.
cmake --build /path/to/aspect-build --target aspect -j4
cmake -S /path/to/aspect-source/benchmarks/reconstructed_fault/bp5 \
      -B /path/to/bp5-diagnostic-build -DAspect_DIR=/path/to/aspect-build
cmake --build /path/to/bp5-diagnostic-build \
      --target bp5_steady_initialization bp5_normal_stress_diagnostic -j4
```

Retain the original compiler/MPI/deal.II modules. The diagnostic library is
additional to the required BP5 library, not a replacement for its registered
plugins or serialized state. Additional libraries in the actual production
input are retained by staging and also need ABI-compatible rebuilds.

## Stage a separate branch (no launch)

Prefer the original output's fully resolved `parameters.prm`. If it was not
retained, use the exact flat input that launched that checkpoint. The supplied
eight-panel `first_event.prm` is a template, not authority over the actual run.

```bash
python3 /path/to/aspect-source/benchmarks/reconstructed_fault/bp5/stage_normal_stress_diagnostic.py \
  --checkpoint /directory/of/output/restart/01 \
  --input /directory/of/output/parameters.prm \
  --job /directory/of/original/job \
  --destination /directory/of/new/bp5-normal-diagnostic \
  --bp5-library /path/to/bp5-diagnostic-build/libbp5_steady_initialization.release.so \
  --diagnostic-library /path/to/bp5-diagnostic-build/libbp5_normal_stress_diagnostic.release.so
```

The destination must not exist. This copies the full checkpoint, fixtures, and
libraries, verifies checkpoint hashes, copies its saved output metadata, and
writes `normal_stress_diagnostic_restart.prm`. It uses no writable hard links.
The immutable original input is included as `production_input.prm`; subsequent
overrides change only paths and diagnostic termination/output. Existing output
metadata may name graphical files not copied here; no old VTUs are needed.
The new cumulative-slip CSV begins with the resumed states; slip history itself
comes from the checkpoint, never the CSV. If prior timestep-selection records
exist, their accepted prefix is copied for the already-selected first dt.

Overrides: remove the first-event/absolute-step/wall stop from BP3; retain its
bookkeeping and output-completion postprocessor; suppress heavy native files;
enable independent diagnostic CSVs; request a normal checkpoint on diagnostic
termination. The end-time safety bound from production remains in force. The
limiter remains **max (b/a) abs(log(Theta_pred/Theta)) <= 0.02**, with the
configured friction-law derivatives providing b/a and its exact aging map.
This is not a bound on raw log state change or a guaranteed bound on the change
realized using the next solve's V. Both predicted and realized values are output.

## Run within an existing allocation

```bash
cd /directory/of/new/bp5-normal-diagnostic
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
export ASPECT_FAULT_EXPLICIT_B=1 ASPECT_FAULT_EXPLICIT_G=1
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
# Unset prior investigation/profiling selectors; retain the qualified production
# environment. Do not source an old script that overwrites the parameter file.
/path/to/aspect-build/aspect-release --validate normal_stress_diagnostic_restart.prm
# Same rank count as the checkpoint. Use the site's normal launcher, e.g.:
mpirun -np NUMBER_OF_ORIGINAL_RANKS /path/to/aspect-build/aspect-release \
  normal_stress_diagnostic_restart.prm > normal_diagnostic.log 2>&1
# On Stampede3 in the matching allocation, use ibrun instead of mpirun -np ... .
```

Do not launch a long job. If the wall stop intervenes, keep the partial results.
For the one-step unchanged-mechanics control, stage a second independent
destination with `--control --steps 1`, using the identical source checkpoint,
binary, plugin, ranks, and physical settings. Do not reuse the diagnostic's
already advanced output as the control's starting state.

## Output definitions and checks

- `normal_profile_STEP.csv`: all stable vertex IDs, in stored order (down-dip
  coordinate is explicitly present and plots sort it). Weak loads use the exact
  production `JxW*chi*N_i`; the mass uses `JxW*chi*N_i*N_j`. In 2-D loads are
  Pa*m, mass entries/row sums are m, projected coefficients and f/m are Pa.
  The actual matrix bands are included. All BP5 nodes are physical free fault
  nodes; bound-active mechanics rows do not constrain the physical output
  projection. The `prescribed` column makes other configurations explicit.
- Pressure is the physical **perturbation** pressure supplied to mechanics.
  Deviatoric traction is `-stress:N` from that same constitutive response,
  including incoming history and current crack strain. The off-diagonal
  contraction has factor two. Reference normal traction is assembled separately
  from the response's actual fixed background (nominally 50 MPa here).
  There is no unexplained closure correction. Friction uses the **raw total at
  each QP**, not the mass-inverted nodal diagnostic.
- `normal_qp_STEP_rankR.csv`: unsmoothed owned Stokes QPs in 22–35 and 60–90 km,
  using mapped surface positions for window selection. Includes physical and
  projected coordinates, signed normal offset, cell ID/level/diameter,
  quadrature index, full stress tensor, reference, phase, I_h, chi and actual
  weight. The source cache includes the two endpoint wedges exactly once.
  Unassociated positive-phase QPs in the windows are counted separately, never
  given invented surface coordinates or weights. Raw output is first and final
  accepted diagnostic state, including a wall-limited final state.
- `normal_summary.csv`: closure maximum/mass-weighted RMS in Pa, fresh matrix
  projection residual divided by row mass in Pa, accepted clock, state limiter,
  geometry/I_h differences from restoration, ownership counts and elapsed wall.
- `normal_totals_STEP.csv`: independent owned-QP integrals versus MPI-assembled
  row sums, testing partition of unity/reduction without gathering bulk data.
- `normal_checks_STEP.csv`: accepted phase norm and particle H/stress sums and
  squared sums (unique real parents). Compare with the control in addition to
  all nodal history/rate/slip and production normal-traction fields.

Closure/projection error is reported in Pa and compared to the configured
mechanical accuracy scale, not a universal hard-coded stress threshold. A
meaningful split must close much more tightly than the original nonlinear
relative tolerance times the traction scale. Independent tests check tensor
signs and mass inversion; the runtime checks expose their actual roundoff.
The existing production fresh-linear/nonlinear checks still accept mechanics.

```bash
python3 /path/to/aspect-source/benchmarks/reconstructed_fault/bp5/plot_normal_stress_diagnostic.py \
  output-normal-diagnostic --compare-control /path/to/control/output-normal-diagnostic
```

The plots retain broad trends, compare consistent coefficients with f/m, and
show final-minus-baseline increments. Raw QPs are scatter plots in (s,r), not
misleading lines joining different normal offsets. Cell-diameter maps expose
refinement association without claiming causation. Chord departure is labeled
a roughness proxy, particularly important near a physical rupture front.

At initial delivery the late server checkpoint was not available locally;
the initial verification report records that limitation. See the follow-up
below for the subsequently supplied five-state output.

## Follow-up: sub-MPa plots and constitutive history split

The subsequently supplied `output-normal-diagnostic` contains accepted steps
5613–5617. Run the same plotting command above to write **`normal_plots_revised`**;
the original plots and CSVs are retained. The upper panels now show p, d and
sigma minus the separately exported background in kPa. Additional plots show
pressure/deviatoric chord-departure scatter and 70–71 / 79–80 km raw QPs at
normal offsets −50, 0 and +50 m, each within ±1 m. They are scatter plots, not
normal interpolation. Companion CSVs give full cell IDs, rank, QP, offset and
stress. Dotted lines mark actual fault vertices. Cell labels do not pretend to
locate physical cell boundaries. The first/final matching checks exact QP
identity/geometry and reports the tiny work-weight differences separately.

The original exports contain **only total stress**. The opt-in capture now also
reports the following terms from the existing constitutive evaluation:

```
history = beta * incoming_working_FE_stress
strain  = 2*kappa * strain_rate
slip    = -2*kappa*(history_localization + chi*V)*S
stress  = history + strain + slip
```

The physical stress expression and its evaluation order are unchanged. The
additional terms are evaluated only when requested by this observer. They are
not formed by subtracting large final/history stresses, and never use newly
committed particle history. Raw CSVs retain each tensor, the unscaled incoming
FE tensor, their normal projections and closure. Nodal CSVs retain the three
normal weak loads and consistent projections using the unchanged mass matrix.
Global-total component IDs 3, 4, 5 mean history, strain and slip respectively.
For the straight mature fault the slip tensor is orthogonal to N; its direct
normal projection should be roundoff zero, though bulk-mediated normal feedback
is present.

Rebuild ASPECT and **all** loaded libraries again for the extended response type.
Merge only these diagnostic edits into the server tree: retain its separately
approved restart-only normalization allowance; do not change the 1e-10
integration tolerances. Repeat from a fresh copy of the **original step-5612
checkpoint**, not the diagnostic's final step-5617 checkpoint. The same staged
five-step configuration suffices. The plotting script detects new columns and
adds history/current-step panels. It explicitly reports the split as unavailable
for old exports, rather than inventing it from current total stress or history
checksums. This extended late restart has not been executed locally.
