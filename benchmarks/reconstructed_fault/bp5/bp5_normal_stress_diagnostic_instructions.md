# Codex task: bounded BP5 restart diagnostic for normal-stress noise

Implement the diagnostic plugins and a ready-to-run restart `.prm` for the current reconstructed-fault BP5 model. The purpose is to locate the oscillation in pressure, deviatoric normal traction, or weak projection. This task is diagnostic only; keep the existing mechanics, history update, boundary conditions, mesh, friction parameters, localization, and timestep criteria unchanged.

The available checkpoint is at approximately **5310111071 s**. It already contains accumulated noise, but is suitable for decomposing the current traction and observing a few subsequent increments. Do not replay the first 168 years. Do not claim that a short late restart establishes the historical origin or long-term growth rate of the noise.

## 1. Inspect and reuse the actual implementation

Read the repository instructions and inspect the current BP5/BP3 benchmark plugin, the reconstructed-fault normal-traction assembly/projection, the Maxwell stress evaluation, and restart hooks before editing. Use actual source interfaces and parameter declarations; do not invent ASPECT parameter names or rebuild the constitutive law independently inside a postprocessor.

The supplied production input currently loads `./libbp5_steady_initialization.release.so`, uses material model `phase field fault`, and includes the postprocessors `reconstructed fault BP3` and `BP3 output complete`. Its output directory is `output-30km-loading-surface8`. Preserve the existing required plugins and their restart state. Treat the input and checkpoint associated with the actual run as authoritative if they differ from the supplied `first_event.prm`.

Prefer a small diagnostic postprocessor plugin, plus an output-only helper or hook in the production evaluation if needed. Keep diagnostics disabled by default. Avoid adding persistent particle or fault properties that change checkpoint compatibility. Temporary diagnostic buffers may hold values captured from the converged evaluation.

## 2. Define the stress split precisely

With unit fault normal n and tension-positive deviatoric stress tau, define

\[
q_p=p,\qquad q_d=-\mathbf n^{\mathsf T}\boldsymbol\tau\mathbf n,
\qquad q_n=q_p+q_d.
\]

In 2D,

\[
q_d=-(n_x^2\tau_{xx}+2n_xn_y\tau_{xy}+n_y^2\tau_{yy}).
\]

Use the physical pressure sign and scaling used by the production traction calculation, not the algebraically scaled pressure block. Use the same normal and fault association as that calculation. Include the off-diagonal factor of two correctly.

Distinguish perturbation fields from total stresses. If production adds a reference/prestress normal traction separately, output that contribution separately and verify

\[
\sigma_n^{\mathrm{production}}=P_h[q_p]+P_h[q_d]+\sigma_n^{\mathrm{reference}}.
\]

If the reference is already contained in p and tau, the separate reference contribution is zero. Do not double-count it. Do not arbitrarily assign the nominal 50 MPa to pressure just to make this identity hold. If the code has further additive corrections, expose and name them explicitly. Derive the decomposition from assembly; do not define an unexplained correction as the leftover closure error.

The tau evaluated here must be the stress used for the accepted mechanical solve, including its incoming-history and current constitutive contributions. It must not silently be the old particle stress, a smoothed visualization stress, or `2 * reference_viscosity * strain_rate`.

## 3. Respect the split history cycle

Capture all components from the same converged mechanical evaluation, before history is committed, and publish them only after that step is accepted. Discard captures from rejected iterates/steps. Diagnostics must never advance state, slip, stress history, particles, or the simulation clock.

In particular, do not reevaluate an incremental constitutive update using already committed history and the previous timestep: that can effectively apply an update twice. If a postprocessor lacks the correct incoming history, capture output-only data through the existing converged-solve path instead.

Attempt a baseline dump immediately after restart only if the stored fields and metadata suffice for a consistent stress reconstruction. A mechanics-based checkpoint dump must use the correct history stage. If that cannot be done safely, document the limitation and use the first newly accepted step as the mechanical baseline. Label a directly restored-history dump separately; never present it as the preceding solve's traction.

Record the checkpoint time, absolute accepted-step ID, number of newly accepted steps, and data stage. Use at least 17 significant digits for absolute time and also output elapsed seconds from restart. Do not use rounded log timestamps as state identifiers.

## 4. Essential output: fault profiles before and after mass inversion

Reuse the actual consistent weak projection: identical quadrature, localization weights, geometry factors, normalization/completion terms, mass matrix, and constraints. Do not replace it with centerline sampling or smoothing.

Write one CSV per diagnostic accepted state, sorted by stable fault/vertex ID and down-dip distance. Include the full fault, which is inexpensive compared with a volume dump.

Required columns:

- time_s, elapsed_s, accepted_step, diagnostic_step, data_stage;
- fault_id, vertex_id, down_dip_s_m, x_m, y_m;
- weak_pressure_Pa, weak_minus_n_tau_n_Pa;
- any separately assembled reference/correction terms, with explicit names;
- weak_normal_sum_Pa, existing_production_weak_normal_Pa, closure_error_Pa;
- pressure_load, deviatoric_normal_load, and normal row mass;
- pressure_load_over_row_mass and deviatoric_load_over_row_mass;
- I_h, V_m_per_s, committed_Theta_s, cumulative_slip_m, where already available.

For each contribution use the production matrix M:

\[
M\mathbf p^w=\mathbf f_p,\quad
M\mathbf d^w=\mathbf f_d,\quad
m_i=\sum_j M_{ij}.
\]

The `load_over_row_mass` values are f_p,i/m_i and f_d,i/m_i, not a proposed replacement for the consistent solution. Document their units from the actual assembly. Handle constraints in the same physical space as the existing projection. Do not report divisions at constrained or zero-mass rows as ordinary valid samples.

The purpose of retaining loads and row masses is to distinguish oscillations in the integrated input from amplification by the consistent mass inverse. If the production traction representation is different from the plotting projection, explicitly identify and compare both; explain which enters friction.

An additional fault VTU containing the two weak stress contributions and their sum is useful for ParaView if it can reuse the current output infrastructure cheaply. CSV is the authoritative comparison output.

## 5. Essential output: unsmoothed quadrature samples in two windows

For down-dip distances **22–35 km** and **60–90 km**, export the pressure and deviatoric normal contribution at the actual coupling quadrature points, before weak projection. Use the production evaluation/quadrature, including the integration across the diffuse fault thickness. Do not substitute interpolated VTK values.

Use rank-local CSV files or an existing parallel output facility to avoid gathering the volume on rank zero. Record unique ownership so ghosts are not counted twice.

Required metadata/fields, using existing values wherever possible:

- time, step, stage, MPI rank, stable cell ID and quadrature/sample ID;
- fault ID, along-fault coordinate s, signed normal offset r, x, y;
- cell refinement level and cell size;
- n_x, n_y, p, tau_xx, tau_yy, tau_xy, minus_n_tau_n, their normal sum;
- reference/correction contributions where applicable;
- phase field, I_h, localization weight chi, and the integration weight actually used by assembly.

If a sample has several fault associations, identify each association and its contribution. Preserve quadrature identity and sign conventions. Report samples that cannot be associated reliably rather than inventing a coordinate.

Keep raw data unsmoothed. Along-fault comparisons must use comparable normal offsets or the actual weighted cross-fault integration; do not concatenate samples from different offsets into a misleading profile.

Default to quadrature dumps at the baseline/first valid state and final diagnostic state. Fault-profile output is required at every newly accepted diagnostic step. Do not dump the entire bulk or all particles every step.

## 6. Restart input and bounded execution

Deliver `normal_stress_diagnostic_restart.prm`, based on the actual production input, plus a short launch/staging script if checkpoint handling requires it.

- Enable resume using ASPECT's supported checkpoint mechanism and the existing benchmark helpers. Read the exact time and accepted-step number from checkpoint metadata; the user-provided 5310111071 s is approximate.
- Use an independent diagnostic output directory. Stage a copy of the selected complete checkpoint set and required fixtures there using the mechanism this code version supports. Preserve the original run's checkpoints and outputs. Avoid writable hard links to restart files that may be overwritten.
- Retain all physical parameters, pressure normalization, phase-field and fault geometry, saved mesh, initialization reference data, particle layout, and the 0.02 timestep setting.
- Ensure the steady-state/prestress initializer is not rerun and the fault/property history is not reset on resume.
- Enable the diagnostic plugin and its two windows.
- Stop after **five newly accepted steps**, counted relative to this restart. This is not absolute step 5 and must not inherit an already-expired stop counter.
- Use a restart-relative stop criterion/helper if necessary. Do not choose a fixed physical duration of several seconds: at these timesteps it may trigger thousands of solves.
- Audit existing `BP3 long run complete`, `Stop after first event`, wall-time, and output-completion hooks so they neither stop before the diagnostic states are captured nor unintentionally extend the diagnostic. Override only diagnostic termination/output scheduling as needed and document each change.
- Keep diagnostic CSV independent of the existing slip/time output triggers. Disable redundant heavy full-volume/particle output for this short run. Preserve internal postprocessors needed for mechanics/history bookkeeping.
- Set a reasonable wall-time ceiling, for example 10 minutes after startup, using supported controls; stop cleanly with partial diagnostics if the ceiling is reached. Do not launch a long server job automatically.

For timestep interpretation, reuse the existing timestep-selection record if possible. At each diagnostic step record the accepted dt, available candidate bounds, controlling criterion/location, V_max, and actual versus predicted logarithmic state change. Inspect whether the 0.02 criterion bounds raw |Delta ln Theta| or (b/a)|Delta ln Theta|; report the implemented expression. Do not change its behavior in this task.

If the checkpoint is only available on the server, provide exact build and server commands with clearly marked filesystem-path substitutions. Complete compilation and any available short local verification, then explicitly state that the server restart has not been executed.

## 7. Minimal verification and report

Required checks are narrowly scoped:

1. Confirm the code builds and the `.prm` uses registered parameters/plugins.
2. Verify the algebraic pressure/deviatoric split and the weak projection closure, including prestress and constraints. Report maximum and weighted RMS errors in Pa and the projection solve residual; use a tolerance justified by actual solver accuracy, not an invented universal threshold.
3. Verify restart invariants: time, accepted-step ID, geometry, history, fault V/Theta/slip, and I_h are restored. If restart reconstruction is not exact, quantify and identify it before attributing a discrepancy to physical evolution.
4. If the checkpoint is available, compare the first accepted step with diagnostics enabled versus disabled in separate scratch runs, using the same configuration/MPI size. The added diagnostic must not change dt, V, Theta, slip, or production normal traction beyond solver reproducibility. One control step is sufficient; no long paired run.
5. Confirm unique MPI ownership and closure of global assembled totals. Reuse existing MPI tests where practical; do not launch a second full-resolution MPI configuration solely for this diagnostic.

Provide a lightweight Python plotting script for the diagnostic CSV files. Plot p^w, d^w, their reference-adjusted sum, and production normal traction in the two windows; compare f/m with consistent coefficients and plot final-minus-baseline increments. Overlay refinement transitions where the metadata support them. Preserve the broad physical trend. A neighboring-point linear-interpolation residual can be used as a diagnostic roughness measure on nonuniform spacing, but label it as a roughness proxy, not a proven numerical error, especially near rupture fronts.

The report should answer:

- Which contribution carries the oscillation, and do pressure and deviatoric fluctuations cancel or reinforce?
- Is it already present at quadrature points, introduced during weighted assembly, or amplified by projection?
- Are current oscillations associated with refinement transitions or variations in I_h? Report association without claiming causation.
- Is there a measurable new increment over these five steps? Lack of measurable change at deep creeping locations over milliseconds does not demonstrate long-term stability.
- Are the diagnostic stress values the same ones used by friction?
- Which timestep criterion is active?

Deliver the plugin/helper source, build instructions, the concrete restart `.prm`, any necessary staging script, plotting script, and a short evidence-based report. Keep all changes opt-in. Do not add smoothing, alter I_h, refine the mesh, change fault properties/boundary conditions, relax the limiter, or run a parameter sweep as part of this task. Recommend one subsequent targeted test only after examining these outputs.
