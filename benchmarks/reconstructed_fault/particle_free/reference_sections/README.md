# Small material-section geometry controls

For an exact replay of the frozen localized control, extract the [canonical reference](../../../../doc/reconstructed_fault/WORKTREE_CLEANUP.md) in a fresh directory and use its bundled replay and checker. The following source-tree commands also support new experiments with explicitly supplied, matched binaries. The historical temporary build defaults and timing-matrix directories require reconstruction. The width comparison retains its coarse outputs and pinned runtime in the research worktree.

The fixture generator starts from the native finite-tip input. Its arguments specify the mesh rotation and material angle separately. The supplied covariance check rotates the box, velocity and scalar seed together about the coordinate origin. The checker can copy the same raw scalar FE coefficients by cell/local shape before initialization. Use the same MPI partition for that benchmark hook. ASPECT applies its native limiter, constitutive update, solver and checkpoint machinery. Zero function topography selects the affine-capable Q1 mapping for the rotated box.

The following commands reproduce the bounded checks from the worktree root. Set `exe` and `checker` to freshly built paths and choose unused run labels. The local verification used `build-particle-free/reference-sections/aspect-centered` and `build-particle-free/reference-sections/checks/libbulk_tip_checks.release.so`.

```sh
bench=benchmarks/reconstructed_fault/particle_free
tests=$bench/reference_sections
work=build-particle-free/reference-sections
runs=build-particle-free/bulk-tip/runs
exe=$work/aspect-centered
checker=$work/checks/libbulk_tip_checks.release.so
mkdir -p "$work/reproduce"
run_section() {
  label=$1
  input=$2
  shift 2
  python3 "$bench/bulk_tip/run.py" "$label" --fixture "$input" --executable "$exe" --checker "$checker" \
    --ranks 2 --bind-cores --seconds 180 --time-update midpoint --dt .005 --end .025 \
    --end-step 2 --checkpoint-steps 1 "$@"
}
for angle in 0 90 30; do
  input="$work/reproduce/rot$angle.prm"
  python3 "$tests/fixture.py" "$input" --mesh-angle "$angle" --frame-angle "$angle" --rotated-mesh --normal-rate .003
  if [ "$angle" = 0 ]; then
    ASPECT_SECTION_SAVE_SEED=1 run_section "rep-rot$angle" "$input"
  else
    ASPECT_SECTION_SEED_FROM="$runs/rep-rot0/output" run_section "rep-rot$angle" "$input"
    python3 "$tests/compare.py" "$runs/rep-rot0/output" "$runs/rep-rot$angle/output" --degrees "$angle"
  fi
done
input="$work/reproduce/cut.prm"
python3 "$tests/fixture.py" "$input" --frame-angle 30
run_section rep-cut "$input" --rejection --rejection-step 2 --tangent --tangent-step 1
run_section rep-restart "$input" --resume-from "$runs/rep-cut/output" --copy-selected-snapshot-only --snapshot-id 2
python3 "$tests/compare.py" "$runs/rep-cut/output" "$runs/rep-restart/output"
python3 "$tests/compare.py" "$runs/rep-cut/output" "$runs/rep-cut/output" --cut --degrees 30
```

The cut reducer evaluates a quadratic and a Gaussian against analytic antiderivatives on the complete physical chord. It also uses the exported native area and section weights to reassemble signed slip rate and friction power. `cold_section_score.py FIXED_OUTPUT` independently scans the first-step criterion at section 20 using the same oblique specimen run with the existing fixed-direction option. The native run directories retain exact invocations, parameter files, binary and checker hashes, point exports and restart records.

## Localized forward-tip controls

The bounded fixture runner is `localized_controls.py`. From the project root build the native executable and matching bulk-tip checker. Use `--executable` and `--checker` to select another matched build. By default the runner uses the local native build in `/tmp/aspect-reference-sections-native` and checker in `build-particle-free/localized-sections/checks`. It creates a fresh `localized-<label>` directory for each command. For example:

```sh
python3 benchmarks/reconstructed_fault/particle_free/reference_sections/localized_controls.py demo1 --single-tip --step 1
python3 benchmarks/reconstructed_fault/particle_free/reference_sections/localized_controls.py demo2 --single-tip --parent demo1 --step 2
python3 benchmarks/reconstructed_fault/particle_free/reference_sections/localized_controls.py demo2-fixed --single-tip --parent demo1 --step 2 --prescribe demo2
```

For a single forward-front run, `--single-tip` extends the formed seed to the upstream boundary and preserves the original right taper. Omitting it retains the archived two-ended specimen, which can activate its opposite end outside the selected disk. The fixed replay consumes the original parent's checkpoint and the final plane table for that one step. The runner suppresses new replay checkpoints. Use `--degrees 30` for an oblique fault on the unchanged mesh, `--exact` for global lines, `--rotate-mesh` for covariance, and `--probes` for the tangent/rejection observer. `localized_analysis.py` evaluates independent area and frozen radius references. `pilot_frozen.py` reads the recorded horizontal archive schema with hash and endpoint validation. `edge_score.py` directly reconstructs the cold angular boundary case. `localized_results.py` collects the named timing pairs, endpoint errors and checkpoint sizes.

Build the optional `tip_section_geometry_cost` target in the checker project and run it with arguments `256 128 30` for native geometry construction on the 8 × 4 m server specimen. It creates no physical state or checkpoint. [Results and qualification limits](../../../../doc/reconstructed_fault/particle_free/TIP_DIRECTION_AND_STRIP_RESOLUTION.md).

For the first-registration width comparison, run two fresh controls with the same matching executable and checker. Each takes one 5 ms midpoint interval. The first command saves the initial DG seed by cell and coefficient. The second reads it before initialization.

```sh
tests=benchmarks/reconstructed_fault/particle_free/reference_sections
runs=build-particle-free/bulk-tip/runs
python3 "$tests/localized_controls.py" width1 --degrees 30 --single-tip --nx 16 --step 1 --save-seed --executable "$exe" --checker "$checker"
python3 "$tests/localized_controls.py" width2 --degrees 30 --single-tip --nx 16 --step 1 --strip-subdivisions 2 --match-seed width1 --executable "$exe" --checker "$checker"
python3 "$tests/strip_width.py" "$runs/localized-width1/output" "$runs/localized-width2/output"
```

For the sliding interval reuse the retained coarse reference and advance only the 36-strip checkpoint. The reducer takes both parent/endpoint pairs in order. It checks native section history, reconstructs the intact-section raw maxima, and sums each pair of fine slip increments over its parent strip.

```sh
tests=benchmarks/reconstructed_fault/particle_free/reference_sections
runs=build-particle-free/bulk-tip/runs
work=build-particle-free/strip-width
python3 "$tests/localized_controls.py" sliding-replay --degrees 30 --single-tip --nx 16 --step 2 --strip-subdivisions 2 --parent width2 --snapshot 2 --executable "$work/aspect" --checker "$work/checker.release.so"
OPENBLAS_NUM_THREADS=1 python3 "$tests/sliding_width.py" "$runs/localized-oneo1/output" "$runs/localized-oneo2/output" "$runs/localized-width2/output" "$runs/localized-sliding-replay/output"
```
