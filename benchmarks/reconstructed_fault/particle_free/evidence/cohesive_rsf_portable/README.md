# Portable setup and complete-support checks

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_RSF_PORTABLE_REPORT.md)
· [active workflow](../../cohesive_rsf_native/ACTIVE_WORKFLOW.md).
Baseline f2f730529 plus local cleanup. Existing qualified pilot outputs and
failed findings are preserved in their original locations.

- `summary.json`: source identities, counts and scope; no new propagation claim.
- `verification.json`: twelve fixed native cases and two optional retained-pilot
  controls, exact commands/binaries/inputs, complete restart comparisons and
  expected failure markers. Missing support/configuration/history is an expected
  negative outcome, not a successful physical timestep.
- `clearance.csv`: actual certificate values in metres, limiting walls,
  trial/initial/restart stages and inside flags. The complete disk and all
  temporary/formed lifting support are checked, independently of formed length.
- `parameter-validation.json`: all ten native inputs pass syntax validation.
- `geometry-unit.log`: analytic turned-disk/interior-extremum and dense enclosure
  checks, including extrema outside the box while Gauss3 points fit.
- `direct-launch.json`: direct native launch and runner have bitwise-identical
  surface, geometry and reconstructed bulk exports at five accepted states.
- `active-results.json`: short fresh API smoke metrics; no measurable damage
  growth. Near-zero dissipation makes relative work/sliding ratios uninformative;
  the absolute remainder and its denominators remain explicit.

Raw runs/checkpoints stay under `build-particle-free/cohesive-rsf-native-portable`.
The direct run and interrupted non-yielding OpenMPI launch are preserved under
`build-particle-free/cohesive-rsf-native`. The interrupted run accepted only its
initialization and is not combined with completed data. Direct OpenMPI launch
with the bounded runner's idle-yield setting completes normally.

Reproduce the artifact-free controls after building host/plugin:

```sh
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/qualify.py \
  portable-check --label NEW-CHECK --aspect "$ASPECT_BUILD/aspect-release" \
  --plugin "$PLUGIN_BUILD/libcohesive_rsf_native_checks.release.so" \
  --mpiexec "${MPIEXEC:-mpiexec}"
```

Only optional `--regression-from RETAINED_OUTPUT` reads the original pilot archive.
The retained controls reproduce step175→176 and its final checkpoint fields
bitwise, with right-wall sampling clearance **.0277772 m**. This certificate
prevents omitted support; it does not establish boundary independence or repair
the original normal/point-traction, spatial and path-hypothesis limitations.
