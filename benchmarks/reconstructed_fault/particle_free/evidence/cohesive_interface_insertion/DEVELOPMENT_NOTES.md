# Retained limits and implementation diagnostics

- The first finite-fixture invalid-publication injection corrupted Gauss record 0,
  outside the installed surface. That is not a physical record; the expected
  rejection did not occur and the fixture reported `Atomic rejection failed`.
  The injection now targets the first actual installed Gauss record. Subsequent
  owner/restart tests pass. The original output directory is retained as a prior
  directory; this was a test-indexing defect, not a constitutive failure.
- The uncached 128×80 run was deliberately stopped with SIGTERM after accepted
  checkpoints, before its 900 s bound, to avoid continuing an unnecessary repeated
  contraction. Its log, input and prior output/checkpoints remain in WORK. No
  scientific result is inferred from that incomplete run. The cache retains the
  original summation order; old/current and final full-trajectory identity checks
  pass bitwise.
- The coarse physical growth run stops before publishing trial 2.44 s at the
  declared .25 m candidate buffer. The last accepted trajectory state is 2.435 s.
  The finer controls use the unchanged loading through a common 2.42 s endpoint;
  there is no later hold or endpoint extension in those trajectories.
- ASPECT takes a rounding-sized final step to reach exactly 2.42 s. The original
  field-export selector found two rows within its time tolerance. It now selects
  by accepted publication index, preserving both actual steps rather than deleting
  history or changing a solver tolerance.
- Negative unrestricted release curvature is retained, not clipped. Independent
  analysis finds a positive strict intact-strength margin, positive first-order
  gaps in both signs of the negative mode, and positive admitted curvature.
- The final insertion implementation reuses the front weak certificate at the
  unchanged accepted state/rate before geometry publication. This factoring does
  not change the evolution equations; the complete common coarse trajectory is
  bitwise identical (`final_kernel_identity.json`).

WORK is `build-particle-free/cohesive-interface-insertion`. None of these entries
supersedes prior milestone failures, production/BP3 evidence or the limitations
in the main report.
