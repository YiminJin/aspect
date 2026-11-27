# Independent review of Assignment A on pf-rsf

Review the five commits in the following range. The baseline is the previously reviewed C2 tip, and the endpoint includes the implementation, tests, example and documentation.

```text
14358d2a56c2542aace26e91adafbc69e1ee71e2..0e6b2233cb98fbe2ce7cca63c2e5f2de39b1eb15
```

This is a review-only assignment. Read the worktree’s AGENTS.md and the Assignment A entry at the top of the working record. Use an ordinary installed Release executable for physical checks, a retained C2 executable for before/after comparisons where available, fresh scratch outputs, and at most four total local workers. Record the executable and commit used. The original findings covered shared material settings, phase identity, friction initialization and restart, output integrity, and the canonical/legacy comparator. Mature weak-work coupling, strip-8 direction selection, CZM and Stage D have separate assignments.

### 1. Can an old input silently acquire different physics?

This is the first test to run. Start from a valid historical growing input with `Model name = growing fault`. Explicitly omit the growing viscosity and the public fault viscosity. Keep its growing seed, friction and all other settings fixed. Compare the baseline and candidate with and without the following addition.

```text
subsection Material model
  subsection Simple model
    set Viscosity = 1e6
  end
end
```

Before A, growth read its own viscosity, with a default of 1e8 Pa s. The new resolver records Simple viscosity on both routes. Use the same physical point and clock to compare accepted shear stress in the four runs. State whether the intent of a legacy growing input is preserved, whether a previously irrelevant value becomes the only source, and how this relates to the documented alias. Some cross-subsection edits were expressly meant to gain effect under the new public name. Distinguish that intended change from a historical material-name regression.

Check public-only, legacy-only and omitted settings on both routes, numerically equal spellings, an explicit value equal to a declared default, and a genuine conflict. Anchor the public-G control to accepted shear stress at 5 ms, cell `0_0:`, point 0. Check the effective modulus or viscosity shown to the user when all shared settings are omitted on growth. Inspect the complete parameter listing as well as the guide. If an emitted parameter file cannot be reused, distinguish the documented purpose of that file and behavior on the baseline before attributing a regression to A.

### 2. Would an ASPECT maintainer accept the parameter resolver?

Trace commit `402cc4a71` from the common material declaration into the two active material parsers. Every material declaration also constructs temporary fault declarations, prints and reads their JSON schema, infers numeric types from pattern-description text, and installs callbacks tracked per parameter handler. Explain what a fault-free run pays or observes. Inspect the deal.II API available in the supported version and the nearest ASPECT precedent. Identify which part needs to observe explicit assignments and whether declaration or alias handling could live closer to the fault and friction owners while preserving independently compiled law plugins and route-specific defaults. Give one concrete lower-impact design or explain the dependency that makes the current one appropriate. Do not propose a replacement framework.

Try an external law with a nested parameter and a pair of equivalent typed values, for example a list of doubles entered as `1, 2` and `1.0, 2.0`. Check the promise in the guide that shared values agree “after parsing”. A passing scalar dashpot alone does not cover that path. Also identify the production callers of the public `FaultParameters::explicitly_set` function. The only caller I found was the legacy alias in the growing-fault benchmark plugin. Decide whether that installed API earns its place. A declaration-count or lifetime issue needs a concrete consequence to qualify as a defect.

```text
source/material_model/interface.cc
source/material_model/fault_parameters.cc
include/aspect/material_model/fault_parameters.h
benchmarks/growing_fault/growing_fault_checks.cc
```

### 3. Can an apparently successful overwrite still mix two frames?

Inspect the separate exporter commit `60115004e`. In a scratch output directory for an ordinary one-rank growing run, pre-place **only** a valid `points-0001-r2.csv` from an older layout for the 5 ms frame. Leave the rank-0 and rank-1 files and section table absent at that step. A copy of a genuine old point shard with the rank-2 filename is sufficient to exercise discovery. Run the real exporter. Read all matching shards with the shipped reader and compare against a clean serial frame. Report the exit status, surviving filenames, and any duplicate record. I tested the exact discovery expression in isolation: rank 1 is detected and a lone rank 2 is missed. Confirm the effect through the production writer. The existing test covers a contiguous two-to-one overwrite.

Inspect cleanup ordering too. Root can reach deletion before the collective write-error reduction. In a scratch overwrite with an injected non-root write failure, determine what a reader sees after failure and after a successful retry. Verify that unrelated frames survive. The working record already discloses that a failed write can leave an incomplete frame. Prioritize successful publication of stale data or deletion of previous valid data over requests for a new transactional output format.

```text
source/postprocess/growing_fault_history.cc
benchmarks/growing_fault/check_history_output.py
benchmarks/growing_fault/compare.py
```

### 4. Check the remaining contracts and the evidence used to certify them

- Trace the configured phase index through initialization, stress-slot validation, mechanics and energy reporting. Check renamed static phase with the reporter off and on, and confirm that a transported phase still uses its bounded basis. Inspect the distinction between explicit geometry and legacy automatic inference. The guide explains the new name near line 30 but still describes a literal `phase_field` and older role restrictions around lines 135–136.
- Check that an explicit initial trace rate and the RSF reference speed have independent physical effects. Verify new mature interior checkpoints reject a change to a physical coefficient of the *same law* before installing history. Check unchanged continuation, the stateless law and a real pre-A archive without a signature. Distinguish verification possible on each archive version. The recap records successful controls; focus reruns on what inspection leaves uncertain.
- Invoke the real canonical/legacy comparison on the unchanged capture, the reversed strip-8 rate and zero changed to 1e−25 m/s. The last two should fail at the intended physical record. Check what comes from archived sections, what comes from installed simulation output, and what is mocked. Preserve the distinction between this equivalence test and the independently solved friction root.
- Reindent a scratch copy of the public growing input without changing any settings. Can the material-settings checker still construct its canonical/legacy tests? Its subsection regex around line 23 assumes one exact indentation level. Also check whether the G test can pass with a materially wrong response simply because the two stresses differ by more than 1 Pa. Recommend evidence proportionate to the contract the test claims. Leave the wider example rewrite for Stage D.

```text
source/material_model/phase_field_fault.cc
benchmarks/fixed_fault/check_mature_settings.py
benchmarks/growing_fault/check_material_settings.py
benchmarks/growing_fault/check_interface.py
refactoring/phase-field-fault/IMPLEMENTATION.md
```

### Return to the author

Lead with **accept** or **request changes** and the finding that decides it. For each actionable issue give severity, commit/file/line, the smallest triggering input, observed versus expected behavior, and the physical or user-visible consequence. Distinguish a reproduced failure, a source-level diagnosis and a maintainability recommendation. List the native checks actually rerun and identify anything that prevented reproduction. Use copyable worktree-relative paths with line numbers outside path blocks. End with the issues that must be fixed before Assignment B. Leave the branch and its scientific references intact.
