# Addressing the C2 maintainer review in Codex

Run the assignments from the root of the ASPECT `pf-rsf` worktree. The companion document is `PF_RSF_C2_MAINTAINER_REVIEW.md`. The finding numbers below refer to that review. Save both documents where the implementation agent can read them. The prompts below assume this location.

```text
refactoring/phase-field-fault/PF_RSF_C2_MAINTAINER_REVIEW.md
refactoring/phase-field-fault/C2_REVIEW_REPAIR_TASKS.md
```

Give Codex one assignment at a time. Start with A. Use its implementation prompt in an ordinary coding conversation. Each assignment is self-contained with the shared instructions and the review. Between assignments, the short review prompt at the end gives a second reader a specific question and an exact range of changes.

## Shared instructions

Read the worktree’s AGENTS.md and the numbered findings assigned below. The reviewed commit is 14358d2a56c2542aace26e91adafbc69e1ee71e2. Record the actual starting commit and inspect local modifications. Work forward from the current state. If a finding has already been fixed, exercise its reproducer on the current code and report the evidence.

Implement and verify the named assignment through the ordinary installed ASPECT entry points where applicable. Keep edits with the owner of the physical quantity or numerical operation. Explain a change to shared ASPECT code by naming the production caller that needs it. Use the existing guide, working record and scoped tests. Preserve the reference data, thresholds and existing changes in the worktree. Keep implementation commits local and reviewable. Use at most four total local workers. Put the start and end commit, commands, and relevant measurements in the handoff. List paths relative to the worktree root in fenced text, one path per line, with line numbers outside the block.

## Assignment A — Finish the public material interface and its checks

**Implementation prompt**

```text
Implement Assignment A in refactoring/phase-field-fault/C2_REVIEW_REPAIR_TASKS.md. Apply its shared instructions. Use findings 1, 4, 5, 6 and 7 of the companion maintainer review. Complete the implementation and run the acceptance checks. Make the output repair a separate commit. Finish with a recap for the author and the commit range for independent review.
```

1. **Canonical material settings and historical inputs (finding 1).** Start with the installed growing seed. Demonstrate which modulus reaches the Maxwell response when the public Phase field fault subsection says 2e6 Pa and the inherited Growing fault subsection says 1e6 Pa. Inspect both material parsers and the resistance factory. Give shared physics one public address for both routes; preserve each route’s historical defaults when entries are omitted. Honor old inputs that specify only the legacy entry. Compare parsed values for equal duplicates, including explicit values equal to a default. Reject conflicting explicit values with both paths in the diagnostic. For a genuinely route-specific setting, either consume an explicitly supplied value on the appropriate route or diagnose its incompatibility. Keep ASPECT’s adiabatic pressure path and the dedicated friction law reader. The two existing numerical routes can retain distinct state. Limit core changes to what routing actually needs.
   
   Use clean canonical-only and legacy-only variants as the equivalence controls. In a canonical-only input, change G by itself and measure an accepted stress component in a case sensitive to G. Check that the equivalent legacy value gives the same response. Include a conflicting pair and an equal pair. Exercise the public resistance-law selector with the test dashpot on both routes and check the force and absence of age. Inspect any old test that inherits irrelevant settings from another route. Keep exercising the behavior the old test was built to check when making its input unambiguous.

2. **Phase-field identity (finding 4).** Have mature mechanics, growing mechanics and the energy observer obtain the selected phase composition consistently. Retain the historical default name. Distinguish the selected phase from the three mature stress slots. Rename the field to damage in a valid mature input and compare the physical result and energy with the original. Exercise the observer both enabled and disabled, rejection of a mature stress slot as phase, and a fault-free energy calculation. Treat automatic geometry inference and FE basis selection as compatibility behavior. A renamed field needs no new discretization.

3. **Law-owned history and initial slip rate (finding 5).** On newly written mature interior checkpoints, record and verify the active law’s identity and physical parameter signature. Use the existing law interface. Demonstrate unchanged continuation and detection of a changed parameter of the same law. Read a previously supported archive lacking the new record using its documented compatibility behavior. Retain the qualified checkpoint/repartition cases. Give initial trace slip rate a clear owner independent of the RSF constitutive reference speed. Preserve the numerical initialization of existing RSF and dashpot inputs through an explicit compatibility rule. Once the initial rate is explicitly supplied, show that an inactive RSF parameter cannot prevent a valid stateless run. For RSF, demonstrate that the constitutive reference speed and the initial rate can be set independently.

4. **A comparison that sees a reversal (finding 7).** Exercise the actual canonical-versus-legacy checker on the archived strip-8 rate and a copy with only its sign reversed. Preserve the evidence that this comparison accepted the mutation before repair. Fix the checker so the sign reversal fails, equivalent inputs pass, and zero rate has a meaningful treatment. Check discrete identities and formation flags exactly. Give physical columns appropriate scales. Use the separate signed-root check for the independent constitutive reference. A passing replay alone cannot demonstrate that the interface comparator detects the mutation.

5. **Reliable growing-history output (finding 6, separate commit).** Check file creation and final writes, including stream close. Surface a useful collective error identifying the failed file and rank. If an earlier two-rank frame is overwritten by a one-rank frame in the same output directory, ordinary readers must see exactly one copy of each physical record. Preserve current filenames and existing reader workflows, or adapt the shipped readers together with any compatible per-frame index you introduce. Protect unrelated frames/files. Test the genuine exporter with a blocked output filename in an otherwise writable directory, and overwrite a two-rank frame with one rank at the same step. Read the result with the shipped reader and compare all native IDs and quantities to a clean one-rank frame. Moving peak-memory or diagnostic dumps behind an option can wait.

**Closeout.** Use the installed material examples and relevant existing controls, including old spellings, the external law, restart and the fault-free phase case. Report a small table showing input edit, physical change or expected diagnostic, and observed result. Include the negative-control outcomes and compatibility results. Update the active guide for any new canonical syntax. The strip-8 angle comparison and the four historical strip-15 differences have their own recorded status. Assignment A ends at the interface and output fixes.

## Assignment B — Repair mature boundary-spanning fault work before BP3

**Implementation prompt**

```text
Implement Assignment B in refactoring/phase-field-fault/C2_REVIEW_REPAIR_TASKS.md. Apply its shared instructions. Address finding 2 of the companion review. Use $verify-fe-operator for the paired FE actions. Capture the counterexample before editing production code, make the smallest repair supported by that evidence, then qualify it on the mature controls. Finish with the measured work on each side and the commit range.
```

Start with the one-cell vertical trace, uniform admissible positive phase, fitting off, full-width normalization, three labels at y = 0, 1/2, 1, a positive base rate, and a perturbation to the middle label only. Use a constant virtual shear. State the sign, units, stiffness, and tensor factors removed in the normalized comparison. Measure the ordinary local Stokes RHS before boundary elimination and the reverse section action for the same virtual velocity. Isolate the coupling contribution from background stress and other RHS terms. A test-only injection at the real assembly boundary is sufficient if the ordinary physical data are hard to prescribe.

The independent split integral is 0.5. The pre-repair Q3 and Q8 rules predict 0.569668517088 and 0.494235968293. Establish whether the actual operators reproduce that mismatch. If they give a different result, locate the first difference in the call path and revise the diagnosis from the evidence. A test that calls one helper twice cannot establish transpose agreement.

Use a common discrete transfer or a consistent projection with the fault contribution. Preserve the ordinary ASPECT Stokes quadrature and fault-free path. Report two separate measurements: discrete work agreement on the same mesh, and error of each action relative to independently integrated work on a resolved, inclined ridge under refinement. Share the reason for the chosen production location. Preserve the physical-stress, rate-tangent, retry, one-/two-rank and existing mature controls that the changed path affects. Record any scientifically explainable shift in a frozen output for the author to decide. Keep its original reference available. For this experiment, include a small reproducible figure showing work error versus resolution, with units or an explicit normalization. This assignment ends with the mature operator.

## Assignment C — Determine what should happen to strip 8

**Investigation prompt**

```text
Carry out Assignment C in refactoring/phase-field-fault/C2_REVIEW_REPAIR_TASKS.md. Apply its shared instructions. Investigate finding 3 of the companion review using the archived data and its executable appendix. Deliver the measured comparison, a reproducible plot, and a recommendation. Keep prospective production changes on an isolated local candidate until the comparison below resolves the choice.
```

Reproduce the affine 5 ms reference peak on each archived field. Then examine the *first divergent decision* in the frozen constant-pressure control. Recover the necessary snapshots with the recorded small run if the relevant inputs were not archived. On the same accepted parent and solved field for each comparison, evaluate the production selector, the isolated candidate and an independent high-precision sum over the native points with a signed root at each angle. Record which peak bracket wins, the reference peak within it, both selected angles, the angle/score errors, and the resulting accepted arclength. The affine one-/two-rank production difference was about 8.57e−7 rad and the respective direct peaks differed by 3.67e−12 rad. Treat these as reproducibility checks for that capture.

Use a zoomed score-difference plot around the peak with both selected angles and the direct peak marked. Label the angular and score units. Distinguish an error in evaluating the score from a change in its physical inputs to the extent the measurements support. Compare against the strict affine check, independent off-peak control and frozen nine-strip case. Track the historical four strip-15 failures separately. If a candidate meets the present criteria, report its commit and the evidence. If the candidate better approximates the physical score but changes the frozen trajectory outside tolerance, report which reference would move, by how much, and what scientific choice the author must make. Keep the original thresholds and evidence available. CZM growth and Stage D are later assignments.

## Review prompt between assignments

Open a separate review conversation in the same worktree. Fill in the assignment letter and the actual two commit IDs from the recap. This ordinary message also works without a slash command.

```text
Review the changes from <start SHA> through <end SHA> for Assignment <letter> in refactoring/phase-field-fault/C2_REVIEW_REPAIR_TASKS.md. Use the associated numbered findings in the companion review as the acceptance criteria. Read the diff and follow each changed setting or numerical term to its production consumer. Independently exercise the most discriminating negative control or counterexample. Check what an existing parameter file, supported checkpoint, or unaffected ASPECT caller would experience. Report actionable findings by severity with copyable paths, line numbers, and the smallest failing case. State exactly which checks you reran. This is a review-only task.
```

For A, ask specifically whether any accepted physical input can still be ignored and whether the sign-flip mutation fails in the interface comparator itself. For B, ask whether both measured actions really came from opposite production transfers and whether their common answer converges to the independent integral. For C, ask whether the data justify changing the historical reference and what evidence could overturn that interpretation.
