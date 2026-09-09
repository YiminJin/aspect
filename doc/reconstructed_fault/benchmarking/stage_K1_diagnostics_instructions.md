# K1: initialization diagnostics and visualization

This replaces the previous detailed task brief. Continue from the current working tree. Read the K1 pilot/prerequisite reports, benchmark README, and authoritative `doc/reconstructed_fault/current_design.md` and `specification.tex`; the pilot report governs current execution status.

## Goal and scope

Investigate the initial phase-field discrepancy and provide visualization for my review before any production correction. Initial particle H is independently verified and x-uniform, but the converged Q1 phi already varies along x before mechanics (range approximately 0.0597; peak approximately 0.402 versus initializer input 0.6), despite meeting the nonlinear residual tolerance. No specific defect is established.

Choose the diagnostic approach from the code and evidence. Determine where unintended x-dependence first appears, checking whether the actual discrete problem should preserve that symmetry. Keep the peak-value discrepancy separate, and verify what the initializer's core input guarantees before treating it as an enforced FE value.

## Visualization first

Produce actual ParaView-readable files, reusing existing exporters where possible:

- Bulk: pre-mechanics Q1 phi and native mesh; gradients if readily available.
- Initial particles: positions, IDs, H, and domain volumes.
- Reconstructed fault: actual geometry, I_h, cohesive traction, and available normals; supply the nominal y=0 line separately.

Include a few CSV profiles and preview plots showing phi(x,0), phi along the actual fault, transverse phi(y) at interior and near-periodic-edge locations, and reconstructed y(x). Compare initial H and transverse phi with the existing independent/intended references, labeling references explicitly. Preserve unsmoothed data and show mesh/particle locations.

Include accepted t=0 and t=2 mechanics (V, Theta, bulk velocity and pressure) only where recoverable from existing artifacts. Clearly distinguish initialization stages, diagnostic states, and accepted states; report missing outputs rather than fabricating them.

## Investigation boundaries

Reuse existing artifacts. If necessary, run a bounded, isolated initialization diagnostic with the original fixture and real production phase solve, collecting outputs before mechanics. Do not rerun the full pilot, convergence matrix, or full test suite for this deliverable. Choose only the focused checks needed to distinguish the remaining explanations; avoid a speculative codebase-wide audit.

Preserve approved initialization/history, physical-boundary and rollback rules, and unrelated changes. Leave production algorithms and fixture parameters, constraints, normalization, tolerances and acceptance gates unchanged. Do not impose symmetry or substitute an ideal phase field to force agreement. Add only minimal diagnostic instrumentation, preferably benchmark-local, without changing the measured state or creating a new public interface.

## Deliver and stop for review

Save files persistently, not only in /tmp. Return absolute paths, a brief README with reproduction/opening instructions, and a short evidence-based report: observed spatial pattern, earliest localized inconsistency or remaining alternatives, confirmed findings versus hypotheses, and the smallest next test or proposed correction with source locations and a regression-test proposal.

Deliver the visualization even if the cause remains unresolved. Stop before implementing a production correction or advancing K1/K2. Other K1 gates and the runtime limit remain separate unresolved issues, not presumed consequences of the phase discrepancy.
