Complete the 5B evidence and figures on the local machine. Use the existing runs, reference and observers recorded at the top of STATUS.md. If a needed physical field is missing, rerun its indexed two-rank Release case with one numerical thread per rank. Use fresh run outputs. Preserve the frozen preparation, two schedules, friction law, reference data and original pass/fail thresholds. The 5B audit currently exits nonzero for the original strip-angle criterion. Retain that result in the handoff.

## Resolve what the angle comparison means

Start with the saved moments and strip histories. Verify that the observed selector call produced the accepted angle, or identify any subsequent update. On each captured input, replay the signed full-strip resistance, candidate arclength and centered onset score. Bracket the resistance root and check its residual. Scan the admitted angular sector, identify a competing peak or boundary if present, then refine the winning maximum at high precision. Quantify the selected angle’s error relative to its own maximum, the donor-to-target movement of the maximum and the score lost at the selected angle. Compare both production selections on identical inputs if attribution to the search is required. The binary64 score evaluation and resistance solves can both affect which near-peak sample wins.

As reproducibility pointers, the existing captured inputs give local peaks near −0.1478615881 rad for the donor and −0.1478615924 rad for the target. Compute these independently. Include one clearly off-peak control that the score comparison rejects. Show formed flags, arclength, signed rates, slip and resistance against the unchanged physical budgets. A demonstrated operator error warrants a focused repair and rerun. If the difference is confined to the numerical resolution of the same peak, recommend a documented disposition for this fixture. Preserve the failed strict angle comparison alongside that recommendation. A future general equivalence rule needs a scale and controls that also reject a wrong peak or materially wrong geometry.

## Show where the phase field changed

Produce these two figures from physical output. Keep the plotting script beside the 5B benchmark or results and list a command that regenerates each figure from recorded inputs.

- __Spatial phase increments__. Make three maps of the 2 × 1 m domain: one-step accepted phase at 10 ms minus the common 5 ms phase, two-step accepted phase at 10 ms minus the same parent, and the signed difference between the two 10 ms fields (two steps minus one). Use cell-local DG Bernstein evaluation or native physical samples that respect cell boundaries. Use the same axes, aspect ratio and color limits on the first two panels and a zero-centered scale on the third. Overlay the same 5 ms band contour with its value and the actual material-strip locations where legible. Annotate the number of new registrations. Reproduce the endpoint distance 0.00742665 m with physical-area quadrature, independently of display resolution. If plotting an additional 7.5 ms panel, label its stage or accepted-endpoint status.

- __Strip-15 direction choice__. Show the direction score across the admissible sector and a magnified view near the selected angles. In the magnified view subtract each curve’s own high-precision maximum before converting to plotting precision. Mark both selected angles and the corresponding maxima. Give the angle reference, radians, and score units (Pa² m²). State how close the scores are and whether the peak identity changes.

Export a readable PNG and vector PDF of each figure. Give a short caption specifying the question, what the plotted fields show and the conclusion they support. In the experiment recap, lead with the question and the spatial figure. Include the timestep schedule, signed slip and added fracture energy, and the relation between the work defect and the cohesive compliance remainder. Compare the independently computed downhill diagnostic with donor evidence if available. Keep near-zero ambiguities explicit. Use keyed evidence or a negative control for any new pass/fail claim. Add compact figures, scripts and reproduction paths to the working record. Keep raw run trees in their current run directories.

For future numerical experiments, add one sentence to the existing reporting instructions if this expectation is absent: “Include a reproducible figure from the computed fields that answers the experiment’s main question, with units, a caption, and copyable paths to the figure and plotting command.”

Deliver the original strict result, the numerical interpretation, any review repair and its verification, and a recommendation on closing 5B. An accepted closeout unlocks Increment 6 consolidation. The later physical propagation example requires a checked initial phase balance, a stated front definition, connected advance beyond the initial diffuse seed, and spatial and temporal refinement. Reserve development of a different time integration method for a separate assignment.

## Files to start from, relative to the worktree root

refactoring/phase-field-fault/INCREMENT_5B_TASK.md
refactoring/phase-field-fault/STATUS.md
refactoring/phase-field-fault/VERIFICATION.md
refactoring/phase-field-fault/results/strip18/intervals.json
refactoring/phase-field-fault/results/strip18/frame-comparisons.csv
refactoring/phase-field-fault/results/strip18/strip15.csv
refactoring/phase-field-fault/results/strip18/moments.csv
refactoring/phase-field-fault/results/strip18/turn.py
benchmarks/growing_fault/timestep/audit.py
benchmarks/growing_fault/timestep/frames.py
source/simulator/solver/midpoint_direction.cc
include/aspect/reconstructed_fault/onset_score.h
