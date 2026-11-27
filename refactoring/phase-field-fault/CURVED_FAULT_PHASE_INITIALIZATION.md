# Initialize the public curved-fault phase from the prescribed trace

Continue in the existing implementation thread on `pf-rsf`. The reviewed starting point is `04aaf6db08abec8a06a27134f803cd265e64f7f9`. Record your actual starting commit and preserve subsequent work. Follow the worktree’s `AGENTS.md`. Implement the change and deliver a tested commit range for the existing review thread.

The current public example reads five vertices for mechanics but initializes phase from a separate piecewise function of x, with a Gaussian width measured vertically. A coordinate edit can silently separate the phase ridge from the mechanical fault. Correct the public example, keep the variable-width case as a verification fixture, and measure the separate slip-transfer limitation at a bend. These are three distinct outcomes.

## 1. Make one geometry edit sufficient

Use the ordered vertex table as the source of the prescribed curve. For the public example initialize the phase composition at a physical point x by

\[
\phi_0(x)=0.03+0.32\exp[-(d(x,\Gamma)/w)^2],\qquad w=0.15\ {\rm m},
\]

where d is the minimum Euclidean distance to the **finite segments** joining consecutive vertices. This gives the Gaussian excess above the 0.03 background an e-fold width of 0.15 m measured normal to an isolated segment. At a bend or endpoint the nearest point may be the vertex itself. Use that distance and document the endpoint convention. Keep the same peak, background, width parameter, prescribed fault, loading, mesh and time window for the comparison. The Gaussian is an input for this demonstration; its width is distinct from the mechanical normal-support radius and from a calibrated fracture length scale.

Use the ordinary ASPECT initial-composition path. The initializer owns the distance-to-phase law; geometry input can be shared without giving the initializer a dependency on a live material model, friction law, accepted history or reconstructed-fault manager. Look at the existing vertex-table parser in `source/material_model/phase_field_fault.cc` and `distance_to_line` in `source/utilities.cc`, which already computes distance to a bounded 2-D segment. Mechanics still needs its own rule for admitting a bulk point to a segment’s normal strip. An inactive mechanical projection near a corner can have a perfectly well-defined geometric distance. Keep both meanings clear.

In the shipped workflow a user edits one file of coordinates, and both consumers see that exact file at startup. Preserve supported table syntax, ID ordering, optional law-specific columns, and installed path resolution. Read or prepare the coordinates outside point-evaluation loops. Check the configured phase composition by name and preserve the other initialized fields. Keep the public command runnable from a clean installation without verification plugins or a donor checkout.

## 2. Verify what ASPECT initialized

- Read the **actual initialized DG field** at time zero and the accepted trace. On shallow and steep segment interiors, compare samples on both local normals against an independent calculation. Measure ridge location and the distance to the level φ = 0.03 + 0.32/e where the sampling ray stays in the box and the same segment remains closest. Check a bend and a contact separately against their finite-segment geometry. Use a refined repeat or an independently constructed FE reference to justify interpolation and measurement tolerances. The analytic widths of the old vertical fixture were approximately 0.147, 0.143, 0.108 and 0.094 m on its four segment interiors; the width check should distinguish that case.
- From a fresh directory change **only one interior vertex** in a copy of the installed table. Pick a displacement resolvable on the test mesh. Verify that the mechanical vertex moves, the FE ridge follows it, and the normal widths still agree with the prescribed profile. A stale phase input must fail this comparison even if the mechanical input and solver succeed. Check the same physical samples on one and two ranks. Use a straight inclined or vertical segment as a control for dependence on coordinate axes.
- Retain the original vertical Gaussian under an explicitly named verification input. Preserve its independent normal-integration, signed-work and restart evidence. Trace callers that inherit `curved_mature.prm`. In particular, `check_local_work.py` currently uses a three-label mechanical probe and reads a field from a five-label run. Give that test an explicit manufactured field or observe the field produced by the probe itself; compare like physical inputs. Keep the existing regression tolerances. Regenerate the new public velocities and normal integrals from a fresh solve, and check continuation of the new public case from its own checkpoint.

## 3. Quantify the bend limitation independently of the initializer

The current algorithm integrates compliance along each complete in-box normal ray to obtain the normalization, then assigns each bulk point to at most one eligible segment. Adjacent strips can overlap on the inside of a turn or leave a gap outside. A forward/reverse work identity can pass even when the integrated magnitude of returned slip differs from the prescribed one.

At fixed accepted phase and geometry, use a constant prescribed test speed and the actual scalar transfer weight χ. One useful diagnostic per station i is to compare the bulk integral of χ times its interpolated Q1 station shape with the line integral of that shape along the trace. Both have units of metres in 2-D; multiplying by the test speed gives the corresponding integrated slip-rate contributions. Use the production admission and weights for the measured action and an independent geometric integral for the reference. Define any equivalent diagnostic equally explicitly. Compare a **straight control** with the bent public case, report per-station and total defects, and distinguish contacts from the central bend. Resolve quadrature or mesh sensitivity enough to separate it from the observed effect. Counts of multiply covered or uncovered points alone do not measure a slip defect. If the discrepancy persists, report its size, location and consequence for the regional application. A change to the mechanical normalization requires a separately derived and reviewed assignment.

## Deliver

Replot `curve.png` from the accepted FE output, with equal map scales, the prescribed trace and visible normal cuts or an e-fold contour. Provide a compact width-vs-distance or normal-profile comparison with units and show the result of the one-vertex edit. Distinguish the finite-separation bulk velocity difference from the RSF label speed. Update the installed guide, figure caption and working record. Give the exact fresh-run and plot commands, measured widths and bend defects, affected regression results, and copyable paths. End with separate conclusions for the initializer and for the physical transfer around a bend.
