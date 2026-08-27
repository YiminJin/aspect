# Stage 4 — Reconstruct the initial sharp fault from the solved Q1 phase field

## Purpose

Stage 3 initialized the particle-based crack-driving force \(H\) from prescribed fault geometry and prescribed core phase-field values. The existing phase-field solver then produces the actual discrete \(Q_1\) phase-field solution.

Stage 4 constructs the initial sharp reconstructed fault from that **solved \(Q_1\) phase field**, not from the prescribed stationary profile.

The intended initialization chain is therefore

\[
\text{prescribed fault geometry and core values}
\rightarrow
H \text{ on particles}
\rightarrow
\text{solve } \phi_h \in Q_1
\rightarrow
\text{reconstruct sharp fault } \Gamma_h .
\]

The prescribed polyline supplies topology, ordering, approximate length, and a reference position. The final sharp-fault vertices are centered using the actual discrete phase-field solution \(\phi_h\), so discretization/interpolation errors in the \(Q_1\) field are reflected consistently in the reconstructed geometry.

This stage implements only the **initial reconstruction**. It does not propagate the fault.

---

# 1. Read and inspect first

Before editing:

1. Read `doc/reconstructed_fault/current_design.md`.
2. Read the Stage-3 instructions and inspect the completed Stage-3 implementation.
3. Inspect the existing Stage-1 `ReconstructedFault<dim>` geometry container.
4. Inspect the Stage-2 distributed phase-field point-evaluation implementation.
5. Inspect how and when the initial phase-field solve is completed and where the solved \(Q_1\) phase-field vector is valid.
6. Reuse the prescribed-fault representation introduced in Stage 3. Do not introduce a second representation of the same initial fault geometry or core phase-field data.

The current source tree is authoritative for actual class/function names.

If an instruction below conflicts with the implemented Stage-1/2/3 interfaces in a scientifically meaningful way, report the conflict before redesigning the algorithm.

---

# 2. Architectural requirement

`ReconstructedFault<dim>` remains a geometry/topology container.

Do **not** put phase-field-specific reconstruction logic into `ReconstructedFault<dim>` merely for convenience.

The reconstruction code may live in `PhaseFieldHandler<dim>` or in a small phase-field reconstruction component if that fits the current source tree better. Codex may choose the concrete organization after inspection.

The important dependency direction is

\[
\text{phase-field reconstruction code}
\longrightarrow
\text{ReconstructedFault geometry},
\]

not

\[
\text{ReconstructedFault geometry}
\longrightarrow
\text{phase-field solver}.
\]

No RSF-specific data or material-model-specific fault state is introduced in this stage.

---

# 3. Input and output

For each prescribed initial fault, Stage 3 already provides an ordered 2-D polyline

\[
\mathcal G
=
\left\{
\mathbf G_0,\mathbf G_1,\ldots,\mathbf G_M
\right\}.
\]

Stage 4 additionally uses the solved discrete phase field

\[
\phi_h(\mathbf x).
\]

For each prescribed fault, produce one ordered reconstructed sharp-fault polyline

\[
\Gamma_h
=
\left\{
\mathbf X_0,\mathbf X_1,\ldots,\mathbf X_{N_\Gamma-1}
\right\}.
\]

Use the existing Stage-1 `ReconstructedFault<dim>` interface to store the result.

If the current Stage-1 design represents one connected fault per `ReconstructedFault` object, preserve that design and reconstruct each prescribed fault independently. Do not redesign multi-fault ownership in this task unless the existing code requires it.

---

# 4. Structural point spacing and arc-length resampling

The sharp fault requires a structural discretization that is independent of the spacing of the user-supplied control/polyline points.

Let the cumulative arc length of the prescribed polyline be

\[
s_0=0,
\]

\[
s_j
=
s_{j-1}
+
\left|
\mathbf G_j-\mathbf G_{j-1}
\right|,
\qquad
j=1,\ldots,M,
\]

and let

\[
L=s_M.
\]

Given a target structural spacing \(h_\Gamma>0\), define

\[
N_{\mathrm{seg}}
=
\max
\left(
1,
\left\lceil
\frac{L}{h_\Gamma}
\right\rceil
\right),
\]

\[
\Delta s_\Gamma
=
\frac{L}{N_{\mathrm{seg}}},
\]

\[
N_\Gamma=N_{\mathrm{seg}}+1.
\]

Construct reference structural points at

\[
s_i=i\Delta s_\Gamma,
\qquad
i=0,\ldots,N_{\mathrm{seg}},
\]

using piecewise-linear interpolation along the prescribed polyline:

\[
\boxed{
\mathbf R_i=\mathcal G(s_i)
}.
\]

The point spacing of the reconstructed fault must therefore not depend on how densely the user happened to specify the input polyline.

### Source of \(h_\Gamma\)

First inspect whether the completed Stage-1/3 implementation already has an appropriate structural-spacing parameter.

- If such a parameter already exists, reuse it.
- If it does not exist, propose the smallest clean way to provide \(h_\Gamma\) before implementing it.
- Do not infer \(N_\Gamma\) from the number of active phase-field nodes.
- Do not silently use the original polyline vertex spacing as \(h_\Gamma\).

This is a genuine sharp-fault discretization parameter and is not the same thing as the phase-field length scale.

---

# 5. Reference tangents and normals

For the resampled reference points, define the tangent using centered secants at interior points:

\[
\widetilde{\mathbf t}_i
=
\mathbf R_{i+1}-\mathbf R_{i-1},
\]

\[
\boxed{
\mathbf t_i
=
\frac{\widetilde{\mathbf t}_i}
{\left|\widetilde{\mathbf t}_i\right|}
}.
\]

At the first endpoint use

\[
\mathbf t_0
=
\frac{\mathbf R_1-\mathbf R_0}
{\left|\mathbf R_1-\mathbf R_0\right|},
\]

and at the last endpoint use

\[
\mathbf t_{N_\Gamma-1}
=
\frac{
\mathbf R_{N_\Gamma-1}-\mathbf R_{N_\Gamma-2}
}{
\left|
\mathbf R_{N_\Gamma-1}-\mathbf R_{N_\Gamma-2}
\right|
}.
\]

For the current 2-D implementation, define

\[
\boxed{
\mathbf n_i
=
(-t_{i,y},t_{i,x})
}.
\]

Normal orientation is arbitrary for the centering calculation, provided it is internally consistent.

Reject degenerate reference geometry that cannot define a nonzero tangent.

Higher-order/spline interpolation of the prescribed fault is outside this stage. Use the current piecewise-linear representation.

---

# 6. Normal phase-field profiles

At each reference point, sample the actual solved \(Q_1\) phase field along

\[
\boxed{
\mathbf x_i(\eta)
=
\mathbf R_i+\eta\mathbf n_i
}.
\]

The sampling interval must be wide enough to contain the active diffuse profile on both sides of the reference fault.

## Preferred sampling half-width

Do not hard-code a universal value such as \(3\ell\) if the existing stationary-profile infrastructure already provides the support of the relevant profile.

Prefer to use the existing `PhaseField::PhaseFieldProfile` information from Stage 3 to estimate a suitable half-width. For a profile with prescribed core phase-field value \(\hat\phi\), its support is available from the final coordinate in its stored profile:

\[
\zeta_{\mathrm{support}}
=
\max \{\zeta:\phi^\star(\zeta)>0\}.
\]

Use a sampling half-width that contains this support plus a small geometric/discretization margin. The exact clean implementation should reuse the Stage-3 profile infrastructure rather than reconstructing the stationary-profile formula.

If several material degradation profiles are relevant at one location, use a half-width large enough to contain all relevant profiles.

The stationary profile is used here **only to determine a safe sampling window**. It must not be used to determine the sharp-fault center.

The center must come from the solved \(Q_1\) field.

## Sampling resolution

Choose a uniform transverse spacing fine enough to resolve the \(Q_1\) profile. As an initial numerical target,

\[
\Delta\eta
\lesssim
\frac{\ell}{8}.
\]

Reuse the existing phase-field length scale \(\ell\); do not create a duplicate length-scale parameter.

Choose an even number of transverse intervals so that

\[
\eta=0
\]

is included exactly.

Generate all sample positions for a fault, or a reasonably large batch of them, before distributed evaluation. Do not perform a new distributed point-location operation separately for every structural point.

---

# 7. Reconsider and simplify the Stage-2 point-evaluation abstraction

Stage 2 introduced

```cpp
PhaseFieldHandler::phase_field_values_at_points(...)
```

which calls

```cpp
PhaseFieldUtilities::distributed_scalar_values_at_points(...)
```

and the latter is only a small wrapper around the actual deal.II distributed point-evaluation machinery.

Stage 4 is now the first real reconstruction caller, so reconsider this abstraction now.

### Required policy

Inspect all current callers.

If both Stage-2 functions exist only to support reconstructed-fault sampling and have no meaningful independent use:

1. remove the redundant public/helper layering;
2. place the small distributed \(Q_1\) point-evaluation code directly in the Stage-4 reconstruction implementation where the batch values are needed;
3. continue to use the tested deal.II/ASPECT distributed point-evaluation infrastructure internally;
4. update/remove Stage-2 tests so the same behavior is covered through Stage-4 reconstruction tests.

Do not keep a general utility function solely because Stage 2 previously introduced it.

If either function now has a genuine independent caller or provides a meaningful reusable abstraction after inspection, report that fact and keep the smallest justified interface rather than deleting it blindly.

Do not implement a custom MPI cell search.

---

# 8. Weight used to locate the diffuse-fault center

Obtain the lower acceptable phase-field bound

\[
\phi_{\min}
\]

from the existing

```cpp
MaterialModel::PhaseFieldModel<dim>::get_phase_field_range()
```

interface.

Do not introduce a second reconstructed-fault active threshold.

For a sampled phase-field value \(\phi\), define the geometric reconstruction weight

\[
\boxed{
w(\phi)
=
\max(\phi-\phi_{\min},0)
}.
\]

This choice intentionally uses the phase field itself rather than material-model-specific quantities such as slip rate, degradation, or RSF variables.

It must work for a partially mature pre-existing fault whose peak phase field is substantially below one.

Do not require a near-unity phase-field ridge.

Do not add an exponent or other weighting parameter in this stage unless testing demonstrates a need for one.

---

# 9. Transverse moment reconstruction

For reference structural point \(i\), let

\[
\phi_{ik}
=
\phi_h
\left(
\mathbf R_i+\eta_k\mathbf n_i
\right),
\]

and

\[
w_{ik}
=
\max(\phi_{ik}-\phi_{\min},0).
\]

Use composite trapezoidal integration to calculate

\[
M_{0,i}
=
\int w_i(\eta)\,d\eta,
\]

\[
M_{1,i}
=
\int \eta w_i(\eta)\,d\eta,
\]

\[
M_{2,i}
=
\int \eta^2w_i(\eta)\,d\eta.
\]

For a uniform transverse grid,

\[
M_{m,i}
\approx
\Delta\eta
\left[
\frac12\eta_0^m w_{i0}
+
\sum_{k=1}^{N_\eta-1}
\eta_k^m w_{ik}
+
\frac12\eta_{N_\eta}^m w_{iN_\eta}
\right],
\]

for

\[
m=0,1,2.
\]

Define the normal offset by

\[
\boxed{
\delta_i
=
\frac{M_{1,i}}{M_{0,i}}
}.
\]

Also compute the transverse-width diagnostic

\[
\boxed{
\sigma_i^2
=
\frac{M_{2,i}}{M_{0,i}}
-
\delta_i^2
}
\]

and the sampled peak

\[
\boxed{
\phi_{\mathrm{peak},i}
=
\max_k\phi_{ik}
}.
\]

Clamp only very small negative values of \(\sigma_i^2\) caused by floating-point roundoff before taking a square root. A materially negative value indicates an implementation error.

---

# 10. Initial reconstructed sharp-fault position

The reconstructed point is

\[
\boxed{
\mathbf X_i
=
\mathbf R_i+\delta_i\mathbf n_i
}.
\]

Only a **normal correction** relative to the prescribed reference geometry is performed.

Do not move initial structural points tangentially.

Do not extend or shorten the prescribed initial fault.

Do not perform tip propagation in this stage.

After reconstruction, the geometry of `ReconstructedFault` is

\[
\Gamma_h
=
\{
\mathbf X_0,\ldots,\mathbf X_{N_\Gamma-1}
\}.
\]

Future geometric operations should use the reconstructed points, not the prescribed stationary profile, as the sharp fault.

---

# 11. Minimal validity checks

For every structural point, at minimum require

\[
M_{0,i}>0
\]

and

\[
\phi_{\mathrm{peak},i}>\phi_{\min}.
\]

Also detect and reject:

- invalid/degenerate reference tangents;
- failed distributed point evaluation;
- non-finite phase-field values;
- non-finite reconstructed offsets;
- sampling windows that clearly truncate an active profile.

A useful truncation check is that the active phase-field signal should have decayed below the accepted lower bound near both transverse ends. If the sampled values at the ends remain above \(\phi_{\min}\), report that the transverse sampling range is insufficient rather than silently computing a biased centroid.

Do not introduce additional empirical acceptance thresholds yet.

Include the fault index and structural-point index in error messages where useful.

---

# 12. Diagnostics

For each reconstructed structural point, make the following quantities available during reconstruction/testing:

\[
\delta_i,
\qquad
M_{0,i},
\qquad
\sigma_i,
\qquad
\phi_{\mathrm{peak},i}.
\]

These are reconstruction diagnostics, not persistent material-model state.

Do not add them to a generic fault-property system in Stage 4.

A temporary result/diagnostic structure is acceptable if it makes tests and debugging cleaner.

---

# 13. Multiple prescribed faults

Reconstruct each Stage-3 prescribed fault independently using its own reference polyline.

Stage 3 already requires non-overlapping prescribed-fault initialization regions. Do not add branching, merging, or topology discovery here.

If the solved \(Q_1\) phase field makes the transverse reconstruction of two prescribed faults ambiguous despite the Stage-3 non-overlap assumption, report the ambiguity rather than silently merging them.

---

# 14. Timing in the initialization sequence

The reconstruction must run only **after the initial Q1 phase-field solve has converged** and the final initial phase-field solution is available.

Do not reconstruct from:

- the prescribed stationary profile;
- the particle \(H\) field;
- an intermediate nonlinear phase-field iterate.

The source of truth for the initial sharp geometry is the converged discrete Q1 field.

Do not yet reconstruct or move the sharp fault inside later nonlinear Stokes iterations.

---

# 15. Keep Stage 4 narrow

Do **not** implement in this stage:

- smoothing/regularization of the reconstructed offsets;
- PCA;
- fault propagation;
- crack-tip search;
- generic fault-property storage;
- particle-to-fault projection;
- fault-to-QP interpolation;
- RSF state or return mapping;
- 3-D reconstruction;
- spline/B-spline interpolation;
- branching or merging.

First inspect the raw reconstructed offsets

\[
\{\delta_i\}
\]

and the resulting fault shape.

If mesh-scale oscillations are visible, smoothing will be designed as a separate follow-up stage using actual numerical evidence.

---

# 16. Required tests

## A. Q1-consistent straight fault

Construct a straight prescribed reference fault and a solved/test Q1 phase-field field symmetric about the same line.

Verify

\[
\delta_i\approx0.
\]

This checks that the reconstruction does not move an already consistent fault.

## B. Known normal displacement

Use a Q1 phase-field profile whose center is displaced normally by a known amount

\[
\Delta
\]

relative to the prescribed reference line.

Verify

\[
\delta_i\approx\Delta
\]

within the expected FE/sampling tolerance.

## C. Partially mature fault

Use a profile with peak phase field substantially below one but above the accepted lower bound.

Verify that the reconstructed center remains correct.

This test is required because the reconstruction must not rely on a mature \(\phi\approx1\) ridge.

## D. Nonuniform input-polyline spacing

Provide a reference polyline with nonuniform control-point spacing.

Verify that the Stage-4 structural reference points are approximately uniformly spaced in arc length according to the requested \(h_\Gamma\).

## E. Curved piecewise-linear reference fault

If practical in the current test infrastructure, use a gently curved reference polyline and verify that normal centering behaves correctly.

This test does not require spline interpolation.

## F. MPI/distributed point evaluation

Preserve the Stage-2 guarantee that requested Q1 sample points may lie in cells owned by other MPI ranks.

If the Stage-2 wrappers are removed, replace their direct tests with Stage-4 tests that exercise the same distributed behavior where practical.

---

# 17. What Codex should report before editing

Before changing files, report briefly:

1. where the Stage-4 reconstruction operation should live and why;
2. how it will access the converged initial Q1 phase-field solution;
3. how it will reuse the Stage-3 prescribed-fault representation;
4. how \(h_\Gamma\) will be supplied;
5. how the transverse sampling width will reuse existing `PhaseFieldProfile` information;
6. whether the two Stage-2 point-evaluation functions have any callers that justify keeping them;
7. which files will be changed;
8. which tests will be added/modified;
9. any ambiguity that changes scientific behavior.

If there is no unresolved scientific/architectural conflict after this inspection, proceed with implementation.

---

# 18. What Codex should report after implementation

After implementation, report:

1. files changed;
2. the final reconstruction interface;
3. how arc-length resampling is performed;
4. how Q1 phase-field samples are evaluated in parallel;
5. how the Stage-2 abstraction was simplified or why it was retained;
6. how the transverse centroid and diagnostics are calculated;
7. how reconstruction is triggered after the initial phase-field solve;
8. test/build results;
9. any remaining limitations or questions.

Do not proceed into smoothing, properties, coupling, or propagation without a separate task.
