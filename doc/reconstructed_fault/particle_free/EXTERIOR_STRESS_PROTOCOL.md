# Coupled prescribed exterior stress — 5 November 2026

Current user selection supersedes the proposed frozen mechanical audit. Implement
one opt-in exterior-stress callback and complete five new coupled runs; no new
physical campaign, general refactor or production conversion. Work in existing
root/branch at cf06525471ce89789f1ac10905dc79f3aac7562a, preserving dirty docs,
prior experiments and protected executables. Entry hashes and baseline executable
copy precede edits/build. Build/test workers <=4; native runs sequential, four ranks.

## Boundary and state contract

Keep native DGQ2, SSPRK3, accepted-gradient Jaumann spin, endpoint exponential
Maxwell/source evolution, phase transport, coupled projection correction and
acceptance unchanged. Add `Material tensor inflow = prescribed exterior stress`
with an explicit provider identity included in existing checkpoint material
identity. Defaults and prior identities remain byte-for-byte unchanged. Initially
support direct physical stress only: one physical exterior tensor does not
specify independent separated auxiliary histories. Reject unsupported selection,
empty/mismatched identity or missing provider. Reattach on restart; never infer
or convert a production exterior law.

Callback arguments: x, accepted parent time t_n, RK stage time t, attempted
physical dt. Return the exterior for **pure transport** from that accepted
parent; rotation/reaction subsequently applies exactly once. Native inflow
quadrature invokes it on every locally incoming boundary, including top portions.
Provider is immutable during an attempt, deterministic and has no MPI collectives.
No pointer enters checkpoint state. No geometry or accepted history is published
by the provider. The small benchmark provider restricts itself to its declared
uniform accepted schedule and rejects a different dt; this is not general
adaptive exterior-history integration.

For G=eta=10, shear gamma=.5, p_ref=0 and zero stored initial stress, write
z=(tau_xx-tau_yy)/2+i*tau_xy, with zero mean/zz for homogeneous simple shear:

    r=exp((-G/eta-i*gamma)*dt), F=eta*gamma*(1-exp(-G*dt/eta))
    z_n=iF*(1-r^n)/(1-r), z_0=0.

Use tau=(Re z,-Re z,Im z). During transport step n+1 return z_n at every stage:
spatially homogeneous advection does not change it. Do not supply z_(n+1).
Compatible top traction at the physical endpoint is (Im z_(n+1),-Re z_(n+1))
at p=0. At simulator step zero only, top traction is
(5*(1-exp(-.1)),0): the separate .1-s initial mechanics interval is **not**
published as accepted history. Pressure normalization stays `no`; no new gauge.
Use the previous isolation's top formula exactly to reuse its trace controls.

## Bounded crossed/refinement matrix

All use the existing finite mature-fault fixture, compact Q1 phase, growth off,
18 identical initial material labels (spacing .1 m), same fields/parameters,
total duration .4 s and ordinary DGQ2/SSPRK3. Four-rank runs, fresh histories.

| Case | Mesh | dt | Top | Tensor inflow |
|---|---|---|---|---|
| retained shear32-dt025 |32²|.025|zero|numerical parent trace|
| retained corner-coupled025 |32²|.025|compatible|numerical parent trace|
| new exterior-zero025 |32²|.025|zero|prescribed parent exterior|
| new exterior-compatible025 |32²|.025|compatible|prescribed parent exterior|
| new exterior-compatible050 |32²|.05|compatible|prescribed parent exterior|
| new exterior-compatible0125 |32²|.0125|compatible|prescribed parent exterior|
| new exterior-fine0125 |64²|.0125|compatible|prescribed parent exterior|

Use the existing representable-clock rule to avoid an extra ulp terminal step.
Record failures rather than change tolerances or physics. Independent compatible
runs continue if the original top case fails. No further refinement matrix.

## Verification and interpretation

Rebuild a matching checker in a separate directory after the material header
changes; preserve old DSOs and runtime libraries. Run existing unit suite and
check independent real-component recurrence against exterior/traction formula,
zero first-step inflow and all incoming face counts. Verify identical initial
fields across temporal cases and publication once per physical step. Checkpoint
identity prevents old trace/zero trajectories from silently changing physics.

Reuse regional stress/pressure-aware traction, slip-rate/cumulative-slip,
state, geometry, phase/bounds/normalization, work and energy reductions from the
shear/isolation reports. Report global and upper-left-corner growth, central/end
regions, signed and absolute energy remainders, actual boundary work/flux and
computational costs. Temporal comparisons are pairwise sensitivities; discrete
background traction also changes with dt. Spatial comparison fixes dt and labels.
Distinguish raw interior-trace continuum flux from prescribed incoming flux and
the stage-specific transport law. Do not relabel a boundary mismatch as physical
dissipation or close a physical energy balance by construction.

The crossed comparison tests whether prescribed inflow removes the original
corner growth, whether compatible top loading is also required, or whether the
consistent-boundary case still fails. Do not select another frozen audit before
these results. Maintain production/cohesive limitations and inherited loss of
39/45 native-pilot frames and earlier temporary logs.
