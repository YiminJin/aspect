# Limits of this control

[Guide](../TIP_TREATMENT.md)

The physical closure is explicit for a fixed-direction single-fault specimen;
it is not a completed general tip model. All bands at a given section coordinate
share one age and resistance. Opposing rates can reduce net slip and section
friction loss while leaving transverse deformation and cohesive variance.
General turning/branching requires a rule for orientations, intersecting bands,
material age ownership and a work-consistent nonlocal mapping. Reconstruction
alone cannot supply those laws.

Current-pressure friction, opening/contact, variable coefficients, nonzero spin,
transport, AMR and arbitrary off-quadrature stress sampling are unsupported here.
The mature bulk-slip-subtracted spin selector remains separately qualified; it
cannot be extended by removing a guard. A growing model must first define the
full fault velocity-gradient contribution J_f, not just its symmetric part.

General motion also adds fracture-gradient deformation work. For constant K,
\[
D_t\nabla\phi=\nabla D_t\phi-(\nabla v)^T\nabla\phi,
\]
\[
D_t\psi_f=K\alpha' D_t\phi+2K\ell^2\nabla\phi\cdot\nabla D_t\phi
-2K\ell^2(\nabla\phi\otimes\nabla\phi):D.
\]
A material-volume balance adds ψ_f div(v); an Eulerian balance needs boundary
flux. A configurational/microstress closure or explicitly prescribed geometry
work is required before moving growth is enabled. An observed residual cannot
be renamed a shape-work correction.

Dissipation positivity and a converged KKT system do not prove uniqueness,
ellipticity, global stability or physical validity. The earlier Coulomb
localization/gradient findings remain evidence for those different laws. No
uncalibrated slip-gradient energy, damping, phase cap, or tolerance relaxation
has been imported to bypass them. The admitted endpoint section law is monotone and has a unique scalar root;
this does not exclude global phase instability or steady velocity weakening.
Failed evidence must remain visible.

The active coupling uses distributed sparse FE ownership and ASPECT Stokes
inverses. The root-direct mixed solve is now an explicit benchmark-only route.
Distributed storage is not a performance qualification: difficult phase searches
can still consume many Krylov and Stokes iterations.
Point-history quadrature and phase-space/obstacle refinement are distinct
accuracy questions. The [mature guide](../MATURE_MOVING_FAULT.md) and its four-run
closeout remain unchanged qualifications of a different model and integrator.

## Measured qualification boundary

The [original report](../TIP_TREATMENT_IMPLEMENTATION_REPORT.md) retains the
coarse/half-dt passes and refined .05 failure of the old add-only procedure.
The [integration report](../TIP_SOLVER_INTEGRATION_REPORT.md) records its corrected
failure-window replay and current continuation limits. They are different solver
generations, not contradictory descriptions of one executable.

The original loading reaches the homogeneous intact threshold near .05 s and
activates broad weak damage. Its unloaded profile relaxation and outer-boundary
rates prevent interpreting that experiment as localized propagation. The new
fixture equilibrates the declared seed before its physical clock and checks
remote driving, but its load-driven diffuse increment and BE work remainder must
still be judged against their measured sizes. No phi≈1 advancing core, stable
branch selection, mesh/quadrature qualification or production readiness follows
from successful residual/KKT/ledger tests.

The [bounded efficiency/growth task](../TIP_EFFICIENCY_AND_GROWTH_REPORT.md)
retains progress-based search ordering on measured cost and local checks. The
same gates permit tens of Pa of resistance variation at rates around 1e-13 m/s;
independent RSF evaluation explains that sensitivity, but it is not uniform
reaction accuracy. A unique physical stationary state is not established.
The localized extension reaches remote Y/Kf=.98149 at .09 before any resolved
advance of the measured phase or H contours. It shows halo adjustment, not an
advancing weakened front. Compact loading and a separately derived time-accuracy
strategy are next proposals; neither a new load nor integrator is enabled.


## Native basis boundary

The committed implementation at `e8b6dd091` stores native Bernstein phase in the
active fixtures. The [basis report](../NATIVE_DG_BERNSTEIN_REPORT.md) establishes
operator and bounded coupled equivalence, including a same-parent difficult
step and an active MPI restart/rejection check. It does not add an endpoint
integrator, compact physical loading, stress rotation or transport to bulk-tip
growth. A passed shared phase test on bilinear/nonconforming cells does not
remove the tip control's Cartesian aligned-section and fixed-mesh restrictions.

Element polynomial interpolation/embedding/restriction is not a general
irreversible AMR transfer. Ordinary checkpoint loading requires the same numerical
basis as well as the same physical law. The small benchmark parent converter
is separate from that production policy. Native Bernstein is not admitted by the
historical replicated `benchmark direct` or cohesive endpoint solvers; their
nodal routes are retired from the current executable and preserved through
their reference binaries.

The basis change removes repeated conversions and a redundant ghosted trial
copy, but does not eliminate Krylov cost: the difficult step increased from
101.36 to 109.94 s while the localized total fell from 12.49 to 10.61 s. The
near-roundoff physical comparison supports the new representation, not a general
speedup. Endpoint time accuracy and an interpretable localized-growth specimen
remain the next scientific questions; no longer trajectory is authorized by
this documentation update.

## Selectable midpoint and deferred cleanup

The [time-integration page](time_integration.md) defines one candidate, not an
integrator framework or an instability qualification. Its sufficient unique-root
condition differs from BE. It excludes sign-changing free Maxwell relaxation
through ΔtG/eta≤2, but this is not an accuracy guarantee. Active-set changes,
intact birth, endpoint stationarity defects and nonlinear cohesive chain-rule
error remain measurable temporal limits. Stage rates are not endpoint rates.
Time-dependent boundary data and stiff viscous-crust evolution are unqualified.

[The time-update report](../TIP_TIME_UPDATE_AND_GROWTH_REPORT.md) records the
bounded comparison and specimen decision. A compact boundary profile alone does
not establish tip localization. The [deferred cleanup inventory](../TIP_DEFERRED_CLEANUP.md)
identifies current callers and evidence dependencies; broad deletion is deferred
until this numerical/specimen decision, before expensive publication runs.

The completed study keeps midpoint opt-in: none of its three common-parent
refinements finishes the .0575→.06 window. A smaller work remainder in the
separate old-loading control does not qualify that missing endpoint. The compact
pilot stops at boundary damage after 23 steps; its high phase contours do not
advance. Opposite-sign local fault rates and multiple bands nearly cancel the
section slip, despite finite cohesive energy and local work of both signs.
The closure guarantees integrated section power, not pointwise frictional
positivity. These are measured limits, not reasons to clip rates or modify the
localization. The subsequent bounded failed-row/search diagnosis is closed without a local
repair; see [consolidation](../TIP_CONSOLIDATION_REPORT.md). A revised loading
screen or broader globalization study needs a separate selection.

## Specimen selection boundary

The [scaling and retained-field assessment](../PROPAGATION_SPECIMEN_ASSESSMENT.md)
rejects the present 4×2 m, .8 m half-seed, Gc=26.6667 input as the next
propagation specimen. Its 26.67 m energy/strength length, finite wall distances
and measured tip/remote driving provide no suitable separation of scales.
The uniform shear control avoids the compact case's final signed cancellation,
but this alone does not establish an energetically suitable growth test.
No new fixed-phase pair was run after the failed scaling gate. Revised toughness
or larger seed/body dimensions require a new calibrated selection; neither
loading amplification nor a relaxed solver tolerance supplies that selection.

The subsequent [selected synthetic specimen](../SYNTHETIC_PROPAGATION_BENCHMARK.md)
changes those material/geometric scales explicitly. Its
[free-phase BE pair](../SYNTHETIC_FREE_EVOLUTION.md) completes 24/48 intervals
from t=0 to .06 s: broadening begins near .03 s and weak axial extension near
.055 s. Threshold motion is sub-cell; the high-phase front does not advance.
At the smaller timestep, physical-work error still exceeds added fracture
energy by 10.42x over the full history and 3.50x over .04–.06 s, despite small
phase/slip differences. This is a local response comparison, not energetic or
spatial qualification. The [subsequent midpoint assessment](../MATERIAL_MODEL_CLEANUP_AND_OVERNIGHT_RESULTS.md)
reduces work loss on three complete coarse histories. Its fine-mesh prefix shows
large growth sensitivity. A verified scalar stopping-floor defect is corrected
and the failed coupled interval passes, but the full fine endpoint and affordable
spatial qualification remain outstanding. The older outer80 nonlinear-search
sentinel remains unqualified under the final kernel. Longer growth and an RSF
instability campaign remain deferred.
