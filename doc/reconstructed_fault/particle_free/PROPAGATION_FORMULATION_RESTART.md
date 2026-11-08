# Propagation formulation and restart — 7 November 2026

**Physical question:** can unstable sliding create a cohesive zone, fully
decohere new material, and leave a fault that continues frictional sliding,
with one consistent stress law and one fracture-energy inventory?

Use the retained exact-jump cohesive–RSF formulation as the **primary controlled
propagation reference** for answering this question on a fixed small-strain
specimen. It already distinguishes intact ties, partially damaged bonds and
mature sliding, and has an independently checked native workflow. This is a
recommendation for the next experiment, not adoption as the general paper model.
Its path hypothesis, normal-traction accuracy and fracture-relative temporal
error remain open. In particular, the recent mature direct-stress/spin/phase
implementation does not automatically turn this reference into a moving-growth
model. First retain the physically interpretable reference; derive and compare
any change of law before a longer trajectory under that change.

The [mature closure](MATURE_FAULT_TEMPORARY_CLOSEOUT.md) parks the four-run
reference and its unresolved refinement. No mature rerun, architecture change,
new constitutive law, checkpoint conversion or server job is selected here.

## Three formulations, not three names for the same model

Let tau be deviatoric physical stress (including zz in plane strain), p the
pressure with total stress sigma=tau−pI, V signed slip rate, and theta the
physical RSF age. Surface normalized age below is Theta=V0 theta/Dc.

| Route | Physical state and mechanical unknowns | Growth and geometry | Energy inventory |
|---|---|---|---|
| Mature moving diffuse fault | Direct DGQ2 tau, material phase phi, curve labels, age, signed cumulative slip; velocity, pressure and V solved together. Separated T/R storage is an alternative reconstruction, not another physical law. | Growth off. Labels move with accepted material velocity; phase and obstacle transport with that state. No new fault material is admitted. | Bulk tau:tau/(4G), Maxwell loss, RSF/radiation work and flux. Auxiliary separated slip memory is not cohesive energy. |
| Exact-jump cohesive interface | Independent DG tensor plus native FE remainder and relaxed lifting memory; Q1 signed jump j and relaxed jump m; Gauss damage a, bond reference b, Theta and accumulated absolute slip. Mixed displacement/pressure and surface increment solve together. | Weak intact admission; irreversible a. Temporary tied candidate geometry has no formed identity. Formed geometry/histories persist, with a causal ahead-stress path rule. Existing formed material is not materially advected in the retained pilot. | Full physical bulk energy, recoverable bond energy, **Gc integral(a ds)**, Maxwell, damage-weighted friction and radiation. No extra bulk-phi fracture charge. |
| Fixed-fault diffuse cohesive phase | Independent trial stress/accepted native memory, physical elastic slip and phi; endpoint mechanics and constrained phase solve include normalization derivatives. | Phi grows against the accepted physical obstacle on fixed geometry. DGQ2 SIPG changes its representation; it does not implement interface admission or free-path propagation. | Recoverable bulk/cohesive energy plus diffuse Gc/(c0 ell) integral[alpha(phi)+ell²|grad phi|²], including discrete SIPG face terms. This is a separate fracture model. |

The current direct mature law is

    (d/dt + v·grad)tau − Wb tau + tau Wb + (G/eta)tau
        = 2G [dev3 D(v) − L_phi[V]],
    Wb = skew(grad v − localized full slip-rate dyad).

The selected bulk-spin closure uses accepted V, phase, normalization and signed
frame. Transport precedes finite rotation and endpoint reaction. Exterior stress
supplies the transported parent, not an already reacted endpoint. Total-material
spin remains the default/historical comparison. The direct representation
removes auxiliary reconstruction bookkeeping, not physical history. For a
separated identity tau=T−R−2G L_phi[s], dropping R is generally a law change.
See [current guide](MATURE_MOVING_FAULT.md) and
[bulk-spin derivation](BULK_SPIN_CORRECTION_REPORT.md).

For the diffuse cohesive alternative, s is physical elastic slip. The simpler
trial-stress/elastic-slip law implies the evolving-profile correction
Q_geo=−2G C[s], where C includes the material/objective derivative of localization
minus localization of the surface history derivative. The mature Q_geo=0 law
cannot be obtained by merely renaming this s or dropping C. The
[particle-free derivation](../particle_free_history.tex) makes this distinction;
it is not permission to change either tested model silently.

## Intact → cohesive → mature limits of the retained interface law

Set C0=2Gc/tau_p², delta_f=2Gc/tau_p, e=j−b. The actual point law is

    a+ = max(a−, min(|e+|/delta_f, 1)),
    q = (1−a)e/(C0 a),                 0<a<1,
    psi_bond = (1−a)e²/(2 C0 a),
    r = q + a[sign(V) pn mu(|V|,Theta) + eta_rad V],
    Theta+ = (Theta− + V0 dt/Dc)/(1+|Delta j|/Dc).

The existing regularized RSF coefficient is
mu=A_rsf asinh(|V| exp[(mu0+B_rsf log Theta)/A_rsf]/(2V0)).
The same signed V drives jump, aging, friction and radiation. Signed slip and
accumulated absolute slip are different histories. This is a **parallel
bond/contact model**: partial damage activates the fraction a of frictional
contacts, while remaining bonds store energy. It is not a series plastic-slider
decomposition. Loading beyond the previous damage envelope softens cohesion;
unloading/reversal retains a and b and follows the remaining elastic stiffness.
There is no healing or age reset at reversal.
At fixed jump, the damage driving force is Y=e²/(2 C0 a²); minimizing
psi_bond+Gc a over a−≤a≤1 gives the stated max-history law. The added
damage-weighted friction law is a constitutive selection, not a proof that
the full nonsymmetric RSF/current-pressure system minimizes one potential.

At a=0, e=0 is tied; bounded intact multipliers collectively balance the weak
surface load with |q|≤tau_p. A pointwise projected stress cutoff is not the
discrete admission condition. At a=1, q=psi_bond=0 exactly, and full RSF/radiation
sliding remains. Existing bulk memory and age persist. Closed contact requires
pn>0; opening is unsupported, not silently clipped. pn includes the specified
confinement p0 and solved current normal response. The mixed pressure perturbation
has zero mean. Changing the pressure datum to improve a relative error is invalid.
The a=1 limit gives the same local RSF/aging form as mature sliding, but does
not make the exact-jump and diffuse mechanical discretizations identical.
Nor does it identify their different currently selected stress-spin laws.

The bulk recurrence retains the complete physical stress:

    Tbulk = TDG + 2G dev3 eps(dR) − Pi3[2G dev3 eps(dR)],
    tau = Tbulk + 2G sum_j m_j dev3 sym grad(L_j),
    beta = 1/(1+G dt/eta),
    TDG+ = beta[TDG− + Pi3(2G dev3 eps(Delta u))],
    dR+ = beta(dR−+Delta u),       m+ = beta(m−+Delta j).

The native FE remainder and analytic lifting carry real stress modes; deleting
them because mature history is now called “true stress” is not an exact change
of variables. An exact replacement would have to represent their complete sum,
retain the same work adjoint/source and demonstrate projection/transfer errors.
Ordinary uncut DGQ2 alone failed the cohesive objective history-space test.

The retained episode uses its original component-rate small-strain recurrence.
The optional newer cohesive objective control rotates complete physical stress
in a formed-piece/side Q2 space before reaction, retaining the native/lifting
remainder. Its gradient includes the analytic lifting contribution. It does
**not** implement the mature bulk-slip-subtracted closure or coupled material
transport/growth. Selecting that mature closure for cohesive history remains
unsupported. No older trajectory acquires it by checkpoint reinterpretation.

The condensed equation and current-pressure tangent are unchanged:

    f0 + beta K Delta j + Q^T W r = 0,
    J = beta K + Q^T W [r_j Q + r_p beta N].

Here Q evaluates surface unknowns and W contains physical surface weights;
this W is not a spin. General incoming stress enters f0 and normal offset
through the native mixed solve. Recoverable bulk/bond increments, Gc Delta a,
Maxwell, friction and radiation are measured separately. The exact BE ledger
includes bulk increment-norm dissipation and the surface endpoint remainder;
small algebraic closure does not establish physical trapezoidal-work accuracy.
Both signed cumulative and accumulated absolute work errors must be compared
with **new fracture work**, not only the much larger sliding loss.

New candidates start tied with zero new bond/fracture/slip/Maxwell memory and
the selected virgin age. Admission adds metadata but no second fracture charge.
The ahead-disk maximum-shear preference with physical R and curvature persistence
is a constitutive/path hypothesis, distinct from numerical lifting width and
quadrature. Formed records keep their direction and history; temporary tails can
be replaced. Rejection restores the whole parent; acceptance publishes once.
See [full equations/ownership](COHESIVE_RSF_NATIVE_FORMULATION.md) and
[phi/a/curve separation](COHESIVE_INTERFACE_ARCHITECTURE_PROPOSAL.md).

For future moving growth, irreversibility must follow material histories:
phi+≥phi_adv or a+≥the transported material a. Monotonicity at fixed Eulerian
nodes would leave a false damage trail. Transport mean preservation does not
require a growing damage field to conserve its integral. Current-arclength
fracture energy also generates shape/measure work; omitting it or silently
using reference length changes the model. Neither the DGQ2 transport pass nor
the present fixed-geometry ledger qualifies this extension.

## Actual pause point and what has been recovered

The earlier [curved-growth pilot](COHESIVE_RSF_NATIVE_REPORT.md) reached
.462375 s after 179 positive steps. New material was admitted, two new cells
fully decohered, and their post-decohesion slip reached .04749/.01606 Dc. Its
7.5 m formed-length computational guard stopped a later attempted extension;
that is not physical arrest. Point/normal-traction and path-choice limitations
remain. This is useful propagation evidence, separate from the RSF episode.

The later [first RSF episode](RSF_EPISODE_REPORT.md) ends at step 748,
t=374.58003334330067 s, four ranks. Maximum observed speed was .0450089 m/s;
final .0444673 m/s was only 1.2% lower. Fully decohered length remained 3 m;
damage increased in the adjacent cohesive region without new persistent
geometry. The repaired continuous-history estimator, not the obsolete raw
rate-change rejection, is explicitly selected. Its local error budget is
5e−4, not a guarantee of 1% trajectory accuracy. The segment's physical-work
remainder was .0573% of sliding loss but **20.8% of added fracture work**.
The episode and the fracture-relative accuracy screen remain incomplete.

The original step-661 controller-floor failure was repaired before this pause;
it is not the current blocker. Separately, the Stage C localized Airy-tip test
stalls in force-residual merit decrease (objective and legacy traces identical),
before admission, with contact and complementarity passing. The
[diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md) also resolves first-order BE bulk and
surface endpoint energy errors. Those failures are not cured by mature transport
success or by choosing a smaller timestep without evidence.

Current numerical source still matches `84f57503c`; the current core is
SHA256 `6ea374c5…a67d9`. A matching benchmark plugin was built in the new
`build-particle-free/propagation-recovery/checks/` directory without rebuilding
or overwriting any retained core. Historical episode reproduction instead uses
`objective-stress/baseline/1-aspect-release`, SHA256 `7277eebc…83b1d`, and its
matching `2-libcohesive_rsf_native_checks.release.so`, SHA256 `7bfbeaa3…22f5d`.
Both historical hashes match the original record. The original episode restart
and all 13 packaged checkpoint files are present and hashed. They are copied
before any restart. No claim of cross-version checkpoint conversion is made.

Before recovery, inspection of retained run indices and the packaged reference
found no episode result later than step 748. This task adds only four steps,
not a completed event. The source/API audit confirms that `cohesive_interface.cc`
still implements the above law; `cohesive_system.cc` retains strict checkpoint
signatures and the legacy default; `cohesive_bulk_history.cc` and
`cohesive_objective.cc` retain native/lifting and cut-objective ownership.
`solver/cohesive_rsf.cc` owns the native solve, and
`time_stepping/cohesive_rsf.cc` owns the selected estimator. The
`material_model/phase_field_fault/` mature owner and `simulator/phase_field.cc`
DGQ2 phase path remain separate. No source API or physics repair was necessary;
only isolated input/output paths and a freshly compiled benchmark plugin were
needed. Existing trajectories, lost-frame cleanup record and failed evidence
remain unchanged.

## Executed recovery checks

All commands and outcomes are indexed in
`evidence/propagation_recovery/summary.json`; the
[reproduction notes](../../../benchmarks/reconstructed_fault/particle_free/propagation_recovery/README.md)
link exact launch records. Nine retained binary/plugin hashes, 950 numerical
source hashes and 189 reference/input/checkpoint/evidence hashes were rechecked.
The shared-assembler review archive was not rebuilt or rescanned. No full
repository archive, cleanup or source commit was made.

| Check | Outcome | Launch wall seconds |
|---|---|---:|
| Current-source fresh active fixture, scheduled cutback control | Five accepted steps through .0005 s; no physical failure | 8.40 |
| Current-source forced rejection/retry | Five accepted steps; one rejected trial, publication once per accepted step | 7.33 |
| Matched accepted-timestep control | All six snapshots (including initialization) bitwise equal to rejection case: full bulk/native history, surface state, geometry and sampled stress | 18.06 |
| Current-source restart from copied slot 1 | Three common snapshots, including two newly accepted steps, bitwise equal to uninterrupted rejection case | 6.68 |
| Historical episode zero advance, four ranks | Exactly step 748; all saved coefficients, surface histories, geometry and bulk samples bitwise equal to packaged terminal reference | 5.03 |
| Historical episode continuation, four ranks | Four steps 749–752 accepted, exactly four publications | 105.31 |
| Additional existing DGQ2 fixed-fault growth replay, two ranks | Initialization/derivative checks pass; **330 s timeout during first physical step**, no completed physical step exported | 330.00 |
| Prepared next input syntax validation | Pass; no state advanced | 3.88 |

Including the final read-only process check, total launcher/simulation time is
**484.76 s (8.08 min)**, below the one-hour aggregate cap. One simulation ran
at a time, at most four ranks, one numerical thread per rank. The only build
used two workers: approximately 46.4 s from generated CMake cache to linked
plugin, plus about 1.5 s configure (filesystem timing estimate, not a profiler).
No core rebuild. Final process inspection found no remaining recovery jobs.
The fresh current-source checks retain `active.prm`'s legacy stress-rotation
selection; they establish functionality/lifecycle, not a new objective cohesive
law. Completed rotation checks were not repeated.

The diffuse-growth input differs from its retained successful loaded control
only in output paths. It was much slower in this invocation; its last recorded
KKT was 6.94e−18, but final acceptance/output had not completed. Do not infer
a new scientific failure or numerical equivalence from this timeout. Its
initial field matches the retained integral .15636083167622108; the prior
17-step result remains historical evidence, not a new pass. No longer rerun
or solver adjustment was made to obtain a pass.

The new episode endpoint is **t=374.5801868753926 s**, step 752, speed
.0444283050 m/s. Global peak remains .0450089193 m/s and the decay diagnostic
is still false. Added-damage integral relative to original initialization is
.000291644178 m; new persistent length is zero and the 3 m fully decohered
interval is unchanged. The four-step maximum surface weak residual is
3.03e−9 Pa (unchanged 1e−8 bound), exact BE ledger error 3.27e−14 J/m,
continuous-history defect 3.9946e−4 (budget 5e−4). New fracture work is
4.44354e−5 J/m. Signed and absolute physical-work remainder both equal
9.65254e−6 J/m: **21.72% of new fracture work**, .0631% of sliding loss.
The accuracy gap persists despite successful solve/restart. An updated
[event-window figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/propagation_recovery/episode/event-window.png)
uses the original prefix and the new four states; old data/figures are unchanged.

The 105.31 s continuation includes 83.91 s native operator setup. Four steps
are too short for a reliable new warm-throughput estimate; retain the measured
historical 8.11 s/step planning rate rather than extrapolating setup-inclusive
time as steady cost. The new checkpoint is in
`build-particle-free/propagation-recovery/runs/episode-short/output/restart/`;
keep its complete directory and last-good marker **3**, whose hashes are in
the recovery summary. The earlier curved pilot's native .441 s anchor is also
still present as `evidence/cohesive_rsf_native/native-start-checkpoint.zip`;
it was not replayed here or treated as the instability specimen.
The original step-748 checkpoint and all its package copies remain intact.

## Classification of the remaining work

| Kind | Concrete issue | Consequence |
|---|---|---|
| API/build drift | Match benchmark plugin to current exported headers/core; correct output paths | Rebuild only the plugin and check the unchanged fixture; no law change |
| Exact state transformation | Store the full cohesive physical tensor instead of DG/native/lifting pieces | Must preserve all modes, work adjoints and accepted histories; finite-space projection is an additional approximation |
| Constitutive/energetic choice | Surface a versus thermodynamic phi; damage-weighted contact; spin; current versus reference fracture measure; path rule | Derive and compare the changed equations before continuing an old trajectory |
| Numerical accuracy | Endpoint temporal error, finite-end stress, normal traction, surface/bulk resolution, quadrature | Measure sensitivity with unchanged physical inputs; no tolerance relaxation or compensating dissipation |

Primary recommendation: retain the single surface-fracture inventory and the
existing small-strain law for one bounded episode continuation comparison.
For a later diffuse paper model the competing energy would be the DG/SIPG
bulk fracture functional **instead of** Gc integral(a ds), with a derived
matching cohesive resistance and history mapping. A decisive future model
comparison is common loaded-tip loading/unloading at matched strength and Gc,
measuring bond storage, new fracture work and fully decohered advance. It cannot
be performed honestly by merely enabling both fracture terms in one input.

## One next task, prepared for review

Recommend a **bounded continuation of the same first RSF episode**, using the
new step-752 checkpoint and the historical compatible executable, with no new
spin, phase fracture energy or material parameters. The hypothesis is that this
retained fixed-small-strain law supports an interpretable complete acceleration/
decay episode and can distinguish partial cohesive-zone damage from actual new
fully decohered fault. It is not yet a claim of quantitatively converged growth.
The older neutral-state-feedback case and the preserved step-748 state provide
controls; the new four steps establish continuation/restart functionality.

The [ready input](../../../benchmarks/reconstructed_fault/particle_free/propagation_recovery/next-episode.prm)
and [exact command/measurement protocol](../../../benchmarks/reconstructed_fault/particle_free/propagation_recovery/README.md)
retain the original 128×96 mesh, .0625 m surface spacing, G/eta=1e6/1e12 SI,
Gc=3 J/m², peak strength 2000 Pa, p0=10000 Pa, radiation and RSF parameters,
constant loading and residual-resolved error budget 5e−4. They are syntax
validated, **not launched**. Budget at most six local hours in checkpointed
segments ≤1.5 hours, horizon t=374.68003334330067 s. Historical throughput
predicts approximately 5.9 hours per .1 physical seconds, plus restart setup;
this is a budget, not a guarantee of decay.

Use the existing peak/decay diagnostic: after Vmax exceeds 1e−3 m/s, require
global rate below 10% of the observed maximum continuously for
max(.005 s, 5Dc/V_peak). Stop on any unchanged physical/numerical guard,
the reviewed compute/horizon cap, or verified decay. The diagnostic is analysis,
not an automatic native stopping plugin; request accepted-state termination
when met. Wall stops never mean arrest. Measure signed/absolute slip and age,
partial-damage integrals, bracketed a=1 fronts, newly formed identities and
temporary candidate geometry separately. Report work errors against both
sliding and fracture and retain exact BE and physical work ledgers separately.
No waveform/rupture-speed claim follows from this quasi-static,
radiation-damped experiment. Full temporal/fracture accuracy still requires
its own focused comparison before a quantitative paper claim.

This recommendation chooses an existing interpretable small-strain control for
the paper's propagation/RSF question. It does not preselect the same law for
advected growing faults. If the intended paper instead requires the diffuse
thermodynamic phi model, derive that energy/resistance mapping and use the
loaded-tip loading/unloading comparison above before the prepared long run.

## Minimum mature rechecks after a selected change

Historical runs always remain controls at their original identity. Only claims
using changed equations/operators need replacement evidence.

| Proposed change | Affected outputs | Minimum justified recheck before reusing mature claims |
|---|---|---|
| Documentation or algebraically equivalent plumbing | Identity/ownership, possibly floating-point order | Existing unit identities and one short matched direct/separated comparison; no automatic A/B/C/D rerun |
| Growth-only law with exactly unchanged a=1 mature limit | Newly formed state, transition resistance/energy | Intact/unload/mature-limit tests and one relevant coupled smoke; independent mature transport evidence survives |
| Direct cohesive history replacement retaining all physical modes | Stress, work adjoint, projection and checkpoint payload | Independent reconstructed-tau/energy/source comparison, rejection/restart; repeat affected mature control only if its owner/operator changes |
| Shared spin or Maxwell response | Stress, shear/normal traction, slip/state, energy | Rotation/relaxation and accepted-spin checks, constant/layered sheared smoke, law-identity restart; then affected A/B temporal and B/D mesh comparison (C if motion/spin direction is involved) |
| Localization, mechanical quadrature or traction coupling | Stress transfer, normalization, friction, phase forces and energy | Work-adjoint/derivative tests and finite-end control; B/D and relevant temporal/direction pair before transferring their numerical claims |
| Diagnostic integration only | Reported norms/work, not accepted evolution | Reprocess retained states with converged integration; do not rerun trajectories unless required state was never saved |
| Geometry/admission/history ownership or transfer | Material identities, phase obstacle, slip/age and energy | Joint rejection/retry and restart, birth/no-free-fracture check; cell-crossing moving control in both directions if common geometry code changes |
| Phase transport/limiter/time splitting | Bounds, shape, normalization and coupled forcing | Material/divergence/inflow operator tests, mean/bounds checks, moving rejection/restart; A/B and B/D plus C for changed boundary/direction behavior |
| Surface-only propagation path rule | Candidate direction, formation, later cohesion/RSF | Tied insertion, admission/energy and growing restart; no mature trajectory rerun if established geometry and its mechanics are untouched |

Before publication, freeze the selected formulation and establish accuracy for
the actual claimed physical observables on that version. Neither universal
validity nor universal obsolescence of old evidence is justified.
