# Horizontal enlarged-domain control

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Use the qualified native signed component; preserve the law, solver tolerances,
quadrature, physical seed and loading. Production/BP3 are untouched.

Compare the saved fresh 8×1 m half-step trajectory with one fresh **16×1 m**
periodic strip, physical x∈[-4,12], y∈[0,1]. Keep seed [2.5,5.5], Γ at y=.4,
p0=10000 Pa, G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², peak=3000 Pa, original RSF/
radiation values, Δx=.0625 m and Δy=.0125 m. The new mesh is 256×80. Implement
only a coordinate translation in the analysis: the native box is [0,16], seed
[6.5,9.5], physical x=x_native−4. Homogeneous shear loading is translation
invariant. Initial jump/memory are zero, Θ=1, a=1 on the mature seed and zero
elsewhere. Both meshes represent this initialization exactly; no histories are
interpolated or transferred. The archived 8 m run already starts from these
fresh physical data. One fresh 8 m control qualifies the native timestep-function
and read-only snapshot hook: it reproduces all 1,057 accepted states bitwise.
Use that matched run for the domain comparison.

The new candidate is [-3.5,11.5] physical m. Check ALL added records outside
old Γ=[.5,7.5] remain tied with zero damage/jump/fracture/friction work. If so,
removing their exact zero-jump variations leaves the same old-Γ problem; this
separates the period/bulk-domain control from activation on added representation.
If they activate, report a combined domain/admission effect and stop interpreting
it as a pure periodic-distance comparison. Do not reset or mask them.

Native timestep-function control: Δt=.0025 through 2.30 s, .00125 thereafter,
loading speed .001 m/s to 2.45 s, hold to 2.47 s. Verify the schedule against the
archived baseline and a cheap native scheduler control before the expensive
case. Preserve a valid native checkpoint near 2.3 outside rolling overwrite.
Add only a read-only optional snapshot observer to obtain actual Gauss6 bulk
fields at 2.30/2.45/2.47 from the candidate BEFORE publication. It must keep all
existing bulk stress/work/pressure checks. No constitutive/integration changes.

Measure seed-connected .01/.05/.5/a=1 extents, the continuous Q1 full-jump
threshold where monotonicity holds, integrated growth/slip, RSF histories,
reaction, pressure, complete tensor stress and energy. Compare common physical
points in [0,8] and a near-tip box [1,3]×[.2,.6]. Use actual quadrature weights.
Retain the whole raw physical-work error / new fracture and / sliding screens;
1% remains the target. The larger uniform far field adds extensive work/storage:
report total values, common-box energy and fault-induced EXCESS relative to an
independent homogeneous Maxwell strip. Excess is a domain-comparison diagnostic,
never a subtraction to make an acceptance error pass.

Predeclare practical domain-sensitivity targets: <2% late integrated growth/slip,
<.05 m full-front change, <1% near-field full-tensor relative L2 change, and
<10 Pa RMS current normal-pressure change. Report early-onset differences and
absolute errors near zero separately; retain the 65.4% early spatial failure.
Compare the domain effect with the measured temporal/spatial uncertainty. No
convergence order or distant-boundary qualification follows from two periods;
H=1 and its wall influence remain fixed.

Estimated larger-case cost: roughly 15–30 minutes including setup and evolution;
use one rank/thread, a **1800 s wall bound** and **6 GiB observed RSS bound**,
and preserve timeout/failed-state
outputs. Stop on the existing contact, weak/derivative/ledger/curvature limits,
remote weak shear >=3000 Pa, detached a>=.01 growth or the .25 m candidate buffer.
If a real numerical/resource blocker appears, diagnose the saved failure; do not
relax tolerances or change the law to finish. A same-Γ representation may be used
only if needed to isolate a demonstrated capacity limit. Do not run past 2.47 s,
change H, begin a mesh sweep, free path, MPI or a longer event in this task.

## Follow-up selected after the preserved diagnostic stop

The first enlarged attempt stopped before publication at 1.9425 s, accepted
time 1.9400 s, on the existing derivative gate. Its three FD perturbations were
2.02e−14, 1.01e−14 and 5.04e−15 m; measured errors grew as 1/epsilon. Preserve
this failure and its checkpoint. Add only an opt-in extended-precision independent
residual reference, using the **same perturbations and 1e−6 gate**. Retain the
double results and reject a deliberately 1% biased tangent. Independently check
saved operands at 50 digits. This is a diagnostic arithmetic correction, not a
new physical allowance or change to the point return, Newton solve or timestep.

One bounded native restart from accepted 1.8975 s has a 1200 s wall budget and
the same 6 GiB RSS limit. Compare every replayed physical field/history through
1.9400 s with the preserved original attempt; require bitwise identity. Stop if
the extended reference exposes a genuine tangent/branch problem. No additional
domain, physical-time or parameter campaign is authorized by this follow-up.
