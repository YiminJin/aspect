# Fixed-surface front and conditional native seeded growth

Keep the exact-jump mixed mechanics, full current pressure, G=1e6 Pa, eta=1e8
Pa s, Gc=1 J/m², peak=3000 Pa, Dc=.001 m, all RSF/radiation parameters and
original force/energy tolerances. Fresh zero bulk/jump histories; Theta=1.
The periodic 1 m candidate Γ remains at y=.05 m between .05/.075 m layers.
A pre-existing mature seed occupies .375<x<.625 m; its .25 J/m fracture inventory
is explicit. Elsewhere a=0 and jump=0 exactly. No tip-growth mask.

For the Q1 jump and physical Gauss records, tied conditions are Q_T δ=b_T and
f_mech−Q_D^T W r_D=Q_T^T W λ, |λ_q|≤peak. Admission is failure of this **weak
box feasibility**, equivalently a positive direction in the support-function
inequality, not |projected nodal stress|>peak. At forming records λ follows the
unchanged return and a_old≤a≤1. All directions, including negative/alternating
ones, must satisfy the weak certificate at a tied solution.

The monotone positive loading permits a positive-formation predictor, but this
is only a solver strategy: unrestricted bounded Gauss multipliers certify each
accepted candidate. Negative rates in established records remain free. A failed
certificate stops the test; no clipping or directional mask substitutes for it.
Use independent linear programming for the certificate and a native box-residual
solver. Retain Gauss damage/age ownership and immutable accepted state.

Stage 1: small exact-jump native test through the first interface activation;
full pressure derivative, weak tied certificate, rejection/retry on the activating
step, one publication, same-rank restart through activation. Independent checks
include multiplier redistribution, alternating directions, uniform identity and
an inadmissible tied load. Force/weak residual <=1e-8 Pa; complete tangent FD
<1e-6; energy ledger <=1e-10 max(1,work). Numerical equality tolerances are derived
from the existing force allowance, not a phase floor. Local stability is evaluated
on the nullspace of tied Gauss constraints, with formation-release curvature
reported separately. The old phase-potential certificate is not reused.

If stage 1 passes, use a bounded 32/64 surface-segment spatial pair with 160
normal cells and a timestep halving .002/.001 s. Loading remains Udot=.001 m/s.
The cheap equilibrated reference is used only to choose a finite observation
horizon before native execution. Continue far enough to observe whether any
new a=1 region develops; report broad nucleation/process-zone overlap if the
short periodic specimen cannot sustain an isolated front. Add a short hold.
Do not change calibration or enlarge the domain to force a desired mechanism.

Compare physical fronts at a=.01,.05,.1,.5,.9 and a=1, seed-connected lengths,
new fracture inventory, peak/integrated increments, slip, reaction and energy
at common times. Use each mesh's exact initial seed; endpoints align on both.
Practical targets: <5% integrated new fracture/slip/reaction changes, <one coarse
segment contour-position change, and <1% absolute physical-work remainder
relative to sliding loss and to added fracture energy separately where nonzero.
Report failure without retuning. A first activated node is only an onset bracket.
No production driver adoption, new geometry, MPI, general propagation or free-path
implementation. Architectural discussion of phi/a/Gamma is a proposal only.
