# PF4 bounded transport-method comparison

Completed 16 October 2026 UTC in `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus the
preserved working tree. **Retain unlimited DGQ2 for resolved prescribed-history
transport.** Scaling remains a bounded scalar benchmark reference; both WENO
paths remain experimental. No threshold tuning, WENO timestep sweeps, production
backend replacement, constitutive change, coupled evolution or PF5 was done.

## Experiment and actual native paths

The unit square has 16×32 or 32×64 fixed Cartesian cells. Velocity is prescribed
as `(0.4,0)` m/s, final time 0.2 s, physical diffusion/reaction/source zero.
The sharp scalar is +1 on `[.25,.5)`, −1 on `[.5,.75)`, and zero elsewhere,
translated by `0.4t`. The finite-width scalar is
`g(x-.25)-2g(x-.5)+g(x-.75)`, with
`g(z)=(1+tanh(z/delta))/2`, `delta=.15625/ln(9)` m. This entire analytic profile
is fixed across meshes, including overlap and nonzero analytic inflow. Its
actual peak is 0.918505516 and its three reference transition widths are
0.139199795, 0.135360618 and 0.139199795 m.

For scalar cases the tensor is `A=f B`, `B=(1000,-400,700)` Pa in `(xx,yy,xy)`.
All relative field errors and quadratic norms use
`|A|_3²=xx²+yy²+(xx+yy)²+2xy²`; `zz=-(xx+yy)`.

DG uses the unchanged PF4 upwind material operator, Gauss3 mass, Gauss4
transport, SSPRK3, and Gauss8 initialization. Scaling uses the existing largest
common mean-preserving factor at GL9×GL9 points and benchmark scalar envelope
`[-1,1]`; no such envelope is applied to native independent history or to ASPECT's
scalar paths. Defaults remain unchanged.

EV and SUPG are **actual ASPECT continuous-Q2 composition solves**, selected by
`Discretization/Stabilization parameters/Stabilization method`, using
`single Advection, no Stokes` and the native prescribed-Stokes function plugin.
They use native interpolation at initialization, Gauss3 assembly/stabilization,
backward Euler for the first physical step and native variable-step BDF2
thereafter, exact left Dirichlet inflow and natural outflow. The native
composition solver tolerance is 1e-11 in every comparison. No scalar clipping,
normalization, viscosity smoothing, extra diffusion or parameter tuning is used.

For this zero-diffusion constant-velocity problem, native SUPG has
`tau_K=h_K/(2 p |v|)`, `p=2`, with `h_K` the cell diameter. Its test is
`w+tau_K v·grad(w)` against the full BDF/advection residual, including the
transient term. EV uses native defaults `alpha=2`, `cR=.11`, `beta=.052`,
`gamma=0`:

```
fbar_q = (f_old + f_oldold)/2
R_q = (f_old-f_oldold)/dt_old + v·(grad f_old+grad f_oldold)/2
E_q = (fbar_q-a)^2
nu_K = min(.052 h_K |v|, .11 h_K^2 max_q |(fbar_q-a) R_q| / D_E)
```

Here `a` is the midpoint of the native extrapolated field range (Q2 nodes), and
`D_E=max(max(E)-mean(E), mean(E)-min(E))` uses native Gauss3 sampling/integration.
The native startup/zero-variation branch uses the viscosity cap. On 32×64,
`h_K=.0349385621` m and SUPG `tau=.0218366013` s. Each tensor component receives
its own scalar EV residual/normalization; this is not a qualified tensor
invariant stabilization rule.

## Independent projection and error accounting

`assess_methods.py` independently implements physical Q2 Lagrange polynomials,
cell mass matrices and the assembled continuous-Q2 mass solve. Sharp integrals
are split exactly at all three transported jumps. Native-history integrals are
split at every translated source-cell face. Gauss3 then integrates every
polynomial product exactly on these affine cells. The finite-width reference
uses independent Gauss16, verified against Gauss24; no native mass matrix,
projection implementation or error integral is imported.

For either discrete space, `P f` solves `(P f-f,w)=0`. We report
`E=||u-f||/||f||`, representation `Erep=||P f-f||/||f||`, and discrete evolution
defect `Ed=||u-P f||/||f||`, verifying `E²=Erep²+Ed²`. CG projections are onto the
unconstrained continuous-Q2 space, hence lower bounds for strongly constrained
inflow solutions. Native initialization error is recorded separately. The ledger
also evaluates `||u-Advect(u_initial)||/||f||` using exact characteristics and
split quadrature; thus an initialization defect is not silently called a
transport defect. That latter norm includes unavoidable final representation
error and need not equal `Ed`.

CG's sharp initial interpolation error .223607 exceeds its independent best
initial projection error .148651. Consequently the sharp final comparison does
not isolate stabilization from initialization. The finite-width and native-load
results support the recommendation independently of this distinction.

Initial extrema are separate from evolution: the cell-aligned sharp DG
projection is ±1 with zero initial front width; native CG interpolation already
has extrema **−1.125/+1.25** and width .0118900 m. For the finite profile, initial
DG/CG extrema are ±.918508239/±.918500372, versus analytic ±.918505516. Thus
overshoot/attenuation comparisons do not silently attribute initial polynomial
features to the transport method.

A second calculation uses normalized Legendre moments only. For cell width `h`,
`m_k=integral_K f P_k(2(x-a)/h-1) dx`, the represented squared norm is
`sum_K sum_(k=0..2) (2k+1)m_k²/h`. On **32×64 at t=.2**, this gives
**0.149862170276557**, matching **0.1498621703** within 2.35e-11. The independent
nodal mass solve agrees within 2e-13. This representation floor explains much of
the sharp DG error. The best projection itself has one-sided polynomial
extrema +1.32704/−1.65408: least L2 error does not imply small overshoot.

|32×64, 103 steps, sharp pulse|Initialization E|Erep|Ed|Final E|
|---|---:|---:|---:|---:|
|Unlimited DGQ2|6.2e-16|.14986217|.09520790|.17754778|
|Scaling reference|6.2e-16|.14986217|.17016354|.22674722|
|Native scalar EV|.22360680|.16032355|.15839115|.22536947|
|Native scalar SUPG|.22360680|.16032355|.13728335|.21106956|

|32×64, 103 steps|Sharp min/max|Sharp three widths, m|Finite E|Finite initialization E|
|---|---|---|---:|---:|
|Unlimited|−1.159517 / 1.165033|.023526 / .023526 / .023526|.000427638|.000271017|
|Scaling|−1 / 1.00001760|.026566 / .050081 / .047601|.000427638|.000271017|
|Native EV|−1.025054 / 1.032423|.028803 / .038049 / .035877|.001275466|.000493617|
|Native SUPG|−1.180399 / 1.132330|.025510 / .025525 / .025570|.000545821|.000493617|

Extrema include quadratic stationary points, so they can exceed earlier GL9
sampled extrema without any field change. Widths/extrema refer to the `y=.5`
polynomial trace, including both
one-sided DG traces and analytic overlap. The maximum measured transverse
nodal tensor variation is 1.01e-5 Pa, negligible at the plotted 1000-Pa scale but
retained in the ledger. GL9 scaling bounds do not imply a continuous maximum
principle; its small between-control-point excess is explicitly reported.

## Time uncertainty, diffusion and integral errors

The common base time step is `.2/103` s on 32×64 (`.2/52` on 16×32).
Every method is also compared at half this step. Retained unlimited/scaling
quarter-step evidence is reanalyzed; no WENO field is rerun or swept. The scalar
native runtime can take an extra roundoff-sized final step when its accumulated
time falls just below .2; actual times/steps are retained and final fields are
compared at .2 within floating-point precision.

|32×64, relative field difference, dt → dt/2|Sharp|Finite width|Native tensor|
|---|---:|---:|---:|
|Unlimited DGQ2|2.685e-5|1.342e-7|1.429e-7|
|Scaling|6.057e-4|1.342e-7|Not applied|
|Native EV|.00183123|.000186511|.000268314|
|Native SUPG|.00148993|.0000560735|.0000522187|

These are measured uncertainties, not fitted convergence claims. DG's quarter
step changes the native tensor by another 7.25e-8. EV native-tensor E changes
.00639269→.00647328 despite smaller load error; a monotone time-convergence claim
would be unsupported. SUPG changes .00463280→.00462872. For the pulse, EV/SUPG
half-step errors are .225222/.210794 (sharp) and .00120163/.000534468 (finite).
Their uncertainty does not reverse the advantage of unlimited DG on these cases.

Finite-width representation floors are .000271117 (DG) and .000434511 (CG).
At the base step, positive peak attenuation is .003709% for unlimited/idle
scaling, .015576% for EV and −.010522% for SUPG (negative means amplification).
Report both pulse lobes through the signed attenuation/extrema ledger.
Unlimited's first finite transition broadens about .02890%. The following base-step
diagnostics use the full tensor integral norm (Pa m²) and `Q=integral |T|_3²`
(Pa² m²). The analytic finite-profile Q change is −3.0115; Q is not mechanical
energy. CG and DG have different initial Q, retained explicitly in the ledger.

|32×64, 103 steps|Finite three widths, m|Sharp integral error / delta Q|Finite integral error / delta Q|
|---|---|---|---|
|Unlimited|.139240 / .135314 / .139216|.000875915 / −44679.7|.000110240 / −6.4761|
|Scaling|.139240 / .135314 / .139216|.000324449 / −156860.7|.000110240 / −6.4761|
|Native EV|.140102 / .135661 / .139510|2.69834 / −97218.2|.00522603 / −723.403|
|Native SUPG|.139276 / .135318 / .139311|.00861844 / −27676.9|.0000743121 / −75.0795|

Finite negative peak attenuation is −.002029%, −.091553%, +.003196% for
unlimited, EV and SUPG respectively. Thus small signed positive-lobe attenuation
alone would conceal negative-lobe amplification. Full component integral errors,
initial/final extrema and physical-range excursions are retained in the ledger.

On 16×32 the sharp errors are .218133/.317947/.302451/.285664 for
unlimited/scaling/EV/SUPG; finite-width errors are
.00315021/.00315021/.00765902/.00424269. Unlimited finite-width refinement rate
is 2.881. Its existing resolved Gaussian rate 2.997 and operator/publication
qualification are retained unchanged; no new WENO qualification is claimed.

## Native independent history and fault weak loads

The source is **the native `make_dg_history_trial` update**, with a prescribed
continuous-Q2 velocity interpolated from

```
u_x = .001 exp(-((x-.45)/.16)^2) (y-.5)
u_y = .0006 exp(-((x-.55)/.19)^2) [(y-.5)^2+.3(y-.5)]
T0 = 2 kappa dev_3 symgrad(u_h)
kappa = -1e8 expm1(-.1*1e6/1e8) Pa s
```

The accepted unloaded anchor is unchanged. Scratch transport has `v=(.4,0)`,
zero source/reaction, fixed native phase/I and `Z=0`; it is a named transport
subproblem, not a replacement for Maxwell/cohesive evolution. Its exact
characteristic reference transports the **actual native piecewise polynomial**,
including one-sided derivative jumps. Independent differentiation gives a maximum stored-2D-block discrepancy of
5.55e-12 Pa, hence a full embedded-3D bound of 9.62e-12 Pa (using
`|D|_3 <= sqrt(3)|D|_2`). DG initialization relative error is
2.50e-15 on 32×64, versus CG interpolation error .00141567. No scalar benchmark
envelope is applied to this non-proportional tensor.

At actual native mechanical QPs and cached fault associations, both the native
stress evaluator and independent coefficient evaluator form
`L_i^s=sum_q JxW chi N_i (T:S)` and
`L_i^n=-sum_q JxW chi N_i (T:n⊗n)`, with actual accepted `chi=h(phi)/I_h`.
These are history-induced deviatoric loads; pressure, background, friction and
cohesive resistance are outside this linear functional. The exact translated
history is evaluated at the **same fixed work map**.
This checks the native weak-load functional; it does not qualify quadrature for
moving geometry or run a coupled residual/solve. At 32×64 there are 36,864
associated QPs, `sum JxW chi=1+6.4e-15`, and both load references are nonzero.

|32×64, dt/2|Tensor E|Erep|Ed|Combined weak-load relative error|
|---|---:|---:|---:|---:|
|Unlimited DGQ2|.004595418|.002531359|.003835374|2.27232e-5|
|Native scalar EV|.006473279|.002902725|.005785976|1.13968e-4|
|Native scalar SUPG|.004628717|.002902725|.003605441|3.64764e-5|

At the common base step the load errors are 2.27316e-5 / 1.90086e-4 / 5.88685e-5.
The **load-vector** dt→dt/2 changes, normalized by the reference load norm, are
DG 1.912e-8, EV 8.626e-5 and SUPG 2.421e-5. DG's next halving changes loads
2.395e-9. SUPG's uncertainty is comparable to its gap from DG, so this is not a
temporally converged ranking of the native-history methods. The recommendation
also uses the resolved finite-width control and actual cost.
Native/independent DG weak-load contractions differ by at most 2.14e-14 Pa m.
All scratch trials leave accepted DG/phase/Z/I unchanged. The 16×32 history and
load controls are retained in the ledger; this two-mesh test does not establish
full history, restart or coupling qualification.

## Cost, verification and decision

At 32×64 there are **18,432 DG scalar DoFs versus 8,385 CG DoFs**, or
55,296 versus 25,155 coefficients for three independent tensor components.
At 16×32 the scalar counts are 4,608 versus 2,145. Exporting CG polynomials in a
cellwise Q2 format does not change its actual continuous DoF count.

At 103 steps, unlimited pulse transport costs about **.407 s for all three
tensor components**, versus scaling **1.201 s**, including its diagnostics.
Native EV/SUPG composition assembly, preconditioning and solve average
**18.891/19.282 s per scalar** across the five independent fields; initialization,
temperature, output and prescribed-velocity work are excluded. At 206 steps the
three native-history components together cost **86.372/88.591 s**, versus
**.819 s** for DG, plus DG's one-time .0814-s operator and .0163-s source setup.
These are actual implementations, not equally optimized kernels: DG exploits
fixed coefficients and a preassembled matrix; native CG reassembles/evaluates
material/stabilization each step. Per-scalar native averages do not isolate an
individual field's residual-dependent EV cost. An optimized cached SUPG
implementation was not evaluated; the recommendation concerns the measured
paths on this host, with at most two simultaneous jobs. All complete run times, peak RSS,
exact core/plugin identities and input files are preserved.

The eight new scalar controls and two native-history controls finish below
180 s each, at most two concurrent, one rank/thread. The four DG comparison
controls are **reused unchanged-source evidence**, not new WENO runs. All fourteen
records match the same executable SHA256
`3ee53a8645b2975db3e21b7de1e9304754531fc9e7c9a17b6893a2e073881718`
and all 879 current production source hashes.

Projection verification retains zero/constants and complete global Q2 tensors
including `x²y²` (max error 4.89e-14), positive full-rank mass, the independent
Legendre prediction, 40 orthogonal error decompositions (max defect 4.31e-16),
14 finite-width quadrature comparisons (max L2 change 1.93e-16), and six native
weak-load contractions. The native Maxwell/cohesive suite passes 628 assertions
in eight cases; PF1 passes 94 checks. Production source/defaults and PF0–PF4
reports/evidence remain unchanged. The runner's default expansion matches four
pinned legacy fixtures byte for byte.

Source corrections are confined to new diagnostic code: initial compositions
can be queried before mesh construction, so fixed-grid resolution is taken from
parameters; independent CG/DG local numbering is respected; the native fixture
adds an asymmetric source term so normal weak loads do not cancel. The two
initial failed launches and the symmetric preliminary fixture remain preserved
under `build-particle-free/runs/` and are not counted as successful controls.

**Recommendation:** keep unlimited DGQ2 for the next selected resolved prescribed
history experiment. It has the best finite-width accuracy, smaller measured native weak
load errors and substantially lower measured fixed-coefficient cost. SUPG is a
useful lower-DoF comparator, with similar native tensor error but larger weak
load errors, sharp oscillations and higher cost. EV suppresses sharp overshoot
at greater error/diffusion and cost. Scaling reduces oscillations at an accuracy
cost and remains benchmark-only. These results do not justify replacing the
production history backend or adding more elaborate stabilization now.

Particle dependencies remain in startup H/CPDI/domain initialization,
support/association, surface material projection, fixed-particle admission and
derived sampling. Transport advances consume no particle properties. Native
accepted publication, phase/surface-history transport, restart, AMR/MPI,
periodic/moving geometry and coupled evolution remain unsupported or guarded.
No BP3, propagation or PF5 work is authorized by this comparison.

[Source audit](SOURCE_AUDIT.md), [portable evidence/replay and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_methods/README.md).
