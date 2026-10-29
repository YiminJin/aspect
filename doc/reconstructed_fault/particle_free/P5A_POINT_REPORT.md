# P5a — tensor point return qualified

The experimental stateless return and small point-history fixture pass the
selected checks. New flow can turn while established cohesive tensor memory
is retained. The fixed-direction restriction recovers the scalar model.
**Native integration and a directional specimen have not begun.** The scalar
law, native solver/history/checkpoint sources, P4 archives and protected BP3
files retain their entry hashes.

Entry is f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
See the [selection](P5A_POINT_SELECTION.md),
[verification JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/verification.json)
and [compact package](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point.zip).
The [task-only changed-file list](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/changed-files.json)
and [commands/provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/provenance.json)
separate this work from inherited local changes.

## Implemented point law

Retain G=1e6 Pa, eta=1e8 Pa s, R=1000 Pa, m=4, Kf=8 Pa, Gc=1 J/m2 and
ell=.046875 m. All contractions are full 3-D plane strain. A/K/Q occupy the
in-plane trace-free space; T retains its full plane-strain deviator, including zz.
A and K are strain tensors, L an accumulated engineering strain; these are not
surface slip or Z histories. No transport, spin or new resistance law is added.

With beta=1/(1+Delta t G/eta), Tplus=beta(Tn+2G Delta e), abar=beta An and P
the in-plane projection, the finite-h return minimizes

    ||Tplus-2G A||_3D²/(4G) + G A:A/h + R sqrt(2)||A-abar||_3D.

The implementation uses V=h P(Tplus)-2G(1+h)abar, v=M(V)=||V||_3D/sqrt(2).
Stick when v<=Rh: A=abar. Otherwise E=V/v,
Q=(P(Tplus)-RE)/(1+h), A=hQ/(2G), and
Delta A=(v-Rh)E/[2G(1+h)]. This h-scaled form avoids dividing by h near intact
material. Physical stress is tau=Tplus-2GA; W=Delta A/(beta Delta t),
Kplus=Kn+Delta t W and Lplus=Ln+Delta t sqrt(2)||W||_3D.
Old tensors are never rotated or replaced by a scalar along E.

At h=0, A=W=0 and Q is the minimum-norm multiplier outside the residual yield
ball. Y=h' Q:Q/(4G) remains defined. Exact yield ties select the sticking
generalized derivative; the intact phase derivative is right-sided. No phase
floor or director is introduced. The fixed-S control projects onto span(S),
S:S=1/2, and reduces to A=aS, K=kS and Q=2qS.

The analytic action includes
delta E=[delta V-E(E:delta V)/2]/v, plus the complete phase dependence of V,
Q, A, tau, W, Y, storage and dissipation. Incoming history and Delta t are held
fixed during differentiation. A frozen direction is not the consistent tangent.

## Verification results

Scaled defects use max(physical scale, reference norm), with the predeclared
1e-12 identity, 1e-6 return-FD, 1e-7 potential-gradient and 1e-10 BE-ledger bounds.

| Check | Maximum scaled defect |
|---|---:|
| Scalar identity, including the complete fixed-direction path | 4.87e-14 |
| Fixed-direction analytic stress/A/Q/Y tangent identity | 8.88e-16 |
| Complete return finite differences, all output fields | 7.17e-9 |
| Passive basis covariance / tangent covariance | 2.00e-14 / 1.78e-15 |
| Independent 60-digit minimization, including path samples, A/Q/stress/Y | 8.14e-16 |
| Continuous power identity / local BE ledger | 1.20e-15 / 7.99e-16 |
| Pure-shear threshold Y=Kf at abs(Tplus,xy)=3000 Pa | 2.22e-16 |
| Yield admissibility / friction complementarity | 7.96e-16 / 4.16e-17 |

The independent reference solves the energy-plus-friction stationarity equations,
not the C++ radial formula. It covers ten point cases and sixteen path states.
Intact/tiny-h limits, positive/negative slip, both sides of yield, zero flow,
nonzero incoming tensors and full zz response are exercised. S/-S relabeling
preserves physical tensor flow and reverses the projected signed rate exactly.
The frozen-direction negative control has error .0671254 and fails the original
1e-6 bound. Constant coordinate rotations are passive basis checks; they do not
qualify finite rotations or an objective rate.

Each 800-step point path discards two trials, retries deterministically and
publishes once per accepted step. Duplicate/stale publications are rejected.
At step 480 (120 s), a file restart reproduces every actual tensor/scalar field
and every subsequent state through step 800 exactly. A separately populated
record tests nonzero Tzz and all A/K/Q components; comparison reads the fields
independently of the serialization list. This is a fixture-only Boost record,
not a production checkpoint extension. The unchanged scalar suite also passes
all 240 assertions in five cases.

The C++ checks and both paths cost .061 s (.265 s process wall); the independent
point minimization/gradient checks cost about .10 s. Builds use GCC 12.4 and
deal.II 9.6.2; exact commands and identities accompany the evidence.

## Intuitive response and work

One fresh point starts at phi=.2 with T/A/K/Q/L=0. At Delta t=.25 s, impose xy
engineering strain rate 1e-4/s through 50 s, hold to 90 s, impose normal-difference
rate 1e-4/s through 150 s, then hold to 200 s. Both laws receive identical
increments. Phi is prescribed: Y above Kf is a driving diagnostic, not an
accepted phase equilibrium or a measured crack advance.

![Imposed loading and physical response](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/response.png)

Alpha=atan2(Wxx,Wxy) describes tensor flow in the two-component shear space,
not crack geometry. It is plotted only when W is nonzero. The response matches
through the first load/hold; the new loading subsequently turns tensor flow
toward alpha=89.76 degrees. With fixed direction the normal-difference loading
cannot activate that flow mode. These are different constitutive responses,
not accuracy errors against either law.

| At 150 s | Changing flow | Fixed direction |
|---|---:|---:|
| Physical (tauxx-tauyy)/2 [Pa] | 2403.10 | 4507.77 |
| Physical tauxy [Pa] | 581.65 | 800.17 |
| Y [Pa] | 14.3890 | 1.16547 |
| Axx / Axy [1] | .00105233 / .000433129 | 0 / .000323870 |

![Tensor memory and accumulated strain](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/tensor-memory.png)

During 90–150 s, old Axy=.000589688 relaxes to .000323870. The relaxed sum of
new increments contributes Axx=.001052334 and Axy=.000109259. Their sum recovers
the returned tensor to 3.37e-15 scaled error. Holds relax A and physical stress,
while K/L remain unchanged. No memory is erased or rotated.

![Separate work, storage and dissipation](../../../benchmarks/reconstructed_fault/particle_free/evidence/p5a_point/energy.png)

At 200 s, all following terms are J/m3; constant fracture storage is excluded
from changes because phi is fixed. D terms are cumulative physical dissipation.

| Law | Endpoint work | Eb+Ec | Db | Dc | Dfriction | BE numerical remainder |
|---|---:|---:|---:|---:|---:|---:|
| Changing flow | 16.18482 | 1.76187 | 6.53733 | 2.97170 | 4.85905 | .0548684 |
| Fixed direction | 21.89519 | 3.91176 | 14.49548 | 1.27912 | 2.13034 | .0784899 |

Work=storage change+physical dissipation+BE remainder. The independently
reconstructed cumulative defect is <=1.36e-14 scaled. This exact ledger and
continuous point power check do not establish specimen energy/time accuracy.

Preserved development evidence: the first identity harness used a fixed floor
instead of the declared max(scale,reference) normalization and failed at
1.4211e-12; only that normalization was corrected. The independent energy probe
at epsilon=.001 failed at 1.1222e-7. One halving gives 2.8052e-8 (factor 4.0004),
consistent with second-order truncation; extrapolation gives 4.04e-12. The failed
probe remains failed. A separate Python check's missing engineering-rate sqrt(2)
factor was corrected to the full tensor convention; the C++ law was already
correct. Final tiny-h checks reproduced overflow of 2G/h times zero at phi=1e-303.
Multiplication precedes division now; exact component zero checks and hypot
preserve a representable phi=1e-200 flow and its derivative. No floor, clipping,
constitutive or tolerance change is used.

## Next bounded task, proposed only

P5a is ready for **P5b native integration qualification**, not a directional run:

1. Extend the existing accepted-history/checkpoint infrastructure with a separate
   experimental A/K/Q/L mode on the retained
   CellId/Gauss8 layout. Preserve the independent DGQ2 T and native FE source
   complement, Gauss3 projection, accepted physical phase/obstacle owner and old
   scalar formats/default. Do not confuse cohesive A with native displacement
   memory. Reject mismatched modes/layouts; no automatic loaded-model conversion.
2. On one small fixed mesh, initially serial, recover native fixed-S stress/weak-load/history
   identity first, then check one non-proportional radial state. Use the complete
   tensor action in mechanical/phase residuals and tangent, pressure scaling and
   C^T g/C^T H C exactly once. Independently verify actions and the actual
   mechanically relaxed admissible curvature gate for the tensor potential.
3. Exercise native rejection/retry, one accepted publication and same-mode
   restart of all histories and reconstructed stress/weak loads. Retain existing
   source/mechanical/KKT/ledger tolerances; no full directional specimen yet.

The later directional specimen is a **finite physical** [0,1.5]x[0,1] m body:
opposite prescribed top/bottom horizontal motion, uy=0 on all boundaries,
natural lateral horizontal traction and natural phase flux. Retain the stated
P4 strain schedule/seed and fixed R; mechanical pressure remains distinct from
p_N. It is not a distant-boundary approximation. Fresh initialization and new
accuracy checks precede quantitative interpretation. P4's failed temporal/
spatial criteria, 57.26% boundary sensitivity, original energy findings and
strict_reference_match_passed=false remain. No native, MPI, RSF, AMR, transport,
geometry or full directional experiment is qualified by these point checks.
