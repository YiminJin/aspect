# Bounded direction/dissipation reconciliation — selection before tests

Keep production/BP3 and all prior evidence unchanged. Work at
f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work. The old
energetic K-gradient and failed scalar-state candidate are not reactivated.

Recover the historical pf-cpdi selector, including its literal 2-D NaN guard,
and separately test its intended spin-sign rule. Use Lij=partial_j v_i,
omega=(L21-L12)/2, tensile-positive stress and unoriented plane axes. Test
simple shear, zero spin, reversal and a superposed observer spin at fixed
instantaneous stress. No total-spin sign becomes a constitutive requirement.

Implement Feng et al. (2024), DOI 10.1016/j.jmps.2024.105687, equations
17–24/43–66 as a small independent direction/dissipation reference:
N=dot(eps_c)/norm(dot(eps_c)), tr N=0, N:N=1;
pN=min(positive(-alpha.sigma.alpha),positive(-beta.sigma.beta)), beta perpendicular alpha;
F(N)=(sigma-Q):N/sqrt(2)-mu pN;
N is a GLOBAL maximizer when flow is active;
dot(eps_c)=lambda_dot N, lambda_dot>=0, max F<=0, lambda_dot max F=0;
frictional power=sqrt(2) mu pN norm(dot(eps_c)).
Use the actual increment direction, not averaged conjugate rates or the
direction of accumulated crack strain. This is non-associated, zero-dilation
flow. The published pressure definition and norm factors are essential.

Map to the current tensor memory only explicitly: W=dot(eps_c),
tau=T-2G A, Q=2G A/h, Adot+(G/eta)A=W, Kdot=W. Feng's own elastic law has
A=K=eps_c and no Maxwell relaxation; the mapping is not an exact equivalence.
Retain current G/eta, m=4, phase energy and closed-contact inputs for mechanical
controls. In particular, do not import Feng's different phase potential or
fracture-energy calibration. Include the zero-relaxation elastic point control.

At a genuine maximum tie, choose the closest unoriented plane to supplied
fault/accepted directional history. Tie allowance is 64 eps times the actual
stress/pressure force scale; do not override a resolvably larger objective.
If no preference or an equal-distance preference resolves the tie, return
unresolved symmetry. Fixed small-strain material coordinates use identity
history transport between steps; passive coordinate rotations transform the
director and every tensor together. No finite-rotation/objective history law
is added. Do not average normals or freeze a plane across changing maxima.

Check global maximization and actual dissipative power before admitting any
finite-step point response. Compare the old double-system mature flow with
both selected single directions; quantify the actual-direction energy defect
of a conjugate mixture. Repeat the original mixed incompressible pressure
symbol, including changing-direction finite-step derivatives and its rate
limit. Use intact, intermediate and developed phase; both symmetry branches;
the archived strict single-system state; and one reversed-memory control.
Retain FD relative allowance 1e-6 and independent analytic/null allowance
1e-12. No additional viscosity, phase mask or tolerance relaxation.

Produce compact direction/spin/power/symbol CSVs and PNGs. First report which
defect is removed and which remains. Any failure of the mixed mechanical or
maximizing-return prerequisite blocks a propagation test. No new remedy is
selected automatically, and no full specimen/MPI campaign is authorized here.
