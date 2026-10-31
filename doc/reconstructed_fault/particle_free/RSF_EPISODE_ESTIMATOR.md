# Residual-resolved continuous-history estimator

Experimental numerical repair; no physical law, residual tolerance, history or
minimum timestep changes. The legacy raw log-rate controller remains selectable.
The new option replaces its independent rate-jump rejection by continuous-history
local defects; it does not exempt points by damage or slip magnitude.

For the current admissible tangent J and nodal arclength mass M, the existing
positive symmetric-part check gives alpha in a mass-orthonormal constraint basis.
Reconstruct the complete force residual, including tied multipliers. Set
b_i=max(|r_i|,mass_i * 1e-8 Pa)+gamma_k sum(abs(actual force operands)), where
k=2(n+na)+4*nq+3 counts the assembled sum/product evaluation. This uses the
unchanged weak certificate's resolution. Define

    eta_F = sqrt(sum_i b_i^2/mass_i)
    U_q = sqrt(sum_i Q_qi^2/mass_i) eta_F / alpha.

This bounds the *linearized assembled* displacement uncertainty by coercivity.
It is not a nonlinear branch, constitutive stability or spatial accuracy proof.
Native bulk/source/contact/energy checks remain independent. No admissible
mechanical directions means zero displacement uncertainty.

Rates have intervals max(0,|V|-U/dt) to |V|+U/dt. Their log-interval separation
identifies definitely resolved rate changes. Overlap indicates an unresolved
instantaneous rate, never proof of rate accuracy. A conservative maximum of U is
stored as numerical diagnostic metadata on acceptance. Old fixed-geometry checkpoints bootstrap the current force budget using the
smaller saved/current coercivity for their first incoming-rate interval; physical
state is unchanged. This bootstrap is conditional on the assembled linearization,
not an independent reconstruction of a past nonlinear error. Subsequent restart retains the accepted numerical metadata.

For increment dj and incoming rate Vold, the BE/trapezoidal jump defect is
|dj-dt*Vold|/2. The aging defect is

    dtheta = theta+ - theta - dt/2 [f(theta,Vold)+f(theta+,V+)],
    f=(V0-|V|theta)/Dc.

Scale its linearized trapezoidal correction by the smaller endpoint age. Evaluate
the existing irreversible damage map at the trapezoidal jump as a diagnostic;
its difference from the candidate is the damage defect. Subtract only the above
algebraic uncertainty and the counted arithmetic allowance from defect magnitudes.
Also bound U/Dc and U/df explicitly by the continuous-history error budget, so
uncertainty cannot conceal unresolved *history*. Retain and strengthen the old
slip, log-age and damage increment checks by their uncertainty envelopes.

Native acceptance uses the largest continuous defect/resolution and the existing
continuous increment ratios. Smooth BE defects give a square-root timestep
proposal; increment bounds give a linear proposal. Raw and resolved log-rate
changes remain visible diagnostics. A finite rate jump at admission can coexist
with a convergent continuous history error; rejecting it independently for every
dt is not a temporal-error test. Signed reversal is detected by the jump defect
even when the old absolute-speed log indicator is zero.

Initial selected continuous error budget:5e-4 in dj/Dc, relative aging and absolute
damage. This is a declared numerical setting, not a claim of1% global accuracy.
One focused replay must measure temporal uncertainty. Physical energy and every
native acceptance check remain. No damage floor, retie/reset, extra damping or
lower timestep floor is introduced.
