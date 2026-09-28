  # Stage E — Generic Fault Friction

  ## Summary

  Extend `MaterialModel::Rheology::FaultFriction` to support the approved stateful rate-and-state law and a stateless rate-dependent weakening law. Stage E remains
  constitutive-only: it will not add fault-surface residuals, Theta storage, solver coupling, or lifecycle hooks from Stage F or later.

  The stateless law is
  \[
  \mu(V)=\mu_d+\frac{\mu_s-\mu_d}{1+V/V_c},
  \qquad
  \frac{d\mu}{dV}
  =-\frac{(\mu_s-\mu_d)V_c}{(V_c+V)^2}.
  \]
  Material fractions arithmetic-average \(\mu_s\), \(\mu_d\), and \(V_c\) before evaluating the law.

  ## Implementation Changes

  - Preserve the existing rate-and-state formulas, regularization, fixed-state derivative, exact aging update, slip-rate clamping, and timestep restriction.
  - Remove the parser rejection for Friction law = rate dependent.
  - Reuse Reference friction coefficients as:
      - \(\mu_0\) for rate-and-state friction;
      - \(\mu_s\) for rate-dependent friction.

  - Add composition-aware mapped-list parameters:
      - Dynamic friction coefficients, default 0.4;
      - Characteristic weakening slip rates, default 1e-6 m/s.

  - Validate selected-law user data at parsing:
      - common slip-rate bounds are finite, \(V_{\min}>0\), and \(V_{\max}\ge V_{\min}\);
      - rate-and-state scales are positive and its coefficient lists are admissible;
      - rate-dependent values satisfy \(V_c>0\) and \(0\le\mu_d\le\mu_s\) componentwise.

  - For the rate-dependent law, apply the existing common \(V_{\min}/V_{\max}\) clamp and return std::numeric_limits<double>::max() from `compute_time_step()`, because no state variable is evolved.

  - Keep the implementation as one small selector class; do not create a friction plugin hierarchy or duplicate state in `ReconstructedFaultManager`.

  ## Interfaces and Documentation

  - Preserve the existing stateful interfaces:

    friction_coefficient(fractions, V, theta)
    friction_coefficient_derivative_wrt_slip_rate(fractions, V, theta)
    update_state(V, theta_old, dt)

  - Add stateless overloads without a dummy theta:

    friction_coefficient(fractions, V)
    friction_coefficient_derivative_wrt_slip_rate(fractions, V)

  - Make calling a stateful operation for the stateless law, or a stateless overload for rate-and-state, fail with an explicit API-precondition diagnostic before
    accessing law-specific data.

  - Keep has_state_variable() as the solver-facing dispatch operation. Do not add a public selected-enum accessor.
  - Update current_design.md and specification.tex with the exact equation, derivative, mixture rule, parameter meanings/defaults, overload contract, and absence of a
    stateless timestep restriction.

  ## Test Plan

  - Add quantitative rate-dependent coverage for:
      - low-, characteristic-, and high-slip-rate values;
      - composition-averaged (\mu_s,\mu_d,V_c);
      - the exact negative analytic derivative and centered finite-difference agreement;
      - zero weakening when (\mu_s=\mu_d);
      - has_state_variable() == false;
      - no timestep restriction;
      - rejection of stateful-only calls.

  - Add rate-and-state regression coverage for:
      - unchanged regularized friction values;
      - unchanged fixed-state derivative and finite-difference agreement;
      - unchanged exact aging update and timestep restriction;
      - has_state_variable() == true;
      - rejection of stateless overloads.

  - Replace the obsolete “rate dependent is unsupported” expected-failure fixture with a successful rate-dependent integration fixture.
  - Add focused invalid-parameter fixtures for (\mu_d>\mu_s) and nonpositive (V_c).
  - Build debug and release with -j4; run the reconstructed-fault unit-test tags and focused phase-field-fault friction integration tests. Do not run the complete ASPECT
    suite unless requested.

  ## Assumptions and Boundaries

  - Existing uncommitted Stage C/D consistency work remains intact and is not folded into an unrelated rewrite.
  - Arithmetic parameter averaging is authoritative; per-material friction responses are not averaged after evaluation.
  - Reference friction coefficients intentionally has law-dependent notation to avoid introducing a duplicate static-friction parameter.
  - Stage E adds no persistent Theta fault property and no initialization, commit, rollback, surface residual, or mechanical coupling. Those remain later-stage
    responsibilities.

  - Changes will follow the existing ASPECT formatting and assertion policy without unrelated cleanup.
