# Stage I Plan Revision Instructions

## Purpose

Revise `stage3_I_plan.md` before implementation. Keep the existing Stage-I architecture—coupled bulk/fault Newton solve, restricted surface solve, non-committing coupled residual evaluation, separate bulk/surface convergence, and manager-owned \(V\) lifecycle—but change the projected-Newton and line-search details below.

These revisions are intended to make the lower-bound treatment mathematically clear, keep the merit function fixed during each line search, and avoid silently accepting unsuccessful nonlinear steps.

---

## 1. Preserve the current high-level architecture

Keep the following decisions from the current Stage-I plan:

- solve directly for \(V\), not \(\log V\);
- enforce the physical/numerical lower bound
  \[
  V \ge V_{\min} > 0;
  \]
- use a restricted semantic surface solve for active fault vertices;
- let the condensed solver depend only on the semantic \(K_V^{-1}\) operation, not on raw matrix factors;
- use one non-committing coupled residual evaluator for explicit \((x,V)\);
- keep \(\Theta\), cohesive history, previous \(I_h\), and Maxwell stress frozen during the Stage-I mechanical solve;
- commit only \(V\) in Stage I;
- preserve dynamic- and adiabatic-pressure branches;
- use separate normalized bulk and fault residual blocks;
- keep the active set fixed during one linear solve and one line search;
- recompute the active set from scratch at the beginning of the next Newton iteration.

Do not introduce a new nonlinear variable transformation or a generic upper bound on \(V\).

---

## 2. Revise the bound-active-set criterion

The current plan uses a tolerance proportional to the global maximum slip rate,

\[
100\epsilon_{\mathrm{mach}}\max(V_{\min},\|V\|_\infty).
\]

Do **not** use the global \(\|V\|_\infty\) as the scale. Slip rate can vary over many orders of magnitude, so a large value elsewhere on the fault can incorrectly classify a much larger-than-\(V_{\min}\) node as being at the lower bound.

Use a local bound test, conceptually

\[
V_i - V_{\min}
\le
C_{\mathrm{bound}}\epsilon_{\mathrm{mach}}
\max(V_{\min},|V_i|),
\]

with a small fixed internal constant \(C_{\mathrm{bound}}\), for example 100.

A free node becomes bound-active when:

1. it is at the lower bound according to the local tolerance above; and
2. the recovered Newton direction points outside the admissible region,
   \[
   \delta V_i < 0.
   \]

The tolerance is an internal numerical constant, not a new user-facing parameter.

---

## 3. Keep the inner active-set solve monotone, but recompute it each Newton iteration

For one Newton iteration:

1. assemble/linearize the coupled system at the current accepted state;
2. start the projected solve with an empty active set;
3. solve the unrestricted condensed system;
4. detect lower-bound nodes whose recovered \(\delta V_i<0\);
5. add those nodes to the active set;
6. rebuild only the restricted surface solve and resolve the condensed system;
7. repeat until no new active node is found.

Within this inner projected solve, active nodes may only be **added**, not removed.

At the next outer Newton iteration, recompute the active set from empty. This allows previously active nodes to be released naturally if the new Newton direction points back into the admissible region.

This is a deliberate first implementation choice; do not add a more elaborate complementarity solver unless a concrete failure requires it.

---

## 4. Make the restricted \(K_V\) solve mathematically explicit

For a stabilized active set \(\mathcal A\) and free set \(\mathcal F\), the restricted solve must use the principal free block

\[
K_{\mathcal F\mathcal F}.
\]

The implementation may materialize a full-size matrix with identity on active rows **and active columns removed from the free equations**, i.e. conceptually

\[
\begin{bmatrix}
K_{\mathcal F\mathcal F} & 0\\
0 & I_{\mathcal A\mathcal A}
\end{bmatrix}.
\]

Do not merely overwrite active rows with identity while leaving active columns coupled into free equations.

The restricted solve must:

- project the right-hand side to the free set;
- return exactly zero increment on active nodes;
- preserve the existing nonsingular-indefinite UMFPACK strategy and backward-error diagnostics;
- remain invalidated by any new surface linearization.

---

## 5. Replace the current `0.99` fraction-to-boundary rule

Do **not** use

\[
0.99
\frac{V_i-V_{\min}}{-\delta V_i}.
\]

That keeps a node artificially in the strict interior and can make it approach the bound geometrically without ever reaching it.

Because the constitutive law is valid at \(V=V_{\min}>0\), allow a free node to arrive exactly at the bound.

For free nodes with \(\delta V_i<0\), define

\[
\alpha_{\mathrm{bound},i}
=
\frac{V_i-V_{\min}}{-\delta V_i}.
\]

Then use

\[
\boxed{
\alpha_{\max}
=
\min\left(
1,\;
\min_{i\in\mathcal F,\;\delta V_i<0}
\alpha_{\mathrm{bound},i}
\right).
}
\]

If no free node points downward, use \(\alpha_{\max}=1\).

After forming a candidate, if roundoff places a component infinitesimally below or above the bound, project that component **exactly** to \(V_{\min}\) using the same local tolerance. This is nonlinear-variable projection, not constitutive clamping.

The friction law itself must still reject \(V<V_{\min}\) and must not silently clamp its input.

---

## 6. Fix the active set during one line search

Once the projected Newton direction has been computed for the current Newton iteration:

- keep the stabilized active set fixed for all trial step lengths \(\alpha\);
- keep the corresponding projected surface residual definition fixed;
- do not recompute or change the active set for individual line-search candidates.

The active set is recomputed only after a line-search candidate is accepted and the next Newton iteration begins.

This keeps the merit function consistent during the line search.

---

## 7. State the Armijo acceptance condition explicitly

Use the joint merit function

\[
\Phi
=
\frac12
\left[
r_b^2+r_\Gamma^2
\right].
\]

For a candidate formed with step length \(\alpha\), accept it only when

\[
\boxed{
\Phi_{\mathrm{trial}}
\le
(1-c_{\mathrm{Armijo}}\alpha)\,
\Phi_{\mathrm{current}}
}
\]

with the existing Armijo coefficient

\[
c_{\mathrm{Armijo}}=10^{-4}.
\]

Start with

\[
\alpha=\alpha_{\max},
\]

then reduce it using the existing factor \(2/3\).

Every trial must be formed from the same accepted base state:

\[
x^{\mathrm{trial}}
=
x^{(k)}+\alpha\,\delta x,
\]

\[
V^{\mathrm{trial}}
=
V^{(k)}+\alpha\,\delta V.
\]

Rejected trials must not accumulate.

---

## 8. Do not silently accept the last unsuccessful line-search candidate

Revise the current plan here.

If no admissible candidate satisfies the Armijo condition within the allowed line-search iterations, report a **line-search / nonlinear failure**.

Do not accept the final unsuccessful candidate merely because the iteration limit was reached.

The timestep-repetition or outer ASPECT failure handling may later reduce the timestep or abort according to the existing configuration.

This is intentional behavior for the coupled reconstructed-fault solver, even if another ASPECT nonlinear path currently accepts the final trial.

---

## 9. Revise residual normalization floors

Keep block-specific fixed normalization scales during one nonlinear solve:

\[
r_b
=
\frac{\|R_{\mathrm{bulk}}\|}{S_b},
\qquad
r_\Gamma
=
\frac{\|P_{\mathcal F}R_\Gamma\|_\Gamma}{S_\Gamma}.
\]

However, do not guard an exactly zero initial residual using only the smallest positive floating-point number.

Use

\[
S_b
=
\max\left(
\|R_{\mathrm{bulk}}^{(0)}\|,
S_{b,\mathrm{floor}}
\right),
\]

\[
S_\Gamma
=
\max\left(
\|P_{\mathcal F}R_\Gamma^{(0)}\|_\Gamma,
S_{\Gamma,\mathrm{floor}}
\right).
\]

The floor must be tied to a meaningful block scale or an already existing ASPECT residual scale. Before implementation, identify the existing bulk residual scaling used by the current nonlinear Stokes solve and reuse it where possible.

Do not introduce arbitrary dimensional user parameters solely for these floors.

The normalization scales remain frozen throughout one nonlinear solve.

---

## 10. Convergence remains block-wise

Convergence requires **both** blocks to satisfy the nonlinear tolerance:

\[
r_b < \epsilon_{\mathrm{NL}},
\qquad
r_\Gamma < \epsilon_{\mathrm{NL}}.
\]

Do not replace this with a single combined merit-function convergence test.

The merit function is used for line-search globalization only.

For the surface block, evaluate the norm on the free residual:

\[
P_{\mathcal F}R_\Gamma.
\]

Active entries are zeroed only after the active set has stabilized for the current Newton iteration.

---

## 11. Clarify initialization semantics

`prepare_reconstructed_fault_mechanical_solve()` may initialize missing constitutive state only for a **fresh timestep-zero model**.

At fresh \(t=0\):

- recompute current transient \(I_h\);
- initialize cohesive history if required by the initial-condition construction;
- project and initialize the supplied positive \(\Theta_0\) for rate-and-state friction;
- initialize the numerical slip-rate iterate with
  \[
  V=V_{\min}.
  \]

On restart or at \(k>0\):

- preserve committed \(V\);
- preserve committed \(\Theta\), cohesive history, previous \(I_h\), and Maxwell stress;
- missing or partially initialized committed history is an error;
- never silently reconstruct later-time history from initial-condition data.

Stage I still does not evolve or commit \(\Theta\), cohesive history, previous \(I_h\), or Maxwell stress.

---

## 12. Keep a separate working bulk iterate

Do not destructively use the simulator's externally visible committed/current `solution` as line-search scratch.

Maintain a working accepted bulk iterate for the coupled nonlinear solve.

For each trial:

1. construct the trial bulk vector from the accepted working vector;
2. evaluate the trial residual non-committingly;
3. accept by replacing the working vector, or reject by discarding it.

Only after nonlinear convergence copy the working bulk state into the production `solution`.

On exception or nonlinear failure:

- restore/retain the pre-solve bulk state;
- roll manager \(V\) back to timestep-committed \(V\);
- leave all deferred constitutive history unchanged.

---

## 13. Revised solve / line-search algorithm

Use the following high-level algorithm.

```text
prepare mechanical solve

working_x = current production bulk solution
initialize/begin manager current V

compute fixed normalization scales from the first residual

for Newton iteration = 0, 1, ...:

    assemble A and complete bulk residual at (working_x, current_V)
    linearize surface system at exactly the same state
    freeze B coefficients

    active_set = empty

    repeat:
        solve condensed Newton system using current restricted surface solve
        recover delta_V

        newly_active =
            free nodes that are locally at Vmin
            and have delta_V < 0

        if newly_active is empty:
            break

        active_set += newly_active
        rebuild only the restricted surface solve
    end repeat

    project active entries of R_Gamma to zero

    if both normalized residual blocks converge:
        commit working_x to production solution
        commit manager current V
        return success

    compute alpha_max from FREE nodes only
    allow exact arrival at Vmin

    alpha = alpha_max
    accepted = false

    for line-search iteration = 0, 1, ...:
        trial_x = working_x + alpha * delta_x
        trial_V = current_V + alpha * delta_V

        project roundoff-level bound contacts exactly to Vmin
        never evaluate trial_V < Vmin

        evaluate complete coupled residual non-committingly
        using the SAME active set and SAME normalization scales

        if Phi_trial <= (1 - c_armijo * alpha) * Phi_current:
            working_x = trial_x
            accept manager trial V
            accepted = true
            break

        discard trial state
        alpha *= 2/3
    end for

    if not accepted:
        rollback manager nonlinear V state
        leave production bulk solution unchanged
        report nonlinear/line-search failure
end for

if Newton iteration limit reached:
    rollback manager nonlinear V state
    leave production bulk solution unchanged
    report nonlinear failure
```

---

## 14. Tests to add or revise

Keep the current Stage-I test scope and add explicit checks for the revised rules:

- local lower-bound tolerance is independent of a large slip rate elsewhere on the fault;
- a free node can reach \(V_{\min}\) exactly;
- no `0.99` strict-interior behavior remains;
- active columns are removed from free equations in the restricted \(K_V\) solve;
- the active set remains fixed across all trials of one line search;
- a node can be released when the next Newton iteration recomputes the active set;
- Armijo acceptance uses the explicit inequality above;
- exhaustion of line-search iterations produces failure rather than accepting the last candidate;
- rejected trials do not modify either the working bulk iterate or manager current/committed \(V\);
- failure leaves the production bulk solution and committed \(V\) unchanged;
- normalization floors remain finite and meaningful when one initial residual block is zero or extremely small;
- both residual blocks must converge independently;
- bound-active cases use projected/one-sided verification rather than centered finite differences across \(V_{\min}\).

Retain the existing end-to-end tests for:

- rate-and-state friction;
- rate-dependent friction;
- dynamic fault pressure;
- adiabatic fault pressure;
- multiple faults;
- one- and two-rank execution;
- indefinite but nonsingular free \(K_V\).

---

## 15. Stage boundary

Do not expand Stage I beyond the mechanical nonlinear solve.

Stage I may:

- initialize fresh-timestep-zero \(V\) and supplied \(\Theta_0\);
- solve the coupled mechanical problem;
- commit \(V\).

Stage I must not:

- evolve or commit \(\Theta\);
- evolve or commit cohesive history;
- advance previous \(I_h\);
- commit Maxwell stress;
- evolve the phase field.

Those remain Stage J.

After revising the plan with these rules, show the updated Stage-I plan for review before implementation.
