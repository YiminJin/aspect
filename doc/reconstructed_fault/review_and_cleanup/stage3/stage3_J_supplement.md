You are re-asking scientific decisions that have already been resolved in this planning session, and in at least one case your new recommendation contradicts the previously approved decision. Stop generating further decision questions for now.

First, collect every Stage-J decision already made in this session and write them into the authoritative current_design.md / specification.tex (or a temporary stage_J_decisions.md reviewed against those files) so they become durable context. Then reread that decision record before asking any further question.

In particular, preserve these resolved decisions:

Timestep ordering:
$$ H_{k-1}\rightarrow\phi_k\rightarrow(u_k,p_k,V_k) \rightarrow\text{history}_k,H_k\rightarrow\phi_{k+1}. $$
Crack-driving force: use the finite-step cohesive-work expression, not the small-step approximation or elastic-stress-only model.
Cohesive history: use projected surface-mixture coefficients consistently in mechanics and commit; bulk Maxwell/B/G retain local bulk coefficients.
Commit \(T_k^{\rm coh}\) through the consistent Q1 projection; use the projected Q1 traction, interpolated back to particle coordinates, for the particle \(H\) update.
Connect the existing law-specific RSF timestep restriction to ASPECT timestep selection; rate-dependent friction contributes no current law-specific restriction; do not add post-solve RSF cutback/repeat yet.
For the finite-step \(H\) expression at an exactly intact current point:
$$ g_k=1,\ h_k=0,\ h_{k-1}=0 $$

use the removable-limit value

$$ \mathcal H_k= \frac{\Delta t_k}{2\kappa_{\Gamma,k}} (T_k^{\rm coh})^2, $$

not zero. If \(g_k=1\) but \(h_{k-1}>0\), diagnose inadmissible healing.

Do not re-ask any of these decisions unless inspection reveals a concrete contradiction with the implementation or theory. If such a contradiction appears, quote the conflicting equations/code and explain it rather than reopening the decision generically.
