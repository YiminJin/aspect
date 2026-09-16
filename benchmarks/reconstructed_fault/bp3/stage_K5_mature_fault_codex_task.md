# K5: implement a mature frictional-fault specialization

Implement an explicit opt-in constitutive mode for the pre-existing, permanently frozen BP3 fault that carries rate-and-state friction without the additional cohesive slip-hardening spring. The cohesion reports already establish that mechanism; proceed to the formulation and implementation change.

1. **Define the specialization.** Update `current_design.md` and `specification.tex` with its traction balance, tangent, and energy accounting. The mature-fault mode has no recoverable cohesive slip energy or evolving cohesive resistance. Retain bulk elasticity/Maxwell history, true normal-stress feedback, radiation damping, and the normalized slip transfer `chi=h/I_h`. Keep the existing cohesive formulation available for other modes. Do not infer maturity merely from “phase frozen,” or implement a damage-to-mature transition in this task. Apply the selected mode throughout this pre-existing fault, including prescribed vertices.

2. **Preserve the initial mechanical problem.** For comparison with the existing frozen-force diagnostic, use the pointwise effective background

   `tau_bg,new(s) = tau_bg,old(s) - C_star(s)`.

   Here `C_star` is the actual evaluated initial cohesive resistance already captured by that diagnostic, not the retained nodal C0. Represent this fixed background accurately at production quadrature coordinates; do not silently project the nonlinear snapshot into Q1. Preserve initial V, Theta, and bulk state. Treat the offset as initialization of prestress, not as an evolving cohesive force. Start from initialization, without converting a late cohesive checkpoint.

3. **Implement consistently.** Remove the cohesive force and its derivative only in the selected mode; preserve the bulk contribution to the slip tangent. Make initialization, accepted updates, outputs, and energy accounting reflect the absence of that spring. Do not publish growing shadow C/H as physical mature-fault histories or allow those histories to feed mechanics. Define the treatment of H from the reduced energy and keep phase evolution disabled for this mode. Use ordinary configuration/checkpoint handling for the mode and its fixed background; reject incompatible histories. Do not set the shared shear modulus to zero, retune I_h, change the 40-km prescribed region, or change Theta discretization.

4. **Verify with a limited budget.** Reuse existing diagnostic machinery. Check that repeated imposed slip creates no cohesive force or cohesive stored energy in this mode, check the affected residual/Jacobian, and confirm the original cohesive mode retains its existing behavior. Compare the new residual on saved fields with the frozen-force residual. Then run at most one fresh four-rank 50-m trajectory through `922804465.5975173 s` (29.2419 yr), using the recorded timestep sequence subject to unchanged production restrictions. Cap simulation wall time at 40 minutes; if a controller or budget prevents completion, report the last accepted comparison rather than retuning or restarting. No later continuation or parameter/refinement campaign.

Compare initial balance, nearby V, accumulated-slip gradient, and raw junction pressure/normal stress with the existing frozen-force reference at matched times. Investigate material discrepancies before claiming equivalence. Success means a consistent constitutive specialization and agreement with that reference; it does not mean every remaining junction or deep-tip anomaly is solved.

Deliver the implementation, concise formulation changes, and one short result report stating what passed and what remains unresolved. Preserve unrelated edits and existing evidence. Stop after this bounded task.
