> Current update: the user-selected narrower continuation is recorded in
> [the coupled material report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md)
> and [energy/tip diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md). Complete mature
> material transport and heterogeneous coupled reaction are now implemented
> with bounded local evidence. Statements below that call those pieces missing
> describe the earlier stage boundary; its measurements/failures are retained.
> Broader Stage C and production readiness remain incomplete.

# Stage C execution record

User selection: Stage C native integration. Entry revision27178813cebd471b93a7a9308f41215b71b4a20b,
required worktree/branch with retained dirty spin work. Current live core controls
were copied and verified before rebuilding; original protected copies and RSF
checkpoints remain untouched. See evidence/transported_objective/stage_c/entry.json.

1. Integrate complete T/R into native mature mechanics with a discrete stress
   identical to published reconstruction. Start matched constants/fixed geometry;
   preserve legacy/default and use the existing Stress rotation selector.
2. Native independent residual/tangent/observer and accepted transaction/restart
   checks, preserving all old guards until the corresponding path is complete.
3. Extend material transport, moving profile/geometry and current coefficients
   with complete scalar/tensor state and actual source/transfer semantics.
4. Audit cohesive native remainder/lifting and evolving energy/shape coupling;
   isolate genuine constitutive conflicts, continue unaffected work.
5. Qualify required distributed/mesh transfer, reconcile reports and readiness.

For matched constants/fixed profile, let P be the common DGQ2 L2 projection,
B−=2chi Z−S, Rrot=exp(dt W−) and

    Tbar=P[Rrot T− Rrot^T],
    Rbar=P[Rrot(R−+B−)Rrot^T−B−],
    tau_frozen= Tbar−Rbar−B−.

Use beta tau_frozen+2eta_ve(D−chi V S) in all native residual/observer paths.
Publish T+=beta Tbar+2eta_ve D, R+=beta Rbar,
Z+=beta Z−+eta_ve V. On affine Q2 cells D lies in DGQ2, so published
T+−R+−B+ equals the solved point stress, including the localized R source.
This finite-step correction is first order and reduces to J R+rR=[W,B].
No source is computed by comparing the reconstruction to itself. Independent
component tests and Galerkin reference remain the validation target.

This simplification is not asserted for changing geometry or unmatched bulk/
surface coefficients: their V-dependent projection terms must be retained in
the actual residual/tangent or a converged declared outer iteration. Moving
and cohesive configurations stay guarded until those operations are complete.


## Measured execution status (3 November 2026)

1. Mature matched/fixed reaction residual, tangent, observer and complete T/R
   owner/schema6: implemented and locally verified; legacy control unchanged.
2. Actual-dt rejection, restart/repartition, restricted T/R AMR and AMR restart:
   pass. Explicit derived point/mass-cache invalidation added at transfer.
3. Distributed DG transport and layered relaxation: implemented operator checks
   pass at three resolutions, approximately third order. Material geometry,
   profile, surface scalar and obstacle publication is still pending; do not
   enable a transported native selector from these operator checks.
4. Cohesive uncut representation fails. Independent cut capacity comparison
   led to an implemented formed-piece/side Q2 owner retaining native/lifting
   state. Three-resolution component, forcing, checkpoint and zero-spin checks
   pass; short coupled phase/rejection/repartition pass numerically. Deformed
   shape-work coupling remains
   a separate governing-energy gate. Tip line search fails in legacy and half
   dt controls; physical energy remains above its1% fracture-relative screen.
5. Final transfer-cache and cut-history checks pass. Same-label material
   geometry staging now preserves properties/rates and invalidates maps, with
   serial/four-rank rollback/archive/shear checks. Full suite152cases/
   183170assertions passes. Reports/maps/specification distinguish this container
   from the still-missing complete material-step transaction.

Next selected implementation remains the complete moving-mature material-step
transaction, followed by native translation with explicit inflow and forced
rejection/restart. This plan is not a Stage C completion claim. No server or
protected-trajectory continuation is authorized by these results.
