Current revision: [BP3 server startup review](BP3_DG_STARTUP_REVIEW.md) adds the
collective wall-stop repair, four validated inputs, bounded startup procedure and
regenerated source package. The original handoff and archive identity below are
retained historical evidence.

---

# DG BP3 first-event source package

17 October 2026. User-selected source handoff; no remote/full-event launch.
[Build/run instructions](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/README.md),
[verification](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/VERIFICATION.md),
[source audit](SOURCE_AUDIT.md).

The distribution supplies actual ASPECT sources (including all uncommitted core
work), bundled World Builder/Catch, regression/unit tests, maintained ordinary
BP3 and accepted-history observer sources, the CMake adapter, standalone fresh/
resume inputs, ordered fault fixture, environment settings, qualification summary,
source diff and checksums. Installed compiler/MPI/deal.II/Voro dependencies are
site supplied; no workstation executable/shared library or run controller ships.

The documented build paths match `$WORK/build/aspect/pf-rsf-dg/build` and
`$SCRATCH/bp3_dg/plugin/build`; CMake exposes the requested direct runtime command
`ibrun ./plugin/aspect-release parameters.prm` from `$SCRATCH/bp3_dg`.

Native adaptive controls are explicitly selected: first/max physical dt 4e6/4e7 s,
maximum logarithmic state change .2, native nonlinear cutback; separate initial
constitutive interval 4e6 s. Stop on native first-event completion, 1500 years or
82800 s accepted-state wall budget. Checkpoint every ten accepted steps and on
graceful termination. Unlimited DGQ2, Gauss6/source3, Helmholtz20, fixed physical
phase/geometry/mesh and zero mature cohesive resistance/storage remain unchanged.

The only runtime C++ change is observer IO: optional checkpointed dense-output
scheduling reuses native time/slip/event helpers; all checks/profiles/root weak
loads remain per step. Existing outputs are unchanged by default. A fresh restart
branch receives correct CSV headers. Regression: 36 default fields and 20 matched
scheduled fields are byte identical; step-three-to-five restart has zero persistent
history, full-tensor stress or weak-load difference against uninterrupted stepping.
Native short tests use two ranks. Full-grid first-event evolution, numerical
accuracy and server scaling remain unqualified; neither backend is the accuracy
reference and `strict_reference_match_passed=false` is preserved.

The exact archive location, hash and fresh distribution checks are recorded with
the delivered package. This changes the selected experiment's run caps, not the
constitutive equations or production defaults. All earlier pilot/reference
reports, archives, checkpoints and original errors remain untouched.
