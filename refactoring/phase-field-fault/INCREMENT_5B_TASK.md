# Increment 5B · Reproduce the two 18-strip midpoint trajectories

**Decision and scope.** Start from the ownership-cleanup baseline on `pf-rsf`, inspected at `2bbeaec54449f7270a91ce4868aad0e4cfb66941`. Use the ordinary ASPECT growing-fault material, its RSF law, the simulator-owned midpoint solve and the existing two-rank Release configuration. Reproduce the frozen 16 × 8 single-front specimen from the 5 ms common parent through 10 ms on both original timestep schedules. The observable for this increment is the reproducibility of the known timestep sensitivity. A conclusion about the physical timing of fault growth requires a separate experiment.

## 1. Identify the reference and fix the comparison contract

Read `AGENTS.md` and the 5A and ownership sections of `refactoring/phase-field-fault/{PLAN,STATUS,VERIFICATION}.md`. Record the actual starting HEAD. In the indexed reference archive, verify the archive SHA-256 `a82d362c3504be188e9602cf10c93cf51a26544be5e6d4a7677016cc52981c57` and its internal manifest. The documented location is `../aspect-pf-rsf-reference-20261121/reference.tar.gz`. Reuse a verified extraction or extract in fresh scratch. The archive README and run manifests are authoritative for the inputs, additional libraries, exact clocks, qualified tip executable and reducers.

| Reference under the extracted runs directory | Purpose |
|---|---|
| `localized-oneo1` | 18-strip preparation through 5 ms, including donor restart generation 02 |
| `localized-oneo2` | One accepted 5 ms interval from that parent to 10 ms |
| `strip18-midpoint-half` | Two accepted 2.5 ms intervals from that parent, with output at 7.5 and 10 ms |
| `strip18-audit-full`, `strip18-audit-half1` | Phase-row observations on the first update of each branch |

Run or reuse the verified archive reductions for `midpoint_timestep.py`, `midpoint_audit.py` and `midpoint_closeout.py`. Compare their results with the archive's expected timestep, point-audit and diagnosis JSON. Run historical reducers with Python assertions enabled. Direct all new outputs to fresh directories. The sibling donor is read-only. If an archived frame is absent, replay its indexed case or record precisely which observable could not be checked.

Write down quantity-specific tolerances and the row/point identity mapping before judging the target outputs. Use archived tolerances where supplied and justified physical scales for additional comparisons, particularly near zero. Use full-precision archived data for acceptance. The rounded table below identifies the intended comparison.

## 2. Reproduce the physical parent, then branch

Use two real MPI ranks, one numerical thread each, and at most four local workers. Match the source input's mesh, frame, seed, 18 complete strips, quadrature, steady loading and material parameters. Keep its original zero-clock initialization. Preserve the incoming Bernstein coefficients, including admissible negative coefficients. Produce the target 5 ms state from initial conditions. Compare target versus donor by stable identities: FE coefficients by cell/component/local basis, native records by cell/point/strip, and strip state by material ID. Include phase/obstacle, physical stress and cohesive strain including zz, both shear inventories, geometry and measures, formed angles, stage rate/resistance, slip, and constitutive ages. Inspect stage equilibrium and accepted clock. Localize any parent mismatch before interpreting downstream results.

Branch from this same target physical state. Prefer copies of its own accepted checkpoint when the ordinary lifecycle permits setting the correct next interval. If the checkpoint already contains a pending timestep that cannot safely be changed through existing facilities, run two deterministic schedules from time zero and verify the same 5 ms parent on both. A small test timestep plugin is appropriate. Never edit the checkpoint time or histories by hand. Keep the donor checkpoint as donor evidence. Verify actual attempted and accepted times from the simulator and report repeat/publication counts.

| Target branch | Parent → midpoint → endpoint in ms |
|---|---|
| Coarse | 5 → 7.5 → 10 |
| First half | 5 → 6.25 → 7.5 |
| Second half | 7.5 → 8.75 → 10 |

The coarse midpoint and the fine branch's first published endpoint both carry the clock 7.5 ms but represent different states. Label them explicitly. Verify the actual pending duration *before* forming wall increments, the Maxwell factor, RSF stage age and obstacle. Archive both branches' physical fields through the ordinary production path. Preserve the frozen midpoint update.

## 3. Check what caused the separation

- **Compare each target trajectory with its own donor trajectory first.** Check every available accepted frame, including 7.5 ms on the fine branch. Verify native tensors and mesh fields, per-strip rates, ages and slip, orientation and admission, independently accumulated transfer work, and accepted energy terms. Then compare the difference between branches. Both donor branches registered zero new sections in this window.
- **Inspect the irreversible phase rows.** Key rows by cell and local Bernstein index. Compare the incoming obstacle, stage coefficients, residuals and active/free classifications at the corresponding stages. The two donor updates directly from the 5 ms parent had 947 lower-active, 205 free and zero upper-active rows. Distinguish the stage KKT residual from admissible downhill increments at the extrapolated endpoint. The latter explain how another phase solve can change the result under steady loading. Preserve both the original active-set definitions and independent phase/energy observations.
- **Reduce physical increments.** Sum the section slip increments with the proper arclength measure on each interval. Compute the endpoint phase increment relative to the common parent with native physical areas. Independently reconstruct external work less the changes in bulk elastic, cohesive and fracture energies and less bulk viscous, cohesive viscous and friction/radiation dissipation. Report each interval, the signed sum and the sum of absolute defects. Report the cohesive compliance remainder and residual work separately. Compare added fracture energy on its own scale. Check the difference between the two endpoint phase fields as well as their peaks.

| Frozen 5–10 ms window | One 5 ms step | Two 2.5 ms steps |
|---|---:|---:|
| Integrated signed slip increment (m²) | 3.748901 × 10⁻⁵ | 5.800755 × 10⁻⁵ |
| Added fracture energy (μJ/m) | 5.550475 | 15.041375 |
| Spatial L² norm of total phase increment (m) | 0.0114891 | 0.0151855 |
| Signed physical-work defect (μJ/m) | +3.960087 | +6.782957 |
| Signed cohesive compliance remainder (μJ/m) | +3.960087 | +6.782315 |

The donor L² distance between the 10 ms phase fields was about 0.00742665 m, while the peak increments were both about 0.0798. A peak-only comparison would miss the separation.

## 4. Keep the evidence independent and the implementation bounded

Inspect the current nine-strip checker before reusing it: it embeds a cell width of 0.25 m, a fixed 0.005 s duration and a three-step loop. Use actual cell geometry, field identities and per-interval clocks for 5B. Keep the independent reductions separate from the code that assembles the residual. A deliberate change to one duration, area weight or intermediate frame in a temporary output copy should fail a discriminating check. Use existing benchmark/output infrastructure. Introduce a production change only for a reproduced defect, isolate it in its own change, and rerun the affected 5A checks. A mismatch is a result to diagnose; a reference number is never an implementation input.

If the repaired binary64 native-history guard fires, retain the failing parent, strip, root, reconstructed increment and storage bound. Check stage inputs and compare the corresponding donor state. A rejection before publication is meaningful evidence, not a qualified trajectory. Preserve the physical sign, work and retry contracts while investigating. Do not tune the seed, alter angles, equilibrate the endpoint, change tolerances to force agreement, or substitute a quarter-step or longer rupture run.

At handoff, record in `STATUS.md` and the verification guide the archive and target identities, exact runnable commands, full tolerances, MPI layout, timings, peak memory and solver iterations. Include separate rows for the common parent, each accepted interval and both 5–10 ms reductions, with donor values, target values and differences. Give the first divergent operation and remaining qualification if either trajectory fails. Recheck the nine-strip two-rank control when shared verification code changes. Stop at the 5B review before changing the propagation method or starting publication-scale examples.
