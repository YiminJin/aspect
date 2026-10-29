# P5c — finite-seed fixed/radial tensor comparison

Completed 22 October 2026, on f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work. **Radial flow delays and reduces growth in the existing diffuse cap; neither mode develops a new crack or branch in this interval.** The measured difference exceeds the focused temporal uncertainty. It is a comparison of constitutive models, not an accuracy ranking.

## Baseline and initialization

The archived P4 resolved inputs, fracture-parameters.csv **and actual step127 checkpoint** agree: cohesion2000 Pa, Gc1 J/m², ell=.046875 m, G1e6 Pa, eta1e8 Pa s, residual R1000 Pa. Thus Kf=Gc/((8/3)ell)=8 Pa and m=Kf/(cohesion²/(2G))=4. P5b's m16 attribution missed fracture.prm's override. Reports/extracted metadata are corrected; the original ZIPs and excluded preliminary m16 run remain unchanged. [Explicit comparison table](parameters.csv) and baseline-correction.json include checkpoint hashes and serialized values.

Both modes start freshly with zero histories and the same continuum seed, peak .6, endpoint (1,.5)m and compact cap radius .0926465m. Initial phase files are bitwise identical; nodal prescription error1.11e-16, L² representation error **0.591954%**, phi in[0,.6]. The first unloaded step has exactly zero phase increment and zero phase KKT residual. Initial fracture storage is .569413700 J/m.

The finite body is 1.5×1m, 96×128 cells: Δx=.015625m, Δy=.0078125m; 12,288 cells, 372,233 total native DoFs (124,228 velocity/pressure/phase), plus 331,776 independent stress DoFs. Retain Gauss8/source3, unlimited DGQ2, native FE source complement, homogeneous coefficients and all tolerances. Released lateral horizontal motion/vertical restraint and top/bottom shear grips are unchanged. Friction pressure10000 Pa is distinct from solved mechanical pressure. No symmetry constraint or phase-growth mask is imposed.

As [declared before execution](P5C_SPECIMEN_SELECTION.md), rest ends at .05s, shear rate .018/s continues to **.140s**, then displacement holds to .145s. This ends loading earlier than P4's .175s experiment; history is identical up to .140s. Main Δt=.00125s. Only .11875–.145s is repeated at .000625s from each mode's unchanged accepted step95 checkpoint. Native checkpoint scheduling explains the .11875s split; clocks/histories are never edited.

## Prerequisite checks and implementation

Exactly-intact native controls recover Y=m(max(|Txy|-R,0))²/(2G) to 3.00e-15 relative to Kf. Both activate in (.200,.2125]s at the 3000Pa pure-shear threshold and accept phi=.292522 at .300s.

Uniform 8×8 non-proportional controls generate nonuniform tensor histories. Across 65 states on 1/4 ranks, every DG coefficient, native FE memory coefficient and point A/K/Q/W/L record agrees within **5.65e-14 fixed / 6.41e-14 radial**, using the unchanged1e-12 comparison bound. Actual rejection/retry leaves accepted state unchanged; same-rank restart reproduces the next complete records, phase, physical means, reactions and phase forces **bitwise** in both modes/rank counts. Publication occurs once.

The only production-source change removes the tensor parser's serial restriction. Existing unique CellId ownership, checkpoint exchange and the distributed no-hanging-phase guard remain. The reused observer now exports full-tensor physical stress, energy, driving, rates and rejection fingerprints. Scalar defaults, point law, solver/tangent, acceptance and 1024-master stability limit are unchanged. See changed-files.txt and SOURCE_AUDIT.

## Measured physical response

| Quantity at .145s, main Δt | Fixed tensor | Radial tensor |
|---|---:|---:|
| Onset bracket [s] | (.13375,.13500] | (.13500,.13625] |
| Maximum new phase | 0.00335139 | 0.00198276 |
| Integrated cap growth [m²] | 1.8687868e-06 | 1.1992414e-06 |
| Added fracture storage [J/m] | 5.2531174e-06 | 3.1583932e-06 |
| Centerline phi=.01 front [m] | 1.087472866 | 1.087107650 |
| Top horizontal reaction [N/m] | 2179.509555 | 2176.037368 |
| Full-run wall time, 4 ranks [s] | 841.60 | 891.05 |

Radial flow reduces integrated growth by **35.83%** and peak increment by **40.84%**. Full plane-strain stress differs by **1.221%** in whole-domain L²; reaction differs by .159%. Active cell-average tensor angle alpha=atan2(Wxx,Wxy) reaches ±21.06° in plotted loading states; fixed flow has alpha0. This is a tensor-component angle, not a crack tangent or finite-rotation qualification.

Both first grow at x=1.078125m and y=.5,.5±.0078125m. Final growth occupies seven nodes in one axial column, y=.5±.0234375m, with reflection defects below 7e-18. No initially intact node activates, no disconnected component forms and phi≥.1 fronts remain unchanged. The phi=.01 front starts at1.086496237m: advances are only .977mm/.611mm, **inside the pre-existing cap**. The hold retains nonzero stress/history; it adds no growth or flow, without imposing such a rule.

At .13375s both phase fields and fracture resistance are identical. The first center node's mass-averaged resistance is2.41753Pa; actual weak mechanical driving is2.41174Pa fixed versus2.09923Pa radial. Additional tensor flow changes stress/history redistribution and lowers local phase driving before phase changes. These are native weak-force averages, not constitutive values reconstructed from averaged histories; see pregrowth-nodal-balance.csv.

## Temporal, stability and work limits

| Focused halving result | Fixed | Radial |
|---|---:|---:|
| Finer onset bracket [s] | (.133750,.134375] | (.135625,.136250] |
| Maximum common-time growth-relative L² difference | 0.00596% | 3.19095% |
| Endpoint growth-relative L² difference | 0.00149% | 0.23078% |
| Half-window wall time [s] | 568.64 | 594.37 |

Both meet the predeclared 5% interpretation target. The common prefix is not refined; two timesteps do not establish an order or qualify the later rapid event.

All four runs publish exactly once per accepted interval and write valid endpoint checkpoints. Maximum source/mechanical/KKT defects are7.10e-15/5.69e-12/9.52e-8 against1e-12/1e-9/1e-7; absolute BE ledger defect≤6.55e-14J/m. Nonempty mechanically relaxed curvature stays≥41.7597, with at most7 critical masters and no stability retry. Maximum rank peak RSS is2.54GiB. All 41 protected files remain unchanged; particles110592 remain fixed with exact integer ID/coordinate fingerprints.

Independent reaction endpoint work agrees with native work within6.7e-16J/m. Fixed/radial endpoint work is1.9149543/1.9140730J/m, trapezoidal work1.8904336/1.8895913J/m; storage changes1.6398905/1.6352609J/m and physical dissipation.2505788/.2543666J/m. Endpoint remainders.0244850/.0244455J/m track the BE numerical terms. Trapezoidal remainders are−3.567e-5/−3.615e-5J/m; sums of absolute step remainders3.5671e-5/3.6151e-5 show no large cancellation. Mechanical/phase residual work is recorded separately. Ledger closure is not physical-energy qualification, and the shared coarse prefix prevents a full-trajectory temporal-energy claim.

## Figures and conclusion

![Loading tip: common scales](fields-0112-tip.png)

![Whole body after hold](fields-0116.png)

![Growth, reaction and physical work remainder](response-energy.png)

[One equal-aspect animation](comparison.gif) shows the whole body, including initially intact/remote regions. White contours are .01/.05/.1/.2/.4/.6; dashed black contours are initial, red crosses mark the prescribed endpoint. Stress is reconstructed physical deviatoric tau, not auxiliary T. Y and W are averages of actual Gauss8 physical values; blank direction means no defined active mean flow. The continuum seed/loading were imposed; coupled mechanics/history/phase were calculated. Crack branching, developed extension, general spatial/quadrature accuracy and objective finite-rotation physics remain untested. Exact plotting arrays, physical-kernel line cuts, units and contour definitions accompany the PNG/GIF.

**P5c passes within this bounded early-response scope.** The tensor law materially changes cap driving while keeping the early response straight and locally stable. Growth spans only one axial node column; initial representation error and spatial sensitivity remain relevant. Recommend one targeted fresh **192×128 comparison through the same .145s endpoint at Δt=.000625s**, to test the constitutive difference under axial cap refinement before a longer directional specimen. Do not begin it automatically.

P4's failed 8.302% temporal/7.442% spatial targets, 57.26% boundary sensitivity and **strict_reference_match_passed=false** remain. This is a finite physical specimen, not a distant-boundary approximation. BP3, production defaults and old failures are protected. AMR, distributed hanging phase, transport, geometry evolution, RSF and general propagation are outside this qualification.
