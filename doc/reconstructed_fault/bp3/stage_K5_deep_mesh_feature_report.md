# K5: deep refinement/source-profile concentration

## Decision

Two nearby features must be distinguished. The larger uniform-sliding stress
concentration at **43.93 km** follows the bulk refinement edge when it is moved
to 48 km: its new position is **47.92 km**. This is direct causal evidence for a
bulk-mesh/profile effect, not the physical 40-km prescribed-slip junction.

The smaller **42.935-km** feature stays in place. It exists in uniform sliding
already at initialization, beside a separate coarse-cell intrusion into the
negative-normal wing of the diffuse profile. That intrusion was unchanged by
the edge-extension test. The source wing changes locally by approximately 12%
at 42.95 km, although the corresponding I_h variation is only about 0.02%.
Thus the small feature does not require RSF evolution and is consistent with
local transverse profile discretization. The one authorized causal test does
**not** separately prove the effect of removing that unchanged coarse patch.

No physical model, support, normalization, history transfer, solver tolerance
or production numerical method was changed. Stop here; do not conflate the two
features or respond by modifying the 40-km RSF junction.

**Follow-up with the supplied figures:** the visible small dipole below the
40-km junction maps to approximately **43.9–44.0 km**, i.e. the main deep-edge
feature above, rather than the separate 42.935-km wing feature. Consequently
the completed edge-shift experiment directly addresses the most visible small
concentration in these images. The 42.935-km finding remains valid but should
not be substituted for the feature the user was pointing out.

## 1. Accepted state and stress lifecycle

The trajectory comparison uses **accepted step 10**,
922804465.59751701 s = 29.24190894 yr. Step 11 from the preceding task was an
end-time remainder artifact and is not used.

The named `delta_p.png`, `tau_xx.png` and `tau_xy.png` were absent during the
initial audit and subsequently supplied and inspected. Their colour-bar extrema
agree, at displayed precision, with the saved step-10 VTU extrema:

| Field | Minimum (Pa) | Maximum (Pa) |
|---|---:|---:|
| delta_pressure | -3852812.75 | 4262437.5 |
| tau_xx | -2569602.75 | 1342401.0 |
| tau_xy | -2481435.5 | 1683265.875 |

The images have no visible time annotation, so this is agreement with the
step-10 data, not independent timestamp metadata. In `delta_p.png`, the physical
100-km square occupies approximately pixels x=197–783, y=49–635. The small blue
dipole at approximately (530,272) coincides with the predicted image position
(530.31,271.87) of the 43.930463-km uniform edge extremum. The separate
42.934745-km wing coordinate would be at (536.04,268.43), only a few pixels away.
The stronger feature at approximately (542,252) is the physical 40-km junction.
This raster localization is only precise to about a pixel (~0.17 km in the
physical box); the saved quadrature coordinates provide the precise locations.

The saved data and writer establish the stress definitions:

- `bulk_10_*.vtu` is written from `get_solution()` in `BP3Output::write_bulk_state`.
  `delta_pressure` is the accepted pressure perturbation at step 10.
- Its `tau_xx`, `tau_yy`, `tau_xy` are the transferred compositional **old FE
  Maxwell-history inputs**. They are not a newly evaluated current stress tensor
  and not the newly committed particle array. In a real step k, mechanics
  consumes the history committed after k-1, after ordinary FE transfer and
  mechanical constraints.
- `work_qp_10_rank*.csv` instead evaluates current constitutive stress using
  the accepted u,p,V and still-frozen working FE old history:
  `tau = 2*kappa*(sym grad u - chi*V*S) + beta*tau_old_FE`.
  Total constitutive normal traction is `50 MPa + p - tau:N`.
- New particle history is independently stored in the accepted history exports;
  it is not recycled through a second Maxwell update for this diagnostic.

New `work_step10_published_fields.png` labels those distinctions explicitly.
It displays 35–48 km, with a marker at the small uniform-feature coordinate.
Published pressure/history CSVs cover 35–52 km. Saved current work-QP exports
cover 37–43 km here; no current stress is invented beyond their saved window.
The uniform-sliding raw constitutive exports cover the full expanded window.

## 2. Exact location and profile/mesh evidence

The persistent small pressure extremum in uniform sliding occurs at

- down-dip coordinate **42934.7446707 m**;
- physical `(x,y) = (57855.8251140, 62554.3311360) m`;
- normal coordinate **r = -526.1785484 m**;
- cell **`0_11:30210100002`**, quadrature index **2**, h=48.828125 m.

It sits immediately beside 97.65625-m cells in the profile wing, notably
`0_10:3003232222` at (s,r)=(43021.331706,-582.523926) m and
`0_10:3003232223` at (42972.503581,-667.096720) m. Along the fault center the
bulk cells remain 48.828125 m through approximately 44 km. Looking only at
center resolution would miss these transverse coarse cells. The surface
elements are 50 m in this region; their change back to 99.960336 m is at
**43998.413448 m**, not at 42.935 km.

At the same physical QP in accepted work step 10:

| Quantity | Value |
|---|---:|
| phi | 0.09743061240 |
| interpolated I_h | 12962.5609221 m |
| actual chi | 6.08259350e-5 /m |
| p | 45688.7989 Pa |
| current tau_xx / tau_yy / tau_xy | 390901.052 / -388211.191 / 257479.123 Pa |
| current tau:N | -26860.4699 Pa |
| total sigma_n | 50072549.2687 Pa |

Those absolute late-time values include the broad evolving RSF stress field.
They are not a measured isolated amplitude of the small discretization feature.
For example the raw maximum over an entire 42–43-km window is influenced by
the 40-km junction's wider field and is not a reliable locator of this small
off-center feature.

### Full transverse source, not only its integral

The offline reconstruction uses actual saved Q1 phase corner values and the
actual surface I_h, retaining the production normal half-width
790.5832804 m. Interpolation is checked against double-precision production
QP samples: max phase discrepancy **2.79118e-8**, consistent with Float32 VTU
storage. Profiles are sampled every 2 m in r and every 50 m along the fault;
their trapezoidal integrals are descriptive diagnostics, **not** new production
quadrature or normalization acceptance tests.

A same-location, fully fine 48.828125-m Q1 interpolation of the same prescribed
distance profile is used as an offline source-shape control, with the **same
actual I_h denominator**. This does not renormalize the source or change a run.

| s (km) | r of largest shape difference (m) | actual chi / fine-grid chi (/m) | local difference |
|---|---:|---:|---:|
| 41.00 control | 20 | 2.73540259e-3 / 2.73540230e-3 | 1.07e-7 relative |
| 42.90 | -738 | 3.23958012e-6 / 1.92020889e-6 | +68.7%, in the weak tail |
| 42.95 | -654 | 1.47828574e-5 / 1.32279296e-5 | +11.75% |
| 43.00 | -570 | 4.07119231e-5 / 3.87938193e-5 | +4.94% |

At 42.95 km, I_h=12962.957132 m versus approximately 12960.24 m in the fully
fine portion. The absolute source-shape L1 difference is 1.9437e-4, while
the diagnostic supported integral is 0.99998233. The source therefore has
a localized wing distortion even when its integrated normalization is nearly
one. Its second normal moment rises from about 31590.61 m² at 41 km to
31673.06 m² at 42.95 km. No center-value or integral-only explanation is used.

At the larger 44-km transition, I_h changes from approximately 12960.24 m to
12692.5 m and the source shape changes much more substantially. The shifted
mesh moves that large transition to 48 km, while preserving the small
42.9–43-km wing defect to roundoff.

## 3. Saved uniform sliding and the single causal test

Saved paired-boundary uniform sliding prescribes Vp everywhere and starts with
zero perturbation-stress history. Its small feature is fixed in space:

| Accepted state | time (s) | p at s=42934.744671 m (Pa) |
|---|---:|---:|
| initialization | 0 | 145.29437 |
| step 1 | 2487214.20566524 | 90.34455 |
| step 2 | 4970143.26691824 | 179.51550 |

Thus local RSF evolution is unnecessary for existence of the feature. Its
amplitude need not remain constant as the Maxwell interval/history changes.

The single new test extends the **bulk** refined strip to 48 km. It preserves
the original surface grid, background, physical 40-km junction, uniform Vp,
material, pressure treatment, phase law, two timestep lengths and both boundary
corrections. Fresh FE phase and projected I_h are recomputed normally. The
outside-box completion table is reused only after confirming identical top and
bottom cells, fault coordinates and profile law: this local interior refinement
does not change its input geometry. Its content is byte-identical.

Realized cells: 42880 → **47140**. Requested cells: 46909; ordinary deal.II
grading introduces 297 descendant leaves in place of requested parents. Changed
leaf centers span s=43.356–48.032 km and r=-5.032–2.991 km (including off-strip
grading). Top/bottom meshes are identical. The entire 42.3–43.4-km neighborhood
of the small feature has unchanged coarse/fine cells. Surface coordinates,
Vp, Theta evolution, C and accumulated slip agree with the saved uniform case.

| Diagnostic | Original 44-km edge, step 2 | Extended 48-km edge, step 2 |
|---|---:|---:|
| maximum absolute p in 43.5–44.5 km | 8422.214 Pa | 54.451 Pa |
| dominant deep-edge p minimum location | 43.930463 km | 47.915901 km |
| p at that minimum | -8422.214 Pa | -8139.028 Pa |
| small 42.934745-km pressure feature | 179.516 Pa | 139.683 Pa |

The old-edge peak falls **99.35%** and a comparably sized dipole appears at
the new bulk edge, despite the surface spacing transition remaining near
44 km. At initialization the minimum similarly moves from 43.930463 to
47.925433 km. This distinguishes the main bulk refinement effect from the
unchanged physical junction and surface grid. The smaller feature's modest
amplitude change with fixed position can include the changed long-range bulk
solution; it is not evidence that its local coarse patch was repaired.

## 4. Termination guard, correctness and cost

The benchmark-only termination plugin `BP3 replay complete` reads the final
time from the existing saved clock. It checks a marker published **only after
the accepted postprocessor/history checks finish**, allowing a remainder of
8*machine-epsilon*max(1,|target|). It changes neither a timestep nor any Newton
or history decision. Failed/trial states cannot trigger it. `run_work_replay.py`
now selects it for future prepared runs; old artifacts are not rewritten.

The standalone C++ test covers the captured two-ULP difference, equality,
overshoot, a physical 1e-4-s gap, a preceding state, and nonfinite/unaccepted
markers. All pass. The causal run accepted **only steps 0,1,2**, and emitted the
completion marker with no `Timestep 3` solve. The first marker printed the
simulator's already-incremented next step index; the message now uses the
stored accepted step index instead. This logging-only follow-up was rebuilt
without rerunning the case or changing the tested guard predicate.

```sh
g++ -std=c++17 -Wall -Wextra -pedantic benchmarks/reconstructed_fault/bp3/test_replay_stop.cc -o /tmp/bp3-test-replay-stop
/tmp/bp3-test-replay-stop
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
MPLCONFIGDIR=/tmp/aspect-work-replay-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_deep_mesh_feature.py --with-shift
```

Six fresh linear checks pass. Each accepted bulk/surface residual is below
the existing nonlinear criterion. Uniform prescribed V is exact, no free/lower
active nodes exist, and Theta-reference errors are at most 2.22045e-16.
Initialization retains zero stress; the first and second updates publish
nonzero stress normally. Paired source/coupling checks retained in the uniform
fixture also complete. No stress/history/solver correction was introduced.

Run command:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_deep_mesh_shift.py prepare
python3 benchmarks/reconstructed_fault/bp3/run_deep_mesh_shift.py run
```

One four-rank run, initialization plus two steps: **148.302 s**, cap 600 s,
no retry. Child peak RSS **1752832 KiB (1.672 GiB)**, not an aggregate four-rank
peak. Build used -j4. No long RSF replay or additional altered-mesh case was run.

## 5. Artifacts and next decision

- `benchmarks/reconstructed_fault/bp3/deep-mesh-shift48-uniform-local4/`:
  exact input/mesh, clock, completion data, provenance, accepted states and log.
- `benchmarks/reconstructed_fault/bp3/deep-mesh-feature-analysis/`:
  `summary.json`, `current_stress_peaks.csv`, `current_stress_bins.csv`,
  `source_shape_checks.csv`, actual transverse profiles and column diagnostics.
- `work_step10_published_fields.png`: accepted p versus explicitly labelled
  old-history tau_xx/tau_xy on 35–48 km.
- `*_overlay.png`: bulk cell size, surface spacing, I_h, source integral and
  current pressure extrema; `*_transverse.png` and `*_wing_defect.png` show
  the source shape, including the weak wing.

The saved uniform output was restored from the recoverable archive; it was
not rerun. Offline VTK sampling uses a cell locator for the disconnected AMR
patches and double arithmetic on the original Float32 nodal values; it is
checked against saved QPs. No production lookup was modified.

**Recommendation:** record the main edge concentration as causally demonstrated
bulk profile-discretization sensitivity. For the user's smaller 42–43-km
feature, retain the narrower conclusion: present without RSF, fixed beside
an unchanged transverse coarse-cell intrusion, with a quantified source-wing
defect. A future targeted test would remove only that intrusion while preserving
the rest of the grid; this task does not authorize/perform another run. Neither
observation justifies changing RSF physics, smoothing I_h, or modifying pressure.
