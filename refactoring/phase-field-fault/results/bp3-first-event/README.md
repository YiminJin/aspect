# Historical BP3 first-event handoff

**Ready for bounded Stampede3 qualification.** The production input spans −60 to +90 km horizontally and 0 to 50 km vertically. All 5,776 IDs from surface to base (57.735026919 km down dip) solve the spatial RSF law. There is no prescribed deep-creep extension. Use the [standalone package](bp3-first-event.zip) and the [fresh, restart and continuation commands](../../../../benchmarks/bp3_qd_thrust/README.md). The full mesh was built on one and two local ranks. The largest local physical solve used the entire historical box on an intermediate mesh. No cluster job or branch-native earthquake is claimed.

The supported **Maxwell initialization interval is 100 s**. The historical 4e6 s value exhausted the specified Picard iterations at time zero on the resolved contact control. The first **physical** step ceiling remains 4e6 s. The native bulk-field MPI comparison at the requested 1e−9 linear tolerance has a measurable exception below. Preserve it when interpreting server qualification.

## Source and input identity

The starting revision was `cfcec4ad0a` on pf-rsf, including spatial log-age interpolation from `264aedc32`. There were no incoming tracked changes. Incoming untracked research material was preserved. Production changes are `f3b13efc0`, benchmark sources and inputs `48dae7aca`, and standalone checks `027e09a75`. Configure the server Release installation from `027e09a75` or the identical scientific sources in the handoff commit. Local CMake compiled the changes before the review commits were created, so the executable banner still reports the starting revision. The exact binary SHA-256 is `3fc422597ecfdbd6f321022320e1c114cd36d866b7a456e47b15d50a7d5d9ba7`. The local build used deal.II 9.6.2 (6890ca740b) and Trilinos 14.2.0. [Numerical run manifest](local-build.sha256) · [final extracted stage](shipped-stage.sha256) · [ZIP, table and input hashes](delivery.sha256).

The attached `parameters.prm` is byte-identical to the initial server run's `output_start/original.prm`. The donor's five-anchor `fault.txt` SHA-256 is `5460e34da479ef16e7dcc95f62669250a891ee8c969df5064dd3e4320bea4a24`. Read the archived donor code with `git show abbc9b0c:<path>`, specifically `source/simulator/phase_field.cc` (PhaseFieldProfile), the server plugin's `profile.cc`, `geometry.cc` and `mesh.cc`, `source/reconstructed_fault/manager.cc` (fixed per-segment resampling and endpoint association), `source/material_model/phase_field_fault/boundary_completion.cc` and `include/aspect/reconstructed_fault/normal_filter_internal.h`. The historical donor profile at time zero is an observation from its own run. The branch starts from initial conditions.

* The donor AT1 first integral uses φ = 0.6 cos²θ and h/m = φ(1+φ)/(1−φ)². The benchmark constructs the 5,000 radius knots, interpolates φ linearly, integrates h and prescribes both components of −Vp times the centered normalized primitive along the down-dip tangent. The positive-normal plate moves up dip. The independent oracle cancels the removable singularities algebraically and integrates on 65,537 angles. The compact radius is 39.529164020 m and the full continuum integral **per unit magnitude** is 108.652564438 m. The donor's m=120.1429512 gives approximately 13,053.8 m. The physical FE normal integral depends on cell sampling.
* Bilinear phase on the finest Cartesian lattice reproduces donor Q1 interpolation inside the static DGQ2 phase storage. Ghost lattice values complete the missing **outside** portion of a normal integral. Within the box, physically positive phase beyond either endpoint uses the endpoint value. The Q1 end value has zero tangential derivative in that wedge. The section and the projected Stokes load consume the same sample. The normalizer uses eight three-point station panels per material segment. Checkpointing saves these fixed completed integrals along with their magnitude and quadrature convention.
* Inside `FaultStokesCoupling::sample_bulk`, Q8 owned bulk samples supply the work measure χ JxW. Before the friction evaluation, the tridiagonal Q1 system (M+20²K)p = b projects the **resolved perturbation**, with zero flux at both endpoints. The 50 MPa background is added back. Friction uses the filtered pressure at the quadrature foot. Post-accept output projects the raw compression, friction compression and returned shear, including damping, onto the same material basis. This path leaves the section rate solve and ordinary Stokes quadrature intact. Straight traces bound candidate segments by station before the existing exact projection tests. Curved traces retain their full search. Candidate working arrays and exported histories scale linearly with the number of labels per rank. The additional traction observation uses one owned Q8 pass on accepted steps. The filter uses tridiagonal arrays and collective sums.
* The registered benchmark resistance retains linear interpolation of the direct effect and log interpolation of positive age. Exact nodal aging and slip run on acceptance, in seconds and metres. Reporting uses Q1 rate/slip and log-linear age at fixed coordinates. There are 25 historical reporting stations, including the actual base, and 17 original pilot stations.

## Geometry, contacts and resolved physics

The shipped ID-ordered table agrees with all donor positions to 7.50e−12 m after reversing donor numbering. The largest relative equilibrium-age error is 2.98e−14, below the independent 1e−13 age tolerance. Segment lengths span 9.99545661–10.00000000 m. The staged SHA-256 check protects the published bytes. Increasing just ID 1775's regenerated age by 1% fails the data check. [Input check](input-check.txt)

The actual 75×25-root production mesh has **552,084** cells, versus the archived 552,081. The independently counted 319,324 physical support cells match the donor. All support cells and all 5,004 cells in the protected face footprints have h=3.90625 m. The largest h is 1 km and the area is 7.5e9 m². One and two ranks gave identical level histograms. The excess of three cells is one net quadtree subdivision in cell-count terms with no change in support count or resolution. The retained donor summary does not identify the differing exterior cell(s) or establish identical global Stokes discretizations. [Mesh measurements](mesh.txt)

**Figure 1.** Normalized slip density in three horizontal transects through the 2 × 1 km resolved verification mesh, and initial raw and filtered compression perturbations along the trace. Positive normal distance points to the hanging wall. The continuum curve uses the analytic AT1 profile. The left panel uses actual FE geometry and the projected normal integrals, checked pointwise against the production transfer on 513 samples per cut. [Resolved figure](contact.png) · [full-box intermediate figure](intermediate.png)

| Local measurement | Resolved contact mesh | Full-domain intermediate mesh |
|---|---:|---:|
| Active cells / RSF labels | 14,483 / 117 | 41,226 / 580 |
| Finest cells / nominal ID spacing | 3.90625 / 10 m | 31.25 / 100 m |
| FE phase maximum absolute error | 4.38e−8 | below 5e−8 |
| Maximum prescribed (ux,uy) errors, m/s | (1.31e−17, 2.13e−17) | (9.62e−18, 7.14e−18) |
| Returned slip / imposed contrast at top, mid-depth, base | 1.00449, 1.00449, 1.00450 | 1.4030, 1.3782, 1.2354 |
| Initial forward RSF at transition quadrature | uniform shallow control | max 1.12e−8 Pa at target speed |

The ratio integrates χV over signed normal distance along a **horizontal cut** at each depth and divides by 1 nm/s. At a contact this cut includes the real in-box continuation wedge. It differs from integrating along a single fixed station normal. Turning completion off in an ordinary initialization gave 0.68933 at the top and 0.68822 at the base for ∫χ dr. It returned zero at 230 top and 234 base points with positive phase above 1e−5. The middle cut stayed at 1.00450. The intermediate grid is useful for lifecycle, transition and throughput measurements. Its return errors of 24–40% make it unsuitable for calibrating contact stresses or event onset.

At resolved time zero, shear / raw / filtered compression at the upper contact were **26,546,122.35875 / 49,999,999.99868 / 50,000,000.00012 Pa**. At the lower contact they were **26,546,122.65892 / 50,000,000.95571 / 50,000,000.08049 Pa**. Independent differentiation of all nine native Q2 velocity nodes and projection of pressure and returned shear gave absolute shear errors of 1.91e−6 and 4.36e−6 Pa. Raw compression errors were 6.42e−8 and 6.30e−7 Pa. The native Float32 output limits this reconstruction. Independently assembled M and K preserved constant compression to 4.37e−14 and attenuated the norm of a 100 m sinusoidal perturbation to 0.38497. Reconstructed filtered compression agreed with the friction input within 7.82e−7 Pa. At 1,600 s, switching off the filter changed base friction compression by 116.86 Pa and base slip rate by 1.0824e−13 m/s. [Resolved checks](resolved-check.txt) · [raw-versus-filtered](raw-filtered.txt)

On the **full-domain intermediate mesh**, the 0 s rates at 15, 16.5 and 18 km were 0.999989916, 0.999993388 and 0.999994223 nm/s. The base rate was 1.000000944 nm/s. Full-box upper/base shear was 26,546,116.78 / 26,546,122.66 Pa and friction compression was 49,999,999.99 / 49,999,998.33 Pa. The donor's archived full-resolution, 4e6 s initialization gives 26,545,846.26 / 26,549,393.27 Pa and 50,000,002.81 / 50,000,834.23 Pa respectively. These are different initial Maxwell intervals and spatial discretizations. The matched local controls distinguish their effects. [Intermediate checks](intermediate-check.txt)

## Accepted-state timing and limitations

The first-step cap is 4e6 s and subsequent cap 4e7 s. At steady 1 nm/s the 0.1 slip budget gives 800,000 s. At 8,000 s initial age, 0.1 log change predicts 842.25 s. In the intermediate calculation the actual limit came from age at **ID 14, 1,400 m**. Almost the entire shallow plateau is tied to the displayed precision. The completed 842.253874 s attempt had |Δlog θ|=0.10000001282 and was rejected. The retry advanced 758.028385 s from the identical parent. The final interval of 841.971615 s was shortened by end time. Its largest slip/Dc was 0.000105248 at deep ID 579 and its largest |Δlog θ| was 0.0917124 at ID 14. Deep steady creep controlled **zero** local physical steps. The row labeled “manager” is a proposal for the following step and is discarded if the just-solved attempt is rejected. [Every attempt and limiting ID](startup-attempts.csv)

* **Maxwell interval.** The delivered 4e6 s negative control ended at residual 1.11587e−6 on Picard iteration 60. ASPECT cannot cut a zero-duration initialization. A diagnostic cap of 180 allowed convergence on iteration 117. Both interval controls used the same 400 s first physical step. The 100 s / 4e6 s initial maxima of |ux| were 2.50128e−10 / 2.50438e−10 m/s. Maximum |p| was 4.56 / 154,618 Pa. Both had zero stored initial DG stress, slip and identical nodal ages. Maximum initial differences were 6.27e−12 m/s in speed, 2,453.61 Pa in filtered compression and 1,895.10 Pa in shear. At 400 s the largest rate and accumulated-slip differences were 1.54e−20 m/s and 6.17e−18 m. This short shallow control does not establish equal long-term loading. [Both sets of native fields and tractions](maxwell.txt) · [production-limit failure](initialization-60.log)
* **Solver tolerance and timestep growth.** Keeping all other inputs fixed on the resolved control with a 400 s first cap, 1e−7 and 1e−8 nonlinear tolerance gave identical material histories at 0, 400, 900, 1,525 and 1,600 s. With 91% growth, the clocks were 0, 400, 1,164 and 1,600 s. At 1,600 s the maximum speed difference was 4.59e−11 m/s and the slip difference 1.06e−8 m. The production candidate retains 25%. [Tolerance comparison](tolerance7.txt) · [growth comparison](growth91.txt)
* **Age budgets.** With the same first cap and growth, 0.2 proposed 1,773.19 s at initialization. The active global caps kept the short trajectory identical. The offline replay evaluates the two inverse exact-age limits at each of the 202 **actual saved donor profiles**. Of 146 pre-onset profiles, 142 were slip-limited below 40 km. All 56 above-threshold profiles were slip-limited. At the first saved threshold crossing both budgets gave 0.771920 s from slip at 10.85 km; the 0.1 age bound was 1.54778 s. This counts sampled profiles, not all time steps. No branch-native acceleration checkpoint exists for a pair of time-integrated budget runs. [Local comparison](age02.txt) · [donor budget replay](acceleration.csv)
* **Restart and MPI.** Intermediate step 1 wrote a terminal checkpoint. The ordinary resumed calculation through step 2 matched all 580 rows, clocks, event state and all 164,904 native patches per output exactly. The serial/two-rank contact comparison matched phase exactly, velocity within 2.78e−17 m/s, speed within 2.16e−20 m/s and friction compression within 7.36e−6 Pa. **The strict native pressure/stress comparison is red at 1e−9 linear tolerance.** Its first physical frame exceeds the existing absolute/relative field budget in some low-amplitude samples. Maximum |Δp| through 1,600 s was 6.15e−5 Pa and maximum |Δτ| was 3.44e−5 Pa. Tightening only linear solve tolerance to 1e−11 on both layouts brought all native fields inside the original check. Keep 1e−9 for the requested production comparison. [Restart](restart.txt) · [strict comparison with sample coordinates](mpi.txt) · [tighter solver control](tighter-mpi.txt)
* **Event lifecycle and pilot.** An independent executable ran the actual event monitor through onset, an intervening trial excluded from observation, an equality reset and five consecutive below-threshold acceptances. Serializing/restoring at every boundary preserved onset 1, equality at 5 and completion at 10. The real simulator calls the observer after the repeat decision and checkpoints the event, last full-profile clock and slip reference. The branch has not yet reached an event. The pilot kept 10,722 cells, 179 IDs, the 0/400/900 s clocks and bulk frames at 0/900 s. Its largest slip-rate difference against the previous pilot was 1.04e−25 m/s at 900 s. The four scoped CTests passed. [Pilot](pilot.txt) · [native comparison](pilot-native-comparison.txt) · [CTest](existing-controls.txt)
* **Cost.** The intermediate serial startup, including the rejected attempt, took 324 s under shared laptop load. Its first physical frame was 16.76 MB. Scaling measured compressed bytes per cell to 552,084 gives approximately **76 MB initially and 225 MB per physical frame**, before changes in event complexity and partitioning. Production uses the donor heavy-output cadence of 1 m or 100 years; text profiles use 0.1 m or one year plus milestones. The launcher reserves 30 minutes for completing the in-flight step and checkpoint. Wall time and memory for a full-grid Stokes solve and checkpoint still require server measurement. The donor's reported 163.893 yr onset is a comparison target.

## Reproduction

Run from the checkout root. The examples require an installed, matching Release executable and SDK. The plugin is compiled during staging. One numerical thread per MPI rank was used, with at most four local workers. The supplied donor checkout is needed only for replaying the archived scientific comparisons. Choose fresh build/output destinations. The following variables and commands reproduce the measurement series (the local run used the paths in the manifests).

```sh
B="$PWD/build-historical-first"
R="$PWD/refactoring/phase-field-fault/results/bp3-first-event"
D="../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/bp3-first-mpl
# Configure with the local ASPECT/deal.II Release toolchain, starting from the existing Release cache.
cmake -S . -B build-6b/release -DCMAKE_INSTALL_PREFIX="$B/install"
cmake --build build-6b/release --target aspect.exe.release -j3
for component in runtime includes Unspecified; do cmake --install build-6b/release --component "$component"; done
sha256sum -c "$R/delivery.sha256"
mkdir -p "$B/public"
unzip -q "$R/bp3-first-event.zip" -d "$B/public"
bash "$B/public/bp3-first-event/prepare.sh" "$B/install" "$B/run"
export BP3_RUN="$B/run"
BP3_LOCAL_RANKS=2 bash "$BP3_RUN/stampede3.sh" contact start
bash "$BP3_RUN/stampede3.sh" pilot start
python3 "$BP3_RUN/check_pilot.py" "$BP3_RUN/output" --read-vtu
python3 "$BP3_RUN/check_historical.py" "$BP3_RUN/contact-output" --figure "$B/contact.png"

Q="$R/qualification.py"
python3 "$Q" "$BP3_RUN" intermediate control "$B/intermediate-clean"
python3 "$Q" "$BP3_RUN" intermediate control "$B/intermediate-resume" --end-step 1
python3 "$Q" "$BP3_RUN" intermediate control "$B/intermediate-resume" --resume
python3 "$R/compare.py" "$B/intermediate-clean" "$B/intermediate-resume"
python3 "$BP3_RUN/check_historical.py" "$B/intermediate-clean" --figure "$B/intermediate.png"
python3 "$R/timesteps.py" "$B/intermediate-clean" "$B/intermediate-clean-start.log"

python3 "$Q" "$BP3_RUN" contact control "$B/contact-serial"
python3 "$Q" "$BP3_RUN" contact raw "$B/contact-raw"
python3 "$Q" "$BP3_RUN" contact uncompleted "$B/cropped-rays" --end-step 0
python3 "$R/sensitivity.py" "$B/contact-serial" "$B/contact-raw"
python3 "$Q" "$BP3_RUN" contact control "$B/fixed-first" --first 400
for option in growth91 tolerance7 age02; do
  python3 "$Q" "$BP3_RUN" contact "$option" "$B/$option" --first 400
  python3 "$R/sensitivity.py" "$B/fixed-first" "$B/$option"
done
# Expected to exit nonzero in initialization at the production limit.
python3 "$Q" "$BP3_RUN" contact maxwell4e6 "$B/final-60" --first 400 --end 400 --ranks 2
python3 "$Q" "$BP3_RUN" contact maxwell4e6 "$B/final-long-initialization" --first 400 --end 400 --diagnostic-iterations 180
python3 "$R/sensitivity.py" "$B/fixed-first" "$B/final-long-initialization" --fields
# Reports the retained strict native-field exception and exits 1.
python3 "$R/compare.py" "$B/contact-serial" "$BP3_RUN/contact-output"
python3 "$Q" "$BP3_RUN" contact linear11 "$B/linear11-serial"
python3 "$Q" "$BP3_RUN" contact linear11 "$B/linear11-mpi" --ranks 2
python3 "$R/compare.py" "$B/linear11-serial" "$B/linear11-mpi"

python3 "$R/inputs.py" "$BP3_RUN" "$D/output_start/profiles/fault_0.csv"
python3 "$R/acceleration.py" "$D/output_start" "$D/output_resumed" > "$B/acceleration.csv"
python3 "$BP3_RUN/history.py" "$B/intermediate-clean/statistics" "$B/accepted-history"
ctest --test-dir build-6b/release -R '^(unit_tests|layered_moving_fault|layered_fault_traction_step|growing_fault)$' --output-on-failure -j1
g++ -std=c++17 -I/opt/boost/1.87.0/include benchmarks/bp3_qd_thrust/check_event.cc -L/opt/boost/1.87.0/lib -Wl,-rpath,/opt/boost/1.87.0/lib -lboost_serialization -o "$B/check-event"
"$B/check-event"

# Optional full-grid measurement. The observer intentionally exits before allocating a Stokes system.
cmake -S "$R/mesh-inventory" -B "$B/mesh-plugin" -DAspect_DIR="$B/install/lib/cmake/Aspect"
cmake --build "$B/mesh-plugin" -j1
cd "$BP3_RUN"
printf 'include production.prm\nset Output directory = footprint-only\nset Additional shared libraries = ./libbp3_qd.so, ../mesh-plugin/libmesh_inventory.so\n' > footprint.prm
"$B/install/bin/aspect-release" footprint.prm > "$B/footprint.log" 2>&1
grep 'Physical support\|Historical mesh total\|Mesh inventory complete' "$B/footprint.log"
```
