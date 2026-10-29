# P5b compact native tensor evidence

Read doc/reconstructed_fault/particle_free/P5B_NATIVE_REPORT.md and P5B_NATIVE_SELECTION.md. This is a uniform damaged material-block integration control, not a directional specimen. m=4 requires native cohesion=2000 Pa at the selected G/Kf; P5c erratum: the actual selected P4 fracture.prm also has cohesion=2000 Pa/m=4. P5b's former m16 attribution missed that override. The original ZIP remains unchanged; its manifest describes the original archived files. See ../p5c_native/baseline-correction.json for checkpoint proof and p5b-erratum.json for corrected extracted-file hashes.

native-response.png and native-energy.png use actual exported histories and accepted ledgers. *-history.csv stores volume-weighted means on the same physical block (no history remapping). Alpha=atan2(Wxx,Wxy) is a shear-tensor flow angle, not crack geometry. Undefined directions are blank, and active-direction.csv excludes zero flow. Native phi is spatially uniform to roundoff/nonlinear tolerance in this control. Point energy densities are J/m3, stress/driving Pa, A/K/L dimensionless and W 1/s. Integrated storage/work/dissipation and weak phase derivatives are J/m per out-of-plane thickness. Native weak-load CSV headers stress_N_m/total_N_m mean N per m of thickness. The .1 J/m initial AT1 storage is subtracted in the energy plot. Each signed BE remainder and residual work remains separate in *-ledger.csv; closure does not qualify physical energy/time accuracy.

verification.json covers native mapped identity, source/residual/energy bounds, actual mixed relaxed stability and 24-state radial restart/rejection. constraint-stability.json independently constructs C from native rules and verifies C^T g/C^T H C, with no artificial solve rows or double reduction. uniform-control.json supplies the identity case; history-recurrences.json independently checks forcing and tensor advancement. fixed-restart.json covers the other archive tag; negative-restart-checks.json requires the intended error markers and unchanged checkpoint bytes. end-cap-diagnostic.json preserves the incorrect fixed-dt probe assumption; development-failures.json preserves the other failures. point-regression.json and curvature-kernel.json retain fresh focused regressions.

run-commands.json records exact bounded commands and wall times; provenance.json records source/build identities. Use only the tensor fixture library, whose established alias names permit reuse of the native parameter include. Production paths/defaults are unchanged. Full raw outputs, coefficients and native checkpoints remain outside this review package in build-particle-free/p5b-native (large completed exports may be losslessly indexed there). The original P4/P5a/BP3 evidence is protected. strict_reference_match_passed=false remains; no MPI, intact native activation, general quadrature, objective rate or propagation qualification follows.

Direct native presets after building the recorded ASPECT and bulk_tensor_checks target:

    export ASPECT_SOURCE_DIR="$PWD"
    export ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1
    ./build-particle-free/p5b-native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/tensor_native.prm
    ./build-particle-free/p5b-native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/tensor_fixed.prm
    ./build-particle-free/p5b-native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/tensor_radial.prm

The large-step preset is an explicit preserved failure control. All four inputs validate; only the small-step controls are qualified. Rebuild plugins including bulk_cohesive_internal.h against the new executable. Old matched P4/BP3 binaries/plugins remain protected.

Restore completed raw exports before re-running full analysis (preview by omitting --write):

    python3 benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_22/restore.py build-particle-free/p5b-native/runs/radial-action-fixed --manifest benchmarks/reconstructed_fault/particle_free/evidence/p5b_native/raw-exports.json --write

Use analogous prefixes for other compared branches. Restoration verifies hashes and refuses changed-file overwrites. Native checkpoints remain extracted. Full raw archives are separate from this compact ZIP.
