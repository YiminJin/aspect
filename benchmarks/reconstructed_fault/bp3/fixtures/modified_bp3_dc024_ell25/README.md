# Prepared width-sensitivity reference — not run or qualified

Dc=0.024 m, ell=25 m, intended AT1/core-.6 support diameter 98.82291 m.
The separate 300-by-100-km mesh has 459,696 cells, finest cell side
6.103515625 m and maximum cell side 12,500 m. Its 4,137,264 particles would
make this a substantially larger simulation; no ASPECT run was launched.

`manifest.json` records the new Q1 endpoint-completion inputs and hashes.
The physical fault and captured background prestress are unchanged. See the
ell50 fixture README for the meaning of the prestress data and the parameter
consistency change. Do not substitute this completion file in an ell50 run.

Preparation from an empty destination:

```
benchmarks/reconstructed_fault/performance/build-gmg/bp3_length_scale_mesh 25 6.103515625 0 benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3_dc024_ell25/target_cells.txt
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py fixture --ell 25
```

This mesh has the same ell/h as the failed ell50 profile-width candidate;
preparation is not proof of resolution. Do not start evolution or treat it as
a resolved narrower-band reference without its own profile/mechanical gates.
