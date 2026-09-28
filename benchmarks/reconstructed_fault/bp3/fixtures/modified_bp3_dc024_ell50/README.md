# Modified BP3: Dc = 0.024 m, ell = 50 m

This is a separate research fixture, not official BP3 and not a replacement for
`modified_bp3_long_run_300km`. Its physical initialization is fresh, never a
restart of the old Dc/ell model. `manifest.json` identifies all input hashes.

The 300 by 100 km box, straight 60-degree fault, approximately 100 m surface
grid, mature C=0 law, work measure, paired endpoint treatment, horizontal
friction/state interfaces and outer loading are unchanged. The finest bulk
cell side is 12.20703125 m over the entire nonzero band, including both ends;
the maximum outer cell side remains 12500 m. The material PRM is the sole
authority for Dc. Both initial Theta and the independent aging checker read it.

`prestress.txt` is intentionally byte-identical to the maintained old fixture:
it defines the accepted *physical frozen background function*, including its
captured rational correction. Its denominator is not this case's I_h. Replacing
that denominator or fitting the background to the new equilibrium would change
the physical loading. The production weak projections, current I_h and mass
matrices are rebuilt on the new mesh; `completion.txt` is newly integrated from
this mesh's Q1 sampling of the new stationary profile.

Preparation from an empty destination, at repository root:

```
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target bp3_length_scale_mesh fault_mechanical_modes -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3 -j4
benchmarks/reconstructed_fault/performance/build-gmg/bp3_length_scale_mesh 50 12.20703125 0 benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3_dc024_ell50/target_cells.txt
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py fixture
```

The generator refuses to replace an existing manifest. Keep completed evidence.
The local-reference mesh is made with the same command and reference flag `1`,
writing `length-scale-study/target_cells_reference.txt`. This halves the band
and surrounding halo resolution only near the frozen probe, not at the ends.

Frozen comparisons (already run; do not overwrite their directories):

```
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py prepare --label probe-candidate
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py prepare --label probe-reference --reference
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py run --label probe-candidate
python3 benchmarks/reconstructed_fault/bp3/length_scale_study.py run --label probe-reference
python3 benchmarks/reconstructed_fault/bp3/analyze_length_scale.py
```

Four ranks, Release, working AMG, explicit B/G, pivoted surface inverse; no
physics/tolerance changes. A verified frozen probe exits intentionally with
status 1 and `MECHANICAL MODES VERIFIED`, before any state is accepted. This
is not an initialization-convergence claim. The existing hook also computes
one uncommitted coupled initialization direction before its three A/B probes.

**Evolution is blocked by the 1% profile-width gate at 12.207 m.** Do not launch
the eight-step PRM as a qualified case. The 6.104-m patch cannot qualify a
whole-fault 6.104-m mesh or its endpoint treatment. See the task report for the
measured comparison and the next decision. The separately prepared ell=25 m
fixture is unrun and is reserved for later width sensitivity.

The subsequently authorized **coupled diagnostic exception**, not a relaxation
of that gate, uses `length_coupled.py`. Its reference generator mode `2` also
refines both endpoint neighborhoods and has separately regenerated completion.
Evidence is in `length-scale-study/coupled-diagnostic/`; see
`doc/reconstructed_fault/bp3/stage_K5_length_coupled_qualification.md` for the
accepted-state, endpoint-mechanics and spatial comparison. Neither the
diagnostic flag nor this README promotes the candidate to production.
