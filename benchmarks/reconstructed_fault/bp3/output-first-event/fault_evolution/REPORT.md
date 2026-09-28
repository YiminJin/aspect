# Copied first-event run: saved fault evolution

Prepared September 28, 2026. No simulation was run and no numerical source,
input, checkpoint or original output was changed.

## Figures

| Figure | Contents |
| --- | --- |
| [Time–distance maps](space_time_shallow.png) / [PDF](space_time_shallow.pdf) | All 124 snapshots, 0–45 km down-dip: V, committed Θ, accumulated slip, shear/normal traction changes and slip gradient |
| [Time histories](time_histories.png) / [PDF](time_histories.pdf) | Existing vertices nearest 0, 5, 10, 15, 18, 25 and 40 km; dots denote saved states |
| [Full-fault profiles](profiles_full_fault.png) / [PDF](profiles_full_fault.pdf) | Nine selected times, covering the entire 57.735 km fault and both endpoints |
| [Shallow profiles](profiles_shallow.png) / [PDF](profiles_shallow.pdf) | The same selected times, restricted to 0–45 km |
| [Early native properties](native_property_ranges.png) / [PDF](native_property_ranges.pdf) | Nine native snapshots through 10.627 years only; no late native-property data were supplied |

## Data coverage and observations

The new index is `../profiles/profiles.csv`; its relative payloads are in
`../profiles/profiles/`. It contains 124 profiles of 2889 vertices, spanning
steps 0–594, 0–4493773756.523243 s (142.399097413 model years; year = 31557600 s).
All indexed files are present. Geometry, node identities, timestamps and finite
values passed the existing reader checks. No duplicate index rows were found.

The top-level index `../profiles.csv`, `../accepted_steps.csv` and native VTUs
cover only steps 0–54, through 10.627038103 years. The nine overlapping profile
payloads are byte-identical between the old and new copies. The later profiles
are plotted as exported and cannot be independently verified against the older
accepted-step/event logs. `../log.txt` ends during step 55, so it does not explain
the reported failure after seven hours. The saved PRM belongs to that older
metadata set and does not independently establish the configuration of every
later snapshot.

The saved profiles show:

- A creeping region extending up-dip from around 15 km toward roughly 8 km.
  The shallower region remains nearly locked, with increasing Θ; passage of
  the creep front produces a sharp reduction in Θ at the 15 and 10 km stations.
- A shear-traction peak traveling with that front and increasing compressive
  normal traction near the shallow endpoint. Full-fault plots also show a
  substantial normal-traction increase at the deep endpoint.
- At the final saved state, slip ranges from 0.0000186066 to 4.51020855 m;
  total weak shear traction from 26.192124 to 36.462869 MPa; and total weak
  normal traction from 49.489919 to 64.223324 MPa.
- The largest instantaneous V anywhere in the saved profiles is
  1.04299733037e-9 m/s, at step 594, down-dip distance 57.575072 km.
  These snapshots do not show a fast seismic event. Their sparse cadence and
  the missing later event/accepted-step logs cannot exclude an unsaved transient.

Tractions are accepted weak-load Q1 representations including background,
not raw stress samples or a re-evaluation of friction using updated Θ.
The state plotted is committed post-update Θ. Slip is the recorded accumulated
quantity; it was not reconstructed by integrating sparse velocities. Time-map
rectangles use actual irregular sample times with midpoint boundaries, without
smoothing. History lines connect saved dots as visual guides. See [README](README.md)
for field semantics and [summary.json](summary.json) for exact ranges and times.

## Reproduction and verification

From the repository root:

```sh
MPLCONFIGDIR=/tmp/aspect-bp3-first-event-mpl PYTHONDONTWRITEBYTECODE=1 \
  python3 benchmarks/reconstructed_fault/bp3/plot_fault_evolution.py \
  benchmarks/reconstructed_fault/bp3/output-first-event \
  --profile-run benchmarks/reconstructed_fault/bp3/output-first-event/profiles \
  --times-years 0 10 30 60 90 120 135 140

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/reconstructed_fault/bp3 -p test_plot_fault_evolution.py
```

All four existing reader tests passed. Plot generation completed successfully
with all 124 profiles and nine native snapshots, without skipping any indexed
payload. The full-fault profiles, time histories and time–distance maps were
visually inspected. This task used Python for plotting only, and did not launch
debugging experiments, numerical tests or simulations.
