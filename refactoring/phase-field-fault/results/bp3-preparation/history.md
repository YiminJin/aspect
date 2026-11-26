# What the retained 60° BP3 histories show

## One acceleration in the 150 × 50 km DG-stress donor

Jobs 3610007, 3613912 and 3616642 cover the same physical trajectory. The first
accepted-state table contains steps 0–815. The second contains a replay of
810–815 with identical clocks, step sizes, maximum rates and down-dip locations.
The third reads the step-815 state, logs accepted steps through 4237
(5,172,072,500.6796865 s) and stops on the allocation wall limit during step 4238.
The full histories locally available for the last allocation are spatial
profiles at selected clocks. The native fields at steps 868–4182 complement
the earlier profiles at steps 0–798. The short six-step replay and the
duplicated profiles are counted once.

The onset bracket at the donor's 0.001 m/s threshold is step 773
(5,172,072,475.779887 s, 0.000610673882 m/s) to 774 (5,172,072,483.057341 s,
0.001036377486 m/s at 10,850 m down dip). The initial maximum was
1.0050647e−9 m/s. At profile 4182, 17.560699 s after the first above-threshold
state, the whole-fault maximum was 1.375532979 m/s at 470 m down dip. This is the
largest **sampled** value. The last logged accepted clock is another 0.061646 s
later. It has no matching local velocity/slip profile.

| Fixed down-dip station | Slip at step 774 (m) | Slip at step 4182 (m) | Added slip (m) |
| --- | ---: | ---: | ---: |
| 5 km | 0.0000196082 | 1.216792983 | 1.216773375 |
| 10 km | 0.544249971 | 1.706338598 | 1.162088627 |
| 15 km | 1.217175644 | 1.941120346 | 0.723944702 |

**Figure.** Whole-fault maximum slip rate during 163.89 yr of loading, the first
threshold crossing at 0 s on the event clock, and increased slip at three fixed
weakening-zone stations. Blue rate data are from the every-step acceptance
table. Red rate data are maxima over all 5,776 Q1 vertices in each saved profile.
Station values are interpolated on the ordered down-dip coordinate. Distances
are measured along the fault. The event time axis covers the available
acceleration and rapid slip. Every post-onset saved profile remains above
0.001 m/s. There is **one documented onset and zero confirmed recoveries**.
After step 815 the sparse samples cannot exclude a short unsampled excursion
below the onset threshold. They do not establish another event or a recurrence
interval.

Input roots below are relative to the worktree. The Release run used 160 MPI
processes. The data reduction and plotting commands use one local worker.

```sh
P=../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free
R=refactoring/phase-field-fault/results/bp3-preparation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python3 -B "$R/history_profiles.py" "$P" "$R"
```

```text
refactoring/phase-field-fault/results/bp3-preparation/history-first-acceleration.png
refactoring/phase-field-fault/results/bp3-preparation/history-profiles.csv
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/output_start/profiles.csv
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/output_resumed/profiles.csv
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free/evidence/server_3610007/accepted_steps.csv
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/accepted_steps.csv
```

There are 148 indexed full-fault profiles in the start and 54 in the resumed
output. All 148 files in `evidence/server_3610007/profile_data` and all 166
files in `evidence/first_event/profile_data` byte-match profiles already in
those two output directories. The seconds stored in `profiles.csv` are
authoritative for both parts. One early plotting summary labels seconds as
years; the figure above divides seconds by 31,557,600.

## Define three distinct events before continuing

The donor's accepted-state observer latches its first upward crossing at
0.001 m/s and marks completion after five consecutive accepted steps strictly
below that speed. It then stops observing. The `Stop after first event` input
controls termination. Disabling it allows more timesteps but does not rearm the
observer. The material trace slip increases using accepted endpoint speed times
accepted interval. `accepted_steps.csv` gives the whole-fault maximum and its
location; `stations.csv` gives accepted speed and cumulative slip at fixed
distances. Rank-zero writes are included in each native checkpoint's
`bp3_output_metadata` prefix.

Use the following proposed operational definition for the three-event run.
It fixes the thresholds before the next observations and preserves the original
onset marker. The stricter return and slip tests are analysis choices for this
experiment; they do not enter mechanics or set the timesteps.

1. Start at the prescribed quiet initial state. Detect a rising, consecutive
   accepted-step crossing of 0.001 m/s in whole-fault maximum speed. Record both
   bounding clocks, the fastest down-dip location and the greatest subsequent
   accepted speed in the episode. The maximum may migrate between vertices.
2. Check cumulative slip at 0, 2.5, 5, 7.5, 10, 12.5 and 15 km down dip. The
   actual station increments must equal the donor's $V_n\Delta t_n$ at that
   same station, to floating-point allowance. Sum the measured increments
   in excess of $V_{pl}\Delta t_n$ while the whole-fault speed is at least
   0.001 m/s and the station exceeds $10V_{pl}$. At least one fixed station
   must accumulate one characteristic slip distance, $D_c=0.008$ m, in these
   fast intervals. A tip-localized pulse missed between the stations remains
   unconfirmed; inspect the full-fault profile at its selected location.
3. Count the event and rearm after at least five consecutive accepted
   observations of whole-fault maximum speed at or below $10V_{pl}=10^{-8}$
   m/s, spanning at least $D_c/V_{pl}=8\times10^6$ s (92.59 days).
   This is one characteristic slip distance of plate loading. A fast secondary
   peak or brief dip before that sustained return belongs to the same episode.
   Keep both third-onset and third-recovery clocks for scheduling; completion
   through three events means the third recovery has been observed.

The offline catalogue uses every physical step beginning at zero. Supply a
complete, matching accept/station prefix (for example from verified native
checkpoint metadata) followed by the same trajectory's continuation files.
Pass a directory with a complete restored prefix only once. It also accepts
split directories with verified overlapping accepted records and rejects gaps
or conflicting branches. After a crash use the last complete checkpoint's
accepted prefix rather than partial in-flight CSV writes. Run this on the
matched donor wall allocation or on copies transferred after checkpointing.
The sparse profiles in the figure cannot substitute for these two time series.

```sh
R=refactoring/phase-field-fault/results/bp3-preparation
python3 -BO "$R/history_events.py" /path/to/complete/output
# Alternatively, when the resume only contains later steps:
python3 -BO "$R/history_events.py" /path/to/verified/prefix /path/to/continuation
python3 -BO "$R/history_test.py"
python3 -BO "$R/history_events.py" ../aspect/benchmarks/reconstructed_fault/bp3/output-first-event-resumed-2
```

The constructed three-episode test checks that a short quiet spell and a
five-step subthreshold afterslip interval stay within an episode, a sub-$D_c$
pulse stays uncounted, and a restart in the middle of an episode produces the
same onset, peak, slip and recovery records as the uninterrupted run. It also
rejects a missing clock, conflicting overlap, a corrupted cumulative slip, and
a missing weakening-zone station. The last command checked 604 consecutive
real accepted clocks and station records through step 603 in an older donor
configuration and returned zero events. The older `output-first-event`
directory has station rows through 55 and accepted rows only through 54, so the
checker reports the unmatched frame. For the DG 160-rank run a complete station
file through 4237 is absent locally. No three-event physics run was inferred
from the constructed catalogue test.

## Other locally retained histories

These sensitivity runs provide independent historical context. The station and
stress histories, physical widths, mesh or software differ from the DG
150 × 50 km server trajectory above. Their modeled years cannot be appended
to that trajectory. The paths in the first four rows share this older root:

```text
../aspect/benchmarks/reconstructed_fault/bp3/
```

| Archive | Last accepted step, physical time | Greatest accepted speed (m/s) | Identification |
| --- | --- | ---: | --- |
| `first_cycle_coarse` | 11, 58.807 yr | 1.02016e−9 | 100 × 100 km, 64 MPI ranks, deal.II 9.6.0, first-event flags 0/0 |
| `first_long_run/output` | 5188, 383.222 yr | 1.53218e−7 | 300 × 100 km, 32 ranks, deal.II 9.6.0, first-event flags 0/0 |
| `modified-long-first-event` | 127, 73.760 yr | 1.00064e−9 | Modified geometry, 4 ranks, deal.II 9.6.2, flags 0/0 |
| `output-first-event`, `output-first-event-resumed-2` | 54, 10.627 yr and 603, 142.933 yr | 1.01633e−9 and 1.59950e−6 | Older 150 × 50 km series, 160 ranks, deal.II 9.7.0, flags 0/0. The recovered endpoint's velocity/pressure hotspot is documented in `resumed_diagnostics/REPORT.md`. |
| Donor `particle_free/rsf_episode_server/analysis/event-summary.json` | 748, 374.580 s | 0.0450089 | Metre-scale cohesive propagation specimen. Its last speed was 0.0444673 m/s; the archive reports zero required-duration decay observed and an incomplete episode. |
| Donor `stage4_rsf` reduced-background peaks | Three peaks in a different reduced calculation | — | The Stage 4 wavelength investigation explicitly leaves event count undetermined. |

Inspect the older accepted and flag tables with:

```sh
python3 - <<'PY'
import csv
from pathlib import Path
base = Path('../aspect/benchmarks/reconstructed_fault/bp3')
for name in ('first_cycle_coarse', 'first_long_run/output', 'modified-long-first-event',
             'output-first-event', 'output-first-event-resumed-2'):
    with (base/name/'accepted_steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    print(name, rows[-1]['step'], float(rows[-1]['time']) / 31557600.,
          max(float(r['max_V']) for r in rows),
          (base/name/'first_event.csv').read_text().splitlines()[-1])
PY
```
