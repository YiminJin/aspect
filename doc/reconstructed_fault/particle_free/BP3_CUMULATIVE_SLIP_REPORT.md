# Server BP3 cumulative-slip plot — 31 October 2026

[Python script](../../../benchmarks/reconstructed_fault/particle_free/plot_bp3_cumulative_slip.py) ·
[PNG](../../../build-particle-free/bp3-cumulative-slip/cumulative_slip.png) ·
[PDF](../../../build-particle-free/bp3-cumulative-slip/cumulative_slip.pdf) ·
[sampling diagnostic](../../../build-particle-free/bp3-cumulative-slip/cumulative_slip.sampling.png).

The plot combines the supplied `output_start/` and `output_resumed/` server
profiles. It uses the axes and colours of Figure8b in the local
[Erickson et al. paper](../bp3/bssa-2022066.1.pdf),
[doi:10.1785/0120220066](https://doi.org/10.1785/0120220066): cumulative slip
horizontally and down-dip distance increasing downward, blue for interseismic
and red for coseismic profiles. Its caption specifies5-year/1-second contours;
the preceding prose says1-year interseismic contours. The code follows the
caption and chooses5 years because the supplied interseismic sampling supports
that interval. Only the available60° thrust run is shown; no other angles,
interevent times or completed cycles are invented.

From the repository root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/plot_bp3_cumulative_slip.py \
  --title 'BP3-derived server run: 60° thrust'
```

No arguments defaults to the two supplied server directories and writes to
`build-particle-free/bp3-cumulative-slip/cumulative_slip`. For other copied
segments or plotting intervals:

```sh
python3 benchmarks/reconstructed_fault/particle_free/plot_bp3_cumulative_slip.py \
  /path/to/output_start /path/to/output_resumed \
  --interseismic-years 5 --coseismic-seconds 1 \
  --xd-max-km 40 --output /path/to/plots/cumulative_slip
```

Dependencies are NumPy and Matplotlib, plus the existing repository reader in
`benchmarks/reconstructed_fault/bp3/plot_cumulative_slip.py`; run the new script
from its repository location. Input directories and checkpoints are read only.

## Time selection and data coverage

Colour uses saved instantaneous max |V| over **all exported fault nodes**,
with coseismic threshold1e−3 m/s, independently of the displayed0–40 km window.
Stored cumulative signed slip is plotted directly, with no offset, slip-rate
integration, temporal interpolation or extrapolation.

At each nominal target time, use the nearest recorded state only when the
same-regime bracketing saved-data gap is no wider than the requested interval.
Exact saved targets are retained directly. Actual times therefore have bounded
sampling jitter rather than pretending to be exact5-year/1-second states.
Retain the first/last saved state of each regime as well; extra transition/final
contours are dashed. CSV sidecars identify every exception and target time.

| Quantity | Observed result |
|---|---|
| Available profiles | 202:146 interseismic,56 coseismic; fixed5776-node geometry |
| Interseismic saved spacing | Median1.114 yr, maximum2.164 yr; adequate for5-year targets |
| Blue regular targets | 33 at0–160 yr, all supported; largest nearest-time shift .628970 yr |
| Blue drawn contours | 34, including the extra last interseismic state |
| Coseismic saved spacing | Median .118224 s; first gap10.689880 s, later gaps at most .469910 s |
| Red target anchor | First **saved** seismic profile, step774, t=5172072483.057341 s; not the exact event onset |
| Red regular targets | 18 at0–17 s from that anchor; targets1–10 s unsupported and omitted |
| Red drawn contours | 9:8 supported targets plus the extra final saved state |
| Supported red target jitter | At most .159790 s |
| Final saved contour | Step4182, t=5172072500.6180401 s,17.560699 s after the anchor |

The preceding interseismic profile is769 at5172072368.423655 s, leaving a
114.633686 s onset bracket. The gap between saved seismic profiles774 and798
cannot resolve the early rupture shape at1-second cadence. Choosing an even
smaller interval would not restore that information. A1-second target interval
is useful for the later dense exports, provided that early gap stays explicit.

The final exported profile remains in the coseismic regime, with sampled
Vmax1.37553 m/s. As in [BP3_REVIEW.md](BP3_REVIEW.md), neither final event peak,
arrest nor completed repeatable cycles are established. The latest logged
acceptance4237 has no full slip-profile payload in these inputs; the script
does not extrapolate a curve to it. These plots do not change prior benchmark
fidelity, spatial/temporal findings or cleanup availability losses.

## Outputs and verification

The output prefix produces:

- `.png` and vector `.pdf`: the cumulative-slip figure.
- `.sampling.png`: saved rate samples, selected red contours and the coarse gap.
- `.selected.csv`: exact source file/hash, physical time, regime and target
  offset for every drawn contour.
- `.targets.csv`: every requested target, its bracket and selected state or
  explicit omission.
- `.summary.json`: interval policy, coverage, all input profile/index hashes,
  source-script hash and command.

Five focused data tests cover missing rupture profiles, full-fault instantaneous
classification, threshold equality, century-scale/subsecond clocks, boundary
contours, restart deduplication and conflicting-history/geometry rejection:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover \
  -s benchmarks/reconstructed_fault/particle_free \
  -p test_plot_bp3_cumulative_slip.py -v
```

Entry root/branch/revision are the required worktree, `pf-rsf-particle-free`,
`27178813cebd471b93a7a9308f41215b71b4a20b`, with the preceding task's dirty
source/docs preserved. All220 supplied server-export files match the prior
review's hashes. The surface VTU inventory is8 files, correcting the previous
report's count of4; the scheduled CSVs provide the needed slip histories.
No executable was used or rebuilt, and no simulation or checkpoint conversion
was run. [Verification record](../../../benchmarks/reconstructed_fault/particle_free/evidence/bp3_cumulative_slip/verification.json).

Next data action: rerun the same script when additional saved profiles arrive,
retaining the1-second coseismic target and an explicit gap wherever the source
exports do not support it. This plotting task does not select the next spin or
production experiment.
