# K2.3: saved-data endpoint localization at 0.5 s

## Decision

Subsequent user decision: K2.3 is accepted as completed **feasibility
verification**, with the spatial limitation quantified below. It is not a
fully spatially converged reference. Do not launch the 128x512 pair now.
The previous resolution-based recommendation below is retained as evidence,
not an active run request. K2.4 preparation uses the accepted 64x256 baseline,
subject to its documented convergence prerequisite; see
`stage_K2_4_preparation.md`. K2.2/K2.3 references remain provisional.

The dominant discrepancy is a grid-scale endpoint layer: its amplitude changes
by less than 0.6% in Delta sigma_n while its physical width approximately halves.
The outer two coarse surface elements at each end contain 98.25% of the squared
32-to-64 difference. However, excluding three elements at each end still leaves
an interior RMS difference of 19.55% of the fine interior Delta sigma_n signal
and 21.92% for -Delta(tau:N). Interior convergence is therefore not yet clearly
established under the requested closure condition. The evidence supports
documenting the endpoint limitation, but not closing K2.3 on that basis alone.

No simulation, production edit, topology change, or acceptance-criterion change
was made. No 128x512 case was launched. K2.2 remains provisional and Gate K2
unmet. The following analysis uses only the accepted 32x128 and 64x256 bumped
and matched homogeneous outputs at 0.5 s; it does not subtract initial errors
or reset histories.

## Measurement and exact partition

For each resolution define Delta z = z_bumped - z_homogeneous, and define
E_z = Delta z_64 - Delta z_32. The fields are the consistent-Q1 representations
of actual constitutive surface moments, not bulk-column averages. In particular,
E_sigma = E_p + E_minus_tauN, with minus_tauN = -Delta(tau:N).

Integrate the piecewise-linear fields on the union of their physical knots.
First and second moments are integrated exactly for that representation.
The length is L=0.25 m, with h_coarse=0.015625 m and h_fine=0.0078125 m.
For a cut a, partition [0,L] into [0,a], [a,L-a], and [L-a,L]. Thus

    Q_total = integral E_z^2 ds = Q_left + Q_interior + Q_right.
    RMS_total = sqrt(Q_total/L).
    Endpoint share = (Q_left + Q_right)/Q_total.

The reported percentage is the share of **squared RMS**, not an additive
percentage of RMS. Endpoint and interior RMS contributions using denominator
L add in quadrature. Conditional interior RMS instead uses denominator L-2a;
its relative error uses the fine-grid signal on that same physical interval.

Total RMS differences are 2.971015e-5 Pa (sigma_n), 1.971941e-5 Pa (p), and
1.021557e-5 Pa (-tau:N). The globally mean-removed sigma_n difference is
2.943932e-5 Pa, so the total difference is not mainly a constant offset.

Here m coarse elements are excluded at **each** end: a=m*h_coarse, equivalent
to 2m fine elements. This avoids comparing different physical interiors.

| m | Cut at each end (m) | Endpoint share, sigma_n | Endpoint share, p | Endpoint share, -tau:N |
|---:|---:|---:|---:|---:|
| 1 | 0.015625 | 80.776% | 81.317% | 76.244% |
| 2 | 0.031250 | 98.252% | 99.045% | 92.494% |
| 3 | 0.046875 | 99.147% | 99.921% | 93.398% |
| 4 | 0.062500 | 99.219% | 99.995% | 93.503% |
| 5 | 0.078125 | 99.254% | 99.999% | 93.751% |
| 6 | 0.093750 | 99.475% | 99.9995% | 95.595% |
| 7 | 0.109375 | 99.568% | 99.9999% | 96.423% |

The small remaining fraction of global squared error does not establish a
small error relative to the much smaller interior signal:

| m | Interior sigma_n RMS difference (Pa) | Difference/fine interior signal: sigma_n | p | -tau:N |
|---:|---:|---:|---:|---:|
| 1 | 1.392598e-5 | 79.59% | 307.89% | 35.91% |
| 2 | 4.536228e-6 | 25.22% | 84.03% | 21.06% |
| 3 | 3.470624e-6 | 19.55% | 26.80% | 21.92% |
| 4 | 3.714150e-6 | 22.71% | 8.57% | 26.38% |

At m=3 the p and -tau:N interior differences are 6.996557e-7 and
3.320077e-6 Pa, respectively: the remaining stress-normal component dominates.
Removing the local interior mean gives relative differences of 19.48%, 26.22%,
and 21.90%, respectively. The result is not explained by an interior offset.
Increasing the cut further does not monotonically improve the relative
comparison; it also removes physical signal. All coarse cuts and all 15
fine-element cuts, including separate left/right contributions, are retained
in the CSV/JSON rather than selecting one favorable exclusion width.

![Error partition and conditional interior comparison](../../../benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/endpoint-error-partition.png)

## Endpoint amplitude and width

The plots use s/h_Gamma at the left and (L-s)/h_Gamma at the right. Only the
horizontal coordinate is scaled; field amplitudes remain in Pa. Width is the
distance to the first half-height crossing of the positive endpoint lobe,
without fitted background subtraction. This is a one-sided half-height width,
not a full FWHM. First-zero widths are also saved. Since both crossings lie in
the first Q1 element, these two width diagnostics are not independent.

| End | Field | Endpoint amplitude 32 / 64 (Pa) | Half-height width 32 / 64 (m) | Width ratio 64/32 |
|---|---|---:|---:|---:|
| Left | Delta sigma_n | 2.207373e-4 / 2.195669e-4 | .00597309 / .00303587 | .5083 |
| Right | Delta sigma_n | 2.209347e-4 / 2.196417e-4 | .00637752 / .00313604 | .4917 |
| Left | Delta p | 1.471028e-4 / 1.464120e-4 | .00611955 / .00307345 | .5022 |
| Right | Delta p | 1.471266e-4 / 1.464202e-4 | .00620889 / .00309551 | .4986 |
| Left | -Delta(tau:N) | 7.363454e-5 / 7.315494e-5 | .00570054 / .00296336 | .5198 |
| Right | -Delta(tau:N) | 7.380804e-5 / 7.322153e-5 | .00674255 / .00322036 | .4776 |

Sigma_n half-width/h changes from .3823 to .3886 on the left and .4082 to
.4014 on the right. Endpoint amplitude changes are below 0.8% for all three
fields. On normalized distance [0,2], profile RMS changes divided by the coarse
endpoint amplitude are 2.23%/2.25% (sigma_n, left/right), .58%/.48% (p), and
5.57%/5.83% (-tau:N). Thus the first lobe collapses well in grid coordinates,
especially in pressure. Further from the endpoint, equal normalized distance
samples different physical positions on the two grids; collapse there is not
a substitute for the common-physical-interval comparisons above.

![Both endpoints in their own grid coordinates](../../../benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/endpoint-scaled-profiles.png)

Conditional on persistence of a fixed-amplitude, O(h)-width layer, its mean
contribution would be O(h), its L2 contribution O(sqrt(h)), and its maximum
would not vanish. This is a scaling implication, not a measured asymptotic
rate or a reason to change an acceptance criterion using only two grids.

## Open surface versus periodic bulk

The saved `pilot64/parameters.prm` Box subsection explicitly sets X periodic
to true. Surface topology is independently open:

- `source/reconstructed_fault/fault.cc`, `Fault::n_cells()`, uses n_vertices-1.
- `source/reconstructed_fault/surface_system.cc` accumulates segment basis
  functions at vertex and vertex+1; K_V sparsity contains adjacent pairs and
  no pair joining the first and last vertices.
- `source/reconstructed_fault/utilities.cc` gives tips constant endpoint
  continuation with clamped segment coordinates, explicitly without joining
  endpoint DoFs. This agrees with `current_design.md` section on polyline
  domain quadrature and `specification.tex`'s corner/tip rules.
- Saved weak-system data have 17/33 independent nodes. Endpoint mass row sums
  are approximately .004842122/.002410889, half the adjacent interior row
  sums .009684245/.004821777; the stored system has no endpoint wrap coupling.

The sharp positive lobe lies in the support of the open endpoint basis, but
the response is **not strictly confined to the two endpoint functions**.
For example, the left first-interior Delta sigma_n is -6.79758e-5/-6.29492e-5
Pa on the coarse/fine grids; the right values are -4.97117e-5/-5.39437e-5 Pa.
Neighboring coefficients and a smaller interior discrepancy remain. The
consistent mass solve couples neighboring represented coefficients.

This is evidence of an endpoint-localized limitation in the current discrete
configuration, not proof that changing the surface topology would remove it.
The endpoint values already nearly match each other. No causal experiment
isolates topology from the other endpoint/domain effects, and no such change
is proposed here.

## Smallest next decision

A matched 128x512 pair, only if separately approved, would distinguish:

1. Whether the endpoint lobe again retains amplitude and halves physical width,
   confirming the grid-scale interpretation beyond a single refinement ratio.
2. Whether the remaining interior -Delta(tau:N) and Delta sigma_n differences
   decrease substantially on **fixed physical interiors**, for example
   [.046875,.203125] m, or persist despite endpoint-layer contraction.
3. Whether signed p and -tau:N cancellation remains stable on those interiors.

The comparison must retain initial-projection differences and cannot treat
subtracting initial errors as removing their influence on subsequent mechanics.
No new mechanism or production correction is inferred from the current data.
The prior tentative cost estimate for 128 remains 10–20 minutes per case and
6–8 GiB, not a measurement or authorization. Approval and an explicit budget
are required before launching it. No third level was prepared or run in this
task. If avoiding that cost is preferred, K2.3 can remain a feasibility pass
with this interior-resolution uncertainty explicitly open, rather than be
declared resolved.

## Reproducibility and checks

Analysis source and generated artifacts are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/`:

- `localize_endpoints.py`: saved-data-only analysis.
- `endpoint-localization.json`, `endpoint-cuts.csv`: exact-Q1 moments, endpoint
  shares, conditional norms, mean-removed checks, and topology measurements.
- `endpoint-scaled-profiles.csv` and `.png`: signed fields for both endpoints.
- `endpoint-error-partition.png`, `endpoint-localization.log`.

Command (exit 0; internal elapsed 0.999 s):

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k23-endpoints-mpl timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/localize_endpoints.py
```

Checks require matched bumped/control coordinates, matching physical domains,
uniform surface spacings, and additive squared-moment partitions to relative
1e-12. The endpoint plots were visually inspected. No ASPECT tests or simulations
were needed or run. Prior production diagnostic changes in the working tree
were preserved, not modified or newly certified by this analysis.
