# Restored BP3 filter comparison — 2026-09-25

## Decision

Use **Helmholtz length 20 m** as the provisional choice for the next long
research run, not 40 m. It is the smaller tested nonzero length, already removes
most grid-scale normal-input variation, and improves the small bottom velocity
artifact. The additional smoothing at 40 m produces only a modest additional
velocity improvement and no solver-iteration benefit. This is a startup-based
choice, not validation of long-term/event dynamics or an official BP3 model.

```
subsection Postprocess
  subsection BP3 restored monitor
    set Friction normal input = helmholtz
    set Normal filter length = 20
  end
end
```

Do not reinterpret raw stress as filtered stress. The filter changes the normal
input to friction (including its consistent coupling derivatives), not the
bulk stress/history representation. Keep this physical length fixed if the
grid changes; this recommendation is for the current ell=20 m fixture.

## Comparable runs and checks

The original PRMs differ only in output directory and filter mode/length.
Each run has accepted states 0--10, with two Newton updates at every state,
493 total Krylov iterations, minimum accepted alpha=1, 2889 free nodes and
zero lower-active nodes. Fresh-linear checks pass at every accepted state;
all normalized nonlinear residuals are below 1e-8. Maximum reported surface
RMS residual is 0.005245 Pa; the independent Theta audit is within 2.23e-16.

Final time is 1133.7390530913 s (18.90 minutes). The maximum corresponding-time
difference is 4.17e-11 s, so no temporal interpolation is needed. The Q1 grid
is identical (19.982--20.000 m spacing). Incoming Theta and committed Theta
are kept distinct in the saved data. No mechanics was rerun for this analysis.

## Measured suppression

Final actual **friction-input** extrema, minus the 50 MPa background:

| Mode | Minimum (Pa) | Maximum (Pa) |
|---|---:|---:|
| Raw | -15.6931 | 38.3880 |
| 20 m | -0.17893 | 1.13515 |
| 40 m | -0.01037 | 0.43387 |

All raw and friction-input stresses remain compressive. In the artificial
timestep-zero initialization, the corresponding ranges are [-53722.6,124943.9],
[-2775.15,26.959] and [-1739.87,26.596] Pa. These initialization responses must
not be confused with growth during real physical timesteps. The first real
step's raw range is already only [-1.539,3.526] Pa.

For tangential roughness, use each node's departure from its neighboring
physical-distance-weighted chord, then take RMS with production row masses.
Endpoints have no two-sided chord and retain the existing zero-chord convention;
endpoint profiles and actual QP extrema are checked separately.

Final normal-traction chord RMS (Pa):

| Window | Raw Q1 observation | 20 m friction Q1 | 40 m friction Q1 |
|---|---:|---:|---:|
| Top 1 km | 0.0049793 | 0.00063314 | 0.00017061 |
| 13--20 km | 0.00063881 | 0.000055589 | 0.000015027 |
| Bottom 1 km | 1.01071 | 0.098948 | 0.026198 |

Thus 20 m suppresses this tangential measure by roughly 8--11 times.
The raw Q1 column is the production consistent projection for observation,
**not** the pointwise raw input actually used by raw friction. There is no
zero-length projected control, so raw-to-filter differences cannot be attributed
solely to the Helmholtz length rather than projection plus filtering.

The separate production-QP comparison uses actual JxW*chi work weights and
includes transverse as well as tangential variation. Final friction-input
standard deviations in the top/bottom 200 m are respectively:

| Mode | Top (Pa) | Bottom (Pa) |
|---|---:|---:|
| Raw | 0.63706 | 2.69703 |
| 20 m | 0.0045805 | 0.20184 |
| 40 m | 0.0028349 | 0.073902 |

At 15 km +/-100 m they are 1.74809, 3.457e-5 and 2.441e-5 Pa.
These are variations across the measured QPs, not exclusively grid-scale
roughness. Actual raw stress persists: the filtered runs' bottom raw-QP RMS
values remain 2.696996 and 2.696989 Pa. The filter does not repair bulk stress.
The filter preserves its own global weak mean to about 4.2e-9 Pa; this does
not require the independently re-equilibrated runs to have identical means.

## Mechanical response and growth

Final bottom-1-km velocity chord RMS / Vp:

- Raw: 5.9573e-7.
- 20 m: 2.1584e-7 (64% reduction).
- 40 m: 1.9408e-7 (only 10% further reduction relative to 20 m).

The top values are 1.3140e-8, 9.1670e-9, 8.8631e-9. These are very small
absolute velocity effects, not evidence of a resolved instability.
The endpoint roughness increases over physical steps 1--10 in all modes;
filtering reduces its amplitude/growth, but this short interval cannot establish
long-term boundedness.

The localized velocity feature at 15 km remains: the largest neighboring-chord
departure at the final state is **1.0892% of Vp in all three runs**. The
13--20 km velocity-chord RMS decreases from about 0.00124 Vp at step 1 to
0.000948 Vp at step 10, with essentially coincident trajectories. Increasing
the normal filter length does not remove the transition feature, so it should
not be increased to try to cure that feature.

Relative to raw, final maximum nodal |delta V|/Vp is 1.9854e-6 (20 m) and
2.2832e-6 (40 m), concentrated near the bottom. Whole-fault work-weighted RMS
differences are 3.7503e-8 and 4.1428e-8 Vp. Final maximum slip differences are
1.2244e-12 and 1.4063e-12 m (relative differences at most 1.08e-6 and 1.24e-6).
Maximum relative committed-Theta differences are 1.53e-10 and 1.76e-10.
Tiny differences should not be given more significance than solver accuracy.

Initial projections/response have not been subtracted away: the maximum
initial |delta V|/Vp is 0.001517 (20 m) and 0.001595 (40 m); initial Theta is
identical. Initial bottom velocity-chord RMS is reduced from 5.216e-4 to
2.084e-4 and 1.891e-4 Vp. These are distinct from later history evolution.

Logged wall times are 750, 675 and 593 s. Identical iteration counts and a
single sequential run per mode do not establish a filter-induced speedup.

## Artifacts and limits

- `final_profiles.png`: matched final profiles, endpoint close-ups and delta V.
- `growth.png`: step-1 onward roughness, separate from artificial initialization.
- `nodal_metrics.csv`, `quadrature_metrics.csv`, `trajectory_differences.csv`,
  `summary.json`: quantitative values and definitions used above.

Reproduce from the repository root with:

```sh
python3 benchmarks/reconstructed_fault/bp3/compare_restored_filters.py \
  benchmarks/reconstructed_fault/bp3/filter-test
```

No production parameters or source were changed. No new trajectories were run.
This study cannot qualify normal-stress feedback during nucleation/rupture:
it spans only 19 minutes and physical-step normal-stress changes are tiny.
Continue monitoring raw versus filtered traction and endpoint/interior rate
roughness in the long run; do not increase the filter in response to a failure
without diagnosing that failure.
