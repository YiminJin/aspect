# Corrected full-normal-support coupled measurements

Q → M denotes Q1 remap → bounded MacCormack. Errors are percentages relative to the same 64², dt=.025 bounded reference for each coefficient/direction. These are numerical references, not exact solutions.

| Viscosity, direction | Cells, dt | Profile L2 | I L2 | Stress L2 | Point traction L2 | Work traction L2 |
|---|---|---:|---:|---:|---:|---:|
| Constant, + | 16², 0.1 | 15.241 → 12.789 | 16.376 → 12.019 | 13.552 → 13.147 | 7.652 → 7.886 | 10.024 → 8.653 |
| Constant, + | 32², 0.05 | 4.699 → 2.882 | 5.898 → 2.912 | 3.895 → 3.559 | 2.752 → 2.415 | 2.897 → 2.120 |
| Constant, − | 16², 0.1 | 15.241 → 12.789 | 16.376 → 12.019 | 13.524 → 13.120 | 8.009 → 8.359 | 9.920 → 8.572 |
| Constant, − | 32², 0.05 | 4.699 → 2.882 | 5.898 → 2.912 | 3.806 → 3.461 | 2.717 → 2.394 | 2.844 → 2.075 |
| Layered, + | 16², 0.1 | 15.241 → 12.789 | 16.376 → 12.019 | 13.619 → 13.224 | 7.701 → 7.995 | 9.992 → 8.609 |
| Layered, + | 32², 0.05 | 4.699 → 2.882 | 5.898 → 2.912 | 3.863 → 3.524 | 2.712 → 2.385 | 2.874 → 2.090 |
| Layered, − | 16², 0.1 | 15.241 → 12.789 | 16.376 → 12.019 | 13.610 → 13.201 | 7.889 → 8.231 | 10.051 → 8.683 |
| Layered, − | 32², 0.05 | 4.699 → 2.882 | 5.898 → 2.912 | 3.845 → 3.496 | 2.743 → 2.409 | 2.902 → 2.122 |

## Separate spatial/time controls, positive direction

| Viscosity | Cells, dt | Profile Q → M (%) | Stress Q → M (%) |
|---|---|---:|---:|
| c | 16², 0.1 | 15.241 → 12.789 | 13.552 → 13.147 |
| c | 32², 0.1 | 4.326 → 2.877 | 4.034 → 3.729 |
| c | 16², 0.05 | 15.448 → 12.756 | 13.480 → 13.116 |
| c | 32², 0.05 | 4.699 → 2.882 | 3.895 → 3.559 |
| l | 16², 0.1 | 15.241 → 12.789 | 13.619 → 13.224 |
| l | 32², 0.1 | 4.326 → 2.877 | 3.978 → 3.676 |
| l | 16², 0.05 | 15.448 → 12.756 | 13.564 → 13.206 |
| l | 32², 0.05 | 4.699 → 2.882 | 3.863 → 3.524 |

## Shape, state, normalization and cost

| Positive, 32² dt=.05 | Peak | Mass drift (%) | Normal variance (m²) | Cumulative slip error (%) | V error (%) | Age error (%) | Z error (%) | Wall (s) | Linear solves / iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c, q | 0.339196 | 0.00006 | 0.0333232 | 0.04569 | 0.12213 | 0.000166 | 0.04677 | 33.29 | 17 / 635 |
| c, m | 0.346404 | 0.00986 | 0.0326425 | 0.04398 | 0.11837 | 0.000159 | 0.04498 | 30.38 | 17 / 635 |
| l, q | 0.339196 | 0.00006 | 0.0333232 | 0.04577 | 0.12190 | 0.000166 | 0.10982 | 75.03 | 25 / 948 |
| l, m | 0.346404 | 0.00986 | 0.0326425 | 0.04408 | 0.11818 | 0.000160 | 0.10787 | 71.26 | 25 / 948 |

Maximum measured normal support loss: 0.000e+00. Maximum absolute integral normalization error |∫χ/LΓ−1|: 4.399e-04.
Native profile replay maximum: 4.996e-16; donor-bound violation: 5.551e-17. Age/memory source maxima: 1.776e-15 / 5.421e-20.
Projection orthogonality defect maximum: 9.093e-24 J/m. Cell crossings summed over labels/steps: 340.

## Energy and material flux, positive direction

All signed and absolute sums are retained separately; no cancellation-based pass criterion. Power scale is the time-integral of the absolute active bulk terms, not the much larger cancelling background/friction work.

| Case | Endpoint signed / absolute (J/m) | Trapezoid signed / absolute (J/m) | Endpoint / trapezoid relative (%) | Net material flux integral (J/m) | Flux error / reference active scale (%) |
|---|---:|---:|---:|---:|---:|
| c-q-space | -2.5889e-09 / 2.5889e-09 | 1.2361e-10 / 1.2361e-10 | 33.297 / 1.590 | -1.0109e-11 | 0.0254 |
| c-m-space | -2.6042e-09 / 2.6042e-09 | 1.2779e-10 / 1.2779e-10 | 33.312 / 1.635 | -1.0112e-11 | 0.0254 |
| c-q-joint | -1.3309e-09 / 1.3309e-09 | 2.0199e-11 / 2.0768e-11 | 17.197 / 0.268 | -1.0533e-11 | 0.0201 |
| c-m-joint | -1.3389e-09 / 1.3389e-09 | 2.3339e-11 / 2.3339e-11 | 17.204 / 0.300 | -1.0534e-11 | 0.0200 |
| c-m-ref | -6.9035e-10 / 6.9035e-10 | 1.4241e-12 / 5.7213e-12 | 8.741 / 0.072 | -1.2117e-11 | 0.0000 |
| l-q-space | -2.4543e-09 / 2.4543e-09 | 1.4621e-10 / 1.4621e-10 | 31.453 / 1.874 | -8.6808e-12 | 0.0244 |
| l-m-space | -2.4690e-09 / 2.4690e-09 | 1.5046e-10 / 1.5046e-10 | 31.471 / 1.918 | -8.6845e-12 | 0.0244 |
| l-q-joint | -1.2644e-09 / 1.2644e-09 | 2.9936e-11 / 2.9936e-11 | 16.280 / 0.385 | -9.2097e-12 | 0.0177 |
| l-m-joint | -1.2722e-09 / 1.2722e-09 | 3.3094e-11 / 3.3094e-11 | 16.289 / 0.424 | -9.2116e-12 | 0.0177 |
| l-m-ref | -6.5664e-10 / 6.5664e-10 | 6.0169e-12 / 6.0169e-12 | 8.287 / 0.076 | -1.0616e-11 | 0.0000 |

## Independent coupled work checks

| Case | Coupled signed / absolute (J/m) | Relative active / gross (%) | Max physical-step surface work residual (W/m) | Final raw face / weak reaction power (W/m) |
|---|---:|---:|---:|---:|
| bpf-c-p-q-space | -2.5889e-09 / 2.5889e-09 | 33.297 / 0.001022 | 5.666e-15 | -2.461e-06 / -1.212e-20 |
| bpf-c-p-q-joint | -1.3309e-09 / 1.3309e-09 | 17.197 / 0.000526 | 2.940e-16 | -2.809e-06 / 2.389e-20 |
| bpf-c-p-m-space | -2.6042e-09 / 2.6042e-09 | 33.312 / 0.001028 | 4.432e-15 | -2.462e-06 / 6.034e-21 |
| bpf-c-p-m-joint | -1.3389e-09 / 1.3389e-09 | 17.204 / 0.000529 | 2.982e-16 | -2.810e-06 / -8.733e-22 |
| bpf-c-p-m-ref | -6.9035e-10 / 6.9035e-10 | 8.741 / 0.000273 | 2.206e-17 | -3.103e-06 / 1.613e-20 |
| bpf-l-p-q-space | -2.4543e-09 / 2.4543e-09 | 31.453 / 0.000969 | 4.970e-15 | -2.356e-06 / -8.518e-21 |
| bpf-l-p-q-joint | -1.2644e-09 / 1.2644e-09 | 16.280 / 0.000500 | 2.891e-16 | -2.655e-06 / 4.857e-21 |
| bpf-l-p-m-space | -2.4690e-09 / 2.4690e-09 | 31.471 / 0.000975 | 4.413e-15 | -2.357e-06 / 1.896e-21 |
| bpf-l-p-m-joint | -1.2722e-09 / 1.2722e-09 | 16.289 / 0.000503 | 2.905e-16 | -2.655e-06 / 1.697e-21 |
| bpf-l-p-m-ref | -6.5664e-10 / 6.5664e-10 | 8.287 / 0.000260 | 2.573e-17 | -2.836e-06 / -8.178e-22 |
