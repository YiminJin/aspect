# Bounded cap-flank and right-boundary diagnosis

20 October 2026. Entry f1f9f29cf plus all retained bulk/P4a/manual work. The
current user selects exactly three fresh trajectories, in the order below.
No production source, constitutive parameters, continuum seed, loading or
acceptance tolerances change. Earlier physical energy and spatial failures remain.

| Control | Domain [m] | Cells | dx, dy [m] | Bulk/history rule |
|---|---|---|---|---|
| Fine quadrature | [0,1]×[0,1] | 32×64 | 1/32, 1/64 | Gauss8 |
| Square cells | [0,1]×[0,1] | 32×32 | 1/32, 1/32 | Gauss6 |
| Right distance | [0,1.5]×[0,1] | 48×64 | 1/32, 1/64 | Gauss8 |

The distance control follows inspection of the numerical controls. Compare it
with the fine Gauss8 run using the same near-field cells and quadrature. Its
purpose is one-boundary sensitivity, not domain independence. Gauss8 is chosen
before seeing the results, to avoid relying on the earlier coarse-rule control.
Each new layout samples the unchanged continuum seed at its own vertices and
starts unloaded; no accepted local quadrature history is interpolated.

Keep dt=.0025 s, physical accepted end=.16 s, the .05–.15 s affine simple-shear
rate .018 s⁻¹, final zero-motion hold, all-boundary ux_dot=rate*(y−.5), uy=0,
and all original material, KKT/residual/projection/ledger tolerances. Moving the
right side does not change any prescribed velocity at a given physical y.
Retain Gauss3 source projection, unlimited DGQ2, constant p_N and stationary
diagnostic particles. All are serial bounded local runs, timeout 1800 s for
fine-rule, 900 s for square and 2400 s for the distance case. No server job.

Measure initialization against the same continuum separately from increments.
Compare common physical times and the actual weak nodal contributions:
g_f, g_m=−integral N Y, g=g_f+g_m, plus their lumped-mass densities [Pa].
Those densities are diagnostic representations of weak quantities, not a new
pointwise phase equation. For display only, interpolate ordinary FE nodal fields;
never use that interpolation to transfer local a/k/L/q/rate histories.

Fixed diagnostic masks: lower/upper cap flanks x in [.625,.8125], y in
[.25,.4375] / [.5625,.75]; core-tip window x in [.45,.85], |y−.5|<=.05;
the earlier tip window |y−.5|<=.2 is also retained. Separate right-face and
last 1/32 m strip damage from those masks and from left-boundary damage.
Line cuts use y=.3125,.5,.6875 and x=Lx. Track seed-connected centerline
contours .001,.01,.1,.3,.5, explicitly distinct from a disconnected boundary
island. Also check the connected component's off-axis right extent.

Use the previously declared quadrature comparison screens (phase Linf .001,
storage/work differences .1% on max(1 J/m,reference)) without relaxing them.
Report the fine-rule flank amplitude difference and local weak residual even
when the broad screen passes. Square/distance cases diagnose sensitivities,
not establish a new convergence threshold. Keep the original failed 1% endpoint
work screen and its dt-halving explanation; no further timestep sweep is selected.
Stop after the three controls and recommend the smallest supported specimen
change, without implementing a changed physical experiment automatically.
