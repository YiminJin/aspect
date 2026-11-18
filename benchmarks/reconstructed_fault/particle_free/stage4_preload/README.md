# Native fixed-phase preload calibration

This bounded benchmark follows the completed Stage 4 phase-release pair. The
single changed physical preparation parameter is the affine **elastic preload
amplitude** gamma_e. Its boundary displacement is

    u_x = gamma_e (y-2) - k gamma_e (x-4),
    u_y = k gamma_e (y-2),       k = .41954981558864.

The retained native mixed equilibrium calculation supplies stress, pressure and
cohesive memory at the exact imported phase. It holds the common initial
section resistance fixed. The mechanical correction uses the original native
operator, quadrature and criteria. Gamma_e is not inferred by multiplying the
boundary velocity by a ratio of local force to strength. The subsequent boundary
velocity remains .01/s shear with the same extension ratio. This is a family of
fresh synthetic preparations, not an elastic replay of an earlier trajectory.
Changing preload changes cohesive elastic memory; it does not change the initial
age/rate preparation or prescribe an extra slip history.

`prepare.py` copies the tested release checker into a separate build directory,
parameterizes only its benchmark preload amplitude, and includes `scan.h`.
`run.py` selects one fresh preparation and zero to eight physical intervals.
Zero intervals uses ASPECT's existing end-step criterion; it still performs
native initialization and the static equilibrium checks. `batch.py` executes an
explicit small case list sequentially with four ranks total, stopping on any
failure. No production source, old checker or checkpoint is overwritten.

## Admissible damage directions and sign

The original two smooth probes and finite-difference check remain enabled;
their derivation is in `../stage4_release/README.md`. `scan.h` additionally
exports for each native nonnegative Bernstein basis function N_i:

    m_i = integral N_i dx,
    d_i = integral N_i Y dx,
    g_i = (native full AT1/SIPG fracture gradient)_i.

Positive d_i-g_i means that an arbitrarily small positive coefficient increment
lowers the reduced functional, provided that coefficient is below the physical
upper obstacle. The artificial fixed-phase condition is not that physical
obstacle. Face penalties and consistency terms are included in g_i. A favorable
infinitesimal row does not establish a finite-amplitude patch or movement of the
absolute phi=.05 connected contour. Smooth piecewise-linear tents are formed by
nonnegative combinations of these rows, with breaks at existing faces. Their
transverse support is y=1.875..2.125 m and axial half-width .0625 m.

The initial predictor and resistance are the native equilibrated values. Formed
sections keep their historical direction. Completely virgin sections have
exactly phi=A=V=f=0; this is asserted. For these sections the native orientation
objective at zero elapsed time reduces to

    integral [u sin(2 alpha) + v cos(2 alpha)]^2 dx,
    u = -(tau_xx-tau_yy)/2, v = tau_xy.

Its maximizing forward-sector direction is
alpha = atan2(2 integral uv, integral(v²-u²))/4. The observer evaluates this
direction without publishing it. It also integrates the pointwise bound
m(u²+v²)/(2G), which bounds any common section orientation. At later steps it
reads the actual native accepted-stage driving, including the solver's ordinary
virgin-plane selection. This is not a new direction-selection law.

At fixed phase and prescribed initial resistance the native static problem is
linear. Its physical predictor is affine in gamma_e. Fixed-plane driving and
the virgin pointwise envelope are therefore quadratic in gamma_e. Three static
native solves identify those quadratics. Selected-plane driving is not asserted
to be exactly quadratic when its maximizing angle changes: interpolation only
predicts a candidate, which its fresh native preparation must verify.

The phase is inherited unchanged, not silently re-equilibrated at each load.
Existing band/taper adjustment must be reported separately from connected
extension and initially intact nucleation. The law comparison shares the same
preload and 1% sin² age perturbation over x=2.5..5.5 m, and compares native
midpoint stages at identical times. A short release pair is conditional on
measurable weakening-specific excess and initial stability of connected advance;
failure of that criterion ends preload tuning rather than authorizing a longer
growth campaign.

The completed static calibration selects gamma_e=.000365. The first row in the
two axial-tip cells reaches zero at .0003753278135, whereas broad tip/ahead
probes reach zero only near .00064..00068. Fully intact rows can already favor
damage near .000633. This order is why the selected comparison stays below the
first tip-cell row threshold, rather than calibrating to a broad probe alone.
`selection.json` records this decision before the physical pair is launched.
The paired window is eight intervals of 8e-5 s, to .00064 s. `calibrate.py` and
`compare.py` run after numerical ranks exit, and `retain.py` verifies protected
identities and preserves compact evidence without duplicating checkpoint trees.

Both selected eight-interval histories completed. Weakening raises peak slip
rate by 29.42% but gives lower driving in every checked axial-tip row at every
accepted stage. The limiting-row difference is -22.6 microPa against a remaining
.040 Pa deficit. Preload tuning is closed; no conditional release is run.
See `doc/reconstructed_fault/particle_free/STAGE4_PRELOAD_CALIBRATION_REPORT.md`
for the decision, qualifications and next nearer-tip slipping-state preparation.
