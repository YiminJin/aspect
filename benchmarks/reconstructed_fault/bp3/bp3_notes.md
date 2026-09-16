# Modified BP3-QD: intentional differences and initialization

Physical authority: supplied SEAS BP3-QD (2021-10-01), equations 4--19,
25--26 and Table 1. Erickson et al. supplies domain/resolution context, not
new physical parameters. Old BP3 files are not the implementation baseline.

- ASPECT uses x=78867.5134594813-x_BP3, y=100000-z_BP3 (meters). This proper
  180-degree rotation makes positive manager V equal official positive thrust
  slip without changing the framework's CCW-normal convention. The displayed
  fault consequently dips left, at 60 degrees. Official station distance is
  measured down dip, not vertically. Endpoints and the 40-km transition are
  retained explicitly. Independent surface endpoints are not identified.
- The 100x100-km box truncates the half-space. Side velocities equal the
  official rigid translations; top and bottom have zero **perturbation** traction.
  Deep surface nodes have exact
  V=Vp, replacing their friction equations. Finite-domain uncertainty is not
  investigated by this short pilot. Bulk pressure is compression-positive
  **Delta p**, with no normalization/shift and no independent pore-pressure field.
  Actual fault normal stress is 50 MPa + Delta p - Delta tau:N, not a
  prescribed/adiabatic replacement for normal-stress feedback.
- The current incompressible Maxwell bulk is not the official nu=.25 elastic
  bulk. G=rho*cs^2=32038120320 Pa and damping=rho*cs/2=4624440 Pa s/m.
  eta=1e26 Pa s gives a relaxation time about 99 million years, making Maxwell
  relaxation negligible over a short run but not changing incompressibility.
- ell=400 m, AT1, core phi=.6, cohesion=1 MPa, Gc=1e5 J/m^2 and curvature=1
  define the additional finite-width/cohesive model; these are not official
  BP3 data. The directly prescribed Q1 stationary distance profile is frozen,
  including its boundary values. H uses the current stationary law and .1
  activation. Phase is frozen; retained histories still follow the production
  history-publication cycle. No homogeneous-Neumann
  initialization solve is used to define the ridge.
- Official Vinit and Theta0 are retained as initialization targets. The
  numerical Maxwell initialization interval is 4e6 s, not the first physical
  timestep; it retains supplied histories under the current timestep-zero
  lifecycle. Real steps use the existing controller, without a benchmark cap.
- The bulk strengthening and initial-Theta extensions use physical depth,
  `xd_equiv=(100000-y)/sin(60 degrees)`. Their interfaces are horizontal at
  y=87009.6189432334 and 84411.5427318801 m. This is a 2598.0762113533 m
  vertical interval, preserving the official 3 km transition along the fault.
  Surface Theta0 retains the official sharp-fault value; deep prescribed Vp
  still uses true down-dip distance, not the bulk depth extension.
- All three particle Maxwell components start at zero. A frozen generic
  two-component surface property supplies background shear/normal traction.
  Its shear is the consistent Q1 weak projection of initial cohesion +
  friction + damping at sigma_bg=50 MPa and official supplied nodal Theta0.
  Evaluated initial cohesion is kappa_Gamma*Vinit/Ih0 + beta_Gamma*C0, not
  merely stored C0. The discrete correction to tau_BP3 + projected C_eval
  is recorded, not absorbed into Theta. Offsets are never updated with C.
- The analytic Airy implementation in bp3_model.h, the frozen Airy plugin in
  airy_dt0/airy.cc and the prestress_audit/ evidence remain diagnostics only.
  No Airy field enters current bulk history, pressure, or boundary loads.

Only initialization and three real steps are intended. Smoke near-fault
spacing is 97.65625 m; pilot spacing is 48.828125 m. Neither is a converged
BP3 reference. The rejected Airy initialization is preserved in
`stage_K5_initialization_report.md` and `stage_K5_prestress_audit.md`.
The current stress-change result is documented separately in
`stage_K5_perturbation_report.md`. First run `perturbation_initialization.prm`;
`perturbation_smoke.prm` is the gated three-real-step case. The plugin requires
genuine convergence and verifies retained/updated Theta and zero t=0 stress.

`fault_N.csv` shear and `constitutive_normal_N_rank0.csv` normal/pressure weak
loads belong to the accepted pre-publication mechanics, never a second update
from already committed history. `analyze_perturbation.py` joins these into
`perturbations_N.csv` and `stations_N.csv`, exporting tau_bg, delta_tau,
tau_total, sigma_n_bg, delta_sigma_n, sigma_n_total separately. `C_evaluated`
is the accepted mechanical resistance; `C` and `Theta` are retained/committed
histories. A consistent-mass projection of residual on all nodes can be nonzero
near the deep essential/free interface even when every free weak row converges:
check `weak_residual` and the solver's restricted surface RMS, not that projected
field alone. Deep prescribed rows carry reactions, not solved friction equations.

Bulk VTU `tau_xx/tau_yy/tau_xy` are the transferred **old perturbation histories**
used by mechanics, not total BP3 stress or the newly committed stress at that
output time. `history_N.csv` separately records the committed particle stress
maximum and independent split-aging-law check. No live FE fields are refreshed
for output. Restart packaging is not implemented; the plugin rejects resume
explicitly before reading a checkpoint. The smoke passed initialization and
one real step, then hit its wall cap in step 2. Do not treat `bp3_smoke.prm`
or `bp3_pilot.prm` as a verified three-step dynamics configuration yet.

Latest bounded follow-up: exact I_h reuse reduces repeated preparation from
about 68 s to 1.4 s. `cached_smoke` records the 1e-12 active-bound audit and
the user-requested stop. `cached_smoke_low_floor` tests the authorized 1e-20
bound and horizontal initial-Theta extension. It retains identical accepted
surface states 0/1, but step 2 exposes cancellation in the bound-contact trial
and subtract/add trial publication. No numerical fix or fine pilot was run;
see `doc/reconstructed_fault/bp3/stage_K5_cache_and_bound_report.md` for the
separate correction proposal. The base smoke's explicit 1e-12 setting remains
unchanged; select the comparison wrapper to reproduce the lower-floor test.
