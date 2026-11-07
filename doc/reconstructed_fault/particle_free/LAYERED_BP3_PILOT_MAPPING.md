# Mapping the layered qualification to a BP3-like pilot

This is a parameter/implementation mapping, not a submitted production input or
permission to resume a retained checkpoint. Final readiness is assessed in
[LAYERED_BULK_CAMPAIGN_REPORT.md](LAYERED_BULK_CAMPAIGN_REPORT.md).

The present qualification is a bounded algorithmic lower/upper-viscosity example,
not the full viscoelastic-crust paper campaign. Its mature frictional physical
stress and age/slip histories must not be merged with cohesive elastic-slip,
fracture inventory or tip-growth claims.

| Item | Current layered qualification | Retained BP3 first-event input | Required pilot decision |
|---|---|---|---|
| Geometry | Interior finite dipping segment, compact Q1 profile, fixed mesh |150×50 km domain, restored fault-support mesh, boundary completion |Choose fixed-fault or advected material geometry explicitly; define surface intersection/completion and preserve physical tips |
| Shear modulus |10 Pa|32,038,120,320 Pa|Use the BP3 modulus; retain full 3D plane strain |
| Viscosity |1–10 Pa s, tanh width .35 m|1e26 Pa s|Specify actual lower-crust relaxation times eta/G, transition depth and physical width; this is a physical model change |
| Bulk boundary velocities |Equilibrated scalar shear profile plus U|BP3 plate loading, Vp=1e-9 m/s|Construct a background compatible with the actual BP3 loading, not the present horizontal-layer shear formula |
| Top traction / pressure |(q,0), p=tau_yy, no gravity|Natural zero total traction, no gravity; prescribed 50 MPa fault background|Preserve the intended total traction and effective normal loading; the synthetic pressure reference is not a replacement |
| Background shear on fault |3.7+.15 cos(3 x_initial) Pa (selected finite-slip comparison)|26.546122365 MPa|Retain/derive BP3 preload and avoid counting it again in exterior physical stress |
| RSF |f0=.6, a=.025, b=.015, Dc=.008 m, Vref=.001 m/s|f0=.6, a=.010/.025, b=.015, Dc=.008 m, Vref=1e-6 m/s|Restore spatial velocity-weakening/strengthening distribution and BP3 initial state |
| Radiation damping |1 Pa s/m|4,624,440 Pa s/m|Restore physical impedance; event-rate timestep requirements differ |
| Initialization interval |.1 s; mechanics gamma accepted, stress remains zero|4e6 s|Audit the physical initial equilibrium, stored history and accepted spin before fresh startup |
| Stress history |Direct DGQ2, SSPRK3, accepted bulk-slip-subtracted finite rotation|Retained stationary DG default evolution|Fresh selected objective/direct trajectory only; no automatic checkpoint conversion |
| Profile / labels |Bounded Q1 transport, .1 m initial labels|20 m length scale, 10 m structural spacing, frozen profile|Resolve physical localization and label measures; fixed support and material motion are distinct physical examples |
| Timestep |Bounded uniform cap plus existing slip/state controls|Maximum4e7 s, first4e6 s, CFL .5|Select dt relative to relaxation, transport cell crossing, accumulated spin, slip/Dc and state change; synthetic seconds do not transfer directly |
| Checkpoint |Native material identity plus accepted benchmark background|Existing production first-event schema/physics|Start fresh with explicit exterior identity; store its actual accepted sequence and test a matching continuation |

The BP3 values above are read from
`benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/parameters.prm`
and that directory's retained README loading table. Those inputs and server
outputs are unchanged.

For the fixed-fault lower-crust example, the scalar background construction,
pressure balance, layered constitutive coefficients and timestep diagnostics
carry over. Geometry/profile transport is deliberately absent there and must
be qualified through a supported stationary layered configuration; setting U=0
in the present transported example does not create it. For the advected example,
use the complete material transaction, measured profile/stress transfer errors,
accepted-time exterior and restart semantics. Neither example is a realistic
subduction model. Changing spatial loading or adding an event requires its own
physical comparison, even when this benchmark's numerical checks pass.

The corrected campaign preserves the constitutive distinction explicitly:
`Constitutive spin = bulk slip subtracted` is the selected new law; total spin
is only its paired control. Neither a retained total-spin first event nor an
old accepted-trace trajectory can be relabeled or resumed under the corrected
law/prescribed exterior. Use fresh initialization and a provider identity that
encodes the actual dimensional boundary model.

For pilot parameter selection, record all of `dt*G/eta(y)`, material cell-crossing
CFL, `dt*omega_b`, `dt*chi*V/2`, `V*dt/Dc`, and predicted `abs(log(Theta_next/Theta))`.
Uniform U affects cell crossing but not spin directly. The present RSF parameters
are velocity strengthening everywhere, so the existing operator-split slip CFL
is inactive; the age-change bound supplies the state-based restriction. Restoring
BP3's velocity-weakening region activates that slip limit and changes the
physical regime. The synthetic cap in seconds must therefore not be copied into
BP3. Final measured cost/accuracy and the pilot readiness decision belong to the
corrected campaign report, not this mapping table alone.

A uniform unit rescaling is insufficient. The synthetic background compression
ratio is 5/10=.5, whereas retained BP3 uses 50e6/32038120320≈.001561. Choosing
units to match G therefore cannot also match normal loading. Likewise, the
compact transported profile and BP3's phase length/normalization must be mapped
through their actual chi=h(phi)/I, rather than identifying different width
parameters. A useful dimensional pilot specification records G, sigma_eff/G,
eta(y)/G, Dc, actual chi, imposed plate velocity and the compatible far-field
stress/traction together. The retained eta=1e26 Pa s corresponds to eta/G≈3.12e15 s;
it is not the relaxing lower layer in this synthetic experiment.

The completed uniform/adaptive comparisons narrow the synthetic policy: cap
dt at .00625 s and use the existing .05 predicted log-age setting when resolving
the initial age transient is useful. Its actual maximum age change is .08273,
not .05; timestep selection precedes mature publication and endpoint V is unknown
to the predictor. The adaptive run costs 97.95 minutes versus 71.37 minutes for the
uniform cap on four ranks. These measured costs and sensitivities apply only to
this velocity-strengthening fixture. Matching 4→2-rank restart and provider
rejection checks pass, but the adaptive profile mass defect is 7.26%; spatial
refinement lowers the corresponding uniform-step defect to 2.39%. Thus a fresh
advected BP3-like pilot still needs a material-equation-consistent bounded profile
transport before spending production event-scale cost. A smaller dt alone is
not that fix. A stationary layered example avoids this particular remapping
defect, but still requires the supported stationary constitutive/boundary
configuration and the dimensional decisions in the table.
