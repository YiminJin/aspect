# K5 mature frictional specialization

The opt-in mature mode is implemented and the single bounded comparison
passed. It reproduces the frozen-force diagnostic without evolving shadow
cohesion: C and cohesive storage are zero, H is retained per particle, and
bulk stress/Theta follow their existing accepted-step updates. The 4-rank run
completed initialization plus ten real steps through 29.2419 yr in 18.23 min.
Initial mechanics and the subsequent trajectory agree to numerical accuracy;
no solver, support, pressure or timestep tolerance was retuned. This verifies
the specialization, not a cure for the remaining junction/deep-tip features.

## Formulation and scope

The explicit parameter `Material model / Phase field fault / Fault constitutive mode`
now selects `cohesive` (unchanged default) or `mature frictional`. The latter
requires reconstructed faults and `Evolve phase field = false`; it is not
inferred from the old freeze parameter. The phase handler retains the prescribed
initial phase rather than solving a phase equilibrium equation in this mode.
There is no damage-to-mature transition or late-checkpoint conversion.

The mature balance and minus-V derivative are

\[
 F=\tau_{\rm bg}^{\rm eff}+\Delta\tau:S-\mu(V,\widehat\Theta)\sigma_n-\eta^d V,
 \qquad \sigma_n=\sigma_{n,\rm bg}+\Delta p-\Delta\tau:N,
\]
\[
 -F_V=2\kappa_{\rm bulk}(h/I_h)S:S+\sigma_n\mu_V+\eta^d.
\]

Only cohesive force/storage and `kappa_Gamma/I_h` are absent. The bulk Maxwell
law, normalized `upsilon=(h/I_h)V`, B/G, pressure convention, radiation damping,
nodal split Theta update, prescribed deep Vp and solver safeguards are retained.
C is initialized and committed as exactly zero, including prescribed vertices.
Recoverable cohesive energy is zero. The reduced cohesive H candidate is zero;
initialized H remains inert profile/irreversibility metadata, with no influence
on mature mechanics or a phase evolution. Bulk elastic storage/Maxwell losses,
frictional work and radiation loss remain. Signed tensile traction is not
clamped or declared dissipative. Fixed background work is not cohesive storage.

For the comparison, `tau_bg_eff(s)=tau_bg_old(s)-C_star(s)`. The generic optional
background correction stores three fixed Q1 coefficient fields and subtracts
`a(s)+b(s)/d(s)` **after interpolation**. Here `a=beta0*C0`,
`b=kappa0*V0_accepted`, `d=Ih0` use the captured evaluated-initial-resistance
snapshot. The old retained C0 is input prestress data, not mature history.
No Q1 reprojection changes the mechanical background. This preserves the
accepted initial root; timestep-zero's Jacobian intentionally lacks the spring.

The BP3 output projects this background with the production domain rule **for
diagnostics only**, so `fault_k.csv` consistently decomposes projected total
shear into projected background and perturbation. Raw constitutive samples
continue using the exact rational background. H is explicitly exported as
`H_inert`; C is never a growing shadow history in this mode.

## Checkpoints and implementation

Generic manager storage checkpoints the rational coefficients, original
background and a mature reference-geometry marker. The material checks the
mode marker on resume and rejects changed fault coordinates; the common law
rejects nonzero mature C or a changing profile. BP3 checkpoint version 3 stores
its mode; version 2 is still readable as cohesive only. Reattach selectors by
property name on restart, without rereading the fresh initial prestress file.
No late cohesive state is converted.

Implementation files for this task:

- `include/aspect/material_model/phase_field_fault.h`,
  `source/material_model/phase_field_fault.cc`: mode, common response, zero
  initialization, fixed rational background selection/evaluation, restart guard.
- `source/simulator/phase_field.cc`: explicit mature-profile retention; ordinary
  cohesive phase solves and the meaning of their existing freeze parameter stay unchanged.
- `benchmarks/reconstructed_fault/bp3/mature_fault.h`, `bp3.cc`: fresh prestress
  input, checkpoint mode, output-only background projection, zero-C and inert-H diagnostics.
- `tests/phase_field_fault_test_access.h`, `unit_tests/phase_field_fault_cohesive.cc`,
  `unit_tests/reconstructed_fault.cc`: mature response and checkpoint-data tests.
- `run_mature_fault.py`, `analyze_mature_fault.py`: one capped replay and comparison.
- `current_design.md`, `specification.tex`: authoritative specialization,
  tangent, history rules and energy accounting.

Public additions are `is_mature_frictional_fault()`, the background setter's
optional three-component correction-property argument, and the coordinate-level
`reconstructed_fault_background_tractions(fault,segment,xi)` evaluator. Existing
one-argument background selection retains its previous behavior.

## Bounded verification

Debug and Release builds and the BP3 plugin were built with `-j4`. Focused
Release command:

`build-pf-cpdi/aspect-release --test '[phase_field_fault_cohesive],[mature_fault],Stage-I*'`

One rank and `mpirun -np 2` each passed **730 assertions in 19 cases per rank**.
Tests include repeated imposed slip with exactly zero mature C/storage,
unchanged growth in cohesive mode, rejection of nonzero mature history/changing
profiles, coefficient/geometry checkpoint roundtrip, and existing Stage-I
bound/line-search/exhaustion/rollback safeguards. Logs:
`/tmp/bp3-mature-unit-one.log`, `/tmp/bp3-mature-unit-two.log`.

The tests cover serialization of the fixed data and constitutive rejection
of nonzero C/changing profiles. The mode-switch guards are implemented in
the production resume path, but were not exercised through a full restarted
ASPECT process. This is not an uninterrupted-versus-restarted mature trajectory
comparison. No extra mechanical run was made for that purpose.

Replay commands:

```
python3 benchmarks/reconstructed_fault/bp3/run_mature_fault.py prepare
python3 benchmarks/reconstructed_fault/bp3/run_mature_fault.py run
python3 benchmarks/reconstructed_fault/bp3/analyze_mature_fault.py
```

The fresh 4-rank 50-m case targets 922804465.5975173 s (29.24190894 yr), using
the original recorded timestep prefix and unchanged production restrictions.
Expected cost was 15--22 min; hard simulation cap 2400 s, with no retry. The
output directory is `benchmarks/reconstructed_fault/bp3/mature-fault-50-local4/`;
it preserves input, coefficient data, exact clock, source/executable/plugin
hashes, ordinary checkpoints and separate diagnostic CSVs.

## Result

**Passed.** Exit 0, 1093.794 s wall time, 11 accepted states (0--10), final
time 922804465.59751725 s. All 84 fresh-linear checks passed; largest
fresh/target ratio 0.99757223. Maximum final reported relative nonlinear
residuals were 4.62402e-13 bulk and 8.86751e-9 surface, both below the unchanged
1e-8 target. Every accepted state had 440 free nodes and zero lower-active
nodes. The prescribed deep Vp remained exact.

The reused four-rank derivative probe passed in mature mode: K finite-difference
error 3.60656e-7 -> 3.56508e-8 for perturbations 1e-15 -> 1e-16 m/s;
G pressure error 8.16205e-13. The probe's historical log label is
`Frozen cohesion K/G check`; it exercises the actual selected mature response.

| Final quantity at 29.24190894 yr | Mature | Frozen-force reference |
|---|---:|---:|
| V at 39.95 km (m/s) | 7.295463230762560e-10 | 7.295463230762551e-10 |
| V at 39.90 km (m/s) | 8.099503728903774e-10 | 8.099503728903803e-10 |
| V at 25 km (m/s) | 6.128122738671216e-10 | 6.128122738671214e-10 |
| Slip at 39.95 km (m) | 0.754547575189542 | 0.754547575189541 |
| Last-element slip gradient | 0.003365137808159515 | 0.003365137808159524 |
| Raw junction bulk pressure range (MPa) | [-4.883373, 4.8039245] | identical |
| Raw normal traction, last mixed element (MPa) | [44.62388774, 55.25867527] | same to <1e-7 Pa |

At the last free row, mature q is 26.3496888972 MPa. Reference q is
26.8505278527 MPa with C_star=0.5008389555 MPa: the changed total shear is
the intended prestress subtraction, not a changed mechanical perturbation.
Across every weak row and accepted time, `q_mature-(q_reference-C_star)`
differs by at most **1.45336e-6 Pa** after row-mass normalization.

Maximum trajectory differences: V **1.63191e-19 m/s** (at initialization),
Theta **3.57628e-7 s** / **4.64603e-15 relative**, slip **2.10942e-15 m**,
and I_h **exactly zero**. Initial Theta and I_h are identical. Initial full
VTU bulk velocities, phase and FE Maxwell-history components are identical;
maximum initial pressure-output difference is **0.0009765625 Pa**, within
float32 output resolution. Retained initial particle Maxwell stress is zero.
All initial and subsequent fault coordinates are unchanged.

The frozen-field algebraic audit evaluated the new rational prestress and
the independently captured C0/V0/I0 formula at all **2640 saved raw surface
samples** over these times. Maximum residual-density difference is
**3.72529e-9 Pa**. These are the saved samples, not an assertion that every
historical quadrature point was exported. The live trajectory comparisons
and K/G checks use the complete production domain rule.

Across the trajectory the maximum difference between raw normal-traction
extrema is **2.05263e-5 Pa**. Junction pressure comparisons use 1730 identical
VTU vertices at each available heavy-output state; maximum difference is
**5.96046e-8 Pa**, and the final fields are identical at exported precision.
These are not the much smoother projected weak normal-traction averages.

Stable-ID checks passed for **all 385920 particles at every accepted state**:
H is unchanged from each particle's initialized value. C is exactly zero in
both the stored nodal field and evaluated weak force. Its recoverable energy
is identically zero by the reduced law. No new physical H candidate or
growing shadow C/H is published.

`comparison/summary.json`, `field_errors.csv`, `trajectory.csv`,
`raw_normal_extrema.csv`, `pressure_comparison.csv`, and
`saved_field_residual.csv` contain the checks and numbers. `analysis.log`
records the successful final analyzer run. Existing reference artifacts were
read only and preserved. `git diff --check` passed.

## Remaining scope

No later continuation, refinement campaign, parameter scan, or full restart
trajectory was run. The residual junction pressure/slip-gradient feature and
the separate deep-tip response remain those of the frozen-force reference.
This task does not change Theta discretization, history transfer, or endpoint
topology, and does not establish long-event accuracy or eliminate their known
limitations. The mature specialization and this bounded verification are ready
for review; the ordinary BP3 fixtures still default to the cohesive mode unless
the new parameter is selected explicitly.
