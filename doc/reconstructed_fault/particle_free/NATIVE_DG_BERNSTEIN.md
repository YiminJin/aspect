# Native DG Bernstein phase: reading guide

The active bulk-tip fixtures select `Discretization / Phase basis = Bernstein`.
The ordinary default is `nodal`, including existing inputs and checkpoints. This
is a change of coordinates in the same discontinuous Q2 polynomial space, not a
new constitutive law or time integrator. Qualification is in the
[bounded report](NATIVE_DG_BERNSTEIN_REPORT.md).

## Field and element

There is one semantic `phase_field` composition in the simulator's mixed FE and
solution vectors. Its cell coefficients are now b, with

    phi(x) = sum_i b_i B_i(x),     a = E b,
    E_ij = B_j(x_i),              b = inverse(E) a.

Here a denotes the retained nodal interpolant. In two dimensions the Bernstein
ordering is i = i_x + 3 i_y. Current, old and old-old phase vectors, the physical
accepted phase/obstacle, and the coupled phase unknown use the selected native
basis. Samples returned by FEValues are physical phi, never raw coefficients.
Auxiliary tensor histories remain ordinary DGQ2; their representation is unchanged.

[`FE_DGQBernstein`](../../../include/aspect/fe_dgq_bernstein.h) derives from
`dealii::FE_DGQ`'s polynomial constructor. It reuses cell-local L2 topology,
tensor numbering, polynomial FEValues, the all-ones constant mode, refinement
embedding and additive L2 restriction. Generalized Lobatto samples produce the
same interpolant via inverse(E); the element has no fictitious ordinary nodal
support points. Interpolation, exact face support, clone and registered name are
provided by the subclass. Installed deal.II 9.6's Bernstein generator supplies
normalization; its existing polynomial root-product evaluator avoids cancellation
near intact endpoints, also for derivatives. No phase clamp is used.

Degree two is the admitted phase degree. Small element tests cover degrees 0–3;
this is not an arbitrary-order phase qualification. Full-dimensional cells only
are admitted: inherited codimension transfer substitutes a different element.

## Operations and ownership

- [`Introspection`](../../../source/simulator/introspection.cc) isolates the phase
  base element from neighboring ordinary/stress compositions. The existing
  semantic composition registration is used; retired separate native DG
  registration remains rejected. Legacy separate Q1 is unchanged.
- [`Simulator::set_initial_temperature_and_compositional_fields`](../../../source/simulator/initial_conditions.cc)
  samples the existing composition initial condition at generalized points and
  uses ordinary FE interpolation to compute coefficients. Setting b_i=f(x_i)
  would change the seed and is not done.
- [`DGOperator`](../../../include/aspect/phase_field/dg_operator.h) assembles the
  same mapped transport/SIPG operators directly with the selected element.
  No duplicate phase field, face assembler, or evolution driver is introduced.
- [`BulkTipCoupling`](../../../source/simulator/solver/bulk_tip_assembly.cc) now
  takes native Bernstein phase directly. Its velocity and pressure coordinates
  remain u=dt*v and pi=p/G. Fracture rows/columns need no E transformation on this
  path. The nodal reference retains a local compatibility map, without a second
  physical solver.
- One ghosted trial vector supplies mechanics and phase. Fixed quadrature basis
  values and accepted parent phase samples are cached for that coupling object's
  single attempt. Mesh, mapping, quadrature and parent are immutable during its
  lifetime; rebuilding the object invalidates them. Trial point responses and
  section roots are recomputed, not cached across iterates.

The coordinate identities are M_B=E^T M_N E, r_B=E^T r_N(Eb), and
J_B=E^T J_N E. For transport the inverse mass yields L_B=E^-1 L_N E. The material
weak form retains the phi div(v) term, upwind traces, boundary inflow, and physical
JxW. Dense QGaussLobatto(4) mass, Lobatto5 transport, Gauss5 fracture and Gauss6 tip
mechanics remain unchanged. SSPRK3, BE tip evolution and the accepted-gradient
spin ordering are unchanged.

## Two different bounds

Growth keeps the existing coefficient box b_parent <= b <= 1 and the
release-capable KKT solve. This sufficient polynomial obstacle is stricter than
all pointwise-admissible Q2 fields. Zero-clock seed equilibration, lower-bound
arithmetic and final mechanics/KKT/energy gates remain unchanged.

Transport uses [`Q2Bounds`](../../../include/aspect/phase_field/dg_bounds.h)'s
subdivision enclosure, then mean-preserving scaling

    b_i_new = mean + alpha*(b_i - mean).

The mean uses physical JxW; it is not generally the arithmetic coefficient mean
on mapped cells. A bounded polynomial with a coefficient above one is preserved
when its enclosure permits it. An inadmissible mean fails; coefficients are not
clipped. The nodal adapter shares the same enclosure/scaling implementation.
Transported irreversibility still uses the transported material obstacle, not an
old coefficient at the same Eulerian cell. Fixed-reference tip growth and moving
mature growth-off transport remain separate admitted configurations.

## Transfer and checkpoints

Element interpolation and refinement embedding preserve representable
polynomials. Additive restriction reproduces a parent polynomial and its
integral on the tested affine/bilinear mappings. Arbitrary child fields lose
information on coarsening; neither L2 restriction nor this task supplies a
bound-preserving irreversible AMR algorithm. Existing physical AMR guards stay.

`phase-space.txt` records an additional `|Bernstein` identity. Nodal metadata is
unchanged; a mismatch is rejected before solution deserialization. Constitutive
history identity remains a separate check. Mature schema9 stores native phase
coefficients with generalized interpolation anchors and persistent cell IDs;
anchors identify ordering, not nodal physical values. The outer basis guard
protects their meaning. All publication, rejection and accepted-history checks
remain in the ordinary simulator/material transaction.

A benchmark-only, serial, same-mesh parent archive permits the bounded old/new
comparison. It records all three solution copies by CellId/component/local shape,
full tau/A and section state, geometry/material identity and exact clock. Only
phase and every accepted obstacle copy are converted. This is not a production
checkpoint converter.

Native Bernstein is admitted for the integrated bulk-tip route and growth-off
mature material transport. Historical replicated cohesive and endpoint-phase
solvers retain nodal assumptions and explicitly reject this selector. The
change does not cure BE work error, qualify moving cohesive growth/branching,
or establish propagation or production readiness.
