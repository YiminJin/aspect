# Frozen locally split Q2 comparison protocol

Selected after `FROZEN_FINITE_END_REPORT.md`, on source baseline
`cf06525471ce89789f1ac10905dc79f3aac7562a`. No native solver, physical input,
rotation, transport, coupled evolution, checkpoint or production change.

Use the saved `frozen_finite_end/frozen-input.npz`: the same 64² Q1 profile,
18 labels/I, full-normal support, prescribed V, G=10, m=50, dt=.1, zero incoming
stress and D=(.03,-.03,.01). Compare both eta=10 and eta=10(1+.5 tanh(4x)).
Reuse the completed uncut DGQ2 coefficients and common 128² partition split at
Q1/tensor cells, label/end lines and diagnostic windows. Retain Gauss6→10
Duffy-rule convergence checks; no source smoothing or tangential continuation.

On each 16²,32²,64²,128² tensor grid, cut only cells crossed by either finite-end
normal line. Each nonempty polygon has an independent Cartesian Q2 restriction
(nine coefficients per tensor component). No additional history is allocated
at internal Q1/label/integration-triangle boundaries. End lines span the full
admitted normal support, including regions with zero profile; report this cost
explicitly rather than hide it through a source-amplitude threshold.

Use Cartesian coordinates scaled to each polygon's bounding box: this changes
basis conditioning, not the Q2 polynomial space. Solve full positive-definite
piece mass matrices with diagonal equilibration; never discard a piece, truncate
a mode, add a penalty or change the physical source to make the solve pass.
Report minimum piece area fraction, raw/equilibrated condition numbers and
polynomial reproduction. Uncut cells retain their orthogonal tensor-product
mass. Assess integration and solve cost separately from coefficient storage.

Report full 3D plane-strain tensor/component L2 errors relative to both complete
and localized source, unchanged central/end windows, point shear traction,
common-mass chi-weighted work traction (including central/end errors), signed
weak slip-power and projection-energy errors. Verify nested-space orthogonality
and the projection energy identity using direct residual integration. Repeat
no rotation or lifecycle tests: this is a fixed approximation-space experiment.

Numerical screens: partition/mass identities <2e-11 absolute; polynomial
reproduction <1e-9 absolute; Gauss6→10 field L2 change <1e-9 times source norm
+1e-12, work-traction change <1e-9 Pa; orthogonal decomposition <1e-10 relative
+1e-16 absolute and energy identity <1e-14 J/m. Preserve failures and investigate
only the affected numerical reference/implementation. No invented physical
acceptance threshold: decide whether the measured end/traction improvement
versus added DOFs, conditioning and integration work merits a later bounded
coupled implementation. Coupled transfer, moving cuts, work adjoints and
acceptance/restart remain separate obligations even on a favorable result.
