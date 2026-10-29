# One bounded near-tip mesh-orientation control — pre-run contract

Entry f1f9f29cf40c32d1ddcea000b724375b64853532 plus retained local work.
The existing wide 48x64, [0,1.5]x[0,1] trajectory is the reference. No seed,
material, physical shear tensor, loading, boundary condition, source3,
mechanics8, dt=.0025 s, solver tolerance or publication rule is changed.
The requested interval ends at .145 s (three steps after reference onset).
Histories start freshly unloaded; physical vertex samples use the same table.

The accepted mesh recipe is a rigid 45-degree rotation of the reference
.03125 x .015625 m rectangular cells in a square |u|,|v|<=.3125 m about
(.5,.5). All ideal cap radii R=.30882159 m fit within this core. A conforming
24-ring annulus connects its 120 boundary edges to the exact outer rectangle;
outer corner nodes attach to the nearest core perimeter angle. There are
800 rigid-core plus 2880 transition cells, with no hanging nodes or runtime
refinement. Far-field shapes differ; this is a local-orientation control,
not an identical far-field discretization. A bilinear determinant is affine
in reference coordinates; independent positivity at every cell corner proves
validity throughout each cell. The current generated mesh minimum corner
determinant is 2.67275e-5 m^2. Physical phase is sampled freshly at its vertices.

The first cubic-twist construction failed positive-JxW at initialization, before
any physical interval, and is retained separately. It is not a physical result.
The annulus corrects the demonstrated geometry failure without tuning physics,
source tolerances or the local core resolution. No additional mesh sweep is
selected. Mesh generation and qualification scripts are benchmark-local.

Use the existing native MappingQ1 selection via identically zero function
topography; no height is changed. MappingCartesian must not be used on rotated
cells. An identity-map control verifies this selection before the rotated run.
No production mapping selector is modified.

Acceptance screens remain native: mechanical residual <=1e-9, physical phase
KKT <=1e-7, DG-source/independent mechanical-QP stress error <=1e-12, physical
phase bounds/irreversibility and positive dissipation/ledger gates unchanged.
The synthetic source preflight records all nine scalar-Q2 basis-gradient
projections with Gauss3 and Gauss8 and tests physical affine reproduction.
Positive Jacobians and total area 1.5 m^2 are required. A nonrepresentable
source must not be admitted by loosening a tolerance or bypassing the guard.
An actual native failure ends the run; do not expand into a general source
representation correction during this comparison.

The Cartesian normal-column observer is diagnostic only. In the isolated DSO
it is replaced by the same positive-weight moment decomposition over native
bulk QPs with JxW and an explicit 1.5 m^2 coverage check. Its area moments are
not line normalization or surface history. No constitutive QP is relocated.
The original DSO and its normal-column coverage check remain unchanged.

Compare in physical coordinates with a common compact radial kernel of radius
.0625 m: K_z(x)=(1-|x-z|^2/a^2)^3_+, normalized by its integrated physical
weight. Evaluate q and Y at original accepted QPs before averaging; never
return-map interpolated histories. Evaluate fracture action with the same
kernel, Kf integral[K+2 ell^2 grad(phi).grad(K)], and total action F-Y. Native
nodal KKT is reported separately. Inspect onset radii .2881107643/.2985152059
and actual growing nodes. Plot mesh, physical phase contours, actual Y and
phase increments on equal physical/color scales. A post-onset orientation
conclusion requires both meshes to pass unchanged source/publication screens.
If blocked earlier, label all loaded comparisons as pre-onset and leave both
lobe-orientation and first-node-following conclusions unqualified.

Preserve the outstanding external-energy screen failure, strict reference
mismatch, prior evidence and BP3. No cap change, load increase, transport,
AMR, MPI expansion or physical propagation is authorized by this control.
