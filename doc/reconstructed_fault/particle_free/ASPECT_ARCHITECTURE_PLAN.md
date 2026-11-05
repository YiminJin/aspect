# Selected ASPECT architecture refactor

Implement `CODEX_ASPECT_ARCHITECTURE_REFACTOR.txt` in the current worktree.
The rejected dedicated driver is present in the incoming dirty, tested source;
HEAD a22214d57 alone is not that baseline. Source archive/patch, six binary
copies and the previous seven controls are preserved before editing.

| Owner | Before | Selected boundary |
|---|---|---|
| Simulator | Separate mature timestep driver, rollback helper/enum, shared executor wrapper | Ordinary `solve_timestep` and solver schemes; existing common endpoint acceptance callback, no alternate timestep entry |
| Coupled solver | Projection iteration moved into the special driver file | Back inside reconstructed-fault Stokes solve; same inner solver, final slip guess and gates |
| Material | Distributed history/geometry/rotation staging across driver-facing methods | Explicit material-history preparation service and private constitutive calculations; no solve or acceptance decision in evaluate |
| History discretization | Nested material-owned DG containers and spatial kernels | Independent tensor/space and advection/projection components; numerical coefficients, explicit FE fields, no friction law |
| Geometry manager | Generic clone/swap/property storage; motion formula in material code | Geometry/support material map belongs to reconstructed-fault infrastructure; generic stored surface properties remain uninterpreted |
| Representation | Physical tau stored under `trial_stress` | Active coefficients and physical stress access; isolated separated-memory calculations; checkpoint packet meanings unchanged |
| Observation | Existing audit/checker modules | Preserve them as observation/reference code with no advancement authority |

Before: solve_timestep -> dedicated mature driver -> shared timestep executor
-> coupled solver -> dedicated projected-solve helper -> wrapped publication.
After: solve_timestep -> explicit material-history preparation -> ordinary
solver scheme -> coupled Stokes/projection iteration -> existing pending endpoint
publication -> ordinary timestep controller accept/reject. Material preparation
uses spatial history operators and geometry manager operations, not a solver.

Working ASPECT examples: registered material models compose EOS/rheology;
SimulatorAccess provides read access to FE/mesh; assembler lists implement
coupling; simulator core/solver schemes own ordering; manager clone/swap owns
geometry. Signals are retained for their existing contracts, not used as a
hidden ordering pipeline. Consulted extension-guide plugin, material, signal and
solver sections. Local source governs actual available APIs.

Verification: reuse seven bounded moving controls and declared norms, full units,
matched checker rebuilds, and add baseline/current stationary-fault and ordinary
non-fault smoke comparisons because dispatch changes. Native runs sequential;
at most four workers/ranks. No new physics, production, event replay or broad
campaign. Finish with one maintained direct-stress guide, one report, updated
active instructions, and explicit retirement of the rejected driver pattern.

Completed: all ownership changes above are implemented. The special driver and
executor are absent. Full units, eight independent compilations, seven moving
comparisons, stationary/non-fault dispatch controls and two serial support
fixtures pass. See the report and evidence index for actual norms and artifacts.
The bounded task is complete; no scientific campaign is automatically resumed.
