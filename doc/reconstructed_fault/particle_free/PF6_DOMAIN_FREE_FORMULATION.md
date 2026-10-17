# Homogeneous mesh-data configuration: bounded verification rules

16 October 2026, base a8707307008401ac9cff702a369e2e2a611fa0f7 plus
preserved incoming work. This task removes domain-dependent setup/material
projection from the selected DG consistent-endpoint configuration. It does not
extend CPDI neighborhoods or qualify propagation, transport or automatic AMR.

Use a default-off material selection, restricted to accepted DGQ2 history,
consistent-endpoint phase, homogeneous configured coefficients, the supplied
physical phase profile, unloaded histories, static bulk composition/temperature,
fixed prescribed fault geometry and no particle domains. Legacy CPDI remains
separately selectable. The selected bulk and surface material uses the configured
background values; require every configured material entry to agree. Chemical
labels are immaterial in this homogeneous experiment. Surface chemical slots
are set directly to the background label rather than projected from particles.

The existing normal-ray integration, consistent surface projection and all
normalization derivatives/adjoints remain unchanged. Bulk mesh/fixed fault
geometry own support. Particle associations in this mode contain only point
coordinates and fault coordinates, with no volume or integration weights.
Particle projection operators must reject these associations. Diagnostic fixed
points retain native matching-CellId source/Maxwell publication checks and its
existing arithmetic allowance. No particle stress enters incoming mechanics.

Before native runs, verify that setup rejects heterogeneous coefficients,
domains, particle-driven fields, missing phase seed and changes of mode on
restart. Keep the existing constitutive law, full embedded tensor metric,
Gauss12 physical/Gauss3 DG-source quadratures and unlimited DGQ2. Do not relax
any existing publication, phase/obstacle or CPDI coverage check.

Run fresh matched serial controls with and without the new mode. Compare
initialization, independent T, Z, physical phase/lower obstacle, I, reconstructed
stress, bulk/cohesive/fracture energies and fault weak loads. Differences in
non-authoritative FE stress/chemical diagnostic slots are labeled separately.
Use the previous gates: 1e-9 scaled physical energy/load integration error,
1e-11 normalization, 1e-10 scaled identical-field transfer and 1e-9 scaled
native physical trajectory comparison. Record actual errors and arithmetic
scales, rather than increasing thresholds to pass.

Restore the observer's original 64→256→64 uniform roundtrip (256 total cells),
including all 27 tensor coefficients and physical phase/next-obstacle transfer.
Retain its manufactured complete-Q2, lossy-tensor and nonrepresentable-phase
negative controls. Use the same 144 fixed diagnostic particles: no additions,
removals, domain handler construction or domain access can occur. A successful
run with domains disabled has no domain object; its accessor throws if invoked.
Export that configuration at each checkpoint/remap and retain all point IDs and
coordinates. The old CPDI failure remains failed evidence.

After the serial gate, repeat bounded 1/2/4-rank ownership/lifecycle checks and
same-count/serial-to-four restart with all persistent physical histories and
point properties. Check continued restart versus uninterrupted advancement,
and restore-only stopping. Preserve the native changed-count RNG restriction.
Run the existing frozen FE/CPDI control on the same build. Record source/build/
observer/input hashes, commands, costs and failed attempts. Stop after this
configuration and restored diagnostic; hanging-node phase evolution stays
guarded and lossy physical obstacle transfer remains prohibited.

The unchanged normalization derivative check uses the Q1 nodal direction
`phi_i*(1-phi_i)*(1+.2*x_i)` at accepted step 1, central FD epsilon=1e-5,
and a 1e-7 scaled FD screen. Independently integrate h, h' dphi and
h'' dphi² with Gauss24/32 and the same consistent surface mass solve;
require 1e-9 scaled agreement with native I, DI and D²I contractions.
