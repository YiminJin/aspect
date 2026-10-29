# Closed-contact turning and bounded computed deflection

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_TURN_REPORT.md)
· [formulation/selection](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_TURN_SELECTION.md).

The imposed geometry is a prospective smooth append to an unchanged old straight
fault. Native vector mixed mechanics and the unchanged cohesive–RSF return
compute slip, current pressure and damage. The candidate terminal angle is
selected by a declared finite-horizon damage score. The experiment demonstrates
closed-contact sliding through a turn and a conditional computed deflection;
it does not establish a geometry-independent angle or general curved propagation.

| Evidence | Meaning |
|---|---|
| [turning-mechanics.png](turning-mechanics.png) | Whole finite specimen, physical stress, sustained through-turn slip rate, sided contact pressure and work |
| [selected-evolution.png](selected-evolution.png) | Old geometry/history retained, new tied tail, actual damage, amplified tangential slip vectors and current pressure |
| [direction-comparison.png](direction-comparison.png) | Primary angular comparison and the unresolved turn-width sensitivity, on equal physical scales |
| [selected-deflection.gif](selected-deflection.gif) | 25 actual accepted snapshots; no interpolated constitutive history |
| [work-and-time.png](work-and-time.png) | Physical work remainders, timestep control, separate algebraic ledger |
| [candidate-scores.csv](candidate-scores.csv) | Common .45 s scores; terminal and damage-weighted directions are distinct |
| [focused-comparisons.csv](focused-comparisons.csv) | Common-time absolute/relative spatial and temporal changes |
| [turning-comparison.csv](turning-comparison.csv) | Two resolved turn widths, refinement and timestep halving |
| [energy-accounting.csv](energy-accounting.csv) | Initial storage, storage changes, single fracture inventory, friction/radiation/Maxwell losses, endpoint/trapezoid work |
| [plotted-surface-profiles.csv](plotted-surface-profiles.csv) | Actual Gauss coordinates, arclength weights, damage, signed slip vector/rate, age and sided normal pressures |
| [verification.json](verification.json) | Numerical gates, reflection, front brackets, derivative/integration checks and scope |
| [selected-lifecycle.json](selected-lifecycle.json) | Atomic discard/retry, rebuilt native context/restart, exactly-once publication, fresh accepted branch |
| [default-regression.json](default-regression.json) | Preserved sharp default, recorded roundoff differences, rebuilt smooth bitwise check |
| [plot-definitions.json](plot-definitions.json) | Units, averaging and arrow conventions |
| [COMMANDS.md](COMMANDS.md) | Actual build/run sequence, environment and archive scope |
| [changed-files.json](changed-files.json) | Exact task changes, separate from pre-existing local work |
| [preservation.json](preservation.json) | 1051 protected files and 13 earlier archives unchanged |

Map backgrounds are native Gauss8 **cell-mean physical τxy**, not auxiliary T or
pointwise stress-error estimates. Surface damage a owns the fracture inventory;
bulk φ is not used here. Γ colors denote actual owned damage. White circle,
square and diamond markers show last sampled a≥.05/.5/.999 locations;
whole-cell front brackets and fully decohered cells are separately reported.
Slip arrows are actual signed `t δ`, amplified only for visibility. All spatial
axes are metres; stresses Pa (plots kPa), slip m (plots mm), velocity m/s
(plots mm/s), energy J/m out-of-plane thickness. Θ is dimensionless state.

The 1.5 m-width family distinguishes +30° from +20° under the observed uncertainty
criterion. Changing the width to 3 m favors the +60° sector boundary. These are
separate hypothetical paths, never simultaneous branches or averaged flows.
The actual fully decohered front has turned only about 5° at the reported time.

Original failed/undersized pilots and the complete coarse trials remain indexed
in the package diagnostics. In particular, the original +20° trajectory beyond
.45 s approaches the candidate endpoint and is not angle qualification.
Production/BP3 and earlier evidence retain entry hashes. The compact ZIP is a
source/evidence snapshot for this isolated benchmark, not a production restart.
