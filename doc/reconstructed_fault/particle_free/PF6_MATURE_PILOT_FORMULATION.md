# Bounded mature DG pilot qualification

17 October 2026. Branch pf-rsf-particle-free, entry a8707307008401ac9cff702a369e2e2a611fa0f7 plus retained incoming work. Entry files/patch are captured in build-particle-free/mature-pilot. This task selects unlimited domain-free DGQ2 as an experimental server-pilot candidate, with neither particle nor DG designated as accuracy truth.

Finish one h=1.953125 m mechanical control at exactly prolonged retained FF Q1 phase and unchanged 3.90625 m exterior lattice, Gauss6, fixed 59-node surface, dt=200000 s to T=600000 s. Keep the initial constitutive interval 200000 s. Halve dt on the retained FF mechanics/profile (100000, then 50000 s if necessary to bound temporal uncertainty). Retain Gauss3 source, Helmholtz20 m, all limiter settings, laws and acceptance rules. Compare complete embedded tensors, reconstructed tau, own-mass perturbation loads and the common geometric functional, including 125 m endpoint windows. Exact prolongation must preserve physical phase to 2e-14 and I/chi to 2e-9 relative. A 1% perturbation-load temporal screen remains a screen, not an accuracy assertion. Spatial differences are not accuracy errors.

Reuse accepted DG ownership, CellId checkpoint records, native manager rollback and stationary particles. Extend only the fixed frozen mature mode; the mature checkpoint records auxiliary Z independently of zero cohesive traction. Qualify serial/2/4-rank ownership, one checkpoint/restart continuation and an actual outer rejection/cutback. Compare phase, all 27 coefficients/cell, I/Z, surface state/slip, reconstructed stress and weak loads by physical CellId/coordinates. Require no accepted mutation on rejection and one publication per accepted positive step. Solvers/reductions permit measured rounding differences; no publication tolerance changes. Keep reference-field import serial-only unless a test specifically requires extending it.

Domain-free homogeneous bulk coefficients must coexist with the prescribed physical surface strengthening fraction. Evaluate configured initial composition directly at fixed surface nodes during fresh initialization; subsequently retain native checkpointed surface properties. Verify BP3 a=0.010 below 15 km down dip and a=0.025 past 18 km, linear transition, b=0.015, Dc=0.008 m and native regularized-law steady state. No particle material projection or domains. This is a discrete initialization replacement, not a friction-law change.

Prepare a separate bounded full-geometry stationary-history server input/package. Preserve the existing FE server package and failed evidence. Bound accepted steps, physical horizon and wall time; pin actual source/artifact/config identities and export authoritative stress and unfiltered/filtered perturbation weak loads. No remote launch, full event, phase evolution, transport, AMR, moving geometry or propagation is selected.

## Bounded follow-up selected from the measured control

The FF dt=200000/100000/50000 s comparison leaves a 21.44 Pa endpoint shear
projection difference at the last halving, larger than the 10.95 Pa mechanical
refinement change. Add **one** dt=100000 s run on the 35342-cell mechanics mesh
with the same exactly retained FF phase/exterior lattice, to T=600000 s. This
checks whether temporal biases cancel in the mechanical contrast and supplies
a fixed-mesh time control on the finer mechanics itself. No further mesh level,
WENO timestep sweep, threshold search or physical-law change is selected.
