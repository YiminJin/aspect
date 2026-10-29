# P5b bounded native tensor integration — selected 22 October 2026

Entry f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
First complete indexed lossless cleanup, then extend the existing bulk native
history owner, residual and checkpoint infrastructure. No directional specimen,
MPI expansion, AMR, transport or geometry evolution is selected.

Keep scalar mode/default and its v2/v3 checkpoint records unchanged. Add explicit
experimental fixed-tensor and in-plane radial-tensor selections, serial only for
this qualification. T remains independent DGQ2 plus the native FE source
complement; cohesive A/K/Q/W/L live on the existing CellId/Gauss8 layout. Physical
phase/obstacle has its existing FE owner. Every trial reads the same incoming
owners; publish once after mechanics/KKT/source/energy/relaxed-stability checks.
Version/tag both experimental modes; reject mode/layout mismatches, with no
implicit loaded-model conversion.

Use one small fixed Cartesian mesh, freshly initialized to constant physical
phi=.2 (an explicitly prescribed damage obstacle, not a propagation seed).
This uniform material control isolates native integration from seed resolution.
Retain G=1e6 Pa, eta=1e8 Pa s, m=4, R=1000 Pa, Gc=1 J/m2,
ell=.046875 m and Kf=8 Pa; Gauss8 mechanics/source Gauss3, unlimited DGQ2.
The native cohesion input is 2000 Pa because m=Kf/(cohesion^2/(2G)); this
matches the selected P5a point coefficients and actual P4 fracture.prm override. (P5c correction: the earlier claim of a different P4 tuple was erroneous.)
Use a .25 m square with an 8x8 base and its four central cells refined once
(a fixed patch with native hanging-node constraints), dt=.1 s and affine xy then
normal-difference loading/hold fixture, identical across compared modes.
This is a constitutive integration control, not a finite directional specimen.

First compare scalar and fixed-tensor physical stress, weak loads, phase and
complete mapped history (scaled identity bound 1e-12). Then advance a radial
non-proportional case with nonzero incoming cohesive tensors. Keep all existing
mechanical (1e-9), phase KKT (1e-7), source consistency (1e-12), ledger (1e-10),
complete coupled action FD (1e-6) and potential cross-symmetry (1e-11) bounds.
Verify current strain/phase derivatives, full C^T actions exactly once and
mechanically relaxed curvature, excluding constrained-row diagonals. Preserve
failed diagnostic evidence; do not tune a tolerance to measured results.

Exercise actual outer rejection/retry and single publication, same-mode native
checkpoint/restart of every tensor history, physical stress and weak loads, and
deliberately mismatched restart rejection. Check fixed-direction identity before
radial interpretation. Report timing and limitations; retain P4's failed spatial/
temporal targets, boundary sensitivity and strict_reference_match_passed=false.

Recorded amendments, preserving the original selection: the first radial
interval at dt=.1 exposed globalization failure from amplified boundary-cell
trial gradients. Before validation, dt=.0125 and 64 accepted intervals were
selected from that amplification; loading, coefficients and endpoint remain.
The uniform base-grid identity control uses the same physical data and sequence.
The hanging-row FD diagnostic excludes artificial solve diagonals; its 1e-6
bound is unchanged. A valid previous fixed-mode checkpoint is replayed when the
latest snapshot contains an end-cap-truncated pending interval; native clock
rules are not modified. The initial mismatched m=16 run is excluded from the
m=4 qualification and retained as evidence.
