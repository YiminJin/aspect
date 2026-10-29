# Extended P5c continuation — predeclared campaign

Selected 22 October 2026, f1f9f29cf40c32d1ddcea000b724375b64853532 plus
preserved local work. Continue each mode from its own accepted step127,
t=.13875s checkpoint on192×128/four ranks, Δt=.000625s. No clock or accepted
history edits, interpolation, fresh initialization or material change. Retain
the same finite specimen, unlimited DGQ2/native complement, Gauss8/source3,
physical obstacle, matched coefficients and all native tolerances. The new
benchmark loading plugin changes only the endpoint of the existing .018/s rate.

The primary radial trajectory loads continuously to at most .200s. Native
terminal checkpoints at .155s and .175s are observation/restart boundaries,
**not holds**. Analysis and continuation proceed automatically. Fixed-direction
flow follows the same history through the radial-selected endpoint unless its
own event or native limit intervenes. Each admissible trajectory then holds its
selected displacement for .005s. Report unequal endpoints explicitly.

Diagnostic conventions, fixed before running:

- Initial represented support is the union of Q1 cells having any positive
  initial phase vertex; its centerline endpoint is x=1.09375m. The target is a
  seed-connected phi=.05 front in |y-.5|≤ell reaching x=1.140625m (support+ell).
  Connected phi=.01/.05/.1 fronts use the retained four-neighbor nodal
  components meeting the left seeded band, with Q1 horizontal edge crossings.
- Intact activation is max phase>1e-8 in cells whose **four initial vertices
  are all exactly zero**. Record its bracket and integrated phase there. It
  does not stop loading. Continue beyond tiny activation.
- Significant remote nucleation means a disconnected phi≥.01 component
  contains initially-zero nodal area≥4 cells (.000244140625m²), or has
  peak new phase≥.05 and initially-zero area≥1 cell. Report the component and
  threshold convention; weak isolated increments remain diagnostic only.
- Boundary approach means newly grown (increment≥.01) seed-connected
  phi≥.01 material lies within2ell=.09375m of the right/top/bottom boundary.
  The originally seeded left boundary is excluded explicitly.
- Rapid growth for preservation/replay means maximum accepted-step increment
  ≥.01 or connected phi=.05 front advance≥one Δx=.0078125m in one step.
  Preserve the immediately preceding accepted checkpoint outside the ring.
- Stop forward loading at the target, significant remote nucleation, boundary
  approach or .200s, or when an existing solver/stability/capacity limit fails.
  No detector changes the phase admissible space or modifies a field. An
  external read-only observer requests native collective user-request
  termination; observation/stop latency is at most one accepted timestep and
  the actual final accepted time defines the loading endpoint.

Checkpoint every accepted interval. Copy initial127, pre-intact, pre-rapid,
milestones and terminal/last-accepted states outside the three-entry ring, with
metadata/hashes. Retain failure logs, negative-mode diagnostics and reproduce
any failed trial with additional read-only diagnostics if necessary; never
remove a guard or change its tolerance. Native publication/ownership/frozen
particle checks remain active.

After the radial event or limiting interval is identified, replay a focused
same-history window at Δt=.0003125s, including its loading-to-hold transition
when admissible. Include the corresponding fixed-mode window, with the same
physical loading endpoint. Compare activation brackets, patterns, phi fronts,
increments, reaction and endpoint/trapezoidal physical work. Report differences
relative to growth and separate temporal evidence from the exploratory run.
If a native limit prevents the window, preserve and diagnose that limit rather
than claiming verification passed. No general mesh campaign is selected.

For the local replay, retain the prior5% growth-relative L²/peak interpretation
target, compare onset brackets, and report front discrepancies against Δx.
Use actual nonzero force–displacement work as the energy scale: report the
sum of absolute step trapezoidal physical remainders, plus the separate
common-time Cauchy changes in work, storage and dissipation (the retained
BULK_ENERGY_REPORT criterion). Two local timesteps alone do not establish an
order, a global temporal bound or the earlier three-level energy qualification.
If tiny activation and the principal rapid event are well separated, use a
short onset replay and a short principal-event replay rather than claiming an
uncomputed globally refined trajectory. For matched replay the physical loading
endpoint stays fixed; crossing the target sooner does not end that comparison,
but remote/boundary and native admissibility limits remain active.

Record critical-master size (cap1024), curvature, Newton/linear counts,
assembly/linear/stability times, memory and checkpoint cost. Prior axial
growth-field differences28.68%/37.97%, earlier P4 energy/accuracy failures and
strict_reference_match_passed=false remain. This is a finite physical
specimen, not a distant-boundary approximation. BP3/production defaults remain
protected. No RSF, AMR, transport or geometry change is selected.
