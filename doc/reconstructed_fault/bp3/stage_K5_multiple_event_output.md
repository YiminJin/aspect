# Multiple-event preparation: clean fault output and complete slip history

## Decision

Prepare (do not launch) the **modified-BP3** multiple-event run to 1500 physical
years. First-event termination is disabled. Graceful wall-budget stops occur
at accepted states, with ordinary checkpoints for continuation. Keep the
tested 300 x 100-km box, 60-degree fault, ell=400 m, 97.65625-m near-fault
bulk spacing and 99.9603–100-m fault spacing. Maximum far-field square side
remains 12.5 km. Equations, work measure, histories, support, endpoints,
timestep controller, solver tolerances and GMG/coupling selections are unchanged.

Erickson et al., *BSSA* 113 (2023), pp. 505, 509, 511–512 and 517
(`bssa-2022066.1.pdf` in this directory), recommend 25-m BP3 spacing and
report 100-m sbplib and 60-degree FDCycle runs in Table 4. The nominal
400-m process zone and 2.5-km nucleation length have about 4 and 25 intervals
at the selected fault resolution. The paper demonstrates sensitivity to
200-to-100-m refinement and finite domain size: it does not qualify our
finite-width/Q1 discretization, fully frictional deep region or finite boundaries.
This is a justified starting **research resolution**, not a converged official
BP3 reference. Retaining tested input tables avoids an unqualified fine-grid
initialization change immediately before the server run. The original paper's
1500-year interval includes multiple characteristic events; no fixed event-count
termination or promise of recurrence is introduced.

## Output changes

The native reconstructed-fault writer removes `vertex_id` and `cell_id` arrays,
retaining `fault_id`. An output-only `Excluded properties` parameter hides BP3's
fixed shear correction and mature reference geometry. Those registered values
remain intact for mechanics and restart. Display names use underscores:

| Stored property | VTU name |
|---|---|
| phase field fault state | slip_state |
| phase field fault cohesive traction | cohesive_traction |
| phase field fault previous I h | previous_I_h |
| phase field fault chemical composition strengthening | composition_strengthening |
| background tractions | background_tractions |
| cumulative_signed_slip_m | cumulative_slip |

Unknown property names retain their words with spaces replaced by underscores;
collisions fail explicitly rather than writing ambiguous arrays. `previous_I_h`
is not relabeled as a differently timed quantity. Units remain SI and background
components remain [shear, normal]. Q2 mesh output and Q1 fault geometry are unchanged.

`cumulative_slip.csv` appends all vertices at **every accepted state**, with
`step,time_s,fault,node,s_m,xd_m,slip_m`. Initial slip is zero; real-step slip
adds exactly the existing accepted dt*V. Local arclength s begins at vertex 0
(bottom in this fixture); xd is official down-dip distance from the top.
It is independent of sparse VTU/profile scheduling. Figure 8's annual/one-second
contours can be selected or explicitly interpolated from this file; the writer
does not force output-time timesteps or reconstruct slip from sparse rates.

The slip vector remains checkpointed. The potentially large CSV is not copied
into every checkpoint. `run_long.py` streams its accepted prefix into the copied
restart branch before appending resumed steps, preserving the original branch.
This mechanism requires a complete new-format file, not a silent reconstruction
of missing historical rows. The launcher already requires identical input/binary
hashes; pre-change runs are not automatically promoted to a complete new record.

## Environment and entry points

`benchmarks/reconstructed_fault/bp3/environment.sh` is sourceable in bash/zsh
from any working directory. It derives ASPECT_SOURCE_DIR, removes inherited
ASPECT investigation flags, enables sparse B/G, pivoted tridiagonal K, GMG plus
its required hierarchy, and one library thread per rank. Optional argument
`amg` keeps the reference backend. It leaves machine MPI/library paths alone;
load the server's compatible modules before sourcing. The Python launcher reads
this same file. Bare mpirun is supported; prepared launch.sh additionally checks
input/executable hashes. No server job was submitted.

See `benchmarks/reconstructed_fault/bp3/LONG_RUN.md` for exact preparation,
source/mpirun and restart commands, termination semantics and storage estimates.
`run_long.py` now defaults to recurrence and 1500 years; explicit first-event
mode remains available for bounded pilot work. The first_event.csv observer is
still a **first**-event summary, not a multiple-event catalog; accepted_steps.csv
contains the full every-state maximum-rate record for subsequent event analysis.

## Verification and evidence

Evidence root: `benchmarks/reconstructed_fault/bp3/long-run-output-check/`.
Core and maintained plugin build with `-j4`.

- Output unit tests: **23 assertions in 5 cases passed**, including hidden
  uninitialized properties, shorter names and alias collisions.
- `test_slip_history.py`: **3 tests passed**, including prefix restoration,
  failed-prefix preservation, and bash/zsh GMG/AMG environment checks.
- `fresh-mpi`: four-rank initialization plus two adaptive real steps passed
  in **145.155 s**, peak child RSS **1,342,740 KiB**. All compared numerical
  fields at steps 0,1,2 are **bitwise identical** to the saved 300-km run after
  accounting only for the deliberate output aliases/exclusions. The checker
  handles absent native output at an unrendered step, requiring identical
  output schedules and retaining all available history/weak-load comparisons.
- The complete slip file contains **3 states x 1156 vertices**, 257,948 bytes.
  Every recorded slip matches the existing profile exactly; real-step increments,
  zero initial slip, local coordinates and actual VTU aliases/exclusions passed.

The first attempted MPI launch (`fresh/`) was blocked by sandbox socket
permissions before ASPECT ran. Its log is preserved; `fresh-mpi/` is the
authorized execution, not a numerical retry or changed fixture.

- `resumed`: ordinary checkpoint 02 (accepted step 1), resumed through step 2,
  **119.384 s**, peak child RSS **1,379,704 KiB**. Numerical fields match the
  uninterrupted step 2 bitwise; the entire cumulative-slip CSV is byte-for-byte
  identical. No duplicate, missing or reset history rows. Results are in
  `long_run_equivalence.json` and `cumulative_slip_verification.json` in each
  compared run directory. Both retain the usual convergence checks.
- `server-prepared/`: prepare-only resolved configuration, recurrence mode,
  1500 years = 47,336,400,000 s, 23-hour accepted-state graceful wall bound,
  no short verification step cap. **Not executed.** Local absolute build paths
  in this generated example must be regenerated in the server checkout.

Qualified executable SHA256:
`47786734f2b14ef78c66a0259828726e689a94e71f64308895e679e036fb015e`.
Qualified plugin SHA256:
`0b4889b37149b4403caf107cc6a37ab4fb7eb60a40e58adcde9f1ad0fc8b377a`.
Source base remains `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5` plus the
recorded working diffs; no commit was requested or made. The unrelated working
tree and old evidence were preserved.

No full earthquake event, convergence campaign, server scaling test or
cross-rank restart is claimed. No long simulation was launched. Python syntax
and scoped diff-whitespace checks passed; a broad ASPECT suite was not run.
