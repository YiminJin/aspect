# Native slow loading / RSF instability with cohesive growth

**Experimental finite-wall specimen; default-off native quasi-dynamic path.**
The maintained native workflow and build instructions are in
[../cohesive_rsf_native/ACTIVE_WORKFLOW.md](../cohesive_rsf_native/ACTIVE_WORKFLOW.md).
`event.prm` selects the uniform128×96 event, including the common fresh physical
inputs in `nucleation.prm`. Only the built loading plugin is external.
`event-neutral.prm` changes b to a and compensates mu0 so the initial friction
is exactly matched. It is a state-feedback diagnostic, not a second benchmark.
`parameters.json` records SI values; the scientific selection and limitations
are in `doc/reconstructed_fault/particle_free/COHESIVE_RSF_INSTABILITY_REPORT.md`.
Run includes from the repository root. No historical artifact is required.

Example (set paths for your build/MPI installation):

```
python3 benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/qualify.py active \
  --label rsf-event --input benchmarks/reconstructed_fault/particle_free/cohesive_rsf_instability/event.prm \
  --work-dir build-particle-free/cohesive-rsf-instability \
  --aspect "$ASPECT_BINARY" --plugin "$ASPECT_COHESIVE_PLUGIN" \
  --mpiexec "$MPIEXEC" --mpi-args='--bind-to none' --ranks 2 \
  --timeout 3600 --save-at-speed 1e-6,1e-5,1e-4
```

The runner launches actual ASPECT; it does not compute or replay evolution.
`--operators` exports native reduced K, Q and current-normal N for independent
stability analysis. `stability.py STEM --output DIRECTORY` uses those operators
on admissible mature-patch perturbations with frozen damage, Maxwell memory,
RSF state and radiation, including pressure derivatives. It also checks the
linearized action by finite differences and compares the pressure-free elastic
limit against independent spring-slider characteristic roots.

`--save-at-speed` copies the latest fully written native checkpoint when accepted
speed crosses a threshold, retaining its slot marker and an observation record.
It may precede the observed accepted state by one step; it never copies pending
trial history. These preserved checkpoints sit outside the rolling overwrite
cycle. `--resume-from DIRECTORY` branches from their native `restart/` directory.
A `terminate-aspect` file inside an output directory requests graceful accepted-
state termination. The hard runner timeout is only a last-resort process budget.

Stop bounds: 2000 s, 600 accepted steps (ASPECT's end-step criterion checks the
advanced counter), unchanged contact/source/weak/energy/domain guards. No bulk
inertia or wave-speed qualification. Report physical temporal error separately
from the exact backward-Euler ledger, and do not treat an algebraically positive
step tangent as evidence against a continuous-time RSF instability.

## Bounded remaining comparisons

Use `event-neutral.prm` freshly to the primary accepted endpoint
374.57855897711539 s. Use `half-event.prm` from the native `event/output`
checkpoint at 374.56704168613635 s for a focused replay to the same endpoint.
The latter halves all four adaptive increment limits and the maximum timestep;
it is an adaptive accuracy comparison, not an imposed identical dt/2 schedule.
It deliberately retains the complete incoming state, including prior temporal
error. It does not qualify the absolute onset time of the earlier acceleration.

Both run through the existing `qualify.py active` command above. Pass
`--end 374.57855897711539`; add `--resume-from .../runs/event/output` only
for the half-limit replay. The control has `--timeout 1200
--graceful-stop-after 1080`; the replay has `--timeout 4200
--graceful-stop-after 3900`. The runner requests ASPECT accepted-state stopping
before its hard process limit; terminal output and checkpoint need that margin.
All input/output paths remain configurable. Never resume the weakening law into
the neutral law: that control starts freshly.

Read accepted exports with `analyze.py --run primary=RUN --output EVIDENCE --gif`.
Use `compare.py --primary PRIMARY --neutral CONTROL --half REPLAY --output EVIDENCE`
for common-time histories, final profiles, energy windows and temporal differences.
The CSV fields are SI; surface damage is `a`, physical shear is `tau_xy`, and the
exported bulk column historically named `Txy_Pa` contains reconstructed physical
stress. The maximum recorded speed is an observation on a truncated trajectory,
not a validated pulse maximum or rupture speed.

Completed result: the matched control remains slow; the half-limit replay stops
at step660/t=374.57851954367675 s, before its requested endpoint, when the unchanged
rate controller demands dt below1e-10 s at incipient near-zero damage. The failed
step661 is not published. `cohesive-rate-failure.csv` now preserves the exact point
and both operands before this guard. The original adaptation CSV schema is retained.
No floor reduction or newborn exemption was used. Analysis uses the actual common
endpoint, not an assumption that the requested end was reached.

`energy.py EVIDENCE/review` plots the comparison CSV's work/storage/dissipation.
In the GIF, stress is sampled at native cell centres, lines join surface Gauss
samples for visualization, and zero speeds use a1e-12 m/s plotting floor only.
The earlier half-replay curves reuse the accepted prefix through step368; only
the following11.48ms is recomputed with tighter controls. Absolute onset timing,
point tractions and developed front advance remain unqualified. See the report.
