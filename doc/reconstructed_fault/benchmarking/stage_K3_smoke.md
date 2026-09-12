# K3 smoke: preserved first-attempt failure record

**Superseded execution status:** the user explicitly approved one corrected
invocation, which passes initialization and both real steps. See
`stage_K3_smoke_result.md`. The first attempt's `smoke/`, `smoke.log` and
`smoke.resources.json` named below now reside under
`evolving/attempt1-fingerprint-failure/`; they were moved, not overwritten.
The text below records the original failure and its bounded correction.

The single authorized one-rank Release invocation stopped after **6.126 s**,
before initial mechanics, on SIGSEGV in the new benchmark state fingerprint.
The initial production phase solve converged, but no mechanical state or real
timestep was accepted. This is **not a K3 pass or evidence of a production
numerical failure**. No automatic retry, MPI run, scan or refinement was run.

## Implemented safeguards and pre-run checks

- `evolving/safeguards.h` uses the existing test-access route to the production
  phase assembler. Residual evaluation has scratch matrix/RHS and a copied
  probe vector; only H is substituted, with a scoped destructor restoring it
  on normal return and exception. Before/after fingerprints cover live bulk
  solution/old/linearization vectors, RHS, matrix entries, particle properties
  and positions, surface properties and available V. An injected exception
  after assembly tests restoration after a genuine H perturbation.
- Phase probes run at timestep entry and the temperature-solver signal, which
  is after phase but **before particle advection** in the selected scheme.
  Thus the paired residuals use the same phase-domain geometry and H. Stable-ID
  checks compare entry H against the preceding accepted publication. The
  actual phase convergence criterion remains 1e-8.
- The benchmark snapshots reconstructed coordinates/topology at zero and
  requires exact equality at later accepted states.
- `check_smoke.py` consumes flushed accepted exports before another timestep
  can start. It independently integrates full Q1 h profiles at every bulk QP
  column and surface vertex, and separately records supported instantaneous,
  signed-history and total crack-strain integrals with actual assembler JxW.
  History diagnostics retain pre-commit C/I_h snapshots, not newly committed
  values. No tail renormalization is used. Both 1e-4 gates remain unchanged.
- It checks phi admissibility and whole-fault H/phi/I_h/C/V ranges including
  endpoints/seam. Comparable means reaching the already predicted feedback
  signal, not a threshold fitted to the smoke. These are conservative
  whole-fault gates; saved profiles permit localization if a gate fails.
- `reference.py --accepted-times <output-directory>` now reruns the primary
  independent reference with exactly exported time/dt/U, initializes histories
  once, and reads no evolved ASPECT history. A partial/empty prefix cannot
  report a completed smoke-budget pass. Conditional production-initialized
  results remain a separate diagnostic decomposition.

No production algorithm was changed. The only shared test-header modification
adds a default-true Jacobian flag to `PhaseFieldTestAccess::assemble`; existing
test calls retain their old semantics. The smoke passes false for residual-only
assembly. Full CPDI weight/gradient export remains a fallback, not a new test.

The Release plugin built with `-j4`. Before the invocation, six existing cheap
reference checks and the saved-data gate test passed. The latter deliberately
adds a history-only crack-strain contribution in a disposable copy: containment
still passes, while total normalization correctly fails. Original saved data
are unchanged. After stopping, the exact-sequence/empty-prefix regression also
passes: **seven reference tests plus one gate test** in total. These Python
checks and a successful build do not verify the failed C++ restoration probe.

## Observed failure and confirmed cause

`smoke.log` records 14 initial phase Newton updates, ending at relative residual
**1.220e-9**, below 1e-8, then successful initial fault reconstruction. The new
pre-mechanics restoration test crashed in `K3::fingerprint<2>` before it
substituted any H or called its scratch residual assembly.

The fingerprint called `get_slip_rate(f)[v]` while V had not been initialized.
This is valid production lifecycle state: reconstruction precedes mechanical
preparation. `get_slip_rate` requires initialized V (a debug assertion); in
Release, indexing the empty vector caused the null read. Disassembly of the
failed plugin places the fault exactly at offset `fingerprint+0xae2`, the
indexed load immediately after `get_slip_rate`. This is direct evidence,
not an inference from the crash timing alone.

The smallest benchmark-only correction uses the existing
`slip_rates_are_initialized()` accessor, fingerprints that initialization flag,
and reads V only when it exists. This preserves the distinction between absent
and initialized V instead of initializing physical state for a diagnostic.
The corrected Release target builds successfully. **It has not been run.**
The failed header and plugin were preserved before this correction.

## Resource use, provenance and missing results

Exactly one invocation, with the approved 180-s process-group cap and no retry:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/run_smoke.py
```

It directly executed `build-pf-cpdi/aspect-release` on `evolving/smoke.prm`,
one rank, no MPI launcher. Measured wall time **6.12566257 s**, peak RSS
**325628 KiB (318.0 MiB)**, exit status **-11**. Initialization plus two real
steps was requested, but that trajectory was not produced.

Artifacts under `benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `smoke.log`, `smoke.resources.json`, partial `smoke/` directory;
- `smoke-failed-plugin.so` and `safeguards-failed-snapshot.h`;
- `build-final.log` for the tested binary, `build-fingerprint-fix.log` for the
  subsequent compile-only correction;
- `smoke-gate-tests.log`, `smoke-reference-tests.log`,
  `exact-sequence-tests.log`;
- `run_smoke.py`, `check_smoke.py`, `test_smoke.py`, `safeguards.h` and the
  exact-sequence reference/test modifications.

Tested plugin SHA256:
`60a18413b30d5d409d6f41a2a620c327ffbb8998f910e8d743b269c987c7c917`.
Executable SHA256:
`abd1a3b8ec4cac3fd025f2ce9ee727abdaef4888d5507e859ce3a1915fb9f2a9`.
The resource JSON records parameter hash and source HEAD
`dc1a96d3f72dce123393c90ad025e729885e8ee9`; benchmark changes remain uncommitted.

No accepted time exports exist, so an exact-accepted-time production/reference
comparison cannot yet be made. Actual evolving support, signed normalization,
seam behavior, unchanged geometry across real steps, fresh-linear convergence,
and successful/exceptional probe restoration all remain **unverified in the
smoke**. Do not infer them from the passing preflight, build, or Python tests.

Next decision: approve one new bounded invocation with the corrected benchmark
guard, preserving this failed evidence. No retry or production correction was
performed automatically. K2 remains provisional/Gate K2 unmet; K3 is incomplete.
