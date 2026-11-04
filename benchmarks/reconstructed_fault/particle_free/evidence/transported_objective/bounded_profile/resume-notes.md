# Resume and retained failed attempts

After context resumption, two surface-sensitivity launches were queued again
before the earlier queued launches had created their output directories. The
runner's `exist_ok=False` guard rejected both duplicates with FileExistsError;
no existing run, output or checkpoint was overwritten. The original
`bpf-c-p-m-surface` and `bpf-l-p-m-surface` runs both exit 0 and their exact
commands/hashes remain authoritative. No additional MPI simulation was started
by either rejected duplicate.

Other retained failures: initial C++ Point ordering build (fixed using an
explicit lexicographic comparator); initial sandbox-blocked MPI attempt;
`bpf-c-p-m-base` finite-endpoint full-normal admission rejection (fixed for the
selected mature route); expected method/inflow/support identity and coverage
rejections. One already completed positive constant reference has a terminal
2.77555756e-17 s interval; raw evidence remains, but physical reductions exclude
that roundoff-only step and identify it explicitly. Later fresh uniform runs
use the representable accumulated end clock without extending requested time.

The first complete 28-case reduction passed its numeric assertions but failed
at final JSON writing because a local slip array shadowed the CLI options.
The options variable was renamed. `reduction.log` retains this analysis-only
failure; `reduction-final.log` records the corrected replay. No native test or
physical output changed or needed repetition.

The first provenance reduction classified three retained native build/config
inputs (CMakeLists.txt, revision.h.in, config.h.in) outside its C++ suffix filter
and incorrectly reported a missing-source assertion. All three files were
present. The recorder now verifies every entry path explicitly in addition to
new native files; no source file was missing or restored.
