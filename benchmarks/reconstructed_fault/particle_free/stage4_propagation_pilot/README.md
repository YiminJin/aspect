# Estimate-led native propagation pilot

This package records the current user-selected 256x128 weakening-law pilot.
See `doc/reconstructed_fault/particle_free/STAGE4_PROPAGATION_PILOT_REPORT.md`
from the repository root for decisions and limitations. Historical reference
runs, current production source and protected runtimes are preserved.

`preload.h` is benchmark preparation only. It adds shear at fixed normal
preload, then calls the retained zero-clock native irreversible phase/mechanics
settlement. The rejected localized streamfunction variant is frozen under
`build-particle-free/stage4-propagation-pilot/rejected-localized-mode/`.
`ASPECT_PILOT_EQUILIBRIUM_SCREEN` skips phase settlement for explicitly labelled
fixed-phase, zero-physical-interval margin calculations. It is not enabled for
physical evolution and does not replace physical KKT acceptance.

`run.py` records one fresh or existing-law restarted case with four bound MPI
ranks and one thread per rank. It never overwrites a labelled input/run, and
compresses closed new diagnostics with decompressed SHA256 verification.
`screen.py` integrates native fracture-gradient/driving rows using their actual
quadrature measures. `calibrate.py` fits the exact fixed-phase quadratic
response to three mechanical solutions and bounds intact orientations.
`inspect_candidate.py` checks the single settled candidate, probes connected
contour directions read-only, and freezes a fresh geometric input only after
its stated gates. Its coefficient probe is an infinitesimal diagnostic; it is
not evidence for finite-energy release or connected propagation.

`analyze.py LABEL` reduces accepted native phase/section/rate histories and
plots the signed rate over material station and time alongside connected
absolute-phi=.05 extent. It reports initially intact phase, signed deformation,
formed sections, fixed old-plane angles and physical work before ledger
corrections. Rate times are midpoint stages; phase and slip times are endpoints.
Use `--output` for an explicitly composed analysis view of restart segments.

`audit_restore.py` checks the restored clock, phase and section memories, the
full three-dimensional stress energy, and signed section integration measures.
`compose.py VIEW LABEL...` joins only closed, chronological accepted segments,
preserving original process exits and runtime identities in its provenance.
It accepts intentional exit143 without relabelling the process as successful.
`plot_response.py LABEL --output VIEW` adds phase and physical-work figures.

`compress_closed.py LABEL --through STEP` only compresses immutable frames
after a later timestep has started. It verifies decompressed hashes and records
every replacement in `online-lossless-compression.json`; checkpoints and scalar
history files are excluded. This manifest complements the wrapper's final
compression manifest. Readers accept either original CSVs or their gzip copies.

The pilot uses one preparation and steady loading, with only the initial
smooth age perturbation. No rupture time, position or later state reset is
specified. One coarse pilot cannot attribute a response to velocity weakening
or qualify episodic behavior. Any repeated advances/arrests require the
user-specified time/space checks; a matched law control is necessary for a
friction-law attribution.

`run.py --snapshot-id` forwards the existing native snapshot selector while
leaving the source marker unchanged; this selects the preserved step68 for the
bounded40-microsecond check. `compare_half68.py` compares endpoint phase, age and
slip at5.52/5.60 ms and averages stage rates/resistance over matching80-microsecond
intervals. Common-reference section error norms and each state physical measures
are distinguished. Missing matched endpoints remain explicitly missing.
