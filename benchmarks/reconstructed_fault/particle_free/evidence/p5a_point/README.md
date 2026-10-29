# P5a compact point evidence

Standalone stateless material-point implementation only. No native solver, production checkpoint conversion, coupled phase evolution or directional specimen run.

See P5A_POINT_REPORT.md for equations and interpretation. response.png shows imposed strain, physical stress, active flow and Y; tensor-memory.png separates relaxed old memory from accumulated new increments; energy.png separates recoverable storage, three physical dissipations and the backward-Euler numerical remainder. Every curve comes from the corresponding fixed/radial-history.csv, with physical units in headers. Axx=-Ayy, Azz=0 (likewise K/Q); full T and tau include zz. Rates are per second; energy/dissipation integrals are J/m3. Phi=.2 is prescribed in this demonstration; no spatial fracture/seed energy is evaluated.

The engineering magnitude is sqrt(2)||W||_3D=2*hypot(Wxx,Wxy). Alpha=atan2(Wxx,Wxy) is a two-component tensor-flow angle, not a crack direction. active-directions.csv excludes zero flow; undefined angles are not assigned a direction. Coordinate covariance is passive, not finite-rotation/objective-rate qualification.

verification.json contains measured defects, negative controls, lifecycle counters, original failed probe, and protected-file hashes. independent-point-reference.csv and independent-path-reference.csv solve the point potential at 60 digits. derivative-inputs.csv supplies all states/increments used for the full-return derivative cases. The local Boost point record is not a native ASPECT checkpoint.

changed-files.json distinguishes this task from inherited local changes. provenance.json lists exact configure/build/run/analysis commands and binary identities. Timings are measured for this tiny fixture, with no native/full-event speed claim. The two p5a-*.json files retain wrapper commands and wall times. development-failures.json records corrected harness/reference mistakes and the demonstrated tiny-phase arithmetic repair without changing law or tolerances. Original logs/versions remain separately in the worktree build and refactoring_boundary/evidence directories.

P4 failed temporal/spatial targets, 57.26% boundary sensitivity and strict_reference_match_passed=false remain. P5b native integration/tangent/stability qualification is proposed only. The eventual specimen is a finite physical body with explicit boundaries, not a distant-boundary approximation.
