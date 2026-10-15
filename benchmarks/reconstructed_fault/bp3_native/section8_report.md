# BP3 section 8: bounded diagnostics and fresh preparation

The ten-step unfiltered control passes. The finer initialization check reduces
profile error but leaves measurable localization sensitivity. The fresh ordinary
150 × 50 km FE candidate is prepared and validates; its fine mesh is not launched.
The coarse full-geometry check passes initialization and two physical steps.
All section-8 launches, including rejected diagnostic startups, cost 1382.60 s
(23.04 min), excluding compilation. No long first-event or cluster run is made.

## Filter comparison

Baseline `fault-propagation`, `c33edaaeb145e5d3dce155bb815552ef80efc5f9`.
Read-only reference `pf-rsf-refactor`, `8f126905c559df106f08af4079f678401fec4d06`;
its revised inputs are local evidence, not a verified latest server policy.
The retained `7cd-bulk` filtered trajectory is reused: same pinned executable
`1ba2f7f43b37`, maintained local plugin `676177bc1935`, physical loading
1e-9 m/s, translated 60° geometry, frozen profile, homogeneous G/eta, perturbed
uniform strengthening, stationary samples, full bottom, q6, finest 3.90625 m,
117 surface nodes/9.954315 m intervals. Step-zero constitutive interval remains
200000 s; ten positive 200000 s intervals reach 2000000 s. Phase, I_h, mesh,
quadrature and physical initial particle identities are byte-identical.

The only resolved mechanical settings changed are `Friction normal input=raw`
and its API-required `Normal filter length=0`. Friction uses raw bulk-QP normal
stress, verified on captured samples; Helmholtz length zero alone would retain
projection. In the raw CSV, `filtered_normal_Pa` is the effective unfiltered Q1
diagnostic, equal to its raw column; it does not denote an applied filter.
The new observer's filtered replay reproduces retained step zero byte for byte.
Both paths use the FE consistent 3D deviator; the legacy full-symmetric-gradient
distinction does not enter this pair.

| Final raw-minus-filtered field | Max absolute | Relative scale |
|---|---:|---:|
| V |5.6945e-12 m/s|.56945% of Vp|
| theta |6207.2 s|.07759% of 8e6 s|
| Omega |.0049203|.49203% of 1|
| cumulative slip |6.7878e-6 m|.33939% of Vp T|
| shear |4064.36 Pa|RMS difference 16.71% of perturbation RMS|
| raw normal |3410.96 Pa|RMS difference 20.27% of perturbation RMS|
| effective normal |20145.27 Pa|RMS difference 87.10% of perturbation RMS|
| pressure /tensor normal |2232.59 /1178.37 Pa|8.19 /20.18% of peak response|

Initial theta, slip, I_h and Z coincide; the mechanical initialization already
differs by max V 7.226e-13 m/s, shear 85.23 Pa, raw/effective normal 71.54/1861.36 Pa,
pressure/tensor 44.99/26.55 Pa. All common-time fields, RMS/relative differences
and initial discrepancies are retained in the linked data.

At the lower 200 m, raw-normal local-chord roughness is 19.19/19.47 kPa
(filtered/raw trajectories); effective-normal roughness is .660/19.47 kPa and
V roughness 2.46e-12/8.19e-12 m/s. Pressure/tensor peaks are 25.03/5.84 versus
27.26/4.66 kPa; their cancellation ratios are 1.608/1.455. Full-window Hann
short-wave raw-normal amplitudes remain 2.27/2.24 Pa at 48.53 m wavelength.
The window downweights endpoints: this spectrum does not contradict increased
endpoint roughness. Chords/spectra are roughness diagnostics, not error estimates.

Max normalized residuals are 7.20e-10/6.33e-9; max surface RMS 4.02e-7/.00930 Pa.
Both need 3 Newton updates at step zero, none at step one, 2 per later step;
Krylov counts are 106/103 initially, 0 at step one, 76 per later step.
Accepted direct-Maxwell defects≤2.92e-8 Pa, positive I_h/state, mature cohesive
traction exactly zero, independent evolving Z, frozen phase/H and once-only
history/state/slip/sample publication pass. Wall times 527.26 s reused/510.27 s
fresh; no cutback or tolerance change. This short control does not justify
removing the full candidate's20 m filter or claim late-event accuracy.

## Profile resolution

Initialization only: same physical profile, fixed 117-node surface, q6 and general
cell-interval consistent normalization, including native automatic completion.
Bulk cells 11528→35342; finest 3.90625→1.953125 m, particles 184448→565472.
At 2651 common physical points (514 exterior-completion samples), max phase change
is .00720065; max discrepancy from the unchanged continuum profile decreases
.00773339→.00197062. Min I_h 12818.60795→12993.67519 m (+1.37%). Max localization
change is .00208632/m, 3.71% of peak; exterior-only change .00146930/m, 2.69% of peak.
Both admitted contacts remain valid. Runtime 10.44/22.97 s; no mechanics is run.
These are representation changes, not quadrature errors. The 3.90625 m candidate
remains provisional: coupled traction/localization accuracy at a finer bulk
resolution has not been established. No profile peak, width or parameter is retuned.

## Fresh ordinary BP3 candidate and bounded startup

[bp3-fresh-fe.prm](bp3-fresh-fe.prm) inherits the maintained fresh production
template, not the historical restart. Ordinary plugin `d8a151bbdd33` has empty
extra CXX flags and both local-test definitions absent; physical Vp remains 1e-9.
The actual 150000×50000 m Box, 60° fault, 15–18 km a=.010→.025 transition, b=.015,
Dc=.008 m, native theta inversion, G=32038120320 Pa, eta=1e26 Pa s, ell 20 m,
peak .6, Gc 1e5, cohesion 1e6, damping 4624440, sigma0=50 MPa and background
shear 26.546122365 MPa are restored. Fresh bulk perturbation history is zero;
background tractions remain surface data. Mature FE memory retains
`d_R'=beta d_R+(eta_ve/G)u`, `Z'=beta Z+eta_ve V` independently of cohesive zero.
Here `beta=exp(-G dt/eta)`, `eta_ve=-eta expm1(-G dt/eta)`, and
`tau=2G D3(symgrad d_R)-2(h/I_h) Z S`. Mature Z is not inferred from I_h tau_coh;
the separate cohesive identity remains unchanged. Oblique Q1 normal restrictions
use the general integrator and consistent surface projection, with automatic
paired computational-boundary completion and native endpoint source continuation.
Geometry, phase and mesh stay fixed; no P2 endpoint/phase solve is enabled.

**Stationary history only:** explicitly stationary particle sampling, no insertion,
deletion or load balancing. FE restart/conversion and transport are unsupported.
Checkpoint/resume writers are disabled, with abort on unexpected nonlinear failure.
The 7D identical-dt in-memory retry does not validate automatic cutback.

Timestep provenance is the read-only reference's `bp3_fresh_revised.prm` (Oct 5)
and executed `output-first-event/original.prm` (Oct 6): first cap 4e6 s, maximum 4e7 s,
initial constitutive interval 4e6 s, convection/fault controllers, unweighted
log-state bound .2, default 91% growth cap made explicit. This supersedes the
maintained template's provisional 100 s/4e6/.1 values. The retired weighted BP5
startup predictor is not restored. The ordinary startup selects 1772.48070853 and
2166.15453236 s: the independently evaluated max predicted log-state changes are
.2000000000000002 in both intervals. A forced 200000 s local diagnostic interval
would predict 3.24981, violating .2; it is not copied into BP3.

Prepared fine mesh: maintained measured 552084 cells, 8833344 particles,
approximately 18–21 million total DoFs, provisional 10 m surface spacing/q6.
Conservative preflight estimates are 35–100 GiB and 2000–10000 s serial startup,
not measured full-mesh resource claims. No fine/full first-event run is launched.
[section8-full-startup.prm](section8-full-startup.prm) reduces only bulk finest
spacing to 31.25 m and requested surface spacing to 40 m, retaining the physical
length scale/profile/loading/transition and valid native graded endpoint support.
Native setup realizes 41226 cells, 1492632 DoFs and 659616 stationary particles,
inside the preflight 25000–50000-cell/3–10 GiB estimate. It has a 1200 s external
wall cap, 1190 s graceful cap and two-positive-step limit. Both prepared/test
inputs pass parameter validation. The actual 1446-node surface has mean spacing
39.9550 m. Setup and two positive steps finish in 696.36 s to 3938.63524089 s.
Newton updates 4/4/4 and Krylov counts 179/116/113 (steps 0/1/2) meet the unchanged
criteria: max residual 1.81e-9, surface RMS .00744 Pa, state-update error 2.23e-16.
Initial theta ranges 8000–8000000 s and matches the native spatial inversion to
1.54e-14 relative. All 659616 initial particle stresses and authoritative bulk
perturbation history are zero; background shear/normal remain surface data.
The virtual 4e6 s step-zero QP stress scale is 6.29 MPa; it is not published as
initial history. Positive-step QP scales are 3.17/7.05 kPa. Both real intervals
pass direct Maxwell checks (max defect 9.16e-10 Pa; including step zero 5.05e-9 Pa,
≤.479 of the original operand allowance), fixed phase/I_h/H/particle identity, mature
cohesive zero, nonzero Z and one terminal publication including derived samples.
Min I_h 5519.69 m is finite, but its 5519.69–9368.35 m range makes the coarse
profile visibly inaccurate relative to the continuous 13053.8 m normalization.
Peak virtual memory is 8617.29 MiB (8.42 GiB), system matrix 215.32 MiB;
resident peak was not collected. These are integration results, not accuracy.

## Verification and limits

Only the benchmark observer, six parameter inputs, analysis/report and status/scope
documents change. Core/maintained plugin sources, production defaults and all
incoming evidence are preserved. Observer fixes handle the raw-mode optional
vector and test zero velocity history separately from its stored frozen phase;
profile output and the ordinary-controller schedule guard are test-only.
Failed setup/observer records remain: raw nonzero length, raw optional-vector
access, profile whole-vector zero assertion; one observer compilation also failed
before using the existing diagnostic data instead of undeclared plugin globals.
No failed physical recurrence/residual is relabelled or criterion relaxed.
Files: `coupled_checks.cc`, `analyze_section8.py`, `bp3-fresh-fe.prm`,
`section8-{raw,filtered-gate,profile,profile-fine,full-startup}.prm`, this report,
`STATUS.md` and the existing current-status/design/specification documents.

The next accuracy requirement is a coupled bulk/profile refinement with acceptable
normal traction and localization sensitivity. Long-run use additionally requires
FE checkpoint/restart and physical-step cutback verification, plus an explicit
supported history-transport choice. This candidate uses fixed Eulerian FE memory
with stationary derived particle samples; it is not an advected-history model.
No first-event campaign, cluster submission or later stage is authorized/launched.

Evidence: `build-bp3-native/section8/` contains
[comparison](../../../build-bp3-native/section8/comparison.json),
[all common-time differences](../../../build-bp3-native/section8/common-time-differences.csv),
[profiles](../../../build-bp3-native/section8/profiles.pdf), artifact/source/input
hashes, timestep provenance, preflight resources and preservation records.
Exact commands/exits/times are `refactoring_boundary/evidence/bp3-native-8-*.json`
and matching logs. Reproduce with the existing logged runner, e.g.
`python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py LABEL 1200 build-bp3-native/aspect benchmarks/reconstructed_fault/bp3_native/section8-raw.prm`,
using a fresh output directory/label; profile controls use 450/600 s caps.
Run `python3 benchmarks/reconstructed_fault/bp3_native/analyze_section8.py` for
the comparison. No complete history campaign or full compatibility suite is repeated
for these benchmark-only diagnostic changes; ASPECT astyle 2.04 is idempotent.
The incoming 2385-file manifest, exact observer/status/specification snapshots and
all seven retained binary hashes are preserved; the reference worktree remains
read-only. No reset, commit or push is performed. Section 8 is complete; stop here.
