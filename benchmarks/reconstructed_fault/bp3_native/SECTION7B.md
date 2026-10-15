# Bounded serial paired BP3 — section 7B only

Baseline c33edaaeb, fault-propagation; incoming first-pass work preserved.
No 7C/7D, unfiltered, full-model, checkpoint continuation or transport test.

Predeclared pair: translated 2000×1000 m Box, 60° fault, existing graded
4136-cell Q2/Q1 mesh, ell20/core .6, 59 surface vertices, native consistent I_h
and paired completion, G=32038120320 Pa, eta=1e26 Pa s, uniform a=.025,b=.015,
Dc=.008 m and physical Vp=Vinit=1e-9 m/s. Full bottom/side loading, natural top,
filter20 and zero fresh bulk perturbation. Native stationary samples and no
insertion/deletion in both backends. Existing Newton/linear tolerances stay
1e-8/1e-9. The common mechanical rule is native QGauss3 in each coordinate;
set the existing FE option to 3 and verify exact points/weights. No new core
quadrature option is needed.

Build the same maintained plugin for both with its historical
ASPECT_BP3_LOCAL_BOTTOM_TEST definition, independently of the old local build.
ASPECT_BP3_LOCAL_OSCILLATION_TEST stays off: Vp remains physical, not 1e-3.
The retained local definition selects uniform strengthening and the existing
Theta0*exp(.02*sin(pi*(y-y_origin)/height)^2) initialization control. In the
creep pair its amplitude is zero. No traction/loading/friction adjustment
occurs between the pair members; only history representation differs.

Prescribed dt=200000 s (also initial constitutive interval), ten positive-time
steps to T=2000000 s, separately recorded step zero. First run one positive
step in each backend to gate correctness. Failure strategy aborts rather than
silently admitting different schedules. Existing positivity/solver/history
criteria are retained. Runner caps: 400 s per gate and 700 s per trajectory,
maximum 3600 s simulation budget including gates, excluding compilation.
Fresh output paths, Last accepted step=1/10 and no checkpoint writers.

Compact observer records common-time full fault profiles, raw/filtered normal
traction and pressure/deviatoric/background loads on the same surface mass
measure. It checks frozen fields/positions/population, one mechanical acceptance,
one state/slip/FE-memory publication and direct Maxwell values at realistic
stress scales. A small QP window and deterministic particle subset check samples;
the production FE commit also validates every derived particle sample.
The measured trace term 2 eta_ve div(u)/3 diagnoses the tensor convention;
it is not a production-law switch or a unique causal attribution of differences.

Compare absolute/relative common-time fields using Vp, Dc/Vp, Vp*T, background
tractions and measured perturbation scales; separate mean/chord/oscillation
metrics near endpoints and mesh transitions. Finish one 7B report and update
STATUS.md. A coupled failure must be resolved or reported precisely before
extending either trajectory. No further stage is implicitly authorized.
