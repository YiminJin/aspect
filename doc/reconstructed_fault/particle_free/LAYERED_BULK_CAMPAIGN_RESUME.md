# Corrected-spin layered campaign — closeout handoff

Complete, 6 November 2026. No campaign process or queued test remains active. The
user-selected task is finished; the next scientific experiment requires a new
selection. The earlier shutdown pause and chronological progress notices remain
in `layered_bulk_campaign/documentation-before-closeout/`, `pause-request.json`
and `pause-complete.json`.

Root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `cf06525471ce89789f1ac10905dc79f3aac7562a`, with
preserved local/manual work. No commit, reset or alternate checkout. Read the
[report](LAYERED_BULK_CAMPAIGN_REPORT.md), [protocol](LAYERED_BULK_CAMPAIGN_PROTOCOL.md)
and [evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/README.md).

## Completed — do not repeat by default

- Homogeneous `exterior-compatible0125-bulk` integration bridge.
- Layered `bulk-zero`, `bulk-plus`, `bulk-minus`, `bulk-plus-half`,
  `bulk-total-plus`, `bulk-total-plus-half`, `bulk-plus-fine-cartesian`,
  `bulk-total-plus-fine-cartesian`, `bulk-adaptive` and `bulk-plus-wide`.
  Layered raw folder names include the `layered-` prefix.
- `bulk-adaptive-restart`: two ranks, 58 steps 160–217 from the preserved checkpoint
  after accepted 159. All clocks match within 2.8e-16 s; exported bulk/surface/
  background component differences remain below 1.8e-13 in their native units.
- `bulk-provider-reject` / `bulk-provider-control`: one rejected attempt,
  five accepted steps to .05 s, bitwise matching physical/background exports.
- Cartesian admission fix: 183390 assertions / 157 unit cases, sheared negative
  control, and bitwise original/corrected two-step coarse prefix. Original
  failed non-dyadic fine attempt remains preserved.

All complete comparisons, independent references, central/end norms, signed and
absolute energy diagnostics, figures and a 65-frame cell-crossing GIF are present.
The finite-slip effect persists, while profile mass error worsens under timestep
halving and is essentially unchanged by widening. The report separates physical
boundary response, numerical profile redistribution and finite-end/traction
uncertainty. It does not qualify production or cohesive evolution.

## Source, binaries and checkpoint bookmark

Original candidate: `build-particle-free/transported-objective/stage-c/layered-bulk/`,
core SHA256 `17f7152ca5e5f78b309cfebf4586a82ac1926e1ef2b9231e225d2afdc3e525ed`.
Corrected Cartesian-admission candidate: sibling `layered-bulk-cartesian/`, core
SHA256 `812811cd41f0551ef346117ecbc86ed0e694f8cb1a0229be7b02207fdbb8834a`.
The latter uses byte-identical copied plugins and only changes the roundoff guard.
It is used by the corrected fine, adaptive, lifecycle and wide cases. Protected
canonical/original binaries and their runtime libraries are retained.

The adaptive bookmark is
`stage-c/checkpoints/layered-bulk-adaptive-step160/restart`, after accepted 159
at t=.45019467084587306 s. Its first resumed step is 160, dt=.004066549949190247 s.
It was copied and hashed before normal rotating-slot overwrite; the marker
`adaptive-intermediate-checkpoint.json` records its files. This preserves varying
dt after reload, rather than testing only the late cap. The runner's optional
`--checkpoint-directory` selects it while verifying the original parent fixture
metadata. Native checkpoint meanings and cross-law rejection remain unchanged.

## Reproduction and next decision

Runners and reducers are in
`benchmarks/reconstructed_fault/particle_free/transported_objective/`.
Use `run_layered_bulk_matrix.py`, `run_layered_campaign.py`,
`check_layered_bulk_rejection.py` and `make_layered_bulk_figures.py`; exact commands
and candidate/input identities are in the evidence bundle. Runners refuse
existing output directories. Set `LAYERED_EVIDENCE` to the corrected
`layered_bulk_campaign` directory; the historical default is a different study.
Keep native runs/builds and heavy reductions sequential, with at most four workers.

The next proposed implementation is bounded Q1 transport satisfying the integrated
material equation, including divergence, with the same transported obstacle and
acceptance transaction. Compare against these retained MacCormack cases with
fixed physical inputs/schedules and label spacing. Do not replace that work by
another smaller dt or global renormalization. The finite-domain mechanical
sensitivity remains part of the error budget. A stationary layered pilot needs
its own supported configuration and dimensional mapping; U=0 here still deforms.
No server/production submission or checkpoint conversion is selected. Cleanup
losses remain 39/45 native-pilot frames and earlier temporary logs.
