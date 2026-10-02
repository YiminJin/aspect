#!/usr/bin/env python3
"""Wrap the existing small BP3 and rollback cases; never adjust physical inputs."""
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2];(r/'inputs').mkdir(exist_ok=True)
for v in ('reference','candidate'):
 plugin=r/('reference-plugin-build' if v=='reference' else 'plugin-build')
 for mode,rank in [('off',1),('on',1),('off',2),('on',2),('missing',1),('blocked',2)]:
  label=f'bp3-{mode}-{rank}';out=r/f'output-{v}-{label}'
  out.mkdir(exist_ok=True)
  if mode=='blocked':
   for step in range(1,7):
    for rank_id in range(rank):
     for stream in ('stress_update','continued_source_history'):
      (out/f'{stream}_{step}_rank{rank_id}.csv').mkdir(exist_ok=True)
  s='include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_boundary/inputs/auto-bp3-base.prm\n'
  s+=f'set Additional shared libraries = {plugin}/bp3/libbp3_restore_150x50.release.so, {plugin}/libcache_audit.release.so, {plugin}/libtrace_cells.release.so\nset Output directory = {out}\n'
  s+='subsection Postprocess\n set List of postprocessors = reconstructed fault BP3, visualization, particles, reconstructed faults, BP3 output complete, BP3 restored monitor, normalization cache audit, history trace cells\n'
  s+=' subsection History trace cells\n  set Write selection = '+('false' if mode=='missing' else 'true')+'\n end\nend\n'
  (r/f'inputs/{v}-{label}.prm').write_text(s)
 for mode in ('off','on'):
  for rank in (1,2):
   label=f'rollback-{mode}-{rank}'
   (r/f'inputs/{v}-{label}.prm').write_text('include $ASPECT_SOURCE_DIR/tests/phase_field_fault_stage_i_rollback.prm\n'+f'set Additional shared libraries = {plugin}/libphase_field_fault_stage_i_rollback.release.so\nset Output directory = {r}/output-{v}-{label}\n')
