#!/usr/bin/env python3
"""Reuse existing surface, filter and bounded automatic-completion fixtures."""
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2];inputs=r/'inputs';inputs.mkdir(exist_ok=True)
for v in ('reference','candidate'):
 plugin=r/('reference-plugin-build' if v=='reference' else 'plugin-build')
 for rank in (1,2):
  for name in ('dynamic','adiabatic','rate','explicit','filter','singular','singular-current','pressure','rollback'):
   source={'dynamic':'phase_field_fault_surface_dynamic_pressure','adiabatic':'phase_field_fault_surface_adiabatic_pressure','rate':'phase_field_fault_surface_rate_dependent','explicit':'phase_field_fault_surface_dynamic_pressure','filter':'phase_field_fault_surface_dynamic_pressure','singular':'phase_field_fault_surface_singular','singular-current':'phase_field_fault_surface_singular','pressure':'phase_field_fault_pressure_gauge','rollback':'phase_field_fault_stage_i_rollback'}[name]
   lib=source if name in ('singular','singular-current','pressure','rollback') else 'phase_field_fault_surface_dynamic_pressure'
   if name=='singular-current':lib='singular_current_diagnostic'
   s=f'include $ASPECT_SOURCE_DIR/tests/{source}.prm\nset Additional shared libraries = {plugin}/lib{lib}.release.so\nset Output directory = {r}/output-{v}-{name}-{rank}\n'
   if name=='filter':
    s+='subsection Material model\n subsection Phase field fault\n  set Fault constitutive mode = mature frictional\n end\nend\nsubsection Fault reconstruction\n set Fit prescribed geometry to phase field = false\nend\n'
   (inputs/f'{v}-{name}-{rank}.prm').write_text(s)
  s='include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_boundary/inputs/auto-bp3-base.prm\n'
  s+=f'set Additional shared libraries = {plugin}/bp3/libbp3_restore_150x50.release.so, {plugin}/libcache_audit.release.so\nset Output directory = {r}/output-{v}-bp3-{rank}\n'
  s+='subsection Postprocess\n set List of postprocessors = reconstructed fault BP3, visualization, particles, reconstructed faults, BP3 output complete, BP3 restored monitor, normalization cache audit\nend\n'
  (inputs/f'{v}-bp3-{rank}.prm').write_text(s)

for v in ('reference','candidate'):
 plugin=r/('reference-plugin-build' if v=='reference' else 'plugin-build')
 (inputs/f'{v}-frozen.prm').write_text(
  'include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/frozen_gmg_repair/frozen.prm\n'
  +f'set Additional shared libraries = {plugin}/libbp3_frozen_reference.release.so, {plugin}/libreconstructed_fault_frozen_gmg.release.so\n'
  +f'set Output directory = {r}/output-{v}-frozen\n')
