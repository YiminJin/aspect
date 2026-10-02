#!/usr/bin/env python3
"""Prepare matched projection fixtures and private preserved-checkpoint branches."""
from pathlib import Path
import shutil
r=Path(__file__).resolve().parent;repo=r.parents[2];inputs=r/'inputs';inputs.mkdir(exist_ok=True)
old=r.with_name('refactoring_r5a1')
for v in ('reference','candidate'):
 plugin='reference-plugin-build' if v=='reference' else 'plugin-build'
 def write(name,source,library,extra=''):
  s=f'include $ASPECT_SOURCE_DIR/tests/{source}.prm\n'
  libraries=f'{r/plugin}/lib{library}.release.so'
  if source=='phase_field_fault_surface_adiabatic_pressure':libraries+=f', {r/plugin}/libphase_field_fault_ih.release.so'
  s+=f'set Additional shared libraries = {libraries}\n'
  s+=f'set Output directory = {r}/output-{v}-{name}\n'
  (inputs/f'{v}-{name}.prm').write_text(s+extra)
 # The same existing fifty-iteration initialization budget used by the cache
 # fixtures; no solver tolerance or physical parameter is changed.
 budget='subsection Solver parameters\n subsection Phase field solver parameters\n  set Max nonlinear iterations = 50\n end\nend\n'
 for rank in (1,2):
  write(f'ih-{rank}','phase_field_fault_ih','phase_field_fault_ih',budget)
  write(f'no-composition-{rank}','phase_field_fault_ih_no_composition','phase_field_fault_ih',budget)
  write(f'surface-{rank}','phase_field_fault_surface_adiabatic_pressure','phase_field_fault_surface_system')
  for backend in ('remote','cell'):
   extra=budget+'subsection Postprocess\n set List of postprocessors = particles, verify fault I h cache, normalization cache audit\nend\n'
   if backend=='cell':extra+='subsection Material model\n subsection Phase field fault\n  set I h integration backend = cell intervals\n end\nend\n'
   write(f'cache-{backend}-{rank}','phase_field_fault_ih','cache_audit',extra)
  for name,source in (('pressure','phase_field_fault_pressure_gauge'),('rollback-original','phase_field_fault_stage_i_rollback')):
   write(f'{name}-{rank}',source,source)
 write('cold-warm-2','reconstructed_fault_particle_projection_cache','reconstructed_fault_particle_projection_cache')
 out=r/f'output-{v}-restart'
 shutil.copytree(r.with_name('restart_investigation')/'output-create-one',out)
 s=(old/f'inputs/{v}-restart.prm').read_text().replace('refactoring_r5a1','refactoring_r5a2')
 (inputs/f'{v}-restart.prm').write_text(s)
