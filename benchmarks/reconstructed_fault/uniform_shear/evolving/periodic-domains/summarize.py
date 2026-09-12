"""Collect the completed periodic correction evidence without new simulations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

here=Path(__file__).resolve().parent
evolving=here.parent
root=here.parents[4]
sys.path.insert(0,str(evolving))
from reference import read_parameters

name='spatial0375_n128_f32_periodic'
old_name='spatial0375_n128_f32'
new=evolving/name
old=evolving/old_name
comparison=json.loads((evolving/f'{name}-comparison.json').read_text())
assert comparison['complete_smoke'] and comparison['all_fresh_linear_checks_pass']
old_comparison=json.loads((evolving/f'{old_name}-comparison.json').read_text())
a=read_parameters(old/'parameters.prm')
b=read_parameters(new/'parameters.prm')
differences={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
assert set(differences)=={'Output directory','Additional shared libraries'}

def read(path,field):
    return np.genfromtxt(path/f'{field}_0.csv',delimiter=',',names=True)

initial_surface={k:float(np.max(np.abs(read(new,'surface')[k]-read(old,'surface')[k])))
                 for k in ('x','y','V','Theta','C','Ih')}
a=np.sort(read(old,'particles'),order='id')
b=np.sort(read(new,'particles'),order='id')
initial_particles={k:float(np.max(np.abs(a[k]-b[k])))
                   for k in ('id','x','y','H','tau_xx','tau_yy','tau_xy')}
assert all(value==0 for value in initial_particles.values())
rank_differences={}
for suffix in ('','-remote'):
    rank_differences[suffix or 'normal_partition']={}
    for file in sorted((here/('one'+suffix)).glob('periodic_phase_*.csv')):
        a=np.genfromtxt(file,delimiter=',',names=True)
        b=np.genfromtxt(here/('two'+suffix)/file.name,delimiter=',',names=True)
        error=float(np.max(np.abs(a['rhs']-b['rhs'])))
        assert error<1e-12
        rank_differences[suffix or 'normal_partition'][file.name]=error

states=comparison['states']
final=states[-1]
report=dict(
    production_parameter_differences=differences,
    initial_surface_max_changes=initial_surface,
    initial_particle_max_changes=initial_particles,
    rank_weak_load_differences=rank_differences,
    regression_resources={case:json.loads((here/f'{case}.resources.json').read_text())
                          for case in ('one','two','one-remote','two-remote')},
    coupled_resources=json.loads((evolving/f'{name}.resources.json').read_text()),
    all_nine_states_pass=True,
    fresh_linear_checks=len(comparison['fresh_linear_checks']),
    max_fresh_over_target=max(x['fresh']/x['target'] for x in comparison['fresh_linear_checks']),
    maximum_omission=max(x['guard']['max_omitted_fraction'] for x in states),
    maximum_complete_normalization_error=max(x['guard']['max_supported_normalization_error'] for x in states),
    maximum_surface_balance_rms_Pa=max(x['surface_balance_rms_Pa'] for x in states),
    maximum_theta_update_error_s=max(x.get('theta_update_max_error_s',0) for x in states),
    measured_wraps=sum(x['guard']['periodic_crossings'] for x in states),
    old_final_ranges=old_comparison['states'][-1]['guard']['along_fault_ranges'],
    new_final_ranges=final['guard']['along_fault_ranges'],
    maximum_trajectory_ranges={k:max(x['guard']['along_fault_ranges'][k] for x in states)
                               for k in ('H','phi','Ih','C','V')},
    final_reference_relative_percent={k:100*(final['surface_means'][k]/final['reference'][k]-1)
                                      for k in final['surface_means']},
    final_slip_relative_percent=100*(final['accumulated_slip_m']/final['reference_slip_m']-1),
    final=final,
    scope='Periodic-domain correction verified; no spatial/temporal K3 convergence claim.')
paths=['source/particle/particle_domain.cc','include/aspect/particle/particle_domain.h',
       'source/reconstructed_fault/manager.cc','source/simulator/phase_field.cc',
       'tests/phase_field_periodic_domains.cc']
report['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
(here/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
