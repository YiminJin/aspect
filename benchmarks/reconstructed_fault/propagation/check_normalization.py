#!/usr/bin/env python3
"""Separate normalization implementation checks from the failed P2 accuracy gate."""
import json
from pathlib import Path
import numpy as np
import check_integration as c
here=Path(__file__).resolve().parent;e=here/'normalization_evidence'
result={'P2_normalization_qualified':False,'fresh_coupled_paths_run':False}
result['phase_checks']={label:c.assess(label,kind) for label,kind in [
 ('normalization_consistent','straight'),('normalization_lumped','straight'),
 ('normalization_lumped_mpi_allowed','straight'),('normalization_lumped_padded','padded')]}
result['MPI_phase_history_error']={str(k):c.compare(c.nodes(c.build/'output-normalization_lumped',k),c.nodes(c.build/'output-normalization_lumped_mpi_allowed',k),('phi','old','older'),4.922e-12) for k in range(5)}
result['default_phase_error']={str(k):c.compare(c.nodes(c.build/'output-normalization_consistent',k),c.nodes(c.build/'output-straight_support_fixed',k),('phi','old','older'),0) for k in range(5)}
for label in ['normalization_lumped','normalization_lumped_mpi_allowed']:
 assert all(result['phase_checks'][label]['restart_error_'+str(k)]==0 for k in (3,4))
result['projection_checks']={}
for label in ['normalization_consistent','normalization_lumped','normalization_lumped_mpi_allowed','normalization_lumped_spacing05','normalization_lumped_padded']:
 rows=[c.rows(f)[0] for f in (c.build/('output-'+label)).glob('projection-check-*.csv')]
 assert len(rows)==5
 assert all(r['constant_error']<2e-14 and r['conservation_error']<2e-14 and r['minimum_lumped']>0 for r in rows)
 result['projection_checks'][label]={k:max(r[k] for r in rows) for k in rows[0]}
result['native_frozen_projection_error']={}
for label,spacing in [('normalization_lumped',.1),('normalization_lumped_mpi_allowed',.1),('normalization_lumped_spacing05',.05)]:
 native=np.genfromtxt(c.build/('output-'+label)/'frozen-native-projection.csv',names=True,delimiter=',')
 reference=np.genfromtxt(e/'coarse'/f'nodes-{spacing}-1.csv',names=True,delimiter=',')
 errors={field:float(max(abs(native[field]-reference[field]))/max(abs(reference[field]))) for field in ['lumped','consistent']}
 assert max(errors.values())<5e-13
 result['native_frozen_projection_error'][label]=errors
for label in ['normalization_reject_consistent','normalization_reject_lumped']:
 log=(here/'integration_evidence'/(label+'.log')).read_text()
 assert 'Cannot change I_h projection on restart' in log
 assert json.loads((here/'integration_evidence'/(label+'.json')).read_text())['code']==1
result['restart_switch_rejected_both_directions']=True
assert json.loads((here/'integration_evidence/normalization_legacy_resume.json').read_text())['code']==0
result['legacy_consistent_checkpoint_loaded']=True
for case in ['mature','ih_budget']:
 assert json.loads((e/('normalization_compat_'+case+'.json')).read_text())['code']==0
result['compatibility']=['default mature frozen boundary completion','I_h reference with retained 50-iteration budget']
for label in ['coarse','fine']:
 d=json.loads((e/label/'summary.json').read_text())
 assert d['analytic_vs_Q64_transverse_max_relative']<1e-11
 assert all(r['constant_error']<1e-14 and r['lumped_conservation_relative']<1e-14 and r['lumped_min']>0 and r['minimum_row_sum']>0 for r in d['projections'])
 result[label]=d
assert not result['fine']['finite_transverse_and_surface_integral']
assert result['coarse']['finite_transverse_and_surface_integral']
assert all(r['lumped_relative_L1']>.3 for r in result['coarse']['projections'])
result['implementation_checks_pass']=True
(e/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
print('Implementation checks pass; P2 gate FAILS: exact saturation and unresolved normalization representation.')
