import json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path('benchmarks/reconstructed_fault/particle_free').resolve()))
from analyze_bp3_optimization import trajectory,comparison
work=Path('build-particle-free/bp3-homogeneous-coefficients');runs=Path('build-particle-free/bp3-optimization/runs')
labels=['coefficients-optimized','coefficients-reference','coefficients-split','coefficients-resume','coefficients-rejection']
r=dict(base='b2dfdebe43437d6028b06d4f9f862cc80a00cc71',strict_reference_match_passed=False,runs={},comparisons={})
for name in labels:r['runs'][name]=trajectory(runs/name)
for a,b in [('coefficients-optimized','coefficients-reference'),('coefficients-resume','coefficients-optimized'),('coefficients-rejection','coefficients-optimized')]:r['comparisons'][a+'__'+b]=comparison(runs/a/'run',runs/b/'run',2,2)
for name in ('coefficients-reference','coefficients-rejection'):assert r['runs'][name]['accepted_step_sequence']==r['runs']['coefficients-optimized']['accepted_step_sequence']
assert r['runs']['coefficients-resume']['accepted_step_sequence'][-1]==r['runs']['coefficients-optimized']['accepted_step_sequence'][-1]
text=(runs/'coefficients-before-v3/stdout.log').read_text()
assert 'p.eta_ve==eta' in text and 'Stationary DG coupling requires homogeneous Maxwell coefficients.' in text
r['original_assertion_reproduced']=True
text=Path('benchmarks/reconstructed_fault/refactoring_boundary/evidence/bp3-coefficients-units.log').read_text()
assert 'All tests passed (165094 assertions in 127 test cases)' in text
r['native_tests']=dict(cases=127,assertions=165094,passed=True)
line=next(t for t in text.splitlines() if t.startswith('Homogeneous coefficient regression:'))
r['independent_coefficient_control']=line
r['units_coefficient_spread_removed']=True
c=r['comparisons']['coefficients-rejection__coefficients-optimized'];assert c['stress']['max_full_tensor_Pa']==0 and all(v==0 for d in c['persistent'].values() for v in d.values())
text=(runs/'coefficients-rejection/stdout.log').read_text();assert 'accepted histories unchanged; identical-dt retry follows.' in text
(work/'assessment.json').write_text(json.dumps(r,indent=2)+'\n')
for k,v in r['comparisons'].items():
 print(k,v['stress']);print('Z/rate/state',*[v['persistent']['coupled-profile'][f] for f in ('Z_Pa_m','V_m_s','theta_s')])
for k,v in r['runs'].items():print(k,'seq',v['accepted_step_sequence'],'maxres',v['max_normalized_nonlinear_residual'],'maxpubratio',v['max_publication_allowance_fraction'])
