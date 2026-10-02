#!/usr/bin/env python3
"""Check evidence consistency; explicitly retain the failed qualification gate."""
from pathlib import Path
import csv,json,hashlib
r=Path(__file__).resolve().parent;d=r/'results';repo=r.parents[2]
def rows(name):return list(csv.DictReader((d/name).open()))
checks={}
a=rows('comparison.csv');assert len(a)==8
checks['constant_consistency']=all(float(x['constant_max_error_over_1e8'])<=1e-10 for x in a)
checks['unlimited_affine_consistency']=all(float(x['affine_max_error_over_1e8'])<=1e-10 for x in a if not x['case'].endswith('limited'))
checks['native_realized_totals']=all(int(x['initial'])==2048 for x in a)
checks['actual_transport_births']=all(int(x['transport_births'])>0 for x in a)
checks['actual_startup_removal']=all(int(x['startup_removals'])>0 for x in a if x['case'].startswith('random'))
checks['managed_bounds']=all(12<=int(x['count_min'])<=int(x['count_max'])<=24 for x in rows('stage_fields.csv') if x['stage']!='generated')
checks['positive_input_negative_H_demonstrated']=any(float(x['input_min'])>0 and float(x['proposal'])<0 for x in rows('negative_H_proposals.csv'))
checks['limited_coupled_H_valid']=all(int(x['H_negative'])==0 and float(x['H_min'])>=0 for x in rows('lifecycle_states.csv'))
lifecycle=rows('lifecycle_checks.csv')
checks['restoration_and_restart_exact']=all(x['equal']=='True' for x in lifecycle if x['candidate']!='crossing-retry')
checks['retry_defect_retained']=sum(x['equal']=='False' for x in lifecycle)==3
shared=json.loads((d/'mpi_supports.json').read_text())
checks['shared_Q2_visualization_consistent']=shared['shared_piece_occurrences']>0 and all(x==0 for x in shared['shared_max_abs'].values())
manifest=json.loads((repo/'benchmarks/reconstructed_fault/refactoring_r6b/evidence/candidate-source-hashes.json').read_text())
checks['R6_source_unchanged']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
runs=json.loads((d/'run_metadata.json').read_text())
for c in [x['case']+'-measured-np1' for x in a]+['coupled-regular-Hlimited-np1','coupled-random-5433-Hlimited-np1','crossing-serial-np1','crossing-mpi-np2','crossing-create-np1','crossing-resume-np1','crossing-after-create-np1','crossing-after-resume-np1','crossing-direct-np1','crossing-retry-np1','candidate-parse-np1']:
 checks[c]=runs[c]['exit_code']==0
seconds=sum(x['seconds'] for k,x in runs.items() if '-np' in k)
checks['bounded_runtime']=seconds<900
result=dict(evidence_checks=checks,all_evidence_checks_pass=all(checks.values()),measured_simulation_seconds=seconds,server_continuation_qualified=False,remaining_gate='Particle RNG retry equivalence fails; per-rank checkpoint RNG preservation unverified.')
(d/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert all(checks.values())
