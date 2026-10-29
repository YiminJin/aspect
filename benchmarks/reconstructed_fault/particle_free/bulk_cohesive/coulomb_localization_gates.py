#!/usr/bin/env python3
"""Retained native gates, same-history cutbacks and bounded temporal differences."""
from pathlib import Path
import json,hashlib
import numpy as np
from coulomb_localization_analysis import ROOT,WORK,OUT,read,write,save

entry=json.loads((WORK/'entry.json').read_text())
for name,digest in entry['protected'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
records=[];costs=[];maximum_source=0.;maximum_residual=0.;maximum_ledger=0.;minimum_trial=1e20;minimum_return=1e20
for run in sorted((WORK/'runs').iterdir()):
    ledger=read(run/'bulk-accepted.csv');resources=read(run/'resources.csv')
    assert all(r['passed']=='1' for r in read(run/'gates.csv'))
    assert len({r['particle_fingerprint'] for r in resources})==1 and all(r['freeze']=='1' for r in resources)
    profile=read(run/'bulk-profile.csv');output=read(run/'output-profile.csv')
    command=json.loads((ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('coulomb-localization-'+run.name+'.json')).read_text())
    costs.append({'case':run.name,'command_exit':command['exit_code'],'command_wall_s':command['seconds'],
                  'accepted_intervals':len(ledger),'accepted_final_time_s':ledger[-1]['time'],
                  'assembly_s':sum(float(r['assembly']) for r in profile),'linear_s':sum(float(r['linear']) for r in profile),
                  'history_s':sum(float(r['history']) for r in profile),'output_s':sum(float(r['seconds']) for r in output),
                  'max_rank_peak_KiB':max(int(r['max_rank_KiB']) for r in resources)})
    for row in ledger:
        maximum_source=max(maximum_source,float(row['projection_error']));maximum_residual=max(maximum_residual,float(row['mechanical_residual']))
        maximum_ledger=max(maximum_ledger,abs(float(row['ledger_error'])));step=int(row['step']);dt=float(row['dt']);beta=1/(1+dt/100.)
        data=read(run/f'coulomb-{step:04}.csv');a=lambda k:np.array([float(r[k]) for r in data])
        tau_x=.5*(a('tauxx_Pa')-a('tauyy_Pa'));tau_y=a('tauxy_Pa');p=a('total_pressure_Pa')-.5*(a('tauxx_Pa')+a('tauyy_Pa'))
        minimum_return=min(minimum_return,float(min(p-np.hypot(tau_x,tau_y))))
        minimum_trial=min(minimum_trial,float(min(p-np.hypot(tau_x+2e6*beta*dt*a('Wxx_s-1'),tau_y+2e6*beta*dt*a('Wxy_s-1')))))
        assert float(row['Db'])>=0 and float(row['Dc'])>=0 and float(row['Df'])>=0
    assert not (run/'bulk-stability.csv').exists()
assert maximum_source<=1e-12 and maximum_residual<=1e-9 and maximum_ledger<=1e-10
assert minimum_trial>0 and minimum_return>0

for n in [4,8]:
    a=read(WORK/f'runs/p08-n{n}-full/coulomb-0010.csv');b=read(WORK/f'runs/p08-n{n}-half/coulomb-0020.csv')
    w=np.array([float(r['JxW_m2']) for r in b]);A=np.array([[float(r['Wxx_s-1']),float(r['Wxy_s-1'])] for r in a]);B=np.array([[float(r['Wxx_s-1']),float(r['Wxy_s-1'])] for r in b])
    relative=np.sqrt(np.sum(w[:,None]*(A-B)**2)/np.sum(w[:,None]*B**2))
    records.append({'mesh_n':n,'time_s':.125,'dt_full_s':.0125,'dt_half_s':.00625,'full_tensor_W_relative_L2_difference':float(relative)})
write('temporal.csv',records)
write('execution-summary.csv',costs)
bitwise=[]
for step in range(11):
    for pattern in ['record-{step:04}.bin','means-{step:04}.csv','phase-{step:04}.csv']:
        name=pattern.format(step=step);a=WORK/'runs/p02-n4-full'/name;b=ROOT/'build-particle-free/coulomb-native/runs/nonuniform-r4'/name
        assert a.read_bytes()==b.read_bytes(),name;bitwise.append(name)
for name in ['p08-n16-full','p02-n16-full']:
    log=(ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/('coulomb-localization-'+name+'.log')).read_text()
    assert 'fully compressive physical trial stress' in ' '.join(log.split())
    assert len(read(WORK/'runs'/name/'bulk-accepted.csv'))==8
save('native-gates.json',{'max_source_error':maximum_source,'max_mechanical_residual':maximum_residual,'max_abs_BE_ledger_error_J_m':maximum_ledger,
    'minimum_trial_normal_pressure_Pa':minimum_trial,'minimum_return_normal_pressure_Pa':minimum_return,
    'protected_hashes_unchanged':len(entry['protected']),'unchanged_physics_control_bitwise_files':bitwise,
    'phase_active_enabled':False,'full_step_fine_failure_classification':'Initial Newton lift leaves the closed elastic-trial admission domain; both low/high phi; half-step same-parent cutbacks pass.',
    'decision':'Stop phase-active/large-run progression; mechanical mixed-symbol prerequisite fails.'})
print('Native gates preserved, source/pressure/particle/bitwise controls pass')
