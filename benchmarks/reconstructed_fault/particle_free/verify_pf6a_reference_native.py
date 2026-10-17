#!/usr/bin/env python3
"""Read-only startup verification and exact failure classification of one native run."""
import argparse,json
from pathlib import Path
import numpy as np
from pf6a_spatial_reference import Endpoint,SEED,A,write

p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=json.loads((a.run/'record.json').read_text());log=(a.run/'run.log').read_text();state=a.run/'run/state-0'
def read(name):return np.genfromtxt(state/name,delimiter=',',names=True)
phase=read('nodes.csv');surface=read('accepted-surface.csv');tensor=read('tensor.csv');grad=read('gradient.csv')
ref=Endpoint(512,512);I=ref.direct(ref.initial)[0]
expected=np.interp(phase['y'],np.loadtxt(SEED)[:,0],np.loadtxt(SEED)[:,1]);unloaded=all(np.all(tensor[k]==0)for k in ['xx','yy','xy']) and np.all(surface['Z']==0) and np.all(surface['slip']==0)
g=np.zeros(513);np.add.at(g,np.rint(phase['y']*512).astype(int),grad['gradient'])
energy=np.genfromtxt(a.run/'run/metrics.csv',delimiter=',',names=True)
checks=dict(phase_max_error=float(max(abs(phase['phase']-expected))),unloaded_histories=bool(unloaded),
            normalization_max_error=float(max(abs(surface['I']-I))),
            fracture_energy_error=float(abs(energy['fracture']-(A*ref.mass@ref.initial+.5*ref.initial@(ref.F@ref.initial)))),
            initial_gradient_Linf=float(max(abs(g-A*ref.mass-ref.F@ref.initial))),
            initial_publications=int(energy['publications']))
assert checks['phase_max_error']==0 and unloaded and checks['normalization_max_error']<1e-9 and checks['fracture_energy_error']<1e-9 and checks['initial_publications']==0
failure='Accepted FE strain does not reproduce the native Maxwell publication.'
assert r['exit_code']==1 and failure in log and not (a.run/'run/state-1').exists()
write(a.output,dict(startup_checks_passed=True,checks=checks,native_confirmation_passed=False,
                    native_exit_code=r['exit_code'],seconds=r['seconds'],max_rss_KiB=r['max_rss_KiB'],
                    native_positive_accepted_steps=0,failure_time=.0078125,failure_marker=failure,
                    failure_owner='history.cc::commit_reconstructed_fault_mechanical_history, error <= 128 epsilon max(1,evaluation_scale)',
                    numerical_error_and_bound_emitted=False,actual_mismatch_cause='unresolved; do not infer from exception text',
                    failure_precedes_publication_by_source_audit=True,positive_step_rollback_bytes_verified=False,
                    native_onset_growth_accuracy_verified=False,acceptance_rules_unchanged=True))
