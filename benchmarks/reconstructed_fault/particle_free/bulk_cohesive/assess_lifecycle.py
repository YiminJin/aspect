#!/usr/bin/env python3
"""Actual accepted-history/restart exports and the loaded zero-motion hold."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from assess_native import csv,G,ETA


def digest(file):return hashlib.sha256(file.read_bytes()).hexdigest()


def assess(primary,before,after):
    ledger=csv(primary/'bulk-accepted.csv');gates=csv(primary/'gates.csv')
    assert np.all(gates['passed']==1)
    recovery=['converged trial does not publish','outer rejection restores native solution and all bulk histories',
              'identical retry native solution','repeated rejection leaves accepted histories unchanged']
    for gate in recovery:assert np.sum(gates['gate']==gate)==1
    hold=ledger[ledger['time']>.15000000001]
    errors={key:0. for key in ['T','a','q_on_positive_h','cohesive_energy_decay','bulk_energy_decay']}
    for row in hold:
        step=int(row['step']);beta=1/(1+row['dt']*G/ETA)
        a=csv(primary/f'bulk-{step-1:04d}.csv');b=csv(primary/f'bulk-{step:04d}.csv')
        a.sort(order=['cell','x','y']);b.sort(order=['cell','x','y'])
        for key in ['k','L','phi']:assert np.array_equal(a[key],b[key])
        for key in ['Txx','Tyy','Txy']:errors['T']=max(errors['T'],float(np.max(abs(b[key]-beta*a[key])))/1000)
        errors['a']=max(errors['a'],float(np.max(abs(b['a']-beta*a['a']))))
        positive=a['phi']>0
        errors['q_on_positive_h']=max(errors['q_on_positive_h'],float(np.max(abs(b['q'][positive]-beta*a['q'][positive])))/1000)
        for key,label in [('bulk','bulk_energy_decay'),('cohesive','cohesive_energy_decay')]:errors[label]=max(errors[label],float(np.max(abs(b[key]-beta*beta*a[key]))))
        assert row['Wext']==0. and row['Df']==0.
    assert max(errors.values())<1e-10
    # Prefix is stopped at an exactly matching binary accepted clock, never
    # rounded into a different last dt. Resume contains only continued rows.
    prefix=csv(before/'bulk-accepted.csv');continued=csv(after/'bulk-accepted.csv')
    rows=np.r_[prefix,continued];assert np.array_equal(rows,ledger)
    files=[]
    for path in [before,after]:
        for file in sorted(path.iterdir()):
            if file.suffix not in ['.bin','.csv'] or '-' not in file.stem:continue
            label=file.stem.rsplit('-',1)[-1]
            if not label.isdigit():continue
            ref=primary/file.name
            assert ref.exists() and file.read_bytes()==ref.read_bytes(),file
            files.append(str(file.relative_to(path)))
    assert files
    return dict(recovery_gates=recovery,rejection_retry_passed=True,single_publication_passed=True,
                restart_accepted_ledgers_bitwise_equal=True,restart_binary_and_field_files_bitwise_equal=len(files),
                prefix_accepted_steps=len(prefix),resumed_accepted_steps=len(continued),hold_steps=len(hold),
                hold_phase_signed_absolute_shear_bitwise_unchanged=True,hold_beta_decay_errors=errors,
                automatic_cutback_qualified=False,
                observer_scope='pure straight-column raw-weight moments; no native M3 ridge evolution',
                strict_reference_match_passed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','before','after']:p.add_argument(name,type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=assess(a.primary,a.before,a.after);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
