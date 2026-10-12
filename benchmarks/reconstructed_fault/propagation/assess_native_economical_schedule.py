#!/usr/bin/env python3
"""Field gates and separate path residual-work gates at common physical times."""
from pathlib import Path
import argparse
import numpy as np
import native_endpoint as native
import asymmetric_branch as old
from assess_native_endpoint import STRESS,STORAGE
from assess_coupled_branch import totals

E=Path(__file__).resolve().parent/'native_tangent_evidence'

def assess(level):
    path=E/f'economical-{level}';reference=native.E/'dt-1-2048'
    a,b=[dict(np.load(p/'fields.npz')) for p in [path,reference]]
    ha,hb=[old.read(p/'history.json') for p in [path,reference]]
    assert old.read(path/'summary.json')['completed']
    r=native.SharedFE().r
    for key in ['phase','R','Z']:
        assert np.array_equal(a[key][0],b[key][0])
    targets=dict(phase=.001,q=1.,slip_fraction=.001,interval_rate_fraction=.005,
                 whole_stress_fraction=.01,localized_stress_fraction=.01,bulk_storage_fraction=.01)
    peak=max(x['V'] for x in hb);slip_scale=hb[-1]['slip']
    rows=[];previous_slip=0.;previous_time=0.
    z,w=np.polynomial.legendre.leggauss(24);s=(z+1)/2;weight=np.tile(w/(2*r.n),r.n)
    def stress(f,j,I):
        phi=((1-s)*f['phase'][j,:-1,None]+s*f['phase'][j,1:,None]).ravel()
        R=f['R'][j];D=r.n*((-3+4*s)*R[:-2:2,None]+(4-8*s)*R[1::2,None]+(-1+4*s)*R[2::2,None])
        chi=old.core.h(phi)/I
        return old.G*D.ravel()-f['Z'][j]*chi,chi
    for j,x in enumerate(ha,1):
        k=round(x['time']*2048);y=hb[k-1];assert x['time']==y['time']
        sa,_=stress(a,j,x['I']);sb,chi=stress(b,k,y['I']);delta=sa-sb
        rate=(y['slip']-previous_slip)/(x['time']-previous_time)
        rows.append(dict(time=x['time'],phase=float(max(abs(a['phase'][j]-b['phase'][k]))),q=abs(x['q']-y['q']),
          slip_fraction=abs(x['slip']-y['slip'])/slip_scale,interval_rate_fraction=abs(x['V']-rate)/peak,
          whole_stress_fraction=float(np.sqrt(weight@(delta*delta)))/STRESS,
          localized_stress_fraction=float(np.sqrt((weight*chi)@(delta*delta)/(weight@chi)))/STRESS,
          bulk_storage_fraction=abs(x['bulk']-y['bulk'])/STORAGE))
        previous_slip=y['slip'];previous_time=x['time']
    maximum={key:max(x[key] for x in rows) for key in targets}
    ledger=old.read(path/'energy.json');total=totals(ledger)
    coarse=totals(old.read(native.E/'dt-1-1024/energy.json'))
    # Explicit provisional schedule gate: no more residual work than the
    # verified uniform 1/1024 reference plus 5%, for each channel separately.
    channels=['mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute']
    energy_targets={key:1.05*coarse[key]['signed'] for key in channels}
    answer=dict(level=level,steps=len(ha),maximum=maximum,field_pass={key:maximum[key]<=targets[key] for key in targets},
                energy=total,energy_targets=energy_targets,
                energy_pass={key:total[key]['signed']<=energy_targets[key] for key in channels},
                maximum_KKT=max(x['KKT'] for x in ha),maximum_phase_increment=max(x['phase_increment'] for x in ha),
                maximum_mechanical_residual=max(x['force'] for x in ha),maximum_contact=max(x['contact'] for x in ha),
                by_time=rows,summary=old.read(path/'summary.json'))
    old.write(E/f'economical-{level}-comparison.json',answer)
    print(old.json.dumps({k:answer[k] for k in ['steps','maximum','field_pass','energy_targets','energy_pass']}))
    print({key:total[key]['signed'] for key in channels})
    return answer

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--level',type=int,choices=[1,2],default=1)
    assess(parser.parse_args().level)
