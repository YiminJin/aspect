#!/usr/bin/env python3
"""Exact-jump/native checks against retained independent sharp-layer mechanics."""
import json,csv
from pathlib import Path
import numpy as np
import analyze_cohesive_interface_native as reference
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_exact'
OLD=OUT.parent/'cohesive_interface_native'
read=reference.read

def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def traces(name):
    rows=read(OUT/name/'trace_operators.csv');n=1+max(int(r['i']) for r in rows)
    matrices=[np.zeros((n,n)) for _ in range(4)]
    for r in rows:
        for a,key in zip(matrices,['shear_lower_Pa_m','shear_upper_Pa_m','normal_lower_Pa_m','normal_upper_Pa_m']):a[int(r['i']),int(r['j'])]=r[key]
    return matrices

def main():
    reference.OUT=OUT
    rows=[]
    for name in ['operator-4','operator-8','operator-16','operator-32','width-half']:
        K,N,w=reference.operators(name);n=len(K);sharp=reference.sharp_q1(n);ks=reference.mode_symbol(K);normal=reference.mode_symbol(N);t=traces(name);trace=[reference.mode_symbol(a) for a in t]
        rows.append(dict(case=name,segments=n,width_m=read(OUT/name/'operator_checks.csv')[0]['width_m'],native_K_Pa_m=ks.real,sharp_K_Pa_m=sharp[0].real,native_N_imag_Pa_m=normal.imag,sharp_N_imag_Pa_m=sharp[1].imag,K_relative_error=abs(ks/sharp[0]-1),N_relative_error=abs(normal/sharp[1]-1),shear_trace_jump_over_sharp_K=abs(trace[1]-trace[0])/abs(sharp[0]),normal_trace_jump_over_sharp_K=abs(trace[3]-trace[2])/abs(sharp[0]),mean_shear_trace_error_over_K=abs(.5*(trace[0]+trace[1])+ks)/abs(ks)))
    write('operator_comparison.csv',rows)
    assert rows[-2]['K_relative_error']<.02 and rows[-2]['N_relative_error']<.02
    assert rows[2]['normal_trace_jump_over_sharp_K']<rows[0]['normal_trace_jump_over_sharp_K']
    K,N,_=reference.operators('operator-32');Kw,Nw,_=reference.operators('width-half')
    width=dict(K=float(np.linalg.norm(K-Kw)/np.linalg.norm(K)),N=float(np.linalg.norm(N-Nw)/np.linalg.norm(N)))
    assert max(width.values())<1e-9
    K0,N0,_=reference.operators('operator-4');K6,N6,_=reference.operators('quadrature6')
    quadrature=dict(K=float(np.linalg.norm(K0-K6)/np.linalg.norm(K0)),N=float(np.linalg.norm(N0-N6)/np.linalg.norm(N0)))
    assert max(quadrature.values())<1e-10
    extra=[]
    for m in [1,2,4,8]:
        sh=reference.sharp_q1(32,m);a=reference.mode_symbol(K,m);b=reference.mode_symbol(N,m)
        extra.append(dict(mode=m,wavelength_m=1/m,K_relative_error=abs(a/sh[0]-1),N_relative_error=abs(b/sh[1]-1)))
    write('shorter_modes_diagnostic.csv',extra)
    histories=[];summaries=[]
    for name,mode in [('formation','formation'),('formation-half','formation'),('mature','mature'),('nonuniform','nonuniform')]:
        h,s=reference.response(name,mode);histories+=h;summaries.append(s)
        assert min(r['min_side_contact_pressure_Pa'] for r in read(OUT/name/'history.csv'))>0
    write('native_reference_histories.csv',histories)
    loaded=read(OUT/'resume/restart_loaded.csv')[0]['accepted_time_s']
    restart={}
    for stem in ['states','nodal_history','history']:
        a=read(OUT/f'nonuniform/{stem}.csv');b=[r for r in read(OUT/f'split/{stem}.csv') if r['time_s']<=loaded]+read(OUT/f'resume/{stem}-resume.csv')
        assert len(a)==len(b)
        keys=[k for k in a[0] if k!='step_seconds']
        err={k:max(abs(x[k]-y[k]) for x,y in zip(a,b)) for k in keys}
        assert max(err.values())==0,err
        restart[stem]=err
    # Baseline arithmetic is compared within the pre-existing measured allowance.
    Kf,Nf,_=reference.operators('finite-width-control')
    reference.OUT=OLD;Kold,Nold,_=reference.operators('formation');reference.OUT=OUT
    baseline=dict(K_relative=float(np.linalg.norm(Kf-Kold)/np.linalg.norm(Kold)),N_relative=float(np.linalg.norm(Nf-Nold)/np.linalg.norm(Nold)))
    assert max(baseline.values())<1e-12
    result=dict(operator_modes=rows,shorter_modes=extra,width_invariance=width,bulk_quadrature=quadrature,finite_width_default=baseline,runs=summaries,restart_bitwise=True,restart_errors=restart)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='restart_errors'},indent=2))

if __name__=='__main__':main()
