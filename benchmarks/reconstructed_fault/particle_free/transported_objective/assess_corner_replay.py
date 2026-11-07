#!/usr/bin/env python3
"""Reduce saved replay histories; plot errors, feedback and energy accounting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-corner-replay-mpl')
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay'
RUN=ROOT/'build-particle-free/transported-objective/stage-c/material-runs/corner-replay32'
LABELS=['Exact stage exterior','Exact parent exterior','Numerical parent trace']
def read(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def main():
    command=json.loads((RUN/'command.json').read_text());assert command['exit_code']==0
    rows=read(RUN/'output/corner-replay.csv');assert len(rows)==84
    cases=[]
    for mode in range(3):
        for dt in [.1,.05,.025]:
            data=[r for r in rows if r['mode']==mode and r['dt']==dt]
            assert len(data)==round(.4/dt) and abs(data[-1]['time']-.4)<1e-14
            assert all(np.isfinite(list(r.values())).all() for r in data)
            last=data[-1].copy()
            for key in ['energy_change','flux_supplied','flux_interior','upwind','divergence','source_work','balance']:
                last[key+'_sum']=sum(r[key] for r in data)
                last[key+'_absolute_sum']=sum(abs(r[key]) for r in data)
            last['max_identity']=max(r['max_identity'] for r in data)
            last['peak_over_time']=max(r['max_stress'] for r in data)
            last['subdt']=dt/last['substeps']
            assert last['max_identity']<1e-10
            if mode==0:assert max(r['relative'] for r in data)<1e-9
            cases.append(last)
    assert max(c['subdt'] for c in cases)-min(c['subdt'] for c in cases)<1e-15
    affine=read(RUN.parent/'corner-prescribed32-r1/output/corner-isolation.csv')
    exact_affine={(r['dt'],int(r['step'])):r['energy'] for r in affine if r['reaction']==0 and r['held_trace']==0}
    # Independent earlier reference has different velocity/source, but the same H*.
    energy_reference_error=max(abs(r['energy']-exact_affine[(r['dt'],int(r['step']))]) for r in rows if r['mode']==0)
    assert energy_reference_error<1e-11,energy_reference_error
    affine=[r for r in affine if r['reaction']==0 and r['held_trace']==1 and abs(r['time']-.4)<1e-14]
    result=dict(cases=cases,affine_control_final=affine,cost_s=command['wall_s'],
                exact_energy_error_against_retained_affine=energy_reference_error,
                velocity_audit=json.loads((OUT/'velocity-audit.json').read_text()),
                native_audit=(RUN/'output/replay-native-audit.txt').read_text())
    result['lag_orders']=[float(np.log2(cases[3+i]['relative']/cases[4+i]['relative'])) for i in range(2)]
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    colors=['#2369bd','#df8b20','#b63a4a']
    ax=axes[0,0]
    for mode in (1,2):
        group=[c for c in cases if c['mode']==mode]
        ax.plot([c['dt'] for c in group],[100*c['relative'] for c in group],'o-',color=colors[mode],label=LABELS[mode]+' — box')
        ax.plot([c['dt'] for c in group],[100*c['corner_relative'] for c in group],'s--',color=colors[mode],label=LABELS[mode]+' — corner')
    ax.plot([r['dt'] for r in affine],[100*r['corner_relative'] for r in affine],':',color='.4',label='Retained affine held — corner')
    ax.set(xscale='log',yscale='log',xlabel='Macro timestep (s), fixed RK substep',ylabel='Final relative L2 error (%)',title='Lag decreases; numerical-trace bias persists')
    ax.legend(fontsize=7)
    ax=axes[0,1]
    for mode in (1,2):
        data=[r for r in rows if r['mode']==mode and r['dt']==.025]
        ax.plot([r['time'] for r in data],[100*r['corner_relative'] for r in data],label=LABELS[mode],color=colors[mode])
    ax.set(xlabel='Replay time (s)',ylabel='Corner relative L2 error (%)',title='Non-affine velocity held fixed, dt=.025 s');ax.legend(fontsize=8)
    ax=axes[1,0]
    for mode in range(3):
        data=[r for r in rows if r['mode']==mode and r['dt']==.025]
        ax.plot([r['time'] for r in data],[r['max_stress'] for r in data],label=LABELS[mode],color=colors[mode])
    ax.set(xlabel='Replay time (s)',ylabel='Maximum full tensor norm (Pa)',title='Manufactured stress amplitude');ax.legend(fontsize=8)
    ax=axes[1,1]
    for mode in (1,2):
        data=[r for r in rows if r['mode']==mode and r['dt']==.025]
        t=[r['time'] for r in data];e=np.array([r['balance'] for r in data])
        ax.plot(t,np.cumsum(e),color=colors[mode],label=LABELS[mode]+' signed')
        ax.plot(t,np.cumsum(abs(e)),'--',color=colors[mode],label=LABELS[mode]+' absolute')
    ax.set(xlabel='Replay time (s)',ylabel='Integrated energy defect (J/m)',title='Includes source and divergence work');ax.legend(fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Frozen original corner velocity — native DGQ2 / SSPRK3, 32² mesh')
    for ext in ['png','pdf']:fig.savefig(OUT/f'corner-replay.{ext}',dpi=160)
    print('mode dt box_error corner_error interior_error peak balance_signed balance_abs')
    for c in cases:print(*(f'{c[k]:.8g}' for k in ['mode','dt','relative','corner_relative','interior_relative','max_stress','balance_sum','balance_absolute_sum']))
    print('Lag-only orders:',result['lag_orders'],'; run seconds:',result['cost_s'])
if __name__=='__main__':main()
