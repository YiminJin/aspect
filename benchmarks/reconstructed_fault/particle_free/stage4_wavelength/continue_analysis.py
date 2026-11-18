#!/usr/bin/env python3
"""Join complete frozen actions across interruptions; compare identical spaces."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
from analyze import spectrum,read,LAWS,ROOT,HERE
OUT=HERE.parent/'evidence/stage4_shortwave';OUT.mkdir(exist_ok=True)
parts=[('stage4-wavelength-512-mpi',list(range(5))),
       ('stage4-wavelength-512-resume',[0]),('stage4-wavelength-512-pairs',list(range(4)))]
target=OUT/'native-512';target.mkdir(exist_ok=True)
blocks=[];checks=[];sources=[]
sections=read(ROOT/'build-particle-free/bulk-tip/runs/stage4-wavelength-512-pairs/output/restoring-sections.csv')
for name,ids in parts:
    run=ROOT/'build-particle-free/bulk-tip/runs'/name
    # A shutdown may leave a partial CSV tail. Only complete selected actions
    # enter the joined data; preserve all original files, including that tail.
    lines=(run/'output/restoring-modes.csv').read_text().splitlines()
    rows=list(csv.DictReader(lines[:-1] if len(lines[-1].split(','))!=6 else lines))
    section_lines=(run/'output/restoring-sections.csv').read_text().splitlines()
    if len(section_lines[-1].split(','))!=12:section_lines=section_lines[:-1]
    s=np.atleast_1d(np.genfromtxt(section_lines,delimiter=',',names=True));ch=np.atleast_1d(read(run/'output/restoring-checks.csv'))
    for field in ('x','ds','angle','clamped_compliance_m_Pa','cohesive_compliance_bound_m_Pa'):
        assert np.allclose(sections[field][:len(s)],s[field],rtol=2e-13,atol=1e-18),field
    for mode in ids:
        selected=[r for r in rows if int(r['mode'])==mode]
        assert len(selected)==len(sections),(name,mode,len(selected))
        for i,r in enumerate(selected):
            assert int(r['section'])==i
            r['mode']=len(blocks)
        blocks.append(selected)
        check={k:float(ch[k][mode]) for k in ch.dtype.names};check['mode']=len(blocks)-1;checks.append(check)
    sources.append(dict(run=name,selected_modes=ids,command=json.loads((run/'command.json').read_text()),
      hashes={f:hashlib.sha256((run/'output'/f).read_bytes()).hexdigest() for f in ('restoring-modes.csv','restoring-sections.csv','restoring-checks.csv')}))
for name,rows in [('restoring-modes.csv',[r for b in blocks for r in b]),('restoring-checks.csv',checks)]:
    with (target/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savetxt(target/'restoring-sections.csv',np.array(sections.tolist()),delimiter=',',header=','.join(sections.dtype.names),comments='',fmt='%.17g')
# The interrupted sixth action has 3034 complete rows; compare those exactly.
old=list(csv.DictReader((OUT/'paused-512/restoring-modes.csv').read_text().splitlines()[:-1]))
old=[r for r in old if int(r['mode'])==5]
new=blocks[5][:len(old)]
repeat=max(abs(float(a['slip_response_per_force_coefficient'])-float(b['slip_response_per_force_coefficient'])) for a,b in zip(old,new))
provenance=dict(sources=sources,repeat_complete_rows=len(old),repeat_max_absolute_response_difference=repeat,
    note='Composite frozen actions, not a physical trajectory or invented successful interrupted run.')
(target/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
raw=read(target/'restoring-modes.csv');n=len(sections);m=len(blocks)
P=raw['force_basis_per_sqrt_m'].reshape(m,n).T;C=raw['slip_response_per_force_coefficient'].reshape(m,n).T
operating=read(ROOT/'build-particle-free/stage4-wavelength/input-512/operating.csv')
active=sections['cohesive_compliance_bound_m_Pa']>0
P=P[active];C=C[active];w=sections['ds'][active];x=sections['x'][active]
v=operating['V'][active];theta=operating['theta'][active]
groups=[('smooth6',list(range(6))),('single_96_0',[6]),('single_96_1',[7]),('pair96',[6,7]),
 ('smooth6_pair96',list(range(8))),('single_192_0',[8]),('single_192_1',[9]),('pair192',[8,9]),
 ('smooth6_pair192',list(range(6))+[8,9]),('coupled_through192',list(range(10)))]
rows=[];started=time.monotonic()
for name,ids in groups:
    for law in LAWS if name in ('pair96','pair192','coupled_through192') else ['weakening']:
        result,delta=spectrum(P[:,ids],C[:,ids],w,x,v,theta,law,profile=name=='coupled_through192' and law=='weakening',max_frequency=48.)
        wave=1. if name=='smooth6' else .0625 if '96' in name and '192' not in name else .03125
        row=dict(nx=512,space=name,law=law,patterns=len(ids),shortest_trial_wavelength=wave,
                 cells_per_trial_wavelength=wave/(8/512),**result);rows.append(row)
        if law=='weakening':
            np.savetxt(OUT/f'mode-512-{name}.csv',np.c_[x,delta.real,delta.imag],delimiter=',',header='x,rate_real,rate_imag',comments='')
        print(name,law,json.dumps(result),flush=True)
        (OUT/'spectrum-512-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
record=dict(rows=rows,analysis_seconds=time.monotonic()-started,provenance=provenance,
 checks={name:max(r[name] for r in checks) for name in ('relative_equilibrium_residual','relative_power_error','finite_difference_relative_error')},
 action_seconds=sum(r['seconds'] for r in checks))
(OUT/'spectrum-512.json').write_text(json.dumps(record,indent=2)+'\n')
prior=json.loads((HERE.parent/'evidence/stage4_wavelength/spectrum-256.json').read_text())['rows']
matched=[r for r in prior if r['space'] in ('single_96_0','single_96_1','pair96','smooth6_pair96','smooth6')]+rows
with (OUT/'matched-comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['nx','space','law','shortest_trial_wavelength','cells_per_trial_wavelength','growth','imag','minimum_stiffness','patterns'],extrasaction='ignore');writer.writeheader();writer.writerows(matched)
