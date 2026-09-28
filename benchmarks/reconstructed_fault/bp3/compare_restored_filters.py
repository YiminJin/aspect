"""Read-only comparison of saved restored BP3 filter startups (no new solves).

Writes separate analysis products. Raw-mode Q1 stress is an observation only;
the actual raw friction input is sampled at production QPs, not nodalized.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
out=a.directory/'comparison';out.mkdir(exist_ok=True)
modes=('raw','filter20','filter40')
def read(path):
    return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
def rms(x,w):
    return float(np.sqrt(np.sum(w*x*x)/np.sum(w)))
def write(name,rows):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

data={m:[read(a.directory/f'output-150x50-{m}'/f'restored_fault_{k}.csv')
         for k in range(11)] for m in modes}
accepted={m:read(a.directory/f'output-150x50-{m}'/'accepted_steps.csv') for m in modes}
s=data['raw'][0]['s_m'];length=max(s)
regions={'whole':np.ones(len(s),bool),'top1km':s<=1000,
         'transition13to20km':(s>=13000)&(s<=20000),
         'bottom1km':s>=length-1000,'interior':(s>1000)&(s<length-1000)}
records=[];difference=[];summary={}
for m in modes:
    d=a.directory/f'output-150x50-{m}';ac=accepted[m]
    assert np.array_equal(ac['step'],np.arange(11))
    assert np.all(ac['fresh_linear_checks_passed']==1)
    assert np.all(ac['normalized_nonlinear_residual']<1e-8)
    summary[m]={'final_time':float(ac['time'][-1]),
        'max_time_difference_from_raw':float(np.max(abs(ac['time']-accepted['raw']['time']))),
        'min_alpha':float(min(ac['min_alpha'])),
        'max_active':int(max(ac['lower_active'])),
        'max_theta_error':float(max(ac['Theta_relative_error'])),
        'max_surface_rms_Pa':float(max(ac['surface_RMS_Pa'])),
        'krylov_total':int(sum(ac['krylov_iterations'])),
        'newton_total':int(sum(ac['newton_updates']))}
    assert summary[m]['max_time_difference_from_raw']<1e-7
    for k,x in enumerate(data[m]):
        assert np.array_equal(x['s_m'],s)
        for region,mask in regions.items():
            w=x['work_mass'][mask];rec={'mode':m,'step':k,'time_s':float(ac['time'][k]),'region':region}
            for field in ('V','V_chord','raw_sigma_Q1','raw_chord','friction_sigma_Q1','filtered_chord'):
                v=x[field][mask];mean=float(np.sum(w*v)/sum(w))
                rec[field+'_mean']=mean
                rec[field+'_rms_centered']=rms(v-mean,w)
                rec[field+'_max_abs']=float(max(abs(v)))
                rec[field+'_rms']=rms(v,w)
            records.append(rec)
            base=data['raw'][k];delta=x['V'][mask]-base['V'][mask]
            difference.append({'mode':m,'step':k,'region':region,
                'V_difference_max_over_Vp':float(max(abs(delta))/1e-9),
                'V_difference_rms_over_Vp':rms(delta,w)/1e-9,
                'Theta_relative_difference_max':float(max(abs(x['Theta_committed'][mask]/base['Theta_committed'][mask]-1)))})
    slips=read(d/'cumulative_slip.csv');base_slip=read(a.directory/'output-150x50-raw/cumulative_slip.csv')
    mask=slips['step']==10
    assert np.array_equal(slips['node'][mask],base_slip['node'][mask])
    delta=slips['slip_m'][mask]-base_slip['slip_m'][mask]
    summary[m]['final_slip_max_difference_m']=float(max(abs(delta)))
    summary[m]['final_slip_max_relative_difference']=float(max(abs(delta)/np.maximum(abs(base_slip['slip_m'][mask]),1e-300)))
    growth=read(d/'restored_growth.csv')
    summary[m]['max_filter_mean_error_Pa']=float(max(abs(growth['filter_weak_mean_error']))/sum(data[m][0]['work_mass']))
    summary[m]['max_net_boundary_flux']=float(max(abs(sum(growth[f'flux_{b}'] for b in ('left','right','bottom','top')))))
write('nodal_metrics.csv',records);write('trajectory_differences.csv',difference)

# Production quadrature: retain actual raw and filtered friction inputs and the
# work measure; no interpolation of a CSV is used as a substitute for mechanics.
qp_rows=[]
columns=['s','r','work_weight','sigma_raw','sigma_friction','mu','V']
for m in modes:
    d=a.directory/f'output-150x50-{m}'
    for k in (0,1,10):
        parts=[]
        for path in sorted(d.glob(f'restored_raw_{k}_rank*.csv')):
            with path.open() as f:
                header=f.readline().strip().split(',')
                if not f.readline().strip():continue
            indices=[header.index(c) for c in columns]
            parts.append(np.loadtxt(path,delimiter=',',skiprows=1,usecols=indices,ndmin=2))
        q=np.concatenate(parts);pos=q[:,0];w=q[:,2]
        masks={'top200m':pos<=200,'bottom200m':pos>=length-200,
               '15km':abs(pos-15000)<=100,'18km':abs(pos-18000)<=100,'40km':abs(pos-40000)<=100}
        for region,mask in masks.items():
            assert np.any(mask)
            row={'mode':m,'step':k,'region':region,'sample_count':int(sum(mask)),'weight':float(sum(w[mask]))}
            for field,col in [('sigma_raw',3),('sigma_friction',4)]:
                v=q[mask,col]-50e6;mean=float(np.sum(w[mask]*v)/sum(w[mask]))
                row[field+'_mean_minus_50MPa_Pa']=mean
                row[field+'_rms_centered_Pa']=rms(v-mean,w[mask])
                row[field+'_min_minus_50MPa_Pa']=float(min(v))
                row[field+'_max_minus_50MPa_Pa']=float(max(v))
            row['friction_load_change_due_to_filter_Pa']=float(np.sum(w[mask]*q[mask,5]*(q[mask,4]-q[mask,3]))/sum(w[mask]))
            qp_rows.append(row)
write('quadrature_metrics.csv',qp_rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig,axes=plt.subplots(3,3,figsize=(15,10))
windows=[('Top',0,200),('Transition',14800,18200),('Bottom',length-200,length)]
for col,(title,lo,hi) in enumerate(windows):
    ids=np.where((s>=lo)&(s<=hi))[0];ids=ids[np.argsort(s[ids])]
    for m in modes:
        x=data[m][10];color={'raw':'black','filter20':'tab:blue','filter40':'tab:orange'}[m]
        axes[0,col].plot(s[ids]/1000,x['V'][ids]/1e-9,'.-',ms=2,label=m,color=color)
        label=m+(' (Q1 observation only)' if m=='raw' else '')
        axes[1,col].plot(s[ids]/1000,x['friction_sigma_Q1'][ids]-50e6,'.-',ms=2,label=label,color=color)
        axes[2,col].plot(s[ids]/1000,(x['V'][ids]-data['raw'][10]['V'][ids])/1e-9,'.-',ms=2,label=m,color=color)
    axes[0,col].set_title(title)
    for row in range(3):axes[row,col].set_xlabel('Down-dip distance (km)');axes[row,col].grid(alpha=.2)
    axes[0,col].set_ylabel('V / Vp');axes[1,col].set_ylabel('Q1 normal traction − 50 MPa (Pa)')
    axes[2,col].set_ylabel('(V − V_raw) / Vp')
axes[0,0].legend();axes[1,0].legend(fontsize=7)
fig.suptitle('Matched final state: 1133.739053 s; no spatial smoothing of plots')
fig.tight_layout();fig.savefig(out/'final_profiles.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(14,7))
for col,region in enumerate(('top1km','transition13to20km','bottom1km')):
    for m in modes:
        rows=[r for r in records if r['mode']==m and r['region']==region and r['step']>0]
        color={'raw':'black','filter20':'tab:blue','filter40':'tab:orange'}[m]
        axes[0,col].plot([r['time_s'] for r in rows],[r['V_chord_rms']/1e-9 for r in rows],'.-',label=m,color=color)
        axes[1,col].plot([r['time_s'] for r in rows],[r['filtered_chord_rms'] for r in rows],'.-',label=m,color=color)
    axes[0,col].set_title(region)
    axes[0,col].set_ylabel('Work-weighted V chord RMS / Vp')
    axes[1,col].set_ylabel('Q1 normal chord RMS (Pa)')
    for row in range(2):axes[row,col].set_xlabel('Physical time (s)');axes[row,col].grid(alpha=.2)
axes[0,0].legend();fig.tight_layout();fig.savefig(out/'growth.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
