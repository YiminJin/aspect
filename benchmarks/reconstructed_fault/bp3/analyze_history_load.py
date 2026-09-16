"""Frozen tau11 transfer/load analysis. No constitutive evolution or new solve."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
out=here/'normal-stress-history-load-local4'

def read(paths):
    columns={}
    for path in sorted(paths):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                for k,v in row.items(): columns.setdefault(k,[]).append(v)
    result={}
    for k,v in columns.items():
        try: result[k]=np.array(v,dtype=float)
        except ValueError: result[k]=np.array(v)
    return result

def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

parents,qp,cells,loads=(read(out.glob(f'history_load_{s}_rank*.csv')) for s in ('parents','qp','cells','loads'))
assert len(np.unique(parents['id']))==len(parents['id'])
assert len(np.unique(cells['cell']))==len(cells['cell'])
assert len(np.unique(loads['dof']))==len(loads['dof'])
beta=np.exp(-376359254.49064183*32038120320./1e26)
for state in ('particle','published','working'):
    parents['N_'+state]=.75*parents['xx_'+state]+.25*parents['yy_'+state]-np.sqrt(3)/2*parents['xy_'+state]
    parents['S_'+state]=np.sqrt(3)/4*(parents['yy_'+state]-parents['xx_'+state])-.5*parents['xy_'+state]
parents['transfer_N']=parents['N_working']-parents['N_particle']
parents['constraint_N']=parents['N_working']-parents['N_published']
norm=lambda a: float(np.linalg.norm(a))
rms=lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
summary={'beta':float(beta),'counts':{'parents':len(parents['id']),'cells':len(cells['cell']),'qps_in_export_strip':len(qp['q'])},
         'weak_load':{'all_norm':norm(loads['all']),'normal_tensor_only_norm':norm(loads['normal']),
                      'q4_difference':norm(loads['reference_q4']-loads['all']),
                      'q4_relative_difference':norm(loads['reference_q4']-loads['all'])/norm(loads['all']),
                      'region_sum_relative_error':norm(sum(loads[k] for k in ['near35','near40','transition','bottom','other'])-loads['all'])/norm(loads['all'])},
         'parent_metrics_weighting':'unweighted at identical actual parent positions; not a surface quadrature norm',
         'windows':{}}
for region in ['near35','near40','transition','bottom','other']:
    summary['weak_load'][region+'_norm']=norm(loads[region])
summary['constraints']={'max_component_change_Pa':float(max(np.max(np.abs(parents[f'{c}_working']-parents[f'{c}_published'])) for c in ('xx','yy','xy'))),
                        'max_N_change_Pa':float(np.max(np.abs(parents['constraint_N'])))}
regions={'near35':(34000,36000),'near40':(39000,41000),'transition':(13000,20000),'interior25':(24000,26000),'bottom':(114000,116000)}
for name,(lo,hi) in regions.items():
    mask=(parents['xd']>=lo)&(parents['xd']<=hi)&(np.abs(parents['normal'])<400)
    cmask=(cells['xd']>=lo)&(cells['xd']<=hi)
    result={'parents':int(sum(mask)),'cells_all_normal_distances':int(sum(cmask)),
            'cell_local_load_norm_RMS':rms(cells['local_load_norm'][cmask])}
    for key in ['N_particle','N_published','N_working','transfer_N','constraint_N','S_particle','S_working']:
        a=parents[key][mask]
        result[key]={'min':float(np.min(a)),'max':float(np.max(a)),'rms':rms(a)}
    result['N_transfer_rms_relative']=rms(parents['transfer_N'][mask])/rms(parents['N_particle'][mask])
    cell_normal=(.5e5*(1+1/np.sqrt(3))-cells['x'])*np.sqrt(3)/2-(1e5-cells['y'])*.5
    strip=cmask&(np.abs(cell_normal)<400)
    result['near_fault_cells']=int(sum(strip))
    result['near_fault_local_load_RMS']=rms(cells['local_load_norm'][strip])
    result['near_fault_local_normal_load_RMS']=rms(cells['local_normal_load_norm'][strip])
    summary['windows'][name]=result

# Join saved samples to their exact parent input. Substituting FE history at
# fixed converged u,p below is diagnostic accounting, not a different solution.
index={int(p):i for i,p in enumerate(parents['id'])}
details=[]
for label,directory in [('baseline',here/'normal-stress-complete-local4'),('junction35',here/'normal-stress-junction35-local4')]:
    raw=read(directory.glob('stress_samples_12_rank*.csv'))
    raw['xd']=(.5e5*(1+1/np.sqrt(3))-raw['surface_x'])*.5+(1e5-raw['surface_y'])*np.sqrt(3)/2
    for name,(lo,hi) in regions.items():
        rows=np.where((raw['xd']>=lo)&(raw['xd']<=hi))[0]
        if not len(rows): continue
        for kind,j in [('min',rows[np.argmin(raw['sigma_n'][rows])]),('max',rows[np.argmax(raw['sigma_n'][rows])])]:
            i=index[int(raw['particle'][j])]
            assert max(abs(raw['parent_x'][j]-parents['x'][i]),abs(raw['parent_y'][j]-parents['y'][i]))<1e-8
            keys=['particle','xd','parent_x','parent_y','signed_normal_distance','delta_p','delta_tau_N','sigma_n']
            row=dict(case=label,window=name,extreme=kind,**{k:float(raw[k][j]) for k in keys})
            row.update({k:float(parents[k][i]) for k in ['N_particle','N_published','N_working','transfer_N']})
            row.update(retained_tau_N=float(beta*parents['N_particle'][i]),working_tau_N=float(beta*parents['N_working'][i]),
                       current_strain_tau_N=float(raw['delta_tau_N'][j]-beta*parents['N_particle'][i]),
                       fixed_u_p_sigma_if_FE_history=float(raw['sigma_n'][j]-beta*parents['transfer_N'][i]))
            details.append(row)
write(out/'hotspot_decomposition.csv',details)
summary['hotspots']=details

# Separate normal-side strips: a whole-column average would cancel the dipole.
records=[]
for side in (-1,1):
    mask=(parents['normal']*side>100)&(parents['normal']*side<300)&(parents['xd']>=10000)&(parents['xd']<=45000)
    bins=(parents['xd']/100).astype(int)
    for b in np.unique(bins[mask]):
        select=mask&(bins==b)
        records.append(dict(side=side,xd=float((b+.5)*100),count=int(sum(select)),
                            **{k:float(np.mean(parents[k][select])) for k in ['N_particle','N_working','N_published','transfer_N','S_particle','S_working']}))
write(out/'history_profiles.csv',records)
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex='col')
for col,window in enumerate([(33,42),(13,20)]):
    for side,color in [(-1,'tab:blue'),(1,'tab:orange')]:
        p=[r for r in records if r['side']==side and window[0]*1000<=r['xd']<=window[1]*1000]
        x=[r['xd']/1000 for r in p]
        for state,style in [('particle','-'),('working','--')]:
            axes[0,col].plot(x,[r['N_'+state]/1e6 for r in p],style,color=color,label=f'{state} side {side}')
            axes[2,col].plot(x,[r['S_'+state]/1e6 for r in p],style,color=color)
        axes[1,col].plot(x,[r['transfer_N']/1e6 for r in p],color=color)
    for ax in axes[:,col]:
        for x in (15,18,35,40):
            if window[0]<=x<=window[1]: ax.axvline(x,color='gray',lw=.5)
        ax.grid(alpha=.3)
    axes[2,col].set_xlabel('down-dip distance (km)')
axes[0,0].set_ylabel('old stress : N (MPa)')
axes[1,0].set_ylabel('FE minus particle : N (MPa)')
axes[2,0].set_ylabel('old shear stress (MPa)')
axes[0,0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out/'history_profiles.png',dpi=150); plt.close(fig)
(out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
assert summary['weak_load']['q4_relative_difference']<1e-11
assert summary['weak_load']['region_sum_relative_error']<1e-12
print(json.dumps({k:v for k,v in summary.items() if k not in ['windows','hotspots']},indent=2))
