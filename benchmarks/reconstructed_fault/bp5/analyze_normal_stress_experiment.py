"""Native stress-band plots and matched-time A/B comparison; no solves or smoothing."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_normal_stress_diagnostic import read


def samples(output,step,prefix='normal_qp'):
    arrays=[read(p) for p in sorted(output.glob(f'{prefix}_{step}_rank*.csv'))]
    arrays=[a for a in arrays if a.size]
    return np.concatenate(arrays) if arrays else None


def bands(output,step,dest):
    q=samples(output,step)
    if q is None:raise ValueError('No raw QPs')
    result={}
    for lo,hi in ((70,71),(79,80)):
        a=q[(q['down_dip_s_m']>=lo*1000)&(q['down_dip_s_m']<=hi*1000)&(abs(q['r_m'])<=1)]
        fields=['minus_n_tau_n']
        if 'd_history' in a.dtype.names:fields+=['d_history','d_update']
        fig,axes=plt.subplots(len(fields),3,figsize=(14,3.3*len(fields)),squeeze=False,layout='constrained')
        for row,field in enumerate(fields):
            for ax,xlabel,x in zip(axes[row],('Actual r (m)','Native QP index','Fault-element fraction ξ'),
                                   (a['r_m'],a['qp'],a['xi'])):
                sc=ax.scatter(x,a[field]/1e3,c=a['qp'],cmap='tab10',s=24,vmin=-.5,vmax=9.5)
                ax.set(xlabel=xlabel,ylabel=field+' (kPa)');ax.grid(alpha=.2)
            fig.colorbar(sc,ax=list(axes[row]),label='Native QP index',ticks=range(9))
        fig.suptitle(f'{lo}–{hi} km, step {step}, r=0±1 m; raw samples, no interpolation')
        fig.savefig(dest/f'bands_{step}_{lo}_{hi}.png',dpi=180);plt.close(fig)
        result[f'{lo}-{hi}']={}
        for index in np.unique(a['qp']):
            b=a[a['qp']==index]
            result[f'{lo}-{hi}'][int(index)]=dict(n=len(b),r_range_m=[float(min(b['r_m'])),float(max(b['r_m']))],
                xi_range=[float(min(b['xi'])),float(max(b['xi']))],
                terms_Pa={f:dict(mean=float(np.mean(b[f])),min=float(min(b[f])),max=float(max(b[f]))) for f in fields})
        if 'ref_x' in a.dtype.names:
            fig,axes=plt.subplots(1,len(fields),figsize=(5*len(fields),4),squeeze=False,layout='constrained')
            for ax,field in zip(axes[0],fields):
                s=ax.scatter(a['ref_x'],a['ref_y'],c=a[field]/1e3,s=55,cmap='coolwarm')
                ax.set(xlabel='Reference-cell x',ylabel='Reference-cell y',title=field)
                fig.colorbar(s,ax=ax,label='kPa')
            fig.savefig(dest/f'reference_cell_{step}_{lo}_{hi}.png',dpi=180);plt.close(fig)
            n=a['n_x'];m=a['n_y']
            interp=-(n*n*a['particle_interp_xx']+2*n*m*a['particle_interp_xy']+m*m*a['particle_interp_yy'])
            working=-(n*n*a['incoming_FE_xx']+2*n*m*a['incoming_FE_xy']+m*m*a['incoming_FE_yy'])
            order=np.argsort(a['down_dip_s_m']);x=a['down_dip_s_m'][order]/1000
            fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
            ax.scatter(x,interp[order]/1e3,label='Configured particle interpolation',s=15)
            ax.scatter(x,working[order]/1e3,label='Working FE history used by mechanics',s=15)
            ax.set(xlabel='Down-dip distance (km)',ylabel='Incoming normal component (kPa)',title='Unrelaxed incoming history, r=0±1 m')
            ax.legend();fig.savefig(dest/f'transfer_{step}_{lo}_{hi}.png',dpi=180);plt.close(fig)
            result[f'{lo}-{hi}']['transfer_pointwise_max_Pa']=float(max(abs(interp-working)))
    line=samples(output,step,'normal_line')
    if line is not None:
        for lo,hi in ((70,71),(79,80)):
            a=line[(line['down_dip_s_m']>=lo*1000-1e-7)&(line['down_dip_s_m']<=hi*1000+1e-7)]
            a=np.sort(a,order=['down_dip_s_m','cell'])
            fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
            for field in ('minus_n_tau_n','d_history','d_update'):
                axes[0].scatter(a['down_dip_s_m']/1000,a[field]/1e3,s=5,label=field)
            axes[1].scatter(a['down_dip_s_m']/1000,a['r_m'],s=4,label='actual r')
            axes[0].set_ylabel('kPa');axes[0].legend();axes[1].set(xlabel='Down-dip distance (km)',ylabel='Actual r (m)')
            fig.suptitle('Native r=0 evaluation; both cell traces retained, no CSV interpolation')
            fig.savefig(dest/f'native_line_{step}_{lo}_{hi}.png',dpi=180);plt.close(fig)
            result[f'{lo}-{hi}']['native_line_max_abs_r_m']=float(max(abs(a['r_m'])))
    return result


def compare(a_path,b_path,dest):
    sa=list(csv.DictReader((a_path/'normal_summary.csv').open()))
    sb=list(csv.DictReader((b_path/'normal_summary.csv').open()))
    if len(sa)!=4 or len(sb)!=8:raise ValueError('Require exactly four A and eight B accepted steps')
    result={'common_states':[],'warning':'Incoming histories and per-step updates span different intervals; compare total current stress at matched times.'}
    for i,ra in enumerate(sa):
        rb=sb[2*i+1]
        if float(ra['time_s'])!=float(rb['time_s']):raise ValueError('Common accepted times do not match at full precision')
        a=read(a_path/f'normal_profile_{ra["step"]}.csv');b=read(b_path/f'normal_profile_{rb["step"]}.csv')
        for key in ('fault_id','vertex_id','down_dip_s_m','I_h'):np.testing.assert_array_equal(a[key],b[key])
        w=a['normal_row_mass'];entry=dict(time_s=float(ra['time_s']),A_step=int(ra['step']),B_step=int(rb['step']),differences={})
        for key in ('V_m_per_s','committed_Theta_s','cumulative_slip_m','weak_pressure_Pa','weak_minus_n_tau_n_Pa','existing_production_weak_normal_Pa'):
            e=b[key]-a[key]
            entry['differences'][key]=dict(max_abs=float(max(abs(e))),work_RMS=float(np.sqrt(np.dot(w,e*e)/w.sum())))
        fields=('V_m_per_s','committed_Theta_s','cumulative_slip_m',
                'weak_pressure_Pa','weak_minus_n_tau_n_Pa','existing_production_weak_normal_Pa')
        for lo,hi in ((0,116),(70,71),(79,80)):
            selected=(a['down_dip_s_m']>=lo*1000)&(a['down_dip_s_m']<=hi*1000)
            x=a['down_dip_s_m'][selected]/1000
            fig,axes=plt.subplots(6,2,figsize=(14,16),sharex=True,layout='constrained')
            for row,key in enumerate(fields):
                scale=1e3 if key.endswith('_Pa') else 1.
                # Remove only the explicitly exported background from total
                # normal traction. Never smooth or subtract a fitted trend.
                av=a[key].copy();bv=b[key].copy()
                if key=='existing_production_weak_normal_Pa':
                    av-=a['weak_reference_normal_Pa'];bv-=b['weak_reference_normal_Pa']
                axes[row,0].plot(x,av[selected]/scale,'.-',label='A: normal steps')
                axes[row,0].plot(x,bv[selected]/scale,'.-',label='B: half steps')
                axes[row,1].plot(x,(bv-av)[selected]/scale,'.-')
                label=key.replace('_Pa',' (kPa)')
                axes[row,0].set_ylabel(label);axes[row,1].set_ylabel('B − A')
                for ax in axes[row]:ax.grid(alpha=.2)
            axes[0,0].legend();axes[-1,0].set_xlabel('Down-dip distance (km)');axes[-1,1].set_xlabel('Down-dip distance (km)')
            fig.suptitle(f'Matched time {entry["time_s"]:.17g} s; steps {ra["step"]}/{rb["step"]}\n'
                         'Normal traction shown relative to background; state is committed, not mechanics input')
            fig.savefig(dest/f'comparison_{ra["step"]}_{rb["step"]}_{lo}_{hi}.png',dpi=150);plt.close(fig)
        result['common_states'].append(entry)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--step',type=int);p.add_argument('--other',type=Path,help='B output, when output is A')
    p.add_argument('--destination',type=Path)
    args=p.parse_args();dest=args.destination or args.output/'stress_experiment_plots';dest.mkdir(exist_ok=True,parents=True)
    step=args.step or max(int(p.stem.split('_')[-1]) for p in args.output.glob('normal_profile_*.csv'))
    result={'bands':bands(args.output,step,dest)}
    if args.other:result['comparison']=compare(args.output,args.other,dest)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
