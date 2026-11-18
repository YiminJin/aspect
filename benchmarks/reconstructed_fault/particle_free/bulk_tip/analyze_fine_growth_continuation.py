#!/usr/bin/env python3
"""Join qualified fine prefixes/restarts; reuse original reductions and spatial maps."""
import argparse, json, os, hashlib, csv
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from analyze_material_overnight import fast_frame,fast_balance
import analyze_synthetic_growth as growth
from analyze_time_update import read
from growth_spatial_analysis import spatial_comparison,weighted_error,trace_plot
import numpy as np
ROOT=growth.ROOT;RUNS=growth.RUNS
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation'
OLD=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/material_overnight'
growth.OUT=OUT;growth.frame=fast_frame;growth.balance=fast_balance

def join(label,segments):
    """Analysis-only view; never a new simulation or a checkpoint conversion."""
    directory=RUNS/label/'output';directory.mkdir(parents=True,exist_ok=True)
    commands=[json.loads((RUNS/s/'command.json').read_text()) for s in segments]
    assert all('exit_code' in c for c in commands), 'Only join finalized run records'
    cached=json.loads((OLD/(segments[0]+'-analysis.json')).read_text())
    assert cached['command']==commands[0], 'Prefix provenance changed'
    source=[];last=0
    for segment,command in zip(segments,commands):
        rows=read(RUNS/segment/'output/bulk-accepted.csv')
        assert np.array_equal(rows.step,np.arange(last+1,int(rows.step[-1])+1)),'Missing or duplicate accepted interval'
        last=int(rows.step[-1]);source.append({'label':segment,'first_step':int(rows.step[0]),'last_step':last,'command':command})
        assert command['arguments']['time_update']=='midpoint' and command['arguments']['refinement']==1
        assert np.max(abs(rows.dt-.000625))<1e-14
        for p in (RUNS/segment/'output').glob('tip-*.csv'):
            if p.name in ('tip-profile.csv','tip-solver.csv','tip-power.csv','tip-time-location.csv','tip-remote.csv','tip-checks.csv','tip-step-timers.csv','tip-timers.csv'):continue
            dest=directory/p.name
            if dest.exists():assert dest.resolve()==p.resolve(),dest
            else:dest.symlink_to(p.resolve())
    for filename in ('bulk-accepted.csv','tip-profile.csv','tip-solver.csv','tip-power.csv','tip-time-location.csv','tip-remote.csv','tip-step-timers.csv'):
        rows=[];header=None;seen=set()
        for segment in segments:
            p=RUNS/segment/'output'/filename
            with p.open() as f:
                reader=csv.reader(f);head=next(reader)
                if header is None:header=head
                assert head==header,(filename,segment)
                for row in reader:
                    key=tuple(row[:2]) if filename=='tip-step-timers.csv' else row[0]
                    assert key not in seen,(filename,key)
                    seen.add(key);rows.append(row)
        with (directory/filename).open('w') as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    command=dict(kind='Analysis-only joined trajectory; no simulation was run in this directory',segments=source,
                 arguments=commands[0]['arguments'],exit_code=commands[-1].get('exit_code',124),seconds=sum(c.get('seconds',0) for c in commands))
    (directory.parent/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (OUT/'joined-provenance.json').write_text(json.dumps(command,indent=2)+'\n')
    return directory,{int(s['step']):s for s in cached['steps']}

def section_comparison(coarse,fine,hc):
    fields={};index=np.floor(coarse.s/hc).astype(int);fine_index=np.floor(fine.s/hc).astype(int)
    for name in ('age','slip','V','force'):
        mapped=np.empty_like(fine.s)
        for cell in np.unique(index):
            a=index==cell;b=fine_index==cell
            coefficients=np.polynomial.legendre.legfit(2*(coarse.s[a]-(cell+.5)*hc)/hc,coarse[name][a],5)
            mapped[b]=np.polynomial.legendre.legval(2*(fine.s[b]-(cell+.5)*hc)/hc,coefficients)
        fields[name]=weighted_error(mapped,fine[name],fine.ds)
    return fields

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('continuations',nargs='*');parser.add_argument('--label',default='fine-growth-joined')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--times',nargs='*',type=float,default=[.04,.045625,.04625,.05,.055,.06]);args=parser.parse_args()
    OUT=args.output;OUT.mkdir(parents=True,exist_ok=True);growth.OUT=OUT
    segments=['material-midpoint-S1','material-obstacle-window02']+args.continuations
    directory,cached=join(args.label,segments)
    rows=read(directory/'bulk-accepted.csv');end=float(rows.time[-1])
    result,rows,saved=growth.analyze(args.label,end,True,cached_steps=cached);result['method']='midpoint';result['trajectory_complete']=bool(result['trajectory_complete'])
    power=read(directory/'tip-power.csv');assert np.array_equal(rows.step,power.step)
    for name,start in [('energy_full',0),('energy_window',.04)]:
        mask=rows.time>start+1e-10
        for field in ('friction','radiation'):
            values=power[field][mask];result[name][field]=dict(signed=float(sum(values)),absolute=float(sum(abs(values))))
        result[name]['dissipation_channel_defect']=float(max(abs(rows.Df[mask]-power.friction[mask]-power.radiation[mask])))
    (OUT/'fine-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    coarse_label='material-midpoint-M3';cd=RUNS/coarse_label/'output';coarse=json.loads((OLD/(coarse_label+'-analysis.json')).read_text());cr=read(cd/'bulk-accepted.csv');cr=cr[cr.time<=end+1e-10]
    ca0=fast_frame(cd,0);fi0=saved[0];comparisons={};last_diff=None
    for step in sorted(set(round(t/.000625) for t in args.times+[end] if t<=end+1e-10)):
        time=float(rows.time[rows.step==step][0]);c=fast_frame(cd,step);f=fast_frame(directory,step)
        values,diff=spatial_comparison(c,ca0,f,fi0)
        values['sections']=section_comparison(read(cd/f'tip-sections-{step:04d}.csv'),read(directory/f'tip-sections-{step:04d}.csv'),.125)
        clocks=[read(d/'tip-time-location.csv') for d in (cd,directory)]
        stage=[float(a.stage_time[a.step==step][0]) for a in clocks];assert abs(stage[0]-stage[1])<1e-14
        values['time_locations']={'endpoint_age_slip':time,'midpoint_rate_resistance':stage[0]}
        comparisons[str(time)]=values
        if step==int(rows.step[-1]):last_diff=diff;last_coarse=c;last_fine=f
    (OUT/'spatial-comparison.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False)+'\n')
    coarse['steps']=[s for s in coarse['steps'] if s['time']<=end+1e-10]
    coarse['comparison_prefix_end']=end
    coarse['original_completed_onset']=coarse['onset'].copy()
    coarse['onset']={k:t for k,t in coarse['onset'].items() if t<=end+1e-10}
    # Original coarse reductions may not contain contours at this interrupted end.
    growth.H=.125
    coarse['steps'][-1]['polynomial_sections']=growth.polynomial_sections(last_coarse)
    cp=read(cd/'tip-power.csv');cp=cp[cp.step<=cr.step[-1]]
    assert np.array_equal(cr.step,cp.step)
    for name,start in [('energy_full',0),('energy_window',.04)]:
        coarse[name]=growth.energy(cr,coarse['initial_fracture'],start)
        mask=cr.time>start+1e-10
        for field in ('friction','radiation'):
            values=cp[field][mask];coarse[name][field]=dict(signed=float(sum(values)),absolute=float(sum(abs(values))))
        coarse[name]['dissipation_channel_defect']=float(max(abs(cr.Df[mask]-cp.friction[mask]-cp.radiation[mask])))
    (OUT/'coarse-at-common-end.json').write_text(json.dumps(coarse,indent=2,allow_nan=False)+'\n')
    figures(coarse,cr,result,rows,ca0,last_coarse,fi0,last_fine,last_diff,end)
    trace_plot([(coarse,cr,{0:ca0,end:last_coarse}),(result,rows,{0:fi0,end:last_fine})],OUT,'native-traces')
    print(json.dumps({'end':end,'completed':result['trajectory_complete'],'last_comparison':comparisons[str(end)],'fine_energy':result['energy_full']},indent=2))

def figures(coarse,cr,fine,fr,c0,c,f0,f,diff,end):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from growth_spatial_analysis import CellPolynomials
    mapped=CellPolynomials(c,.125,['phi'],2).evaluate(f.x,f.y)[:,0]-CellPolynomials(c0,.125,['phi'],2).evaluate(f.x,f.y)[:,0]
    fig,axes=plt.subplots(3,1,figsize=(10,6.4),sharex=True,layout='constrained');order=np.lexsort((f.x,f.y));x=np.unique(f.x);y=np.unique(f.y);maximum=max(max(mapped),max(f.phi-f0.phi))
    for ax,values,title in zip(axes,[mapped,f.phi-f0.phi,diff],['Coarse increment (h=0.125 m)','Fine increment (h=0.0625 m)','Coarse − fine increment']):
        v=max(abs(diff)) if ax is axes[-1] else maximum
        im=ax.pcolormesh(x,y,values[order].reshape(len(y),len(x)),shading='nearest',cmap='coolwarm' if ax is axes[-1] else 'magma',vmin=-v if ax is axes[-1] else 0,vmax=v,rasterized=True)
        ax.set(xlabel='x (m)' if ax is axes[-1] else '',ylabel='y (m)',title=title,aspect='equal',xlim=(1.7,6.3),ylim=(1.6,2.4));fig.colorbar(im,ax=ax,shrink=.75)
        for edge in (2,6):ax.axvline(edge,color='white',lw=.7,ls=':')
    fig.suptitle(f'Own-seed phase increments at {end:.6g} s'+(' — target incomplete' if end<.06-1e-10 else ''))
    for ext in ('png','pdf'):fig.savefig(OUT/f'phase-increment.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for r,rows,tag in ((coarse,cr,'Coarse'),(fine,fr,'Fine')):
        t=rows.time;steps=r['steps'];axes[0,0].plot(t,rows.fracture-r['initial_fracture'],label=tag)
        for name,style in [('old_central_band','-'),('old_tip_band','--'),('axial_intact',':')]:axes[0,1].plot(t,[s['regions'][name]['max_delta_phi'] for s in steps],style,color='C0' if tag=='Coarse' else 'C1',label=tag+' '+name.replace('_',' '))
        error=rows.Wext-rows.energy_change-rows.Db-rows.Dc-rows.Df
        line,=axes[1,0].plot(t,np.cumsum(error),label=tag+' signed');axes[1,0].plot(t,np.cumsum(abs(error)),'--',color=line.get_color(),label=tag+' accumulated absolute')
        profile=read(RUNS/r['label']/'output/tip-profile.csv');profile=profile[profile.step<=rows.step[-1]];axes[1,1].plot(profile.step*.000625,profile.step_seconds,label=tag)
    cost_file=OUT/'cost-summary.json'
    if not fine['trajectory_complete'] and cost_file.exists():
        cost=json.loads(cost_file.read_text())
        if cost.get('unfinished_step')==int(fr.step[-1])+1:
            axes[1,1].scatter([end+float(fr.dt[-1])],[cost['unfinished_elapsed_seconds']],marker='x',color='C3',label='Fine unfinished interval')
    for ax,title,unit in zip(axes.flat,['Added SIPG fracture','Regional phase increment','Physical work error','Cost per interval (× unfinished)'],['J/m','phase','J/m','seconds']):ax.set(title=title,xlabel='Endpoint time (s)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=6)
    axes[0,1].set_yscale('symlog',linthresh=1e-4)
    axes[0,1].axhline(1e-4,color='k',lw=.7,ls=':');axes[1,0].set_yscale('symlog',linthresh=1e-8);axes[1,1].set_yscale('log')
    fig.suptitle('Fixed-direction synthetic verification; midpoint stage work and endpoint storage')
    for ext in ('png','pdf'):fig.savefig(OUT/f'onset-energy-cost.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
