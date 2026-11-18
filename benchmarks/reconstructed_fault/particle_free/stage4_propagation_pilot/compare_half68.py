#!/usr/bin/env python3
"""Compare the 80/40-us branches from their identical accepted step68 parent."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import argparse,json,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import coefficients,fronts,read,basis
ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/bulk-tip/runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
COARSE=RUNS/'stage4-propagation-pilot-resume51/output'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fine-run', type=Path, default=RUNS/'stage4-propagation-half68')
parser.add_argument('--destination', type=Path, default=OUT)
args=parser.parse_args()
if args.fine_run != RUNS/'stage4-propagation-half68' and args.destination.resolve()==OUT.resolve():
    parser.error('Choose a separate destination for another fine trajectory.')
OUT=args.destination
OUT.mkdir(parents=True,exist_ok=True)
FINE=args.fine_run/'output'
PARENT_TIME=.00544
parent=coefficients(COARSE,68)
initial=coefficients(RUNS/'stage4-propagation-pilot-adaptive/output',0)
parent_front=fronts(parent,16)
parent_sections=read(COARSE/'pilot-sections-0068.csv')
parent_planes=read(COARSE/'tip-planes-0068.csv')
x=parent_sections['x'];weights=parent_planes['reference_dx']
assert np.all(weights>0) and abs(weights.sum()-8)<1e-12
original_intact=abs(initial).max(axis=(2,3))==0
coarse_energy=read(COARSE/'bulk-accepted.csv');parent_energy=coarse_energy[coarse_energy['step']==68][0]
fine_summary=read(FINE/'pilot-summary.csv') if (FINE/'pilot-summary.csv').exists() else np.array([],dtype=[('step',int),('time',float),('stage_time',float),('dt',float)])
fine_energy=read(FINE/'bulk-accepted.csv') if len(fine_summary) else []
assert all(abs(r['dt']-.00004)<1e-14 for r in fine_summary)
assert all(abs(r['time']-(PARENT_TIME+(r['step']-68)*.00004))<1e-13 for r in fine_summary)
q,w=np.polynomial.legendre.leggauss(6);q=basis((q+1)/2);w=w/2

def phase_l2(a):
    values=np.einsum('ai,yxij,bj->yxab',q,a,q)
    return float(np.sqrt(np.einsum('yxab,a,b->',values**2,w,w)/1024))

def norms(delta,reference,mask):
    weight=weights[mask];d=delta[mask];r=reference[mask]
    rms=float(np.sqrt(np.dot(weight,d*d)/weight.sum()))
    rr=float(np.sqrt(np.dot(weight,r*r)/weight.sum()))
    return dict(max_abs=float(abs(d).max()),reference_dx_rms=rms,relative_rms=rms/rr if rr else None)

regions={'all':np.ones(len(x),dtype=bool),'slipping_patch':(x>=3)&(x<=6),
         'right_taper':(x>=5.5)&(x<=6.25),'left_taper':(x>=1.75)&(x<=2.5)}
cache={}
def frame(out,step):
    key=(str(out),step)
    if key not in cache:
        b=coefficients(out,step);f=fronts(b,16);s=read(out/f'pilot-sections-{step:04d}.csv')
        assert np.array_equal(s['x'],x)
        planes=read(out/f'tip-planes-{step:04d}.csv')
        old=parent_planes['formed']>0
        cache[key]=(b,f,s,dict(new_formed=int(np.sum((planes['formed']>0)&(~old))),
            old_angle_max=float(abs(planes['angle'][old]-parent_planes['angle'][old]).max()),
            any_angle_max=float(abs(planes['angle']-parent_planes['angle']).max())))
    return cache[key]

def ledger(energy,endstep):
    rows=energy[(energy['step']>68)&(energy['step']<=endstep)]
    e=rows[-1];error=rows['Wext']-rows['energy_change']-rows['Db']-rows['Dc']-rows['Df']
    r={key:float(e[key]) for key in ['bulk','cohesive','fracture','phase_KKT','mechanical_residual']}
    r.update(fracture_increment=float(e['fracture']-parent_energy['fracture']),
             physical_work_error_signed=float(error.sum()),physical_work_error_absolute=float(abs(error).sum()))
    for key in ['Wext','Db','Dc','Df','energy_change']:r[key+'_window']=float(rows[key].sum())
    return r

comparisons=[];plot_profiles=[]
for coarse_step,fine_step,time in [(69,70,.00552),(70,72,.00560)]:
    available=fine_summary[fine_summary['step']==fine_step]
    if len(available)!=1:continue
    assert abs(float(available[0]['time'])-time)<1e-13
    cb,cf,cs,cg=frame(COARSE,coarse_step);fb,ff,fs,fg=frame(FINE,fine_step)
    fine_frames=[read(FINE/f'pilot-sections-{k:04d}.csv') for k in [fine_step-1,fine_step]]
    mean_rate=.5*(fine_frames[0]['rate_stage']+fine_frames[1]['rate_stage'])
    mean_force=.5*(fine_frames[0]['resistance_stage']+fine_frames[1]['resistance_stage'])
    cprev=parent_sections if coarse_step==69 else read(COARSE/f'pilot-sections-{coarse_step-1:04d}.csv')
    fprev=parent_sections if fine_step==70 else read(FINE/f'pilot-sections-{fine_step-2:04d}.csv')
    slip_rate_defect=max(float(abs((cs['slip_endpoint']-cprev['slip_endpoint'])/.00008-cs['rate_stage']).max()),
                         float(abs((fs['slip_endpoint']-fprev['slip_endpoint'])/.00008-mean_rate).max()))
    assert slip_rate_defect<1e-10,slip_rate_defect
    fields={}
    for name,mask in regions.items():
        fields[name]={
            'slip_endpoint':norms(fs['slip_endpoint']-cs['slip_endpoint'],cs['slip_endpoint']-parent_sections['slip_endpoint'],mask),
            'age_endpoint':norms(fs['age_endpoint']-cs['age_endpoint'],cs['age_endpoint'],mask),
            'mean_rate_last_80us':norms(mean_rate-cs['rate_stage'],cs['rate_stage'],mask),
            'mean_resistance_last_80us':norms(mean_force-cs['resistance_stage'],cs['resistance_stage'],mask)}
    phase_error=phase_l2(fb-cb);coarse_increment=phase_l2(cb-parent)
    row=dict(time=time,coarse_step=coarse_step,fine_step=fine_step,
        phase_l2_error=phase_error,phase_increment_coarse_l2=coarse_increment,
        phase_increment_relative_l2_error=phase_error/coarse_increment if coarse_increment else None,
        phase_max_coefficient_error=float(abs(fb-cb).max()),
        right_advance_coarse=cf['right_x']-parent_front['right_x'],right_advance_fine=ff['right_x']-parent_front['right_x'],
        right_position_error=ff['right_x']-cf['right_x'],left_position_error=ff['left_x']-cf['left_x'],
        slip_integral_coarse=float(np.dot(cs['ds'],cs['slip_endpoint'])),slip_integral_fine=float(np.dot(fs['ds'],fs['slip_endpoint'])),
        slip_window_integral_coarse=float(np.dot(cs['ds'],cs['slip_endpoint'])-np.dot(parent_sections['ds'],parent_sections['slip_endpoint'])),
        slip_window_integral_fine=float(np.dot(fs['ds'],fs['slip_endpoint'])-np.dot(parent_sections['ds'],parent_sections['slip_endpoint'])),
        section_measure_max_difference=float(abs(fs['ds']-cs['ds']).max()),
        matched_interval_rate_slip_identity_error=slip_rate_defect,section_errors=fields,
        coarse_geometry=cg,fine_geometry=fg,coarse_energy=ledger(coarse_energy,coarse_step),fine_energy=ledger(fine_energy,fine_step))
    for label,phase in [('coarse',cb),('fine',fb)]:
        row[label+'_fresh_max']=float((phase-initial)[original_intact].max())
        row[label+'_fresh_integral']=float((phase-initial)[original_intact].sum()/9/1024)
    comparisons.append(row);plot_profiles.append((time,cs,fs,mean_rate,cb,fb))

trajectory={}
for name,out,steps,dt,energy in [('80us',COARSE,[69,70],.00008,coarse_energy),('40us',FINE,list(fine_summary['step'].astype(int)),.00004,fine_energy)]:
    rows=[dict(step=68,time=PARENT_TIME,right_advance=0.,fracture_increment=0.,signed_work_error=0.,absolute_work_error=0.)];previous=parent
    for step in steps:
        b,f,s,g=frame(out,step);e=ledger(energy,step);rows.append(dict(step=int(step),time=PARENT_TIME+(step-68)*dt,
            right_advance=f['right_x']-parent_front['right_x'],fracture_increment=e['fracture_increment'],
            signed_work_error=e['physical_work_error_signed'],absolute_work_error=e['physical_work_error_absolute'],
            minimum_increment=float((b-previous).min()),minimum_phase=float(b.min()),maximum_phase=float(b.max()),**g))
        previous=b
    trajectory[name]=rows

cost={}
for name,folder in [('80us',COARSE.parent),('40us',FINE.parent)]:
    chunks=re.split(r'\*\*\* Timestep (\d+):',(folder/'run.log').read_text());rows=[]
    for i in range(1,len(chunks),2):
        step=int(chunks[i]);body=chunks[i+1]
        if step<69 or (name=='80us' and step>70):continue
        solves=re.findall(r'Tip bound linear solve: pass=(\d+), iterations=(\d+), residual=([^,]+), seconds=([\d.e+\-]+)',body)
        rows.append(dict(step=step,accepted='Postprocessing:' in body,linear_seconds=sum(float(v[3]) for v in solves),linear_solves=len(solves),iterations=sum(int(v[1]) for v in solves)))
    cost[name]=rows
command=json.loads((FINE.parent/'command.json').read_text())
record=dict(common_parent_step=68,common_parent_time=PARENT_TIME,complete=len(fine_summary)==4 and len(comparisons)==2,
            fine_process_exit=command.get('exit_code'),fine_process_seconds=command.get('seconds'),comparisons=comparisons,
            trajectory=trajectory,cost=cost,
            conventions='Phase/age/slip at identical endpoints. Rates and section resistance averaged over the same preceding80us interval; no unequal midpoint-stage samples labelled as endpoint rates. Section RMS errors use common parent reference_dx; physical integrated slip uses each state ds. Phase L2 uses6x6 Gauss, exact for squared DGQ2 differences on this affine mesh. Error in slip and phase is normalized by evolution since the common parent, not total accumulated history. Full endpoint stress fields were not exported; bulk energy and stage section resistance are compared, not an inferred stress norm.')
# These are trial residuals. They are never plotted as accepted physical fields.
nonlinear=[];linearized=[];outer=-1;last_linear=None
step=None
for line in (FINE.parent/'run.log').read_text().splitlines():
    m=re.search(r'\*\*\* Timestep (\d+):',line)
    if m:step=int(m[1])
    m=re.search(r'Bulk tip coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+)',line)
    if m:
        outer=int(m[1]);last_linear=None
        nonlinear.append(dict(step=step,iteration=outer,mechanics=float(m[2]),phase_KKT=float(m[3])))
    m=re.search(r'Tip bound linear solve: pass=(\d+), iterations=(\d+), residual=([^,]+), seconds=([\d.e+\-]+)',line)
    if m:last_linear=dict(pass_index=int(m[1]),iterations=int(m[2]),reported_residual=float(m[3]),seconds=float(m[4]))
    m=re.search(r'Tip linearized VI: pass=(\d+), feasible=(\d+), phase=([^,]+), mechanics=([^,]+), continuity=([^,]+)',line)
    if m:linearized.append(dict(step=step,outer_iteration=outer,pass_index=int(m[1]),feasible=bool(int(m[2])),phase_KKT=float(m[3]),mechanics=float(m[4]),continuity=float(m[5]),preceding_linear_solve=last_linear))
record['fine_nonlinear_trace']=nonlinear;record['fine_linearized_trace']=linearized
(OUT/'half68-comparison.json').write_text(json.dumps(record,indent=2)+'\n')
flat=[]
for r in comparisons:
    flat.append({k:v for k,v in r.items() if isinstance(v,(float,int))})
if flat:
    with (OUT/'half68-comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
if not record['complete']:fig.suptitle('Bounded check incomplete: only accepted states are shown')
for name,rows in trajectory.items():
    t=np.array([r['time'] for r in rows])*1e3
    axes[0,0].plot(t,[r['right_advance']*1e3 for r in rows],'-o',label=name)
    axes[0,1].plot(t,[r['fracture_increment'] for r in rows],'-o',label=name)
    axes[0,2].plot(t,[r['signed_work_error'] for r in rows],'-o',label=name+' signed')
    axes[0,2].plot(t,[r['absolute_work_error'] for r in rows],'--',label=name+' absolute')
for ax,title,y in zip(axes[0],['Connected right contour','Added fracture','Physical work error'],['Advance from parent (mm)','J/m since parent','J/m since parent']):
    ax.set(title=title,xlabel='Endpoint time (ms)',ylabel=y);ax.legend(fontsize=8)
for time,cs,fs,rate,cb,fb in plot_profiles:
    line,=axes[1,0].plot(x,cs['rate_stage']*1e3,label=f'{time*1e3:.2f}ms, 80us')
    axes[1,0].plot(x,rate*1e3,'--',color=line.get_color(),label=f'{time*1e3:.2f}ms, 40us')
    line,=axes[1,1].semilogy(x,cs['age_endpoint'],label=f'{time*1e3:.2f}ms, 80us')
    axes[1,1].semilogy(x,fs['age_endpoint'],'--',color=line.get_color(),label=f'{time*1e3:.2f}ms, 40us')
axes[1,0].set(title='Matched last-80us mean slip rate',xlabel='Material station x (m)',ylabel='Signed rate (mm/s)',xlim=(2.5,6.25));
if plot_profiles:axes[1,0].legend(fontsize=7)
axes[1,1].set(title='Age at matched endpoints',xlabel='Material station x (m)',ylabel='Age (s)',xlim=(2.5,6.25));
if plot_profiles:axes[1,1].legend(fontsize=7)
if plot_profiles:
    time,_,_,_,cb,fb=plot_profiles[-1];delta=(fb-cb).mean(axis=(2,3));maximum=max(float(abs(delta).max()),1e-20)
    im=axes[1,2].imshow(delta,origin='lower',extent=[0,8,0,4],cmap='coolwarm',vmin=-maximum,vmax=maximum,aspect='auto')
    axes[1,2].set(title=f'40us minus80us phase, {time*1e3:.2f}ms',xlabel='x (m)',ylabel='y (m)',xlim=(5.5,6.3),ylim=(1.7,2.3));fig.colorbar(im,ax=axes[1,2],label='Exact cell-mean difference')
else:
    for ax in axes[1]:ax.text(.5,.5,'No matched endpoint completed',ha='center',transform=ax.transAxes)
for extension in ['png','pdf']:fig.savefig(OUT/('half68-comparison.'+extension),dpi=180)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
if nonlinear:
    for step in sorted({r['step'] for r in nonlinear}):
        group=[r for r in nonlinear if r['step']==step]
        axes[0].semilogy([r['iteration'] for r in group],[r['phase_KKT'] for r in group],'-o',label=f'Step {step}')
    axes[0].axhline(1e-7,color='k',ls='--',label='Retained acceptance gate')
    axes[0].set(xlabel='Newton iteration',ylabel='Phase KKT',title='40-us trial convergence');axes[0].legend(fontsize=8)
if linearized:
    axes[1].semilogy(range(len(linearized)),[max(r['phase_KKT'],1e-30) for r in linearized],color='0.6',lw=.7)
    for feasible,color in [(True,'tab:green'),(False,'tab:red')]:
        indices=[i for i,r in enumerate(linearized) if r['feasible']==feasible]
        axes[1].scatter(indices,[max(linearized[i]['phase_KKT'],1e-30) for i in indices],s=14,color=color,label='Feasible' if feasible else 'Infeasible')
    axes[1].set(xlabel='Logged linearized-VI observation',ylabel='Linearized phase residual',title='Trial iterates, not accepted physical states');axes[1].legend(fontsize=8)
for extension in ['png','pdf']:fig.savefig(OUT/('half68-solver.'+extension),dpi=180)
print(json.dumps(dict(complete=record['complete'],matched_times=[r['time'] for r in comparisons],fine_exit=record['fine_process_exit'],fine_seconds=record['fine_process_seconds']),indent=2))
