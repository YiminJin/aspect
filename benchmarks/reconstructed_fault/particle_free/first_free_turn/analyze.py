#!/usr/bin/env python3
"""Native-weight, native-Bernstein observations for the first free turn.

Read only closed exports. Each run is independent; no interpolation across DG
faces, no contour threshold for history registration, no invented missing data.
"""
import argparse,json,os,sys
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'bulk_tip'))
from analyze_resolution_matched import ROOT,RUNS,read,gather,RectBernstein,masks
from analyze_resolution60_window import contour_segments
from analyze_resolution60_growth import pulse,intervals
import numpy as np
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn'

def native(path,step):return gather(path,f'tip-balance-{step:04d}-r',['cell','i'])
def polynomial(path,step,hx,hy):return RectBernstein(native(path,step),hx,hy)
def contour(poly):
    results={}
    for n in (16,32):
        seg,bounds=contour_segments(poly,n)
        vertices=np.concatenate(seg)
        right=vertices[np.argmax(vertices[:,0])]
        beyond=vertices[vertices[:,0]>6]
        results[str(n)]=dict(bounds=bounds,rightmost=right.tolist(),
            rightmost_lateral=float(right[1]-2),
            ahead_y_range=[float(beyond[:,1].min()),float(beyond[:,1].max())] if len(beyond) else None)
    return results

def connectivity(poly):
    """Four-neighbor sampled superlevel connection to the interior seed.

    Restricted endpoint window; two samplings expose observation uncertainty.
    This is a diagnostic and never controls section membership.
    """
    from scipy.ndimage import label
    result={}
    for n in (16,32):
        dx,dy=poly.hx/n,poly.hy/n
        x=np.arange(5.25+dx/2,6.75,dx);y=np.arange(1.5+dy/2,2.5,dy)
        phi=poly.evaluate(x[None,:],y[:,None]);labels,count=label(phi>=.05)
        anchor=(x[None,:]<5.5)&(abs(y[:,None]-2)<.05)
        seed_labels=np.unique(labels[anchor & (labels>0)])
        connected=np.isin(labels,seed_labels)&(labels>0)
        columns=np.flatnonzero(connected.any(axis=0))
        ahead_labels=np.unique(labels[:,x>6]);ahead_labels=ahead_labels[ahead_labels>0]
        result[str(n)]=dict(connected_x_max=float(x[columns[-1]]) if len(columns) else None,
             connected_beyond_original=bool(connected[:,x>6].any()),
             components_ahead=int(len(ahead_labels)),
             components_ahead_not_connected=int(sum(~np.isin(ahead_labels,seed_labels))),
             sampling_dx=dx,sampling_dy=dy)
    return result

def reduce(label,hx,hy,seed_label=None):
    path=RUNS/label/'output';accepted=read(path/'bulk-accepted.csv')
    first=native(path,int(accepted.step[0]))
    seedpath=RUNS/seed_label/'output' if seed_label else path
    if not seed_label and not list(path.glob('tip-seed-fracture-r*')) and label.endswith('-end'):
        seedpath=RUNS/(label.removesuffix('-end')+'-prefix')/'output'
    seed=gather(seedpath,'tip-seed-fracture-r',['cell','i']) if list(seedpath.glob('tip-seed-fracture-r*')) else None
    summary=dict(label=label,hx=hx,hy=hy,steps=[])
    previous=None
    for a in accepted:
        step=int(a.step);cache=OUT/(label+f'-{step:04d}.json')
        if cache.exists():
            item=json.loads(cache.read_text())
            if item.get('schema')==3:
                summary['steps'].append(item);continue
        rows=native(path,step);poly=RectBernstein(rows,hx,hy)
        section=read(path/f'tip-sections-{step:04d}.csv')
        plane=read(path/f'tip-planes-{step:04d}.csv')
        old=plane.parent_formed>0;new=(plane.formed>0)&~old;virgin=plane.formed==0
        E=.5*rows.B*(rows.fracture_gradient+2*rows.mass)
        assert abs(float(sum(E))-a.fracture)<2e-8
        item=dict(schema=3,step=step,time=float(a.time),accepted={n:float(a[n]) for n in a.dtype.names},
             physical_work_error=float(a.Wext-a.energy_change-a.Db-a.Dc-a.Df),
             contours=contour(poly),width=pulse(poly),
             formation=dict(new_sections=int(sum(new)),new_x=plane.x[new].tolist(),
                 max_adjacent_angle_jump=float(max(abs(np.diff(plane.angle)))),
                 max_adjacent_angle_gradient=float(max(abs(np.diff(plane.angle)/np.diff(plane.x)))),
                 old_angles_max=float(max(abs(plane.angle[old]))) if old.any() else None,
                 old_age_range=[float(min(plane.age[old])),float(max(plane.age[old]))] if old.any() else None,
                 new_age_range=[float(min(plane.age[new])),float(max(plane.age[new]))] if new.any() else None,
                 new_slip_range=[float(min(plane.slip[new])),float(max(plane.slip[new]))] if new.any() else None,
                 new_angle_range=[float(min(plane.angle[new])),float(max(plane.angle[new]))] if new.any() else None,
                 virgin_age_error=float(max(abs(plane.age[virgin]-(.5+a.time)))) if virgin.any() else None),
             integrated_slip=float(sum(section.ds*section.slip)),
             signed_local_rate=dict(positive=float(sum(section.ds*section.positive_w)),negative=float(sum(section.ds*section.negative_w))))
        if new.any():
            z,_=np.polynomial.legendre.leggauss(6)
            y=((np.arange(poly.ny)[:,None]+(z+1)/2)*hy).ravel()
            phinew=poly.evaluate(plane.x[new,None],y[None,:])
            item['formation']['native_new_phase_max']=float(phinew.max())
            item['formation']['new_section_phase_maxima']=np.max(phinew,axis=1).tolist()
        if seed is not None:
            initial=RectBernstein(first,hx,hy,seed.B);E0=.5*seed.B*(seed.fracture_gradient+2*first.mass)
            item['initial_contour']=contour(initial)
            item['added_fracture']=float(sum(E-E0))
            item['regions']={n:dict(added_fracture=float(sum((E-E0)[mask])),max_coefficient_increment=float(max((rows.B-seed.B)[mask]))) for n,mask in masks(rows.x,rows.y).items()}
        if list(path.glob(f'tip-points-{step:04d}-r*')):
            q=gather(path,f'tip-points-{step:04d}-r',['cell','q']);idx=q.section.astype(int)
            V=np.bincount(idx,weights=q.weight*q.w,minlength=len(section))/section.ds
            power=float(sum(q.weight*q.w*section.force[idx])-sum(section.ds*section.V*section.force))
            assert max(abs(V-section.V))<2e-12
            assert abs(power)<1e-8
            item['local_power']=dict(positive=float(sum(np.maximum(q.weight*q.w*section.force[idx],0))),negative=float(sum(np.minimum(q.weight*q.w*section.force[idx],0))))
            item['native_weight_checks']=dict(rate_error=float(max(abs(V-section.V))),power_error=power,
                point_polynomial_error=float(max(abs(poly.evaluate(q.x,q.y)-q.phi))),material_area=float(sum(q.weight)),
                new_phase_max=float(max(q.phi[new[idx]])) if new.any() else None,
                new_A_norm_max=float(max(np.sqrt(q.Axx[new[idx]]**2+q.Ayy[new[idx]]**2+q.Azz[new[idx]]**2+2*q.Axy[new[idx]]**2))) if new.any() else None)
        cache.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n');summary['steps'].append(item)
    errors=np.array([i['physical_work_error'] for i in summary['steps']])
    summary['work']=dict(signed=float(sum(errors)),accumulated_absolute=float(sum(abs(errors))))
    profile=read(path/'tip-profile.csv')
    lifecycle=[]
    for a in accepted:
        plane=read(path/f'tip-planes-{int(a.step):04d}.csv')
        assert max(abs(plane.reference_dx-plane.ds*np.cos(plane.angle)))<2e-14
        assert abs(sum(plane.reference_dx)-8)<2e-12
        if previous is not None:
            old=plane.parent_formed>0
            assert np.array_equal(plane.x,previous.x)
            assert max(abs(plane.angle[old]-previous.angle[old]))==0
            assert max(abs(plane.ds[old]-previous.ds[old]))==0
            assert np.array_equal(plane.parent_formed,previous.formed)
        previous=plane
        lifecycle.append(dict(step=int(a.step),formed=int(sum(plane.formed)),metric_error=float(max(abs(plane.reference_dx-plane.ds*np.cos(plane.angle))))))
    summary['lifecycle']=lifecycle
    summary['cost']=dict(solver_seconds=float(sum(profile.step_seconds)),line_trials=float(sum(profile.line_trials)),stokes_iterations=float(sum(profile.stokes_iterations)))
    (OUT/(label+'-analysis.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary

def verify_parent(parent_label,child_label,step):
    parent=RUNS/parent_label/'output';child=RUNS/child_label/'output';result={}
    def compare(a,b,names):
        errors={}
        assert len(a)==len(b)
        for n in names:
            if a[n].dtype.kind in 'SU':
                assert np.array_equal(a[n],b[n]);continue
            assert np.array_equal(np.isnan(a[n]),np.isnan(b[n]))
            finite=np.isfinite(a[n]);e=float(max(abs(a[n][finite]-b[n][finite]))) if finite.any() else 0.
            assert e<=2e-13*max(1.,float(max(abs(a[n][finite]))) if finite.any() else 1),(n,e)
            errors[n]=e
        return errors
    if list(parent.glob(f'tip-points-{step:04d}-r*')):
        a=gather(parent,f'tip-points-{step:04d}-r',['cell','q']);b=gather(child,'restore-points-r',['cell','q'])
        result['points']=compare(a,b,b.dtype.names)
    else:
        result['points_note']='No parent point export at this step; full tensor rank-transfer is checked independently in the short nonuniform recovery control.'
    a=native(parent,step);b=gather(child,'restore-phase-r',['cell','i'])
    result['phase']=compare(a,b,['cell','i','B']);assert max(abs(b.B-b.solution_B))==0
    a=read(parent/f'tip-sections-{step:04d}.csv');b=read(child/'restore-sections.csv')
    result['sections']=compare(a,b,b.dtype.names)
    a=read(parent/f'tip-planes-{step:04d}.csv');b=read(child/'restore-planes.csv')
    result['planes']=compare(a,b,b.dtype.names)
    clock=read(child/'restore-clock.csv')[0];assert clock.accepted_step==step
    result['clock']={n:float(clock[n]) for n in clock.dtype.names}
    (OUT/(child_label+'-restore.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

def pair(positive,reflected,step,hx,hy):
    pa=RUNS/positive/'output';pb=RUNS/reflected/'output'
    a=polynomial(pa,step,hx,hy);b=polynomial(pb,step,hx,hy)
    reflection=b.coefficients[::-1,:,::-1,:]
    result=dict(step=step,phase_reflection_max=float(max(abs(a.coefficients-reflection).ravel())),
                phase_reflection_coefficient_rms=float(np.sqrt(np.mean((a.coefficients-reflection)**2))))
    sa=read(pa/f'tip-sections-{step:04d}.csv');sb=read(pb/f'tip-sections-{step:04d}.csv')
    result['section_reflection']={n:float(max(abs(sa[n]-sign*sb[n]))) for n,sign in [('age',1),('slip',-1),('V',-1),('ds',1),('force',-1)]}
    qa=read(pa/f'tip-planes-{step:04d}.csv');qb=read(pb/f'tip-planes-{step:04d}.csv')
    if list(pa.glob(f'tip-points-{step:04d}-r*')) and list(pb.glob(f'tip-points-{step:04d}-r*')):
        va=gather(pa,f'tip-points-{step:04d}-r',['cell','q']);vb=gather(pb,f'tip-points-{step:04d}-r',['cell','q'])
        va=va[np.lexsort((np.round(va.y,12),np.round(va.x,12)))];vb=vb[np.lexsort((np.round(4-vb.y,12),np.round(vb.x,12)))]
        assert max(abs(va.x-vb.x))<2e-13 and max(abs(va.y-(4-vb.y)))<2e-13
        fields={}
        for prefix in ('tau','A'):
            err=sum(factor*sum(va.weight*(va[prefix+n]-sign*vb[prefix+n])**2) for n,sign,factor in [('xx',1,1),('yy',1,1),('zz',1,1),('xy',-1,2)])
            den=sum(factor*sum(va.weight*va[prefix+n]**2) for n,factor in [('xx',1),('yy',1),('zz',1),('xy',2)])
            fields[prefix]=dict(absolute_L2=float(np.sqrt(err)),relative_L2=float(np.sqrt(err/den)) if den else None)
        result['physical_tensor_reflection']=fields
    result['angle_reflection_max']=float(max(abs(qa.angle+qb.angle)))
    result['formed_flag_disagreements']=int(sum(qa.formed!=qb.formed))
    result['positive_contour']=contour(a);result['reflected_contour']=contour(b)
    result['positive_connectivity']=connectivity(a);result['reflected_connectivity']=connectivity(b)
    (OUT/f'reflection-{step:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def figure(positive,reflected,step,hx,hy,seed_label=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    p=RUNS/positive/'output';r=RUNS/reflected/'output'
    seedpath=p if list(p.glob('tip-seed-fracture-r*')) else RUNS/(positive.removesuffix('-end')+'-prefix')/'output'
    if seed_label:seedpath=RUNS/seed_label/'output'
    first=native(seedpath,1);seed=gather(seedpath,'tip-seed-fracture-r',['cell','i'])
    initial=RectBernstein(first,hx,hy,seed.B)
    a=polynomial(p,step,hx,hy);b=polynomial(r,step,hx,hy)
    baseline=polynomial(RUNS/'resolution60-coarse-growth/output',step,hx,hy) if hx==.0625 and hy==.03125 else None
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax in axes:
        for field,color,label in [(initial,'black','Initial seed'),(a,'#c44e52','Positive load'),(b,'#4c72b0','Reflected load')]:
            seg,_=contour_segments(field,32)
            ax.add_collection(LineCollection(seg,colors=color,linewidths=1.2,label=label))
        if baseline is not None:
            seg,_=contour_segments(baseline,32)
            ax.add_collection(LineCollection(seg,colors='.5',linewidths=.8,linestyles='--',label='Original straight load'))
        ax.axvline(6,color='.35',ls='--',lw=.8)
        ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    axes[0].set(xlim=(1.6,6.5),ylim=(1.45,2.55),title=r'Absolute $\phi=0.05$ contour')
    axes[0].legend(loc='upper center',ncol=2,fontsize=8)
    axes[1].set(xlim=(5.75,6.3),ylim=(1.78,2.22),title=f'Original right endpoint; accepted step {step}')
    axes[1].set_xticks(np.arange(5.75,6.31,hx),minor=True);axes[1].set_yticks(np.arange(1.75,2.26,hy),minor=True)
    axes[1].grid(which='minor',color='.85',lw=.5)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'free-turn-contours-{step:04d}.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for directory,color,label in [(p,'#c44e52','Positive'),(r,'#4c72b0','Reflected')]:
        sections=read(directory/f'tip-sections-{step:04d}.csv')
        planes=read(directory/f'tip-planes-{step:04d}.csv')
        formed=planes.formed>0
        axes[0,0].plot(planes.x[formed],planes.angle[formed]*180/np.pi,'.',ms=2,color=color,label=label+' formed')
        axes[0,0].plot(planes.x[~formed],planes.angle[~formed]*180/np.pi,':',lw=.7,color=color)
        axes[0,1].plot(planes.x,sections.age,color=color,label=label)
        axes[1,0].plot(planes.x,sections.slip,color=color,label=label)
        axes[1,1].plot(planes.x,sections.V,color=color,label=label)
    for ax,title in zip(axes.flat,['Structural plane (degrees); dots formed','Age (s)','Signed cumulative slip (m)','Stage slip rate (m/s)']):
        ax.axvline(6,color='.6',ls='--',lw=.7);ax.set(xlabel='Material x (m)',title=title,xlim=(5.5,6.5));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'free-turn-histories-{step:04d}.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('labels',nargs='+')
    parser.add_argument('--hx',type=float,default=.0625);parser.add_argument('--hy',type=float,default=.03125)
    parser.add_argument('--pair-step',type=int);parser.add_argument('--restore-parent');parser.add_argument('--seed-label');parser.add_argument('--parent-step',type=int,default=12);args=parser.parse_args()
    for label in args.labels:
        result=reduce(label,args.hx,args.hy,args.seed_label if label==args.labels[0] else None);print(label,len(result['steps']),result['work'])
    if args.restore_parent:
        verify_parent(args.restore_parent,args.labels[0],args.parent_step)
    if args.pair_step:
        print(json.dumps(pair(*args.labels,args.pair_step,args.hx,args.hy),indent=2))
        figure(*args.labels,args.pair_step,args.hx,args.hy,args.seed_label)
