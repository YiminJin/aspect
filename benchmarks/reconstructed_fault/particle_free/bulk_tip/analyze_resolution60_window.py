"""Matched-parent timestep comparison and absolute-phase contour observations."""
import argparse
import csv
import json
import os
from pathlib import Path
from analyze_resolution60_growth import ROOT,RUNS,read,gather,masks,RectBernstein,pulse,intervals,value,PRIMARY
import numpy as np

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window'
BASE=RUNS/'resolution60-coarse-growth'/'output'
FINE=RUNS/'resolution60-halfstep-window'/'output'
HX,HY,K=.0625,.03125,2.


def native(directory, step):
    return gather(directory,f'tip-balance-{step:04d}-r',['cell','i'])


def field(rows):
    return RectBernstein(rows,HX,HY)


def energy(rows):
    return .5*rows.B*(rows.fracture_gradient+K*rows.mass)


def dump_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def line_coefficients(poly, side):
    iy=round(2/HY)-(1 if side<0 else 0)
    weights=np.array([.25,.5,.25]) if abs(side)==2 else np.array([0.,0.,1.] if side<0 else [1.,0.,0.])
    return np.einsum('j,xji->xi',weights,poly.coefficients[iy])


def line_observations(poly, initial):
    records=[]
    for side in (-2,-1,1,2):
        b=line_coefficients(poly,side);b0=line_coefficients(initial,side)
        pieces=intervals(b,HX,.05);seed_pieces=intervals(b0,HX,.05)
        connected=next(p for p in pieces if p[0]<4<p[1]);seed=next(p for p in seed_pieces if p[0]<4<p[1])
        damage=intervals(b-b0,HX,1e-4)
        tip=next((p for p in damage if p[0]<6<p[1]),None)
        records.append(dict(trace=side,level=.05,absolute_left=connected[0],absolute_right=connected[1],
                            initial_left=seed[0],initial_right=seed[1],right_advance_from_initial=connected[1]-seed[1],
                            right_offset_from_original_end=connected[1]-6,
                            increment_1e4_extension=tip[1]-6 if tip else 0.))
    return records


def contour_segments(poly, subdivisions, cell_mask=None):
    """Cellwise native Q2 contour; no interpolation across DG faces."""
    import contourpy
    u=np.linspace(0,1,subdivisions+1);basis=np.array([(1-u)**2,2*u*(1-u),u*u]).T
    coefficients=poly.coefficients
    hx=getattr(poly,'hx',HX);hy=getattr(poly,'hy',HY)
    candidates=np.argwhere((coefficients.min(axis=(2,3))<=.05)&(coefficients.max(axis=(2,3))>=.05))
    segments=[]
    for iy,ix in candidates:
        if cell_mask is not None and not cell_mask[iy,ix]:continue
        z=basis@coefficients[iy,ix]@basis.T
        if z.min()>.05 or z.max()<.05:continue
        generator=contourpy.contour_generator(x=(ix+u)*hx,y=(iy+u)*hy,z=z,name='serial')
        segments.extend(generator.lines(.05))
    vertices=np.concatenate(segments)
    return segments,dict(x_min=float(vertices[:,0].min()),x_max=float(vertices[:,0].max()),
                         y_min=float(vertices[:,1].min()),y_max=float(vertices[:,1].max()),segments=len(segments))


def verify_restore():
    points=gather(BASE,'tip-points-0012-r',['cell','q']);restored=gather(FINE,'restore-points-r',['cell','q'])
    assert np.array_equal(points.cell,restored.cell) and np.array_equal(points.q,restored.q)
    differences={n:float(max(abs(points[n]-restored[n]))) for n in restored.dtype.names if n!='cell'}
    assert max(differences.values())==0
    p=native(BASE,12);rp=gather(FINE,'restore-phase-r',['cell','i'])
    assert np.array_equal(p.cell,rp.cell) and np.array_equal(p.i,rp.i)
    assert max(abs(p.B-rp.B))==0 and max(abs(rp.B-rp.solution_B))==0
    a=read(BASE/'tip-sections-0012.csv');b=read(FINE/'restore-sections.csv')
    section={n:float(max(abs(a[n]-b[n]))) for n in b.dtype.names};assert max(section.values())==0
    clock=read(FINE/'restore-clock.csv')[0]
    assert clock.accepted_step==12 and abs(clock.accepted_time-.06)<1e-12
    result=dict(point_max_differences=differences,section_max_differences=section,phase_coefficient_max_difference=0.,
                clock={n:float(clock[n]) for n in clock.dtype.names},
                clock_note='Audit is emitted before the existing restart-dt signal override. Actual physical dt is verified from accepted rows and timestep banners.')
    (OUT/'restore-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify_contours():
    """An exact quadratic circle checks roots and rendering error independently."""
    from types import SimpleNamespace
    radius=.2
    x=np.arange(128)*HX-6; y=np.arange(128)*HY-2
    xx=np.array([x*x,x*(x+HX),(x+HX)**2]).T
    yy=np.array([y*y,y*(y+HY),(y+HY)**2]).T
    poly=SimpleNamespace(coefficients=.05+radius**2-yy[:,None,:,None]-xx[None,:,None,:])
    roots=intervals(line_coefficients(poly,1),HX,.05)
    assert np.max(abs(np.array(roots)-[[5.8,6.2]]))<1e-12
    results={}
    for n in (16,32):
        _,bounds=contour_segments(poly,n)
        error=max(abs(bounds[k]-v) for k,v in dict(x_min=5.8,x_max=6.2,y_min=1.8,y_max=2.2).items())
        # Linear interpolation error <= spacing^2/4; slope >= 2(r-spacing).
        spacing=HX/n; bound=spacing**2/(8*(radius-spacing))
        assert error<bound
        results[n]=dict(bounds=bounds,maximum_extremum_error=error,interpolation_error_bound=bound)
    assert results[32]['maximum_extremum_error']<results[16]['maximum_extremum_error']
    result=dict(analytic_field='phi=.05+.2^2-(x-6)^2-(y-2)^2',exact_line_intervals=roots,contour_subdivision=results)
    (OUT/'observation-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def reduce_state(directory, step, initial_rows, seed, parent_rows, parent_points, previous_rows, previous_points):
    name=('half' if directory==FINE else 'coarse')+f'-{step:04d}'
    cache=OUT/(name+'.json')
    if cache.exists():return json.loads(cache.read_text())
    rows=native(directory,step);poly=field(rows);initial=RectBernstein(initial_rows,HX,HY,seed.B)
    points=gather(directory,f'tip-points-{step:04d}-r',['cell','q'])
    assert np.array_equal(points.cell,parent_points.cell) and np.array_equal(points.q,parent_points.q)
    assert max(abs(poly.evaluate(points.x,points.y)-points.phi))<2e-12
    assert min(rows.B-previous_rows.B)>-2e-12 and max(rows.B)<1+2e-12 and min(rows.B)>-2e-12
    e=energy(rows);ep=energy(parent_rows);e0=.5*seed.B*(seed.fracture_gradient+K*initial_rows.mass);e_previous=energy(previous_rows)
    accepted=read(directory/'bulk-accepted.csv');a=accepted[accepted.step==step][0]
    assert abs(sum(e)-a.fracture)<2e-9
    nm,pm=masks(rows.x,rows.y),masks(points.x,points.y)
    nm['other']=~np.logical_or.reduce([nm[n] for n in PRIMARY]);pm['other']=~np.logical_or.reduce([pm[n] for n in PRIMARY])
    regions={}
    phi0=initial.evaluate(points.x,points.y)
    for n,m in nm.items():
        q=pm[n]
        regions[n]=dict(max_delta_phi=float(max(points.phi[q]-phi0[q])),max_delta_from_parent=float(max(points.phi[q]-parent_points.phi[q])),
                        max_step_increment=float(max(points.phi[q]-previous_points.phi[q])),
                        phase_increment_integral=float(sum(points.weight[q]*(points.phi[q]-phi0[q]))),
                        added_fracture=float(sum(e[m]-e0[m])),window_added_fracture=float(sum(e[m]-ep[m])),step_added_fracture=float(sum(e[m]-e_previous[m])))
    sections=read(directory/f'tip-sections-{step:04d}.csv');idx=points.section.astype(int)
    power=float(sum(points.weight*points.w*sections.force[idx])-sum(sections.ds*sections.V*sections.force))
    assert abs(power)<1e-9
    assert max(abs(np.bincount(idx,weights=points.weight*points.w)/sections.ds-sections.V))<1e-12
    result=dict(step=step,time=float(a.time),dt=float(a.dt),regions=regions,width=pulse(poly),
                contours=line_observations(poly,initial),window_added_fracture=float(sum(e-ep)),added_fracture=float(sum(e-e0)),
                physical_work_error=float(a.Wext-a.energy_change-a.Db-a.Dc-a.Df),power_adjoint_error=power,
                integrated_V=float(sum(sections.ds*sections.V)),integrated_slip=float(sum(sections.ds*sections.slip)),
                positive_w=float(sum(sections.ds*sections.positive_w)),negative_w=float(sum(sections.ds*sections.negative_w)),
                accepted={n:float(a[n]) for n in a.dtype.names})
    cache.write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_endpoint(coarse_step,half_step,parent_points):
    a=gather(BASE,f'tip-points-{coarse_step:04d}-r',['cell','q']);b=gather(FINE,f'tip-points-{half_step:04d}-r',['cell','q'])
    assert np.array_equal(a.cell,b.cell) and np.array_equal(a.q,b.q) and np.array_equal(a.weight,b.weight)
    weight=b.weight
    def norm(v):return float(np.sqrt(np.sum(weight*v*v)))
    phase=dict(absolute_L2=norm(a.phi-b.phi),maximum=float(max(abs(a.phi-b.phi))),
               relative_to_fine_window_increment=norm(a.phi-b.phi)/norm(b.phi-parent_points.phi))
    fields={}
    for names,label,factors in [(['tauxx','tauyy','tauzz','tauxy'],'physical_stress',[1,1,1,2]),(['Axx','Ayy','Azz','Axy'],'cohesive_A',[1,1,1,2]),(['qcoh'],'qcoh',[1])]:
        err=sum(k*norm(a[n]-b[n])**2 for n,k in zip(names,factors));den=sum(k*norm(b[n])**2 for n,k in zip(names,factors))
        fields[label]=dict(absolute_L2=np.sqrt(err),relative_L2=np.sqrt(err/den) if den else None)
    sa=read(BASE/f'tip-sections-{coarse_step:04d}.csv');sb=read(FINE/f'tip-sections-{half_step:04d}.csv');sb_previous=read(FINE/f'tip-sections-{half_step-1:04d}.csv')
    assert np.array_equal(sa.s,sb.s) and np.array_equal(sa.ds,sb.ds)
    sections={}
    for n in ('slip','age'):
        sections[n]=dict(relative_L2=float(np.sqrt(sum(sb.ds*(sa[n]-sb[n])**2)/sum(sb.ds*sb[n]**2))),maximum=float(max(abs(sa[n]-sb[n]))))
    mean_rate=.5*(sb.V+sb_previous.V)
    sections['interval_mean_V']=dict(relative_L2=float(np.sqrt(sum(sb.ds*(sa.V-mean_rate)**2)/sum(sb.ds*mean_rate**2))),maximum=float(max(abs(sa.V-mean_rate))),
                                     coarse_integral=float(sum(sa.ds*sa.V)),half_integral=float(sum(sb.ds*mean_rate)))
    return dict(coarse_step=coarse_step,half_step=half_step,phase=phase,fields=fields,sections=sections,
                rate_note='Compare one coarse stage rate to the duration-weighted mean of its two half-step rates, not rates at different stage times.')


def analyze(live=False):
    """Reduce only complete exported states; preserve the inherited .060 parent."""
    import re
    initial_rows=native(BASE,1);seed=gather(BASE,'tip-seed-fracture-r',['cell','i'])
    parent_rows=native(BASE,12);parent_points=gather(BASE,'tip-points-0012-r',['cell','q'])
    accepted_path=FINE/'bulk-accepted.csv'
    steps=[] if not accepted_path.exists() else [int(s) for s in read(accepted_path).step if s>12]
    if live:
        banner=max(map(int,re.findall(r'\*\*\* Timestep (\d+):',(FINE.parent/'run.log').read_text())))
        steps=[s for s in steps if s<banner and all((FINE/f'{stem}-{s:04d}-r{r}.csv.gz').exists() and not (FINE/f'{stem}-{s:04d}-r{r}.csv').exists() for stem in ('tip-balance','tip-points') for r in range(2))]
    result=dict(scope='Four midpoint half-steps from the shared accepted .060 s parent; not a full-from-zero convergence test.',coarse=[],half=[],comparisons=[])
    for directory,key,selected in ((BASE,'coarse',[12,13,14]),(FINE,'half',steps)):
        previous_step=12
        for step in selected:
            cache=OUT/f'{key}-{step:04d}.json'
            previous_rows=parent_rows;previous_points=parent_points
            if not cache.exists() and previous_step!=12:
                previous_rows=native(directory,previous_step)
                previous_points=gather(directory,f'tip-points-{previous_step:04d}-r',['cell','q'])
            item=reduce_state(directory,step,initial_rows,seed,parent_rows,parent_points,previous_rows,previous_points)
            if key=='half':assert abs(item['dt']-.0025)<1e-14 and abs(item['time']-(.06+(step-12)*.0025))<1e-12
            result[key].append(item)
            previous_step=step
    for c,h in ((13,14),(14,16)):
        if h in steps:
            cache=OUT/f'comparison-{h:04d}.json'
            if not cache.exists():cache.write_text(json.dumps(compare_endpoint(c,h,parent_points),indent=2)+'\n')
            result['comparisons'].append(json.loads(cache.read_text()))
    for key in ('coarse','half'):
        records=[s for s in result[key] if s['time']>.060000001]
        result[key+'_window_work']=dict(signed=sum(s['physical_work_error'] for s in records),accumulated_absolute=sum(abs(s['physical_work_error']) for s in records))
    inherited=read(BASE/'bulk-accepted.csv')
    inherited=inherited[inherited.step<=12]
    errors=inherited.Wext-inherited.energy_change-inherited.Db-inherited.Dc-inherited.Df
    result['inherited_work_to_parent']=dict(signed=float(sum(errors)),accumulated_absolute=float(sum(abs(errors))))
    for key in ('coarse','half'):
        result[key+'_full_work']={n:result['inherited_work_to_parent'][n]+result[key+'_window_work'][n] for n in ('signed','accumulated_absolute')}
    result['cost']={}
    for key,directory,selected in (('coarse',BASE,[13,14]),('half',FINE,steps)):
        profile=read(directory/'tip-profile.csv')
        rows=[{n:float(r[n]) for n in profile.dtype.names} for r in profile if int(r.step) in selected]
        result['cost'][key]=dict(steps=rows,solver_seconds=sum(r['step_seconds'] for r in rows),
                                line_trials=sum(r['line_trials'] for r in rows),
                                stokes_iterations=sum(r['stokes_iterations'] for r in rows))
    result['command']=json.loads((FINE.parent/'command.json').read_text())
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    records=[]
    for key in ('coarse','half'):
        for s in result[key]:
            row=dict(case=key,step=s['step'],time=s['time'],dt=s['dt'],W_over_hn=s['width'][0]['W_over_hn'])
            for n in ('window_added_fracture','added_fracture','physical_work_error','integrated_V','integrated_slip','positive_w','negative_w'):row[n]=s[n]
            for region in PRIMARY+['other']:
                for n,v in s['regions'][region].items():row[region+'_'+n]=v
            records.append(row)
    dump_csv(OUT/'history.csv',records)
    return result


def plot(result):
    """Native phase profiles and cellwise contours, without smoothing DG jumps."""
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    initial_rows=native(BASE,1);seed=gather(BASE,'tip-seed-fracture-r',['cell','i'])
    initial=RectBernstein(initial_rows,HX,HY,seed.B)
    initial_segments,initial_bounds=contour_segments(initial,32)
    contours=[];profiles=[];polys={}
    for key,directory in (('coarse',BASE),('half',FINE)):
        for s in result[key]:
            poly=field(native(directory,s['step']));polys[key,s['step']]=poly
            segments,bounds=contour_segments(poly,32);_,coarser=contour_segments(poly,16)
            record=dict(case=key,step=s['step'],time=s['time'],**bounds,
                        extraction_xmax_difference=abs(bounds['x_max']-coarser['x_max']),
                        extraction_ymax_difference=abs(bounds['y_max']-coarser['y_max']))
            for cut in s['contours']:
                suffix=str(cut['trace'])
                for k,v in cut.items():
                    if k!='trace':record[k+'_trace'+suffix]=v
            contours.append(record)
            for side in (-2,-1,1,2):
                b=line_coefficients(poly,side);b0=line_coefficients(initial,side)
                for cell in range(84,104):
                    for u in np.linspace(0,1,33):
                        p0=float(value(b0[cell],u));p=float(value(b[cell],u))
                        profiles.append(dict(case=key,step=s['step'],time=s['time'],trace=side,cell=cell,x=(cell+u)*HX,phi0=p0,phi=p,delta_phi=p-p0))
    dump_csv(OUT/'absolute-contours.csv',contours);dump_csv(OUT/'endpoint-profiles.csv',profiles)
    (OUT/'initial-contour.json').write_text(json.dumps(initial_bounds,indent=2)+'\n')
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    def draw_profile(ax,key,step,dt):
        selected=[p for p in profiles if p['case']==key and p['step']==step and p['trace']==1]
        x=[p['x'] for p in selected]
        for n,label,color,style in [('phi0',r'$\phi_0$ (original seed)','black','--'),('phi',r'$\phi$','C0','-'),('delta_phi',r'$\Delta\phi=\phi-\phi_0$','C1','-')]:
            ax.plot(x,[p[n] for p in selected],style,color=color,label=label,lw=1.4)
        ax.axhline(.05,color='purple',ls=':',lw=1,label=r'absolute $\phi=0.05$')
        ax.axvline(6,color='.5',ls='--',lw=1,label='original endpoint')
        ax.set(xlim=(5.75,6.25),ylim=(-.001,.14),xlabel='x (m)',ylabel='Phase / increment',title=f't={selected[0]["time"]:.4f} s; dt={dt:g} s')
        ax.grid(alpha=.2)
    if result['half']:
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for ax in axes.flat:ax.set_visible(False)
        for ax,s in zip(axes.flat,result['half']):
            ax.set_visible(True);draw_profile(ax,'half',s['step'],.0025)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Four half-steps: initial phase, current phase and change together\nOriginal seed reference; native upper trace at y=2')
        save(fig,'halfstep-profiles')
    paired=[c for c in result['comparisons']]
    if paired:
        fig,axes=plt.subplots(len(paired),2,figsize=(11,3.5*len(paired)),squeeze=False,constrained_layout=True)
        for row,c in enumerate(paired):
            for col,(key,step,dt) in enumerate((('coarse',c['coarse_step'],.005),('half',c['half_step'],.0025))):
                ax=axes[row,col];draw_profile(ax,key,step,dt)
                if row==0 and col==0:ax.legend(fontsize=8)
        fig.suptitle('Right endpoint: initial phase, current phase and change together\nNative upper trace at y=2; jumps retained')
        save(fig,'endpoint-profiles')
        for comparison in paired:
            c,h=comparison['coarse_step'],comparison['half_step'];time=.06+(h-12)*.0025
            fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
            for row,(key,step,dt) in enumerate((('coarse',c,.005),('half',h,.0025))):
                poly=polys[key,step];segments,_=contour_segments(poly,32)
                u=(np.arange(12)+.5)/12;edges=np.linspace(0,1,13);basis=np.array([(1-u)**2,2*u*(1-u),u*u]).T
                for col,title in enumerate((r'$\phi_0$',r'$\phi$',r'$\Delta\phi$')):
                    ax=axes[row,col];maximum=.65 if col<2 else .04
                    for iy in range(55,73):
                        for ix in range(84,103):
                            b0=initial.coefficients[iy,ix];b=poly.coefficients[iy,ix]
                            z=basis@(b0 if col==0 else b if col==1 else b-b0)@basis.T
                            im=ax.pcolormesh((ix+edges)*HX,(iy+edges)*HY,z,cmap='viridis' if col<2 else 'magma',vmin=0,vmax=maximum,rasterized=True)
                    ax.add_collection(LineCollection(initial_segments,colors='white',linestyles='--',linewidths=.8))
                    if col:ax.add_collection(LineCollection(segments,colors='cyan',linewidths=1.1))
                    ax.axvline(6,color='.7',ls=':',lw=1)
                    ax.set(xlim=(5.25,6.4375),ylim=(1.71875,2.28125),aspect='equal',xlabel='x (m)',title=title)
                    if col==0:ax.set_ylabel(f'dt={dt:g} s\ny (m)')
                    fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle(f'Right endpoint at t={time:.3f} s\nWhite dashed: initial phi=0.05; cyan: current absolute phi=0.05')
            save(fig,f'phase-fields-{h:04d}')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for key,marker,label in (('coarse','o','dt=.005 s'),('half','s','dt=.0025 s')):
        rr=[c for c in contours if c['case']==key];ss=result[key]
        color='C0' if key=='coarse' else 'C1'
        axes[0,0].plot([c['time'] for c in rr],[c['absolute_right_trace1'] for c in rr],marker+'-',color=color,label=label+' axis')
        axes[0,0].plot([c['time'] for c in rr],[c['x_max'] for c in rr],marker+':',color=color,label=label+' 2D extent')
        axes[0,1].plot([s['time'] for s in ss],[s['width'][0]['W_over_hn'] for s in ss],marker+'-',label=label)
        after=[s for s in ss if s['time']>.060000001];t=[.06]+[s['time'] for s in after]
        axes[1,0].plot(t,[0]+[s['window_added_fracture'] for s in after],marker+'-',label=label+' added fracture')
        axes[1,1].plot(t,np.r_[0,np.cumsum([s['physical_work_error'] for s in after])],marker+'-',color=color,label=label+' signed')
        axes[1,1].plot(t,np.r_[0,np.cumsum([abs(s['physical_work_error']) for s in after])],marker+':',color=color,label=label+' absolute')
    axes[0,0].axhline(6,color='.5',ls='--',label='original endpoint')
    axes[0,0].axhline(initial_bounds['x_max'],color='.6',ls=':',label='initial 0.05 extent')
    axes[0,0].set(ylabel='Absolute phi=0.05 right extent (m)',title='Contour tracking')
    axes[0,1].axhline(2,color='.5',ls='--');axes[0,1].set(ylabel=r'$W/h_n$',title='Central excess-compliance width')
    axes[1,0].set(ylabel='Added fracture since .060 s (J/m)',title='Matched-parent energy growth')
    axes[1,1].set(ylabel='Work remainder since .060 s (J/m)',title='Physical-work accounting')
    for ax in axes.flat:ax.set_xlabel('Time (s)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,'contour-energy-history')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    verify_contours()
    if args.restore:verify_restore()
    result=analyze(args.live);plot(result)
    print(json.dumps(dict(accepted_half_steps=len(result['half']),comparisons=result['comparisons'],coarse_window_work=result['coarse_window_work'],half_window_work=result['half_window_work']),indent=2))
