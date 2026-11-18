"""Common-time native observations for the selected overnight A/B/C comparisons."""
import argparse
import json
import os
import re
from pathlib import Path
from analyze_resolution60_growth import ROOT,RUNS,read,gather,masks,RectBernstein,pulse,intervals,value,PRIMARY
from analyze_resolution60_window import contour_segments,energy,dump_csv
import numpy as np

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight'
CASES={
 'base':('resolution60-coarse-growth',.0625,.005,None,0),
 'half':('resolution60-halfstep-window',.0625,.0025,'base',12),
 'A':('resolution60-quarter-window',.0625,.00125,'base',12),
 'B':('resolution60-axial-growth',.03125,.005,None,0),
 'C':('resolution60-halfstep-late-extended',.0625,.0025,'half',16),
 'Afinish':('resolution60-quarter-final',.0625,.00125,'A',19),
}
HY=.03125


def directory(key):return RUNS/CASES[key][0]/'output'
def native(key,step):return gather(directory(key),f'tip-balance-{step:04d}-r',['cell','i'])
def poly(key,step):return RectBernstein(native(key,step),CASES[key][1],HY)
def seed_key(key):return 'B' if key=='B' else 'base'
def initial(key):
    k=seed_key(key);rows=native(k,1);seed=gather(directory(k),'tip-seed-fracture-r',['cell','i'])
    assert np.array_equal(rows.cell,seed.cell) and np.array_equal(rows.i,seed.i)
    return rows,seed,RectBernstein(rows,CASES[k][1],HY,seed.B)


def lines(field,seed):
    result=[]
    for side in (-2,-1,1,2):
        iy=round(2/HY)-(1 if side<0 else 0)
        w=np.array([.25,.5,.25]) if abs(side)==2 else np.array([0,0,1] if side<0 else [1,0,0])
        b=np.einsum('j,xji->xi',w,field.coefficients[iy]);b0=np.einsum('j,xji->xi',w,seed.coefficients[iy])
        absolute=next(p for p in intervals(b,field.hx,.05,strict=True) if p[0]<4<p[1])
        original=next(p for p in intervals(b0,field.hx,.05,strict=True) if p[0]<4<p[1])
        damage=intervals(b-b0,field.hx,1e-4,strict=True)
        for end,index,sign in ((2.,0,-1),(6.,1,1)):
            # The increment component must extend from inside the old taper through its end.
            component=next((p for p in damage if p[0]<end<p[1]),None)
            extension=sign*(component[index]-end) if component else 0.
            result.append(dict(trace=side,original_end=end,absolute_position=absolute[index],initial_position=original[index],
                               advance=sign*(absolute[index]-original[index]),signed_beyond_support=sign*(absolute[index]-end),
                               advance_over_hx=sign*(absolute[index]-original[index])/field.hx,
                               beyond_over_hx=sign*(absolute[index]-end)/field.hx,increment_extension=extension,increment_extension_over_hx=extension/field.hx))
    return result


def accepted_steps(key,live):
    p=directory(key)/'bulk-accepted.csv'
    if not p.exists():return []
    rows=read(p);rows=rows[rows.step>CASES[key][4]]
    command=json.loads((directory(key).parent/'command.json').read_text())
    if live:
        banners=re.findall(r'\*\*\* Timestep (\d+):',(directory(key).parent/'run.log').read_text())
        last=int(banners[-1]) if banners else 0
        if 'exit_code' not in command:rows=rows[rows.step<last]
        rows=np.array([r for r in rows if all((directory(key)/f'{stem}-{int(r.step):04d}-r{rank}.csv.gz').exists() and not (directory(key)/f'{stem}-{int(r.step):04d}-r{rank}.csv').exists() for stem in ('tip-points','tip-balance') for rank in range(2))],dtype=rows.dtype).view(np.recarray)
    return rows


def reduce_case(key,live):
    accepted=accepted_steps(key,live)
    if not len(accepted):return []
    r0,s0,p0=initial(key);e0=.5*s0.B*(s0.fracture_gradient+2*r0.mass)
    hx=CASES[key][1];parent,parent_step=CASES[key][3:]
    if parent:
        previous=native(parent,parent_step);previous_e=energy(previous);previous_poly=RectBernstein(previous,hx,HY)
        parent_energy=float(sum(previous_e))
    else:previous=None;previous_e=e0;previous_poly=p0;parent_energy=float(sum(e0))
    records=[]
    for a in accepted:
        step=int(a.step);cache=OUT/f'{key}-{step:04d}.json';nr=native(key,step);p=RectBernstein(nr,hx,HY);e=energy(nr)
        assert max(abs(nr.mass-hx*HY/9))<1e-14
        assert min(nr.B)>-2e-12 and max(nr.B)<1+2e-12
        assert min(nr.B-(s0.B if previous is None else previous.B))>=-2e-12
        assert abs(sum(e)-a.fracture)<2e-9
        reference_cache=(OUT.parent/'resolution60_free_growth'/f'step-{step:04d}.json' if key=='base'
                         else OUT.parent/'resolution60_halfstep_window'/f'half-{step:04d}.json' if key=='half' else OUT/'unused')
        if cache.exists():record=json.loads(cache.read_text())
        elif reference_cache.exists():
            record=json.loads(reference_cache.read_text())
            record['regions']={n:{q:(r['max_step_delta_phi'] if q=='max_step_increment' and q not in r else r[q])
                                  for q in ('max_delta_phi','max_step_increment','phase_increment_integral','added_fracture','step_added_fracture')}
                               for n,r in record['regions'].items()}
            record.update(case=key,dt=float(a.dt),hx=hx,hy=HY,window_added_fracture=float(sum(e)-parent_energy),contours=lines(p,p0),
                          reduction_reference=str(reference_cache.relative_to(ROOT)))
            cache.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        else:
            points=gather(directory(key),f'tip-points-{step:04d}-r',['cell','q'])
            assert abs(np.sum(points.weight)-32)<1e-11
            assert max(abs(p.evaluate(points.x,points.y)-points.phi))<2e-12
            nm,pm=masks(nr.x,nr.y),masks(points.x,points.y)
            nm['other']=~np.logical_or.reduce([nm[n] for n in PRIMARY]);pm['other']=~np.logical_or.reduce([pm[n] for n in PRIMARY])
            d=points.phi-p0.evaluate(points.x,points.y);inc=points.phi-previous_poly.evaluate(points.x,points.y)
            regions={n:dict(max_delta_phi=float(max(d[pm[n]])),max_step_increment=float(max(inc[pm[n]])),
                            phase_increment_integral=float(sum(points.weight[pm[n]]*d[pm[n]])),
                            added_fracture=float(sum(e[m]-e0[m])),step_added_fracture=float(sum(e[m]-previous_e[m]))) for n,m in nm.items()}
            assert abs(sum(regions[n]['added_fracture'] for n in PRIMARY+['other'])-sum(e-e0))<1e-10
            sections=read(directory(key)/f'tip-sections-{step:04d}.csv');index=points.section.astype(int)
            power=float(sum(points.weight*points.w*sections.force[index])-sum(sections.ds*sections.V*sections.force))
            assert abs(power)<1e-9
            assert max(abs(np.bincount(index,weights=points.weight*points.w)/sections.ds-sections.V))<1e-12
            rates=points.w[np.lexsort((points.y,points.section))].reshape(len(sections),-1)
            threshold=np.maximum(1e-10,1e-6*np.max(abs(rates),axis=1))[:,None]
            active=abs(rates)>threshold
            starts=active&~np.column_stack([np.zeros(len(sections),dtype=bool),active[:,:-1]])
            local_power=points.weight*points.w*sections.force[index]
            integral_abs=float(np.sum(sections.ds*sections.integral_abs_w))
            record=dict(case=key,step=step,time=float(a.time),dt=float(a.dt),hx=hx,hy=HY,regions=regions,
                        added_fracture=float(sum(e-e0)),window_added_fracture=float(sum(e)-parent_energy),
                        physical_work_error=float(a.Wext-a.energy_change-a.Db-a.Dc-a.Df),width=pulse(p),contours=lines(p,p0),
                        integrated_slip=float(sum(sections.ds*sections.slip)),integrated_V=float(sum(sections.ds*sections.V)),
                        positive_w=float(sum(sections.ds*sections.positive_w)),negative_w=float(sum(sections.ds*sections.negative_w)),
                        opposite_sign_sections=int(sum(np.any(rates>threshold,axis=1)&np.any(rates < -threshold,axis=1))),
                        multiple_band_sections=int(sum(np.sum(starts,axis=1)>1)),
                        cancellation=1-float(np.sum(sections.ds*abs(sections.V)))/integral_abs if integral_abs else 0.,
                        negative_local_power=float(np.sum(local_power[local_power<0])),
                        negative_local_work=float(a.dt*np.sum(local_power[local_power<0])),
                        power_adjoint_error=power,accepted={n:float(a[n]) for n in a.dtype.names})
            cache.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        record['contours']=lines(p,p0)
        records.append(record);previous=nr;previous_e=e;previous_poly=p
    return records


def step_at(key,time):
    p=directory(key)/'bulk-accepted.csv'
    if not p.exists():return None
    a=read(p);match=a.step[abs(a.time-time)<1e-11]
    return int(match[0]) if len(match) else None


def sections_on(key,step,x,quantity):
    """Piecewise P5 interpolation of the six native section quadrature values."""
    a=read(directory(key)/f'tip-sections-{step:04d}.csv');hx=CASES[key][1];nx=round(8/hx)
    assert len(a)==6*nx and np.all(np.diff(a.s)>0)
    nodes=np.polynomial.legendre.leggauss(6)[0];values=a[quantity].reshape(nx,6)
    expected=(np.arange(nx)[:,None]+(nodes+1)/2)*hx;assert max(abs(expected.ravel()-a.s))<1e-12
    cells=np.minimum((x/hx).astype(int),nx-1);u=2*(x/hx-cells)-1
    # Fixed cardinal polynomials avoid stochastic barycentric weights and repeated fits.
    cardinal=np.array([np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyfromroots(np.delete(nodes,i)))/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(6)]).T
    return np.sum(cardinal*values[cells],axis=1)


def interval_rate(key,end,duration,x):
    rows=[(k,r) for k,r in history(key) if end-duration+1e-11<r.time<=end+1e-11]
    assert abs(sum(r.dt for k,r in rows)-duration)<1e-11
    return sum(float(r.dt)*sections_on(k,int(r.step),x,'V') for k,r in rows)/duration


def history(key):
    parent,step=CASES[key][3:]
    before=[(k,r) for k,r in history(parent) if r.step<=step] if parent else []
    return before+[(key,r) for r in read(directory(key)/'bulk-accepted.csv')]


def compare(keys,time,start,duration):
    cache=OUT/f'compare-{"-".join(keys)}-{time:.5f}.json'
    if cache.exists():return json.loads(cache.read_text())
    steps=[step_at(k,time) for k in keys]
    if any(s is None for s in steps):return None
    # The finest axial mesh partitions every native field. Q6 integrates squared Q2 exactly.
    hx=min(CASES[k][1] for k in keys);z,w=np.polynomial.legendre.leggauss(6)
    xx=(np.arange(round(8/hx))[:,None]+(z+1)/2)*hx;yy=(np.arange(128)[:,None]+(z+1)/2)*HY
    x,y=np.meshgrid(xx.ravel(),yy.ravel());weight=np.outer(np.tile(w,128)*HY/2,np.tile(w,round(8/hx))*hx/2)
    absolute=[poly(k,s).evaluate(x,y) for k,s in zip(keys,steps)]
    seeds=[initial(k)[2].evaluate(x,y) for k in keys]
    increments=[p-s for p,s in zip(absolute,seeds)]
    if start==0:reference=increments[-1]
    else:
        parent='half' if keys[-1]=='C' else 'base'
        reference=absolute[-1]-poly(parent,step_at(parent,start)).evaluate(x,y)
    norm=lambda f:float(np.sqrt(np.sum(weight*f*f)))
    denom=norm(reference);pairs=[]
    sx=xx.ravel();sw=np.tile(w,round(8/hx))*hx/2
    snorm=lambda f:float(np.sqrt(sum(sw*f*f)))
    for j in range(len(keys)-1):
        phase=norm(absolute[j]-absolute[j+1]);growth=norm(increments[j]-increments[j+1]);ss={}
        for quantity in ('slip','age'):
            vals=[sections_on(k,s,sx,quantity) for k,s in zip(keys,steps)]
            ss[quantity]=dict(absolute_L2=snorm(vals[j]-vals[j+1]),relative_L2=snorm(vals[j]-vals[j+1])/snorm(vals[-1]))
        vals=[interval_rate(k,time,duration,sx) for k in keys]
        ss['interval_mean_V']=dict(absolute_L2=snorm(vals[j]-vals[j+1]),relative_L2=snorm(vals[j]-vals[j+1])/snorm(vals[-1]))
        pairs.append(dict(pair=[keys[j],keys[j+1]],phase_absolute_L2=phase,growth_absolute_L2=growth,
                          phase_relative_common=phase/denom if denom else None,growth_relative_common=growth/denom if denom else None,
                          sections=ss))
    result=dict(keys=keys,time=time,start=start,matched_rate_interval=duration,common_phase_denominator=denom,pairs=pairs,
                fine_pair_ratio=pairs[1]['phase_absolute_L2']/pairs[0]['phase_absolute_L2'] if len(pairs)==2 else None)
    cache.write_text(json.dumps(result,indent=2)+'\n');return result


def verify_restore(key):
    parent,step=CASES[key][3:]
    a=gather(directory(parent),f'tip-points-{step:04d}-r',['cell','q']);b=gather(directory(key),'restore-points-r',['cell','q'])
    assert np.array_equal(a.cell,b.cell) and np.array_equal(a.q,b.q)
    errors={n:float(max(abs(a[n]-b[n]))) for n in b.dtype.names if n!='cell'};assert max(errors.values())==0
    a=native(parent,step);b=gather(directory(key),'restore-phase-r',['cell','i']);assert max(abs(a.B-b.B))==0 and max(abs(b.B-b.solution_B))==0
    a=read(directory(parent)/f'tip-sections-{step:04d}.csv');b=read(directory(key)/'restore-sections.csv');assert all(max(abs(a[n]-b[n]))==0 for n in b.dtype.names)
    result=dict(parent=parent,step=step,maximum_differences=errors,phase_max_difference=0,section_max_difference=0)
    (OUT/f'{key}-restore.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def seed_comparison():
    cache=OUT/'seed-comparison.json'
    if cache.exists():return json.loads(cache.read_text())
    r,s,p=initial('base');rr,ss,pp=initial('B')
    z,w=np.polynomial.legendre.leggauss(6)
    x,y=np.meshgrid(((np.arange(256)[:,None]+(z+1)/2)*pp.hx).ravel(),((np.arange(128)[:,None]+(z+1)/2)*HY).ravel())
    weights=np.outer(np.tile(w,128)*HY/2,np.tile(w,256)*pp.hx/2)
    a,b=p.evaluate(x,y),pp.evaluate(x,y)
    result=dict(coarse_hx=p.hx,refined_hx=pp.hx,hy=HY,common_measure=float(weights.sum()),
                maximum_phase_difference=float(max(abs(a-b).ravel())),L2_difference=float(np.sqrt(sum((weights*(a-b)**2).ravel()))),
                relative_L2_difference=float(np.sqrt(np.sum(weights*(a-b)**2)/np.sum(weights*b*b))),
                coarse_fracture=float(sum(.5*s.B*(s.fracture_gradient+2*r.mass))),
                refined_fracture=float(sum(.5*ss.B*(ss.fracture_gradient+2*rr.mass))),
                coarse_contours=lines(p,p),refined_contours=lines(pp,pp),coarse_width=pulse(p),refined_width=pulse(pp))
    assert abs(weights.sum()-32)<1e-12
    cache.write_text(json.dumps(result,indent=2)+'\n');return result


def matched_balance(key,time,start):
    rows=[r for k,r in history(key)];end=next(r for r in rows if abs(r.time-time)<1e-11)
    window=[r for r in rows if start+1e-11<r.time<=time+1e-11]
    errors=np.array([r.Wext-r.energy_change-r.Db-r.Dc-r.Df for r in window])
    if start==0:
        r,s,_=initial(key);start_energy=float(sum(.5*s.B*(s.fracture_gradient+2*r.mass)))
    else:
        start_energy=float(next(r.fracture for r in rows if abs(r.time-start)<1e-11))
    added=float(end.fracture-start_energy)
    return dict(start=start,end=time,added_fracture=added,signed_work=float(sum(errors)),absolute_work=float(sum(abs(errors))),
                signed_relative_to_added=float(sum(errors)/added) if added else None,
                absolute_relative_to_added=float(sum(abs(errors))/added) if added else None)


def main(live=False):
    result=dict(cases={},comparisons=[])
    for key in CASES:
        if not (directory(key)/'bulk-accepted.csv').exists():continue
        result['cases'][key]=reduce_case(key,live)
        if key in ('A','C','Afinish') and not (OUT/f'{key}-restore.json').exists():verify_restore(key)
    onset=[(['base','half','Afinish' if step_at('A',t) is None and step_at('Afinish',t) is not None else 'A'],t,.06,.005) for t in (.065,.07)]
    for keys,time,start,duration in onset+[(['base','B'],t,0,.005) for t in (.065,.07,.075,.08)]+[(['base','C'],t,.07,.005) for t in (.075,.08)]:
        if all(k in result['cases'] and any(abs(s['time']-time)<1e-11 for s in result['cases'][k]) for k in keys):
            record=compare(keys,time,start,duration)
            if record:
                record['balances']={k:matched_balance(k,time,start) for k in keys}
                result['comparisons'].append(record)
    if 'B' in result['cases'] and result['cases']['B']:result['seed_comparison']=seed_comparison()
    for key,states in result['cases'].items():
        for s in states:
            preceding=[r for r in states if r['time']<=s['time']+1e-12]
            s['signed_window_work']=sum(r['physical_work_error'] for r in preceding)
            s['absolute_window_work']=sum(abs(r['physical_work_error']) for r in preceding)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    flat=[];contours=[]
    for key,states in result['cases'].items():
        for s in states:
            r={n:s[n] for n in ('case','step','time','hx','hy','added_fracture','window_added_fracture','physical_work_error','signed_window_work','absolute_window_work','integrated_slip','integrated_V','positive_w','negative_w')}
            r['W_over_hy']=s['width'][0]['W_over_hn']
            for n in PRIMARY+['other']:
                for q,v in s['regions'][n].items():r[n+'_'+q]=v
            flat.append(r)
            contours.extend(dict(case=key,step=s['step'],time=s['time'],**c) for c in s['contours'])
    dump_csv(OUT/'history.csv',flat);dump_csv(OUT/'contours.csv',contours)
    table=[]
    for c in result['comparisons']:
        for p in c['pairs']:
            table.append(dict(time=c['time'],pair='/'.join(p['pair']),normalization_start=c['start'],common_phase_denominator=c['common_phase_denominator'],rate_interval=c['matched_rate_interval'],phase_L2=p['phase_absolute_L2'],phase_relative_common=p['phase_relative_common'],growth_L2=p['growth_absolute_L2'],growth_relative_common=p['growth_relative_common'],fine_pair_ratio=c['fine_pair_ratio'],slip_relative=p['sections']['slip']['relative_L2'],age_relative=p['sections']['age']['relative_L2'],interval_V_relative=p['sections']['interval_mean_V']['relative_L2']))
    if table:dump_csv(OUT/'common-time-comparison.csv',table)
    print(json.dumps(dict(completed={k:[s['time'] for s in states] for k,states in result['cases'].items()},comparisons=result['comparisons']),indent=2))
    return result



def seed_component(p,subdivisions=32):
    """Cell graph with exact overlapping positive face traces, seeded at (4,2).

    Each candidate cell must have at most one sampled positive component. This
    guards the cell-graph topology; subdividing 16/32 checks the observation.
    """
    from collections import deque
    from scipy.ndimage import label
    c=p.coefficients;ny,nx=c.shape[:2];candidate=c.max(axis=(2,3))>.05
    u=np.linspace(0,1,subdivisions+1);b=np.array([(1-u)**2,2*u*(1-u),u*u]).T
    active=np.zeros_like(candidate)
    for iy,ix in np.argwhere(candidate):
        _,n=label(b@c[iy,ix]@b.T>.05)
        assert n<=1,('Multiple intra-cell contour components require a finer topology check',iy,ix,n)
        active[iy,ix]=n==1
    connected=np.zeros_like(active);queue=deque()
    for iy in (round(2/p.hy)-1,round(2/p.hy)):
        for ix in (round(4/p.hx)-1,round(4/p.hx)):
            assert active[iy,ix];connected[iy,ix]=True;queue.append((iy,ix))
    def overlap(a,b):
        ia=intervals(np.array([a]),1,.05,strict=True);ib=intervals(np.array([b]),1,.05,strict=True)
        return any(min(x[1],y[1])-max(x[0],y[0])>1e-13 for x in ia for y in ib)
    while queue:
        iy,ix=queue.popleft()
        for dy,dx in ((0,-1),(0,1),(-1,0),(1,0)):
            jy,jx=iy+dy,ix+dx
            if not (0<=jy<ny and 0<=jx<nx) or connected[jy,jx] or not active[jy,jx]:continue
            if dx:a,b=c[iy,ix,:,2 if dx>0 else 0],c[jy,jx,:,0 if dx>0 else 2]
            else:a,b=c[iy,ix,2 if dy>0 else 0,:],c[jy,jx,0 if dy>0 else 2,:]
            if overlap(a,b):connected[jy,jx]=True;queue.append((jy,jx))
    return connected,dict(active_cells=int(active.sum()),seed_connected_cells=int(connected.sum()),disconnected_cells=int((active&~connected).sum()))


def plot(result):
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    display_cases={k:states for k,states in result['cases'].items() if k!='Afinish'}
    if result['cases'].get('Afinish'):
        display_cases['A']=sorted(display_cases.get('A',[])+result['cases']['Afinish'],key=lambda s:s['time'])
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    # Profiles include both endpoints, both DG traces and half-normal-cell cuts.
    profiles=[];topology=[]
    for key,states in result['cases'].items():
        _,_,p0=initial(key)
        for s in states:
            if s['time']<.06-1e-12:continue
            p=poly(key,s['step'])
            for side in (-2,-1,1,2):
                iy=round(2/HY)-(1 if side<0 else 0);w=np.array([.25,.5,.25]) if abs(side)==2 else np.array([0,0,1] if side<0 else [1,0,0])
                b=np.einsum('j,xji->xi',w,p.coefficients[iy]);b0=np.einsum('j,xji->xi',w,p0.coefficients[iy])
                for endpoint in (2,6):
                    for ix in range(round((endpoint-.5)/p.hx),round((endpoint+.5)/p.hx)):
                        for u in np.linspace(0,1,17):
                            initial_phi=float(value(b0[ix],u));phi=float(value(b[ix],u))
                            profiles.append(dict(case=key,step=s['step'],time=s['time'],trace=side,endpoint=endpoint,x=(ix+u)*p.hx,phi0=initial_phi,phi=phi,delta_phi=phi-initial_phi))
    dump_csv(OUT/'endpoint-profiles.csv',profiles)
    for name,keys,times in [('timestep-onset',['base','half','A'],[.065,.07]),('axial-mesh',['base','B'],[.07,.075,.08]),('late-timestep',['base','C'],[.075,.08])]:
        def shown(k,t):
            return 'Afinish' if k=='A' and step_at('A',t) is None and step_at('Afinish',t) is not None else k
        available=[t for t in times if all(shown(k,t) in result['cases'] and any(abs(s['time']-t)<1e-11 for s in result['cases'][shown(k,t)]) for k in keys)]
        if not available:continue
        fig,axes=plt.subplots(len(available),len(keys),figsize=(4.2*len(keys),3*len(available)),squeeze=False,layout='constrained')
        for row,time in enumerate(available):
            for col,key in enumerate(keys):
                key=shown(key,time)
                rr=[p for p in profiles if p['case']==key and abs(p['time']-time)<1e-11 and p['trace']==1 and p['endpoint']==6];ax=axes[row,col]
                for field,label,color,style in [('phi0',r'$\phi_0$','black','--'),('phi',r'$\phi$','C0','-'),('delta_phi',r'$\Delta\phi$','C1','-')]:ax.plot([p['x'] for p in rr],[p[field] for p in rr],style,color=color,label=label)
                ax.axhline(.05,color='purple',ls=':',label='absolute 0.05');ax.axvline(6,color='.5',ls='--')
                label='A' if key=='Afinish' else key
                ax.set(xlim=(5.75,6.5),ylim=(-.002,.22),title=f'{label}: t={time:g}; dt={CASES[key][2]:g}; hx={CASES[key][1]:g}',xlabel='x (m)',ylabel='Phase / increase');ax.grid(alpha=.2)
                if row==0 and col==0:ax.legend(fontsize=8)
        fig.suptitle('Original seed, absolute phase and increase across right endpoint; native upper trace')
        save(fig,name+'-profiles')
    # The left/right agreement is checked explicitly, not inferred from symmetry.
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for key,states in display_cases.items():
        for col,end in enumerate((2,6)):
            selected=[(s,next(c for c in s['contours'] if c['trace']==1 and c['original_end']==end)) for s in states if s['time']>=.06-1e-12]
            axes[0,col].plot([s['time'] for s,c in selected],[c['signed_beyond_support'] for s,c in selected],'.-',label=key)
            axes[1,col].plot([s['time'] for s,c in selected],[c['increment_extension'] for s,c in selected],'.-',label=key)
        for col in (0,1):axes[0,col].axhline(0,color='.5',ls='--')
    for row in range(2):
        for col,end in enumerate((2,6)):
            axes[row,col].set(title=f'{"Absolute phi=0.05 on upper axis trace" if row==0 else "Connected delta phi=1e-4"}; x={end}',xlabel='Time (s)',ylabel='Signed beyond support (m)' if row==0 else 'Extension (m)');axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.2)
    save(fig,'front-tracking')
    # Selected late maps carry the seed-connected absolute contour.
    for key in ('base','B','C'):
        if key not in result['cases']:continue
        states=[s for s in result['cases'][key] if s['time']>=.07-1e-12]
        if not states:continue
        _,_,p0=initial(key)
        seedmask,_=seed_component(p0,32);seedseg,seedbounds=contour_segments(p0,32,seedmask)
        for s in states:
            p=poly(key,s['step']);masks_by_order={n:seed_component(p,n) for n in (16,32)}
            assert np.array_equal(masks_by_order[16][0],masks_by_order[32][0])
            seg,bounds=contour_segments(p,32,masks_by_order[32][0]);_,bounds16=contour_segments(p,16,masks_by_order[16][0])
            topology.append(dict(case=key,time=s['time'],**masks_by_order[32][1],bounds=bounds,initial_bounds=seedbounds,
                                 right_advance=bounds['x_max']-seedbounds['x_max'],left_advance=seedbounds['x_min']-bounds['x_min'],
                                 right_beyond=bounds['x_max']-6,left_beyond=2-bounds['x_min'],
                                 right_advance_over_hx=(bounds['x_max']-seedbounds['x_max'])/p.hx,
                                 right_beyond_over_hx=(bounds['x_max']-6)/p.hx,
                                 extent_extraction_difference=abs(bounds['x_max']-bounds16['x_max'])))
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');u=(np.arange(8)+.5)/8;edges=np.linspace(0,1,9);b=np.array([(1-u)**2,2*u*(1-u),u*u]).T
        for col,title in enumerate((r'$\phi_0$',r'$\phi$',r'$\Delta\phi$')):
            ax=axes[col]
            for iy in range(54,74):
                for ix in range(round(5.25/p.hx),round(6.75/p.hx)):
                    z=b@(p0.coefficients[iy,ix] if col==0 else p.coefficients[iy,ix] if col==1 else p.coefficients[iy,ix]-p0.coefficients[iy,ix])@b.T
                    im=ax.pcolormesh((ix+edges)*p.hx,(iy+edges)*p.hy,z,cmap='viridis' if col<2 else 'magma',vmin=0,vmax=.65 if col<2 else .16,rasterized=True)
            ax.add_collection(LineCollection(seedseg,colors='white',linestyles='--',linewidths=.8))
            if col:ax.add_collection(LineCollection(seg,colors='cyan',linewidths=1))
            ax.axvline(6,color='.7',ls=':');ax.set(xlim=(5.25,6.75),ylim=(1.7,2.3),aspect='equal',title=title,xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,shrink=.65)
        fig.suptitle(f'{key}: t={s["time"]:.3f} s; cyan seed-connected absolute phi=0.05, white dashed initial')
        save(fig,key+'-phase-map')
    if topology:
        (OUT/'contour-topology.json').write_text(json.dumps(topology,indent=2)+'\n')
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for key in ('base','B','C'):
            rows=[r for r in topology if r['case']==key]
            if not rows:continue
            color={'base':'C0','B':'C1','C':'C2'}[key]
            axes[0].plot([r['time'] for r in rows],[r['right_advance'] for r in rows],'.-',color=color,label=key+' right')
            axes[0].plot([r['time'] for r in rows],[r['left_advance'] for r in rows],':',color=color,label=key+' left')
            axes[1].plot([r['time'] for r in rows],[r['right_beyond'] for r in rows],'.-',color=color,label=key+' right')
            axes[1].plot([r['time'] for r in rows],[r['left_beyond'] for r in rows],':',color=color,label=key+' left')
        axes[1].axhline(0,color='.5',ls='--')
        for ax,title in zip(axes,['Motion of seed-connected 2D absolute contour','Signed distance beyond original support']):ax.set(title=title,xlabel='Time (s)',ylabel='Distance (m)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        save(fig,'global-absolute-front')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for key,states in display_cases.items():
        t=[s['time'] for s in states]
        axes[0,0].plot(t,[s['regions']['axial_ahead']['added_fracture'] for s in states],'.-',label=key)
        axes[0,1].plot(t,[s['width'][0]['W_over_hn'] for s in states],'.-',label=key)
        axes[1,0].plot(t,[s['integrated_slip'] for s in states],'.-',label=key)
        origin='Afinish' if key=='A' and result['cases'].get('Afinish') else key
        work=[(float(r.time),float(r.Wext-r.energy_change-r.Db-r.Dc-r.Df)) for k,r in history(origin)]
        color={'base':'C0','half':'C1','A':'C2','B':'C3','C':'C4'}[key]
        axes[1,1].plot(t,[sum(e for time,e in work if time<=s['time']+1e-11) for s in states],'.-',color=color,label=key+' signed')
        axes[1,1].plot(t,[sum(abs(e) for time,e in work if time<=s['time']+1e-11) for s in states],':',color=color,label=key+' abs')
    for ax,title,ylabel in zip(axes.flat,['Axial added fracture from own initialization','Central excess-compliance width','Integrated cumulative slip','Physical-work remainder since t=0\nIncludes inherited parent history'],['J/m','W / hy','m²','J/m']):ax.set(title=title,ylabel=ylabel,xlabel='Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'energy-slip-width')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');parser.add_argument('--plots',action='store_true');a=parser.parse_args();result=main(a.live)
    if a.plots:plot(result)
