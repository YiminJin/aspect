"""Accepted peak-.60 free response: native profiles, regional energy and pulse width.

Caches closed accepted observations so --live never rereads large point exports.
All spatial reductions use actual rectangular cells and native Bernstein fields.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path
from analyze_resolution_matched import ROOT, RUNS, read, gather, masks, RectBernstein
import numpy as np

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth'
LABEL = 'resolution60-coarse-growth'
PRIMARY = ['central_band', 'old_tip_taper', 'axial_ahead', 'remote']


def value(b, u):
    return b[..., 0]*(1-u)**2+2*b[..., 1]*u*(1-u)+b[..., 2]*u*u


def intervals(coefficients, spacing, threshold, strict=False):
    """Exact Q2 superlevel intervals, retaining each cell's independent traces."""
    pieces = []
    previous_right_positive = False
    for cell, b in enumerate(coefficients):
        a, c, d = b[0]-2*b[1]+b[2], 2*(b[1]-b[0]), b[0]-threshold
        roots = np.roots(np.trim_zeros(np.array([a, c, d]), 'f')) if a or c else []
        cuts = [0.] + sorted(float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<1) + [1.]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if value(b, (lo+hi)/2) > threshold:
                bounds = [(cell+lo)*spacing, (cell+hi)*spacing]
                join = not strict or (lo == 0 and b[0] > threshold and previous_right_positive)
                if pieces and abs(bounds[0]-pieces[-1][1])<1e-12 and join:
                    pieces[-1][1] = bounds[1]
                else:
                    pieces.append(bounds)
        previous_right_positive = b[2] > threshold
    return pieces


def peak_of_line(b, spacing):
    candidates = []
    for i, row in enumerate(b):
        denominator = row[0]-2*row[1]+row[2]
        u = (row[0]-row[1])/denominator if denominator else -1
        for t in ([0., 1., u] if 0<u<1 else [0., 1.]):
            candidates.append((float(value(row,t)), (i+t)*spacing))
    return max(candidates)


def pulse(poly):
    results = []
    for side in (-1, 1):
        # Exact x=4 one-sided trace, with independent normal cell polynomials.
        ix = round(4/poly.hx)-(1 if side<0 else 0)
        line = poly.coefficients[:, ix, :, 2 if side<0 else 0]
        peak, location = peak_of_line(line, poly.hy)
        record = dict(x_trace=side, peak=peak, peak_y=location)
        if peak >= 1:
            record.update(W=None, W_over_hn=None, compliance_integral=None,
                          reason='Saturated phase: excess-compliance peak is unbounded.')
        else:
            c = 4*peak*(1+peak)/(1-peak)**2/2
            half_phi = c/(2+c+2*np.sqrt(1+2*c))
            pieces = intervals(line, poly.hy, half_phi)
            connected = [s for s in pieces if s[0]-1e-12<=location<=s[1]+1e-12]
            assert len(connected)==1
            width = connected[0][1]-connected[0][0]
            integrals = []
            for order in (24,48):
                z,w = np.polynomial.legendre.leggauss(order)
                phi = value(line[:,None,:],(z+1)/2)
                integrals.append(float(np.sum(4*phi*(1+phi)/(1-phi)**2*w)*poly.hy/2))
            record.update(W=width,W_over_hn=width/poly.hy,half_height_phi=float(half_phi),
                          half_height_components=pieces,compliance_integral=integrals[-1],
                          integration_difference=abs(integrals[1]-integrals[0]))
        results.append(record)
    return results


def endpoint_profiles(poly, initial, step):
    records, connectivity = [], []
    difference = poly.coefficients-initial.coefficients
    for side in (-2, -1, 1, 2):
        iy = round(2/poly.hy)-(1 if side<0 else 0)
        weights = np.array([.25,.5,.25]) if abs(side)==2 else np.array([0.,0.,1.] if side<0 else [1.,0.,0.])
        y = 2 + np.sign(side)*poly.hy/2 if abs(side)==2 else 2.
        line = np.einsum('j,xji->xi',weights,difference[iy])
        initial_line = np.einsum('j,xji->xi',weights,initial.coefficients[iy])
        for threshold in (1e-8,1e-6,1e-4):
            pieces = intervals(line,poly.hx,threshold)
            for endpoint in (2.,6.):
                touching = [s for s in pieces if s[0]<endpoint<s[1]]
                extent = (endpoint-touching[0][0] if endpoint==2 else touching[0][1]-endpoint) if touching else 0.
                connectivity.append(dict(y_trace=side,y=float(y),threshold=threshold,endpoint=endpoint,
                                          connected_extension=extent,extension_over_hx=extent/poly.hx,
                                          component=touching[0] if touching else None))
        for cell in range(round(5.25/poly.hx),round(6.75/poly.hx)):
            for u in np.linspace(0,1,33):
                records.append(dict(step=step,y_trace=side,y=float(y),cell_x=cell,x=(cell+u)*poly.hx,
                                    phi0=float(value(initial_line[cell],u)),
                                    delta_phi=float(value(line[cell],u))))
    path=OUT/f'endpoint-profile-{step:04d}.csv'
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    return connectivity


def reduce(live=False):
    run=RUNS/LABEL; directory=run/'output'
    accepted_path=directory/'bulk-accepted.csv'
    if not accepted_path.exists():
        print('No accepted physical interval yet.');return
    command=json.loads((run/'command.json').read_text())
    accepted=read(accepted_path)
    if live:
        last_banner=max(map(int,re.findall(r'\*\*\* Timestep (\d+):',(run/'run.log').read_text())))
        accepted=accepted[accepted.step<last_banner]
        accepted=np.array([r for r in accepted if all((directory/f'tip-points-{int(r.step):04d}-r{rank}.csv.gz').exists() and not (directory/f'tip-points-{int(r.step):04d}-r{rank}.csv').exists() and (directory/f'tip-balance-{int(r.step):04d}-r{rank}.csv.gz').exists() and not (directory/f'tip-balance-{int(r.step):04d}-r{rank}.csv').exists() for rank in range(2))],dtype=accepted.dtype).view(np.recarray)
    if not len(accepted):
        print('No closed, compressed physical observation yet.');return
    hx,hy,K=.0625,.03125,2.
    native1=gather(directory,'tip-balance-0001-r',['cell','i'])
    seed=gather(directory,'tip-seed-fracture-r',['cell','i'])
    assert np.array_equal(seed.cell,native1.cell) and np.array_equal(seed.i,native1.i)
    assert max(abs(native1.mass-hx*hy/9))<1e-14
    initial=RectBernstein(native1,hx,hy,seed.B)
    p0=gather(directory,'tip-points-0000-r',['cell','q'])
    assert max(abs(initial.evaluate(p0.x,p0.y)-p0.phi))<2e-12
    seed_energy=.5*seed.B*(seed.fracture_gradient+K*native1.mass)
    previous_B=seed.B; previous_energy=seed_energy; previous_points=p0
    clocks=read(directory/'tip-time-location.csv')
    result=dict(label=LABEL,command=command,hx=hx,hy=hy,seed_width=pulse(initial),
                seed_fracture=float(np.sum(seed_energy)),steps=[])
    for r in accepted:
        step=int(r.step); cache=OUT/f'step-{step:04d}.json'
        native=gather(directory,f'tip-balance-{step:04d}-r',['cell','i'])
        assert np.array_equal(native.cell,seed.cell) and np.array_equal(native.i,seed.i)
        energy=.5*native.B*(native.fracture_gradient+K*native.mass)
        assert abs(sum(energy)-r.fracture)<2e-9
        if cache.exists():
            item=json.loads(cache.read_text())
        else:
            points=gather(directory,f'tip-points-{step:04d}-r',['cell','q'])
            assert np.array_equal(points.cell,p0.cell) and np.array_equal(points.q,p0.q)
            if step>1 and previous_points is None:
                previous_points=gather(directory,f'tip-points-{step-1:04d}-r',['cell','q'])
            poly=RectBernstein(native,hx,hy)
            assert max(abs(poly.evaluate(points.x,points.y)-points.phi))<2e-12
            assert min(native.B-previous_B)>-2e-12 and min(native.B)>-2e-12 and max(native.B)<1+2e-12
            nm,pm=masks(native.x,native.y),masks(points.x,points.y)
            nm['other']=~np.logical_or.reduce([nm[k] for k in PRIMARY]);pm['other']=~np.logical_or.reduce([pm[k] for k in PRIMARY])
            regions={}
            for name,mask in nm.items():
                qmask=pm[name];delta=points.phi[qmask]-p0.phi[qmask];increment=points.phi[qmask]-previous_points.phi[qmask]
                regions[name]=dict(max_delta_phi=float(max(delta)),max_step_delta_phi=float(max(increment)),
                                   max_delta_coefficient=float(max(native.B[mask]-seed.B[mask])),
                                   phase_increment_integral=float(sum(points.weight[qmask]*delta)),
                                   phase_step_integral=float(sum(points.weight[qmask]*increment)),
                                   added_fracture=float(sum(energy[mask]-seed_energy[mask])),
                                   step_added_fracture=float(sum(energy[mask]-previous_energy[mask])),
                                   fracture_directional_work=float(sum(.5*(native.B[mask]-seed.B[mask])*(native.fracture_gradient[mask]+seed.fracture_gradient[mask]))))
            assert abs(sum(regions[k]['added_fracture'] for k in PRIMARY+['other'])-(sum(energy)-sum(seed_energy)))<1e-10
            sections=read(directory/f'tip-sections-{step:04d}.csv');index=points.section.astype(int)
            dn=np.bincount(index,weights=points.weight/sections.ds[index]);assert max(abs(dn-4))<2e-11
            V=np.bincount(index,weights=points.weight*points.w)/sections.ds
            assert max(abs(V-sections.V))<1e-12
            power=float(sum(points.weight*points.w*sections.force[index])-sum(sections.ds*sections.V*sections.force))
            assert abs(power)<1e-9
            rates=points.w[np.lexsort((points.y,points.section))].reshape(len(sections),-1)
            threshold=np.maximum(1e-10,1e-6*np.max(abs(rates),axis=1))[:,None]
            active=abs(rates)>threshold;starts=active & ~np.column_stack([np.zeros(len(sections),dtype=bool),active[:,:-1]])
            absrate=float(sum(sections.ds*sections.integral_abs_w))
            item=dict(step=step,time=float(r.time),stage_time=float(clocks.stage_time[clocks.step==step][0]),
                      regions=regions,width=pulse(poly),endpoint_connectivity=endpoint_profiles(poly,initial,step),profile_version=2,
                      added_fracture=float(sum(energy)-sum(seed_energy)),step_added_fracture=float(sum(energy-previous_energy)),
                      physical_work_error=float(r.Wext-r.energy_change-r.Db-r.Dc-r.Df),
                      integrated_V=float(sum(sections.ds*sections.V)),integrated_slip=float(sum(sections.ds*sections.slip)),
                      positive_w=float(sum(sections.ds*sections.positive_w)),negative_w=float(sum(sections.ds*sections.negative_w)),
                      cancellation=1-float(sum(sections.ds*abs(sections.V)))/absrate if absrate else 0.,
                      opposite_sign_sections=int(sum(np.any(rates>threshold,axis=1)&np.any(rates < -threshold,axis=1))),
                      multiple_band_sections=int(sum(np.sum(starts,axis=1)>1)),power_adjoint_error=power,
                      accepted={n:float(r[n]) for n in r.dtype.names})
            cache.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
        if item.get('profile_version') != 2:
            item['endpoint_connectivity']=endpoint_profiles(RectBernstein(native,hx,hy),initial,step)
            item['profile_version']=2
            cache.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
        result['steps'].append(item);previous_B=native.B;previous_energy=energy;previous_points=None
    result['signed_work_error']=sum(s['physical_work_error'] for s in result['steps'])
    result['absolute_work_error']=sum(abs(s['physical_work_error']) for s in result['steps'])
    result['cost']=dict(process_seconds=command.get('seconds'))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'history.csv').open('w') as f:
        rows=[]
        for s in result['steps']:
            row={k:s[k] for k in ['step','time','stage_time','added_fracture','step_added_fracture','physical_work_error','integrated_V','integrated_slip']}
            widths=[w['W_over_hn'] for w in s['width'] if w['W_over_hn'] is not None]
            row['W_over_hn_min']=min(widths) if widths else None
            row['peak']=max(w['peak'] for w in s['width'])
            for name in PRIMARY+['other']:
                for k,v in s['regions'][name].items():row[name+'_'+k]=v
            rows.append(row)
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(accepted=len(result['steps']),last_time=result['steps'][-1]['time'],width=result['steps'][-1]['width'],
                         last_regions={k:result['steps'][-1]['regions'][k] for k in PRIMARY},signed_work_error=result['signed_work_error']),indent=2))
    return result


def plot(result):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-resolution60-growth-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps=result['steps'];times=[s['time'] for s in steps]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for region in PRIMARY:
        axes[0,0].plot(times,[s['regions'][region]['max_delta_phi'] for s in steps],'.-',label=region.replace('_',' '))
        axes[0,1].plot(times,[s['regions'][region]['added_fracture'] for s in steps],'.-',label=region.replace('_',' '))
    axes[0,0].set_yscale('symlog',linthresh=1e-8)
    for side in (0,1):axes[1,0].plot(times,[s['width'][side]['W_over_hn'] for s in steps],'.-',label=f'x=4 trace {side}')
    axes[1,0].axhline(1,color='k',linestyle=':');axes[1,0].axhline(2,color='gray',linestyle=':')
    signed=np.cumsum([s['physical_work_error'] for s in steps]);absolute=np.cumsum([abs(s['physical_work_error']) for s in steps])
    axes[1,1].plot(times,signed,label='signed work remainder');axes[1,1].plot(times,absolute,'--',label='accumulated absolute remainder')
    axes[1,1].plot(times,[s['added_fracture'] for s in steps],label='added fracture')
    axes[1,1].set_yscale('symlog',linthresh=1e-10)
    for ax,title,ylabel in zip(axes.flat,['Regional phase increments from seed','Regional added fracture (SIPG row partition)','Central excess-compliance pulse','Physical-work error versus added fracture'],['max delta phi','J/m','W / normal cell width','J/m']):
        ax.set(title=title,xlabel='Endpoint time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(OUT/f'growth-history.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    fig.suptitle('Solid/dashed: native y=2 traces; dotted: y=2 ± hy/2')
    for index,s in enumerate(steps):
        p=read(OUT/f"endpoint-profile-{s['step']:04d}.csv")
        for side,style in [(-1,'-'),(1,'--'),(-2,':'),(2,':')]:
            mask=p.y_trace==side
            for ax in axes:ax.plot(p.x[mask],p.delta_phi[mask],style,color=plt.cm.viridis(index/max(1,len(steps)-1)),label=f"{s['time']:.3f} s" if side==-1 else None,linewidth=1)
    for ax in axes:
        ax.axvline(6,color='k',linestyle=':');ax.axhline(0,color='gray',linewidth=.5);ax.set(xlabel='x (m)',ylabel='phi − equilibrated phi0');ax.grid(alpha=.2)
    axes[0].set(title='Across original right endpoint',xlim=(5.25,6.75));axes[1].set(title='Endpoint magnification',xlim=(5.875,6.375))
    axes[1].set_yscale('symlog',linthresh=1e-8);axes[0].legend(fontsize=6,ncol=2)
    for ext in ['png','pdf']:fig.savefig(OUT/f'endpoint-growth.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    axes[0].plot([s['stage_time'] for s in steps],[s['integrated_V'] for s in steps],'.-')
    axes[0].set(xlabel='Midpoint stage time (s)',ylabel='integral V ds (m²/s)',title='Signed section slip rate')
    axes[1].plot(times,[s['integrated_slip'] for s in steps],'.-')
    axes[1].set(xlabel='Endpoint time (s)',ylabel='integral slip ds (m²)',title='Cumulative section slip')
    for ax in axes:ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(OUT/f'slip-history.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');args=parser.parse_args()
    result=reduce(args.live)
    if result:plot(result)
