#!/usr/bin/env python3
"""Physical mass-normalized comparisons and highest-resolution-only figures."""
import argparse
import hashlib
import json
import platform

import scipy

import numpy as np

import asymmetric_branch as a
from assess_coupled_branch import totals


def configs():
    return [(256,256,False),(256,512,False),(256,1024,False),
            (512,512,False),(512,1024,False),(511,1024,False),(512,1024,True)]


def continuation_path(n,d,sign,exact):
    p=a.search_path(n,d,exact)
    return p.parent/(p.name+f'-continue-{sign:+d}')


def curvature_check(p,x):
    status,v=a.spectrum(p,x);H=p.hessian(x);M=a.mass(p.r)
    # Operator differences may leave the obstacle, but never the physical box.
    # Feasible perturbations of candidates are checked separately by search.
    epsilons=[1e-5,2e-5,5e-5]
    derivatives=[]
    for eps in epsilons:
        assert min(x-eps*abs(v))>=0 and max(x+eps*abs(v))<1
        vp,gp=p.evaluate(x+eps*v);vm,gm=p.evaluate(x-eps*v)
        quotient=float(a.A*v@(gp-gm)/(2*eps)/(v@M@v))
        derivatives.append(dict(epsilon=eps,physical_curvature_FD=quotient,
             error=abs(quotient-status['minimum']),tangent_error=float(max(abs((gp-gm)/(2*eps)-H@v)))))
    assert min(x['error'] for x in derivatives)<max(1e-5,abs(status['minimum'])*.01)
    return dict(stability=status,derivatives=derivatives)


def selected_checks(n,d,exact,sign):
    out=continuation_path(n,d,sign,exact);f=np.load(out/'fields.npz');hist=a.read(out/'history.json')
    start=np.load(a.search_path(n,d,exact)/'incoming.npz');p=a.model(n,exact);results=[]
    ids={0,len(hist)-1, max(range(len(hist)),key=lambda j:hist[j]['phase_increment'])}
    for t in [.59375,.609375]:
        for j,row in enumerate(hist):
            if row['time']==t:ids.add(j)
    for j in sorted(ids):
        pa=f['phase'][j];pb=f['phase'][j+1]
        slip=hist[j-1]['slip'] if j else float(start['slip'])
        p.incoming(pa,f['R'][j],f['Z'][j],slip,hist[j]['time'],1/d)
        curvature=curvature_check(p,pb)
        p.evaluate(pb);data=p.last[3];u=p.r.velocity.copy()
        ledger=(a.exact_ledger if exact else None)
        energies=[]
        for order in [16,32]:
            args=(pa,pb,p.Ra,p.Za,p.last[4],p.last[5],u,data['V'],1/d,order)
            energies.append(ledger(p,*args) if exact else a.core.path_ledger(p.r,*args))
        keys=['external_remainder','mechanical_power','friction_residual_power','phase_power',
              'mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute']
        uncertainty=max(abs(energies[0][k]-energies[1][k]) for k in keys)
        assert uncertainty<1e-10
        assert max(x['identity_error'] for x in energies)<1e-10
        normalization=normalization_check(p.r,pb)
        results.append(dict(time=hist[j]['time'],curvature=curvature,
             normalization=normalization,time_quadrature_uncertainty=uncertainty))
    return results


def normalization_check(r,phi):
    I,J=r.direct(phi);measurements=[]
    for order in [12,24]:
        z,w=np.polynomial.legendre.leggauss(order);s=(z+1)/2
        values=(1-s)*phi[:-1,None]+s*phi[1:,None]
        integral=float(r.dy/2*np.sum(w*a.core.h(values)))
        weights=r.dy/2*w*a.core.dh(values)
        left=np.sum(weights*(1-s),axis=1);right=np.sum(weights*s,axis=1)
        derivative=np.r_[left,0.]+np.r_[0.,right]
        measurements.append(dict(order=order,integral=integral,
             relative_error=abs(integral-I)/I,derivative_error=float(max(abs(derivative-J)))))
    assert I>0 and np.isfinite(I) and min(x['relative_error'] for x in measurements)<1e-11
    return dict(analytic_I=float(I),checks=measurements,
         scope='Uniform along fault: consistent surface projection reproduces this constant exactly')


def main():
    started=a.time.perf_counter()
    out=a.E/'assessment';out.mkdir(exist_ok=False);tables=[];verification=[];reflection=[];modes=[]
    for n,d,exact in configs():
        base=a.search_path(n,d,exact);incoming=np.load(base/'incoming.npz')
        p=a.model(n,exact)
        p.incoming(incoming['phi'],incoming['R'],float(incoming['Z']),float(incoming['slip']),float(incoming['time']),float(incoming['dt']))
        symmetric=incoming['symmetric'];basevalue,_=p.evaluate(symmetric)
        baseenergy=sum(p.last[3][k] for k in ['bulk','cohesive','fracture'])
        spec,v=a.spectrum(p,symmetric)
        modes.append(dict(n=n,d=d,exact=exact,check=curvature_check(p,symmetric)))
        write_mode=dict(**spec,time=p.time,potential=float(a.A*basevalue),endpoint_energy=float(baseenergy),
            line_search_arithmetic_allowance=float(a.A*32*np.finfo(float).eps*max(1.,abs(basevalue))))
        a.write(base/'physical-mode.json',write_mode)
        np.savez_compressed(base/'physical-mode.npz',mode=v,y=p.r.y,phase=symmetric)
        candidate_rows=a.read(base/'candidates.json');assert len(candidate_rows)==6
        for x in candidate_rows:
            assert x['KKT']<=a.TARGET and x['stable'] and x['potential_reduction']>0
            assert x['mechanical']['force']<1e-7 and x['mechanical']['contact']<1e-7
            field=np.load(base/f"candidate-{x['fraction']:g}-{x['sign']:+d}.npz")['phase']
            x['phase_difference_from_symmetric']=float(max(abs(field-symmetric)))
            x['phase_L2_difference_from_symmetric']=float(np.sqrt((field-symmetric)@a.mass(p.r)@(field-symmetric)))
            x['physical_energy_reduction']=float(baseenergy-x['endpoint_energy'])
        for fraction in [.25,.5,.8]:
            plus=np.load(base/f'candidate-{fraction:g}-+1.npz');minus=np.load(base/f'candidate-{fraction:g}--1.npz')
            reflection.append(dict(n=n,d=d,exact=exact,fraction=fraction,
                phase=float(max(abs(plus['phase']-minus['phase'][::-1]))),
                R=float(max(abs(plus['R']-minus['R'][::-1]*(1 if exact else -1)))),
                Z=float(abs(plus['Z']-minus['Z']))))
        for sign in [-1,1]:
            path=continuation_path(n,d,sign,exact);hist=a.read(path/'history.json');en=a.read(path/'energy.json')
            f=np.load(path/'fields.npz');summary=a.read(path/'summary.json')
            assert summary['completed'] and hist[-1]['time']==.625
            assert all(x['KKT']<=a.TARGET and x['stability']['stable'] for x in hist)
            assert max(x['force'] for x in hist)<1e-7 and max(x['contact'] for x in hist)<1e-7
            assert min(np.diff(f['phase'],axis=0).ravel())>=-4.922e-12
            assert min(f['phase'].ravel())>=0 and max(f['phase'].ravel())<1
            beta=np.exp(-a.G/(d*a.ETA));kappa=-a.ETA*np.expm1(-a.G/(d*a.ETA))
            assert max(abs((f['R'][1:]-beta*f['R'][:-1]-kappa/a.G*f['velocity']).ravel()))<1e-16
            assert max(abs(f['Z'][1:]-beta*f['Z'][:-1]-kappa*np.array([x['V'] for x in hist])))<1e-10
            assert abs(hist[-1]['slip']-float(incoming['slip'])-sum(x['V']/d for x in hist))<1e-14
            # Common post-entry window excludes distinct first release times.
            common=[x for x in en if x['time']>.59765625]
            table=dict(n=n,d=d,exact=exact,sign=sign,first_entry=float(incoming['time']),
                 mode=write_mode,candidates=candidate_rows,summary=summary,final=hist[-1],
                 max_phase_increment=max(x['phase_increment'] for x in hist),
                 total=totals(en),common_energy=totals(common),
                 minimum_stable_curvature=min(x['stability']['minimum'] for x in hist),
                 endpoint_KKT_max=max(x['KKT'] for x in hist),endpoint_force_max=max(x['force'] for x in hist),
                 endpoint_contact_max=max(x['contact'] for x in hist),
                 max_identity_error=max(x['identity_error'] for x in en))
            tables.append(table)
        verification.append(dict(n=n,d=d,exact=exact,checks=selected_checks(n,d,exact,1)))
        plus=np.load(continuation_path(n,d,1,exact)/'fields.npz')
        minus=np.load(continuation_path(n,d,-1,exact)/'fields.npz')
        reflection.append(dict(n=n,d=d,exact=exact,fraction='trajectory',
            phase=float(np.max(abs(plus['phase']-minus['phase'][:,::-1]))),
            R=float(np.max(abs(plus['R']-minus['R'][:,::-1]*(1 if exact else -1)))),
            Z=float(np.max(abs(plus['Z']-minus['Z'])))))
    compare=compare_fields()
    a.write(out/'tables.json',tables);a.write(out/'verification.json',verification)
    a.write(out/'reflection.json',reflection);a.write(out/'comparison.json',compare)
    a.write(out/'mode-verification.json',modes)
    a.write(out/'initial-and-common-time.json',initial_and_common_time())
    highest_visualizations()
    a.write(out/'decision.json',dict(diagnostic_continuations_complete=True,production_unchanged=True,
            physical_branch_selected=False,P3=False,
            conclusion='Resolved translation-like discrete pinning; stable reflected diagnostic continuations, not a physical asymmetry qualification'))
    a.write(out/'provenance.json',dict(base='de7beaa65846c7372711f7922abf1c5d9a745ee6',branch='fault-propagation',
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        source_sha256={name:hashlib.sha256((a.HERE/name).read_bytes()).hexdigest()
           for name in ['asymmetric_branch.py','assess_asymmetric_branch.py','coupled_branch.py','localization_1d.py']},
        assessment_seconds=a.time.perf_counter()-started,
        assessment_peak_RSS_KiB=a.resource.getrusage(a.resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(tables=len(tables),reflection_max=max(x['phase'] for x in reflection),comparisons=compare)),flush=True)


def initial_and_common_time():
    rows=[];initials=[];y=np.linspace(0,1,8193)
    for n,exact in [(256,False),(512,False),(511,False),(512,True)]:
        if n==256:
            path=a.core.HERE/'coupled_branch_fresh_evidence/dt-1-1024'
        else:path=a.search_path(n,1024,exact).parent
        f=np.load(path/'fields.npz');hist=a.read(path/'history.json')
        p=a.model(n,exact);t=.546875
        j=next(i for i,x in enumerate(hist) if x['time']==t)
        p.incoming(f['phase'][j],f['R'][j],f['Z'][j],hist[j-1]['slip'],t,1/1024)
        status,_=a.spectrum(p,f['phase'][j+1])
        initials.append((n,exact,np.interp(y,f['y'],f['phase'][0])))
        rows.append(dict(n=n,exact=exact,time=t,stability=status,
             initial_max=float(max(f['phase'][0])),centroid=a.centroid(p.r,f['phase'][0]),
             normalization=normalization_check(p.r,f['phase'][0])))
    reference=initials[1][2]
    for row,(_,_,profile) in zip(rows,initials):
        row['initial_phase_difference_from_512']=float(max(abs(profile-reference)))
    return rows


def compare_fields():
    result=[]
    for n,da,db in [(256,256,512),(256,512,1024),(512,512,1024)]:
        r=a.core.Reference(n,1.,12)
        for sign in [-1,1]:
            pa=continuation_path(n,da,sign,False);pb=continuation_path(n,db,sign,False)
            ha=a.read(pa/'history.json');hb=a.read(pb/'history.json')
            fa=np.load(pa/'fields.npz');fb=np.load(pb/'fields.npz');rows=[]
            index={x['time']:j for j,x in enumerate(hb)}
            for j,x in enumerate(ha):
                if x['time']<.6015625:continue
                k=index[x['time']];y=hb[k]
                oldtime=x['time']-1/da
                if oldtime not in index:continue
                va=(x['slip']-ha[j-1]['slip'])*da
                vb=(y['slip']-hb[index[oldtime]]['slip'])*da
                def stress(f,index):
                    phase=f['phase'][index+1]
                    return a.G*(r.D@f['R'][index+1])-f['Z'][index+1]*a.core.h(r.B@phase)/r.direct(phase)[0]
                delta=stress(fa,j)-stress(fb,k)
                rows.append(dict(time=x['time'],phase=float(max(abs(fa['phase'][j+1]-fb['phase'][k+1]))),
                     q=abs(x['q']-y['q']),slip=abs(x['slip']-y['slip']),common_rate=abs(va-vb),
                     stress_RMS=float(np.sqrt(r.w@(delta*delta))),
                     bulk=abs(x['bulk']-y['bulk']),cohesive=abs(x['cohesive']-y['cohesive']),
                     fracture=abs(x['fracture']-y['fracture']),
                     centroid=abs(x['centroid']['phase']-y['centroid']['phase'])))
            keys=['phase','q','slip','common_rate','centroid','stress_RMS','bulk','cohesive','fracture']
            result.append(dict(type='time',n=n,coarse=da,fine=db,sign=sign,
                maximum={key:max(x[key] for x in rows) for key in keys},by_time=rows))
    # Spatial fields on a fixed independently sampled coordinate set; no history interpolation.
    y=np.linspace(0,1,8193)
    for n1,n2,exact2 in [(256,512,False),(511,512,False),(512,512,True)]:
        for sign in [-1,1]:
            pa=continuation_path(n1,1024,sign,False);pb=continuation_path(n2,1024,sign,exact2)
            fa=np.load(pa/'fields.npz');fb=np.load(pb/'fields.npz')
            ha=a.read(pa/'history.json');hb=a.read(pb/'history.json');ia={x['time']:j for j,x in enumerate(ha)};ib={x['time']:j for j,x in enumerate(hb)}
            rows=[]
            for t in [.6015625,.609375,.625]:
                ja=ia[t];jb=ib[t];x=ha[ja];z=hb[jb]
                delta=np.interp(y,fa['y'],fa['phase'][ja+1])-np.interp(y,fb['y'],fb['phase'][jb+1])
                rows.append(dict(time=t,phase_max=float(max(abs(delta))),phase_L2=float(np.sqrt(np.trapezoid(delta*delta,y))),
                    q=abs(x['q']-z['q']),slip=abs(x['slip']-z['slip']),
                    centroid=abs(x['centroid']['phase']-z['centroid']['phase'])))
            result.append(dict(type='space',first=n1,second=n2,second_exact=exact2,sign=sign,by_time=rows))
    return result


def highest_visualizations():
    """This task alone writes figures exclusively from n=512 tests."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=a.E/'assessment';n=512;d=1024
    fig,ax=plt.subplots(2,3,figsize=(12,7));prefix=a.E/'n512-d1024'
    for sign in [-1,1]:
        path=continuation_path(n,d,sign,False);f=np.load(path/'fields.npz');hist=a.read(path/'history.json');en=a.read(path/'energy.json')
        t=[x['time'] for x in hist];label=f'initial sign {sign:+d}'
        ax[0,0].plot(t,[x['phase_max'] for x in hist],label=label)
        ax[0,1].plot(t,[1e3*(x['centroid']['phase']-.5) for x in hist],label=label)
        ax[0,2].plot([x['slip'] for x in hist],[x['q'] for x in hist],label=label)
        for k,style in [('mechanical_power','-'),('friction_residual_power','--'),('phase_power',':')]:
            ax[1,0].plot(t,np.cumsum([x[k+'_absolute'] for x in en]),style,label=f'{sign:+d}: {k}')
        ax[1,1].plot(t,np.cumsum([x['external_remainder'] for x in en]),label=label)
        ax[1,2].plot(t,[x['stability']['minimum'] for x in hist],label=label)
    for panel,x,y in zip(ax.ravel(),['time [s]','time [s]','slip [m]']+['time [s]']*3,
        ['maximum phase','phase centroid shift [mm]','cohesive traction [Pa]',
         'integrated absolute residual power [J/m]','external remainder [J/m]','L2-normalized curvature [Pa]']):
        panel.set_xlabel(x);panel.set_ylabel(y);panel.grid(alpha=.2);panel.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'highest-overview.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(12,7))
    for sign in [-1,1]:
        path=continuation_path(n,d,sign,False);f=np.load(path/'fields.npz');hist=a.read(path/'history.json')
        r=a.core.Reference(n,1.,12)  # Independent postprocessing points only.
        index={x['time']:j for j,x in enumerate(hist)}
        for t in [.5625,.59375,.625]:
            j=index[t];p=f['phase'][j+1];I=r.direct(p)[0];tau=a.G*(r.D@f['R'][j+1])-f['Z'][j+1]*a.core.h(r.B@p)/I
            label=f'{t:g}s, {sign:+d}';style='-' if sign>0 else '--'
            ax[0,0].plot(f['y'],p,style,label=label);ax[0,1].plot(f['y'],a.core.h(p)/I,style,label=label)
            ax[0,2].plot(r.yq,tau,style,label=label)
        phase=np.array(f['phase'][1:]);times=np.array([x['time'] for x in hist])
        if sign>0:
            image=ax[1,0].pcolormesh(f['y'],times,phase,shading='auto');fig.colorbar(image,ax=ax[1,0],label='phase')
            ax[1,0].set_xlabel('y [m]');ax[1,0].set_ylabel('time [s]')
            im=ax[1,1].pcolormesh(f['y'],times,phase-phase[0],shading='auto');fig.colorbar(im,ax=ax[1,1],label='phase growth after entry')
            ax[1,1].set_xlabel('y [m]');ax[1,1].set_ylabel('time [s]')
    mode=np.load(a.search_path(n,d)/'physical-mode.npz');r=a.model(n).r
    derivative=np.gradient(mode['phase'],mode['y']);derivative=-derivative/np.sqrt(derivative@a.mass(r)@derivative)
    ax[1,2].plot(mode['y'],mode['mode'],label='unstable mode');ax[1,2].plot(mode['y'],derivative,'--',label='normalized -phi_y')
    ax[1,2].set_xlabel('y [m]');ax[1,2].set_ylabel('L2-normalized direction');ax[1,2].legend(fontsize=7)
    for panel,y in zip(ax[0],['phase','chi [1/m]','off-assembly stress [Pa]']):
        panel.set_xlabel('y [m]');panel.set_ylabel(y);panel.grid(alpha=.2)
    ax[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'highest-profiles.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    for denominator,exact,label in [(512,False,'FE dt=1/512'),(1024,False,'FE dt=1/1024'),
                                   (1024,True,'exact bulk dt=1/1024')]:
        hist=a.read(continuation_path(512,denominator,1,exact)/'history.json')
        t=np.array([x['time'] for x in hist])
        ax[0,0].plot(t,[x['phase_max'] for x in hist],label=label)
        ax[0,1].plot(t,[1000*(x['centroid']['phase']-.5) for x in hist],label=label)
        ax[1,0].plot([x['slip'] for x in hist],[x['q'] for x in hist],label=label)
    hist=a.read(continuation_path(512,1024,1,False)/'history.json');previous=set()
    for row in hist:
        free=set(np.linspace(0,1,513))-set(row['stability']['lower_y'])
        released=sorted(free-previous)
        if previous:
            ax[1,1].scatter([row['time']]*len(released),released,s=8,color='tab:blue')
        previous=free
    for panel,x,y in zip(ax.ravel(),['time [s]','time [s]','slip [m]','time [s]'],
         ['maximum phase','phase centroid shift [mm]','cohesive traction [Pa]','newly free node y [m]']):
        panel.set_xlabel(x);panel.set_ylabel(y);panel.grid(alpha=.2)
    for panel in ax.ravel()[:3]:panel.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'highest-comparison.png',dpi=170);plt.close(fig)


def check():
    for name,digest in a.read(a.E/'preservation.json').items():
        assert hashlib.sha256((a.ROOT/name).read_bytes()).hexdigest()==digest,name
    decision=a.read(a.E/'assessment/decision.json')
    for name,digest in a.read(a.E/'assessment/provenance.json')['source_sha256'].items():
        assert hashlib.sha256((a.HERE/name).read_bytes()).hexdigest()==digest,name
    assert decision['diagnostic_continuations_complete'] and not decision['physical_branch_selected']
    for x in a.read(a.E/'assessment/tables.json'):
        assert x['summary']['completed'] and x['endpoint_KKT_max']<=a.TARGET
        assert x['minimum_stable_curvature']>0 and x['max_identity_error']<1e-10
    for r in a.read(a.E/'assessment/reflection.json'):assert r['phase']<1e-7
    print(json.dumps(dict(accepted_numerical_checks='pass',physical_branch_selected=False,
        old_files_preserved=len(a.read(a.E/'preservation.json')),P3=False)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    check() if args.check else main()
