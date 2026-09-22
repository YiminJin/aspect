"""Offline BP5-friction gates, using actual Q1 inputs and native work loads."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'bp3'))
from analyze_mechanical_modes import table
from analyze_mechanical_width import records
from analyze_length_scale import moments
from run_mechanical_width import stationary, VirtualProfile, NORMAL, G, XT, SN

OUT=HERE/'dc010-ell100'
KAPPA=-1e26*np.expm1(-4e6*G/1e26)


def realized_profile(profile, origin):
    e=profile.extent
    cuts=[-e,e]
    for d in range(2):
        limits=origin[d]+NORMAL[d]*np.array([-e,e])
        for j in range(int(np.floor(min(limits)/profile.h)),int(np.ceil(max(limits)/profile.h))+1):
            z=(j*profile.h-origin[d])/NORMAL[d]
            if -e<z<e: cuts.append(z)
    cuts=np.unique(cuts)
    lo,hi=cuts[:-1],cuts[1:]
    def value(z): return profile.q1(origin[0]+NORMAL[0]*z,origin[1]+NORMAL[1]*z)
    mid=(lo+hi)/2
    f0,f1,f2=value(lo),value(mid),value(hi)
    active=f1>0
    # Q1 restricted to each straight ray/cell interval is quadratic. Check its
    # interior stationary point as well as panel endpoints for the actual peak.
    A=2*(f0-2*f1+f2);B=f2-f0-A
    z=np.divide(-B,2*A,out=np.zeros_like(B),where=A!=0)
    z=np.clip(z,0,1)
    peak=max(np.max(f0),np.max(f2),np.max(value(lo+(hi-lo)*z)))
    return dict(realized_peak=float(peak),support_lo=float(min(lo[active])),
                support_hi=float(max(hi[active])),realized_support=float(max(hi[active])-min(lo[active])))


def kernel(k, dn=.125):
    r,p,m=stationary(100)
    n=np.arange(-256.,256.+dn/2,dn)
    phi=np.interp(abs(n),r,p,right=0.)
    h=m*phi*(1+phi)/(1-phi)**2
    ih=np.sum(h)*dn
    chi=h/ih
    size=1 << (2*len(n)-1).bit_length()
    corr=np.fft.irfft(abs(np.fft.rfft(chi,size))**2,size)[:len(n)]*dn
    distance=np.arange(len(n))*dn
    weights=corr*dn
    weights[1:]*=2
    result=np.zeros(len(k))
    for start in range(0,len(k),128):
        z=abs(k[start:start+128,None])*distance
        result[start:start+128]=abs(k[start:start+128])*(((1-z)*np.exp(-z))@weights)
    return G*result,dict(Ih=ih,rms_width=np.sqrt(np.sum(h*n*n)*dn/ih),support=2*r[-1])


def spectrum(nodes,dn):
    order=np.argsort(nodes['xd'])
    x,a=nodes['xd'][order],nodes['deltaV'][order]
    lengths=np.diff(x)
    mass=np.sum(lengths*(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2)/3)
    active=abs(a)>0
    spacing=float(np.mean(lengths[abs(a[:-1])+abs(a[1:])>0]))
    assert max(abs(lengths[abs(a[:-1])+abs(a[1:])>0]-spacing))<1e-7
    length=51200.
    k=2*np.pi*np.arange(8193)/length
    transform=spacing*np.sinc(k*spacing/(2*np.pi))**2*(np.exp(-1j*np.outer(k,x[active]-16500))@a[active])
    weights=abs(transform)**2
    weights[1:]*=2
    stiffness,profile=kernel(k,dn)
    return np.dot(weights,stiffness)/(length*mass),mass,profile,spacing,k,weights


def main():
    coefficients,columns,profiles=[],[],{}
    for mesh,h in [('candidate',24.4140625),('reference',12.20703125)]:
        run=OUT/('probe-'+mesh)
        assert json.loads((run/'execution.json').read_text())['expected_completion']
        modes,nodes,surface=[np.atleast_1d(table(run/file)) for file in
                            ('mechanical_modes.csv','mechanical_mode_nodes.csv','surface.csv')]
        assert set(modes['mode'])=={'target_3125m','alternating_200m'}
        for mode in modes:
            data=nodes[nodes['mode']==mode['mode']]
            dv=data['deltaV']
            mass=mode['mass_norm']
            direct=np.dot(dv,data['K_deltaV']+data['minus_sigma_muV_deltaV']+data['minus_damping'])/mass*G/KAPPA
            relaxation=np.dot(dv,data['G_shear_delta_x'])/mass*G/KAPPA
            net=mode['mechanical_shear']*G/KAPPA
            assert abs((direct-relaxation)/net-1)<1e-10
            prediction,line_mass,continuum,spacing,k,power=spectrum(data,.125)
            check,*_=spectrum(data,.0625)
            assert abs(check/prediction-1)<1e-5
            assert mode['fresh_relative']<1e-10 and mode['work_pair_relative']<1e-8 and mode['action_relative']<1e-8
            coefficients.append(dict(mesh=mesh,mode=str(mode['mode']),stiffness=net,direct=direct,
                bulk_relaxation=relaxation,continuum_actual_Q1=check,continuum_quadrature_change=check/prediction-1,
                relative_to_continuum=net/check-1,spacing=spacing,mass_over_line_mass=mass/line_mass,
                iterations=int(mode['iterations']),fresh_relative=mode['fresh_relative'],
                work_error=mode['work_pair_relative'],action_error=mode['action_relative']))
            if mesh=='candidate':
                records(OUT/(str(mode['mode'])+'_spectrum.csv'),[dict(k=kk,power=ww) for kk,ww in zip(k,power)])
        profile=VirtualProfile(100,h)
        production=table(run/'stationary_profile.csv')
        r,p,m=stationary(100)
        assert max(abs(r-production['r']))<1e-7 and max(abs(p-production['phi']))<2e-15
        xd=(100000-surface['y'])/SN
        order=np.argsort(xd)
        for s in np.arange(13000.,20000.01,125.):
            origin=np.array([XT-.5*s,100000-SN*s])
            j,second=moments(profile,origin)
            jj,ss=moments(profile,origin,32)
            assert abs(jj/j-1)<1e-9 and abs(ss/second-1)<1e-9
            ih=np.interp(s,xd[order],surface['Ih'][order])
            columns.append(dict(mesh=mesh,xd=s,J=j,Ih=ih,normalization_error=j/ih-1,
                peak=float(profile.q1(*origin)),rms_width=np.sqrt(second/j),
                width_error=np.sqrt(second/j)/continuum['rms_width']-1,**realized_profile(profile,origin)))
        completion=np.concatenate([np.atleast_1d(table(path)) for path in run.glob('ih_bottom_completion_rank*.csv')])
        endpoints=[]
        for row in completion[completion['outside']>0]:
            origin=np.array([row['x'],row['y']]);e=profile.extent
            total=profile.integrate(origin,-e,e)
            inside=profile.integrate(origin,max(-e,-origin[1]/NORMAL[1]),min(e,(100000-origin[1])/NORMAL[1]))
            endpoints.append(dict(id=int(row['id']),xd=(100000-origin[1])/SN,
                inside_fraction=inside/total,completed_error=row['completed']/total-1,
                inside_error=row['inside']/inside-1))
        selected=[v for v in columns if v['mesh']==mesh]
        profiles[mesh]=dict(continuum=continuum,max_width_error=max(abs(v['width_error']) for v in selected),
             max_normalization_error=max(abs(v['normalization_error']) for v in selected),endpoints=endpoints)
    comparisons=[]
    for mode in modes['mode']:
        a,b=[c for c in coefficients if c['mode']==mode]
        change=a['stiffness']/b['stiffness']-1
        comparisons.append(dict(mode=str(mode),coarse_over_fine_minus_one=change,pass_5_percent=bool(abs(change)<.05)))
    # The short-wave screen is distinct from long-wave nucleation and transient
    # stability. Include Q1 aliases and the consistent line mass, not a point grid.
    wavelength=np.arange(200.,1500.1,10.)
    k=2*np.pi/wavelength
    aliases=k[:,None]+2*np.pi*np.arange(-20,21)[None,:]/100
    stiffness,_=kernel(aliases.ravel(),.125)
    screen=np.sum(stiffness.reshape(aliases.shape)*np.sinc(aliases*100/(2*np.pi))**4,axis=1)/((2+np.cos(k*100))/3)
    records(OUT/'short_wave_screen.csv',[dict(wavelength=l,stiffness=v,Kc50=13e6,Kc60=15.6e6) for l,v in zip(wavelength,screen)])
    records(OUT/'coefficients.csv',coefficients)
    records(OUT/'profile_columns.csv',columns)
    gate=all(c['pass_5_percent'] for c in comparisons) and all(
        p['max_width_error']<(.02 if mesh=='candidate' else .01) and p['max_normalization_error']<.01
        and max(abs(e['completed_error']) for e in p['endpoints'])<1e-6 for mesh,p in profiles.items())
    result=dict(gate=bool(gate),original_one_percent_width_gate=False,diagnostic_exception='candidate width <=2%, this task only',
                coefficients=coefficients,refinement=comparisons,profiles=profiles,
                short_wave_interval=[200,1500],screen_min=float(min(screen)),screen_alternating=float(screen[0]))
    (OUT/'probe_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for mesh in profiles:
        data=[v for v in columns if v['mesh']==mesh]
        axes[0].plot([v['xd']/1000 for v in data],[100*v['width_error'] for v in data],label=mesh)
        axes[1].plot([v['xd']/1000 for v in data],[100*v['normalization_error'] for v in data])
        c=[v for v in coefficients if v['mesh']==mesh]
        axes[2].plot([3125,200],[v['stiffness']/1e6 for v in c],'o-',label=mesh)
    axes[0].axhline(1,color='k',ls='--');axes[0].axhline(2,color='r',ls=':')
    axes[0].set(xlabel='Down dip (km)',ylabel='RMS width error (%)');axes[0].legend()
    axes[1].set(xlabel='Down dip (km)',ylabel='Column normalization error (%)')
    axes[2].set(xlabel='Nominal wavelength (m)',ylabel='Work stiffness (MPa/m)');axes[2].legend()
    fig.tight_layout();fig.savefig(OUT/'probe_comparison.png',dpi=160)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
