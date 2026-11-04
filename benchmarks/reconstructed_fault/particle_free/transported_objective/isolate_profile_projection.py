#!/usr/bin/env python3
"""Independent Q1 projection controls and replay of native accepted profiles."""
import hashlib
import json
import os
from pathlib import Path
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DEAL_II_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
from assess_material_native import State, read, basis, RUNS, ROOT

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile'


def compact(x):
    return .35*np.maximum(0.,1-(x[...,0]/1.4)**2)**4*np.maximum(0.,1-((x[...,1]-.3*x[...,0])/.6)**2)**4


def grid(n):
    a=np.linspace(-2,2,n+1)
    return np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1)


def q1(nodes,x):
    n=nodes.shape[0]-1;h=4/n
    ij=np.floor((x+2)/h).astype(int)
    valid=np.all((x>=-2-1e-13)&(x<=2+1e-13),axis=-1)
    ij=np.clip(ij,0,n-1);uv=(np.clip(x,-2,2)+2)/h-ij
    i,j=ij[...,0],ij[...,1];u,v=uv[...,0],uv[...,1]
    return valid*((1-u)*(1-v)*nodes[i,j]+u*(1-v)*nodes[i+1,j]+(1-u)*v*nodes[i,j+1]+u*v*nodes[i+1,j+1])


def quadrature(n):
    g,w=np.polynomial.legendre.leggauss(8);h=4/n
    x1=(-2+h*(np.arange(n)[:,None]+.5*(g+1))).reshape(-1)
    w1=np.tile(.5*h*w,n)
    return np.stack(np.meshgrid(x1,x1,indexing='ij'),axis=-1).reshape(-1,2),np.outer(w1,w1).reshape(-1)


def profile_controls():
    records=[]
    for n in [16,32,64]:
        X=grid(n);x,w=quadrature(n);initial=compact(X)
        for speed in [-.4,.4]:
            for dt in [.1,.05,.025]:
                steps=round(.4/dt);a=initial.copy();foot=X.copy();foot[...,0]-=speed*dt
                for _ in range(steps):a=q1(a,foot)
                exact_foot=x.copy();exact_foot[:,0]-=speed*.4;exact=compact(exact_foot)
                final_nodes=X.copy();final_nodes[...,0]-=speed*.4
                endpoint=compact(final_nodes)
                one_shot=q1(initial,final_nodes)
                numerical=q1(a,x);best=q1(endpoint,x);direct=q1(one_shot,x)
                total=numerical-exact;remap=numerical-best;representation=best-exact
                integral=lambda z:float(w@z)
                norm=lambda z:float(np.sqrt(integral(z*z)))
                mass=integral(numerical);mean=integral(numerical*x[:,0])/mass
                variance=integral(numerical*(x[:,0]-mean)**2)/mass
                identity=integral(total**2)-integral(remap**2)-integral(representation**2)-2*integral(remap*representation)
                assert abs(identity)<1e-14
                records.append(dict(n=n,speed=speed,dt=dt,steps=steps,CFL=abs(speed)*dt/(4/n),
                    total_L2=norm(total),relative_L2=norm(total)/norm(exact),remap_L2=norm(remap),
                    representation_L2=norm(representation),cross_term=2*integral(remap*representation),
                    decomposition_defect=identity,one_shot_L2=norm(direct-exact),mass=mass,
                    mass_change=mass-integral(q1(initial,x)),maximum=float(a.max()),mean_x=mean,variance_x=variance))
    # An independent probabilistic moment identity for repeated linear interpolation.
    variance=[]
    for extent in [2.,8.]:
        for n in [16,32,64]:
            h=4/n;x=np.arange(-extent,extent+h/2,h);initial=np.maximum(0,1-(x/1.0)**2)**4
            for speed in [-.4,.4]:
                for dt in [.1,.05,.025]:
                    steps=round(.4/dt);a=initial.copy()
                    for _ in range(steps):a=np.interp(x-speed*dt,x,a,left=0.,right=0.)
                    def moments(z):
                        mass=h*z.sum();mean=h*(z*x).sum()/mass
                        return mean,h*(z*(x-mean)**2).sum()/mass+h*h/6
                    mean0,var0=moments(initial);mean,var=moments(a)
                    f=(abs(speed)*dt/h)%1;predicted=steps*f*(1-f)*h*h
                    error=var-var0-predicted
                    if extent==8.:assert abs(error)<1e-12
                    variance.append(dict(n=n,domain_halfwidth=extent,speed=speed,dt=dt,
                        variance_increase=var-var0,predicted=predicted,defect=error,
                        centroid_error=mean-mean0-speed*.4,mass_fraction_lost=1-a.sum()/initial.sum()))
    return records,variance


class ClosedState(State):
    def remote_at(self,x):
        # deal.II RemotePointEvaluation's default 1e-6 unit-cell tolerance
        # admits adjacent cells; EvaluationFlags::avg averages their clamped
        # reference-cell evaluations. Keep this distinct from exact pullback.
        base=np.floor((x+2)/self.h).astype(int)
        out=np.zeros((len(x),self.values.shape[1]));count=np.zeros(len(x))
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                ij=base+np.array([dx,dy]);u=(x+2)/self.h-ij
                valid=np.all((ij>=0)&(ij<self.n)&(u>=-1e-6)&(u<=1+1e-6),axis=1)
                i,j=ij[valid].T
                out[valid]+=np.einsum('ni,nij->nj',basis(2*np.clip(u[valid],0,1)-1),self.coeff[i,j])
                count[valid]+=1
        return out/np.maximum(count,1)[:,None]

    def at(self,x):
        # Native nodes include the closed outer boundary. The earlier Gauss-point
        # comparison did not need this endpoint convention.
        n=self.n;h=self.h
        valid=np.all((x>=-2-1e-13)&(x<=2+1e-13),axis=1)
        clipped=np.clip(x,-2,2);ij=np.clip(np.floor((clipped+2)/h).astype(int),0,n-1)
        out=np.zeros((len(x),len(self.values[0])));i,j=ij[valid].T
        unit=2*((clipped[valid]+2)/h-ij[valid])-1
        out[valid]=np.einsum('ni,nij->nj',basis(unit),self.coeff[i,j])
        return out


def native_replay(name):
    p=RUNS/name;meta=json.loads((p/'command.json').read_text());assert meta['exit_code']==0
    folder=p/'output';summary=read(folder/'material-summary.csv');level=meta['arguments']['level']
    m=float(read(folder/'material-model.csv')[0]['m']);n=2**level;X=grid(n);nodes=X.reshape(-1,2)
    states={int(row['step']):ClosedState(folder,int(row['step']),level,m) for row in summary}
    results=[]
    for row in summary:
        step=int(row['step']);dt=float(row['dt'])
        if step==0 or dt<1e-10:continue
        old,new=states[step-1],states[step]
        v=old.remote_at(nodes)[:,1:3];foot=nodes-dt*v
        mapped=old.remote_at(foot)[:,0].reshape(n+1,n+1)
        mathematical=q1(old.at(foot)[:,0].reshape(n+1,n+1),new.x)
        predicted=q1(mapped,new.x);error=predicted-new.values[:,0]
        direct=old.at(new.x-dt*old.at(new.x)[:,1:3])[:,0]
        defect=new.values[:,0]-direct
        weights=new.weights
        interior=np.max(np.abs(new.x),axis=1)<1.25
        L2=lambda mask:float(np.sqrt(np.sum(weights[mask]*defect[mask]**2)))
        results.append(dict(step=step,time=float(row['time']),dt=dt,nodal_replay_Linf=float(abs(error).max()),
            point_lookup_tolerance_Linf=float(abs(predicted-mathematical).max()),
            one_step_remap_L2=L2(np.ones(len(new.x),dtype=bool)),interior_L2=L2(interior),boundary_band_L2=L2(~interior)))
        assert abs(error).max()<1e-11,(name,step,abs(error).max())
    return dict(case=name,steps=results,inputs_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [p/'command.json',*folder.glob('material-*.csv')]})


def main():
    records,variance=profile_controls()
    names=['matched-positive','matched-half-dt','matched-fine-final']
    names += [name for name in ['compact-profile-full','compact-profile-half'] if (RUNS/name/'command.json').exists()]
    replay=[native_replay(name) for name in names]
    out=dict(profile_controls=records,variance_controls=variance,native_replays=replay,
        interpretation='Repeated Q1 nodal pullback, not a conservative DG phase method. L2 signed decomposition includes the cross term. Finite-boundary variance defects are reported rather than zeroed. No production algorithm changed.')
    (OUT/'profile-projection.json').write_text(json.dumps(out,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for n in [16,32,64]:
        r=[x for x in records if x['n']==n and x['speed']>0]
        axs[0,0].plot([x['dt'] for x in r],[x['relative_L2'] for x in r],'.-',label=f'{n}² cells')
        axs[0,1].plot([x['dt'] for x in r],[x['total_L2'] for x in r],'.-',label=f'{n}² repeated')
        axs[0,1].plot([x['dt'] for x in r],[x['one_shot_L2'] for x in r],'--',label=f'{n}² one-shot')
    for dt in [.1,.025]:
        r=[x for x in variance if x['speed']>0 and x['dt']==dt and x['domain_halfwidth']==8.]
        axs[1,0].plot([4/x['n'] for x in r],[x['variance_increase'] for x in r],'.-',label=f'dt={dt}')
        axs[1,0].plot([4/x['n'] for x in r],[x['predicted'] for x in r],'kx',label=f'prediction dt={dt}')
    for case in replay:
        r=case['steps'];axs[1,1].plot([x['time'] for x in r],[x['interior_L2'] for x in r],'.-',label=case['case'])
    axs[0,0].set(xlabel='Timestep (s)',ylabel='Relative L2',title='Compact profile: final error')
    axs[0,1].set(xlabel='Timestep (s)',ylabel='L2 (m)',title='Repeated versus one-shot remapping')
    axs[1,0].set(xlabel='Cell width (m)',ylabel='Variance increase (m²)',title='Interpolation diffusion identity')
    axs[1,1].set(xlabel='Time (s)',ylabel='Interior L2 (m)',title='Native one-step Q1 defect')
    for ax in axs.flat:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig(OUT/'profile-projection.png',dpi=160)
    print(json.dumps(dict(max_replay_error=max(r['nodal_replay_Linf'] for c in replay for r in c['steps']),
        compact_relative_errors=[(r['n'],r['dt'],r['relative_L2']) for r in records if r['speed']>0],
        max_variance_defect=max(abs(r['defect']) for r in variance if r['domain_halfwidth']==8.))))


if __name__=='__main__':main()
