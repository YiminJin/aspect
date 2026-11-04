#!/usr/bin/env python3
"""Independent characteristic/rotation/reaction audit of native material exports.

Interpolates exported complete Q2 polynomials; does not call production code.
The characteristic comparison is first order and its error is measured, not
assumed zero. It separates native publication/lifecycle identities from PDE
discretization error. All tensors use the full 3D plane-strain metric.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled'
NAMES=['phi','vx','vy','Txx','Tyy','Txy','Rxx','Ryy','Rxy','tauxx','tauyy','tauxy','Lxx','Lxy','Lyx','Lyy','eta']

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def norm3(a):return np.sqrt(a[...,0]**2+a[...,1]**2+(a[...,0]+a[...,1])**2+2*a[...,2]**2)

def basis(u):
    a=np.stack([np.ones(len(u)),u[:,0],.5*(3*u[:,0]**2-1)],axis=1)
    b=np.stack([np.ones(len(u)),u[:,1],.5*(3*u[:,1]**2-1)],axis=1)
    return (a[:,:,None]*b[:,None,:]).reshape(-1,9)

class State:
    def __init__(self,path,step,level,m):
        self.h=4/2**level;self.n=2**level;self.m=m
        self.direct=float(read(path/"material-model.csv")[0].get("direct",0))==1
        records=[r for p in sorted(path.glob(f'material-{step}-rank*.csv')) for r in read(p)]
        records.sort(key=lambda r:(r['cell'],int(r['q'])))
        self.x=np.array([[float(r['x']),float(r['y'])] for r in records])
        names=NAMES+(['pressure'] if 'pressure' in records[0] else [])
        self.values=np.array([[float(r[k]) for k in names] for r in records])
        self.coeff=np.zeros((self.n,self.n,9,len(names)))
        for i in range(0,len(records),9):
            points=self.x[i:i+9];center=points.mean(axis=0)
            ix,iy=np.rint((center+2)/self.h-.5).astype(int)
            self.coeff[ix,iy]=np.linalg.solve(basis(2*(points-center)/self.h),self.values[i:i+9])
        self.surface=np.array([[float(v) for v in row.values()] for row in read(path/f'material-{step}-surface.csv')])
        # i,x,y,Theta,Z,I,V,width,eta
        g,w=np.polynomial.legendre.leggauss(3)
        self.weights=np.array([w[int(r['q'])%3]*w[int(r['q'])//3]*self.h**2/4 for r in records])

    def at(self,x):
        ij=np.floor((x+2)/self.h).astype(int);valid=np.all((ij>=0)&(ij<self.n),axis=1)
        out=np.zeros((len(x),self.values.shape[1]));i,j=ij[valid].T
        unit=2*((x[valid]+2)/self.h-ij[valid]) - 1
        out[valid]=np.einsum('ni,nij->nj',basis(unit),self.coeff[i,j])
        return out

    def memory(self,x,phi,which=4):
        s=self.surface;X=s[:,1:3];z=s[:,which];I=s[:,5];width=s[:,7]
        best=np.full(len(x),np.inf);out=np.zeros((len(x),3))
        for j in range(len(X)-1):
            d=X[j+1]-X[j];t=d/np.linalg.norm(d);n=np.array([-t[1],t[0]])
            xi=((x-X[j])@d)/(d@d);dist=np.abs((x-X[j])@n)
            select=(xi>=0)&(xi<=1)&(dist<best)&(dist<=(1-xi)*width[j]+xi*width[j+1])
            q=xi[select];p=phi[select]
            h=self.m*p*(1+p)/(1-p)**2
            amp=2*h*((1-q)*z[j]+q*z[j+1])/((1-q)*I[j]+q*I[j+1])
            S=np.array([t[0]*n[0],t[1]*n[1],.5*(t[0]*n[1]+t[1]*n[0])])
            out[select]=amp[:,None]*S;best[select]=dist[select]
        return out

    def stress(self,x):
        v=self.at(x)
        return v[:,3:6] if self.direct else v[:,3:6]-v[:,6:9]-self.memory(x,v[:,0])

def rotation(tau,angle):
    c=np.cos(angle);s=np.sin(angle);xx,yy,xy=tau.T
    return np.stack([c*c*xx+s*s*yy-2*c*s*xy,s*s*xx+c*c*yy+2*c*s*xy,c*s*(xx-yy)+(c*c-s*s)*xy],axis=1)

def assess(name):
    p=RUNS/name;command=json.loads((p/'command.json').read_text());assert command['exit_code']==0
    folder=p/'output';metadata={k:float(v) for k,v in read(folder/'material-model.csv')[0].items()}
    summary=read(folder/'material-summary.csv')
    if summary and 'step' not in summary[0]:
        # Early restart checker appended to a new output directory without
        # writing a header. Preserve those original exports verbatim.
        with (folder/'material-summary.csv').open() as f:
            summary=list(csv.DictReader(f,fieldnames=['step','time','dt','history_publications','geometry_publications','retries']))
    G=metadata['G'];m=metadata['m'];level=command['arguments']['level']
    states={int(r['step']):State(folder,int(r['step']),level,m) for r in summary}
    results=[]
    for row in summary:
        step=int(row['step']);new=states[step]
        recon=norm3(new.stress(new.x)-new.values[:,9:12]).max()
        assert recon<1e-10,(name,step,'independent localization reconstruction',recon)
        if step-1 not in states:continue
        old=states[step-1];dt=float(row['dt']);vold=old.at(new.x)
        foot=new.x-dt*vold[:,1:3]
        anchor=rotation(old.stress(foot),.5*dt*(vold[:,14]-vold[:,13]))
        v=new.values;beta=np.exp(-dt*G/v[:,16]);kappa=-v[:,16]*np.expm1(-dt*G/v[:,16])
        trace=v[:,12]+v[:,15]
        D=np.stack([v[:,12]-trace/3,v[:,15]-trace/3,.5*(v[:,13]+v[:,14])],axis=1)
        expected=beta[:,None]*anchor+2*kappa[:,None]*D-kappa[:,None]*new.memory(new.x,v[:,0],6)
        error=norm3(v[:,9:12]-expected);scale=norm3(expected)
        interior=np.max(np.abs(new.x),axis=1)<1.25
        material_x=new.x[:,0]-command['arguments']['speed']*float(row['time'])
        center=interior & (np.abs(material_x)<.55) & (np.abs(new.x[:,1]-.3*material_x)<.4)
        relative=lambda mask:float(np.sqrt(np.sum(new.weights[mask]*error[mask]**2)/max(1e-300,np.sum(new.weights[mask]*scale[mask]**2))))
        omitted_R=norm3(v[:,3:6]-new.memory(new.x,v[:,0])-expected)
        phase_one=new.values[:,0]-old.at(foot)[:,0]
        X0=old.surface[:,1:3];X1=new.surface[:,1:3]
        curve_error=np.linalg.norm(X1-X0-dt*old.at(X0)[:,1:3],axis=1).max()
        V=new.surface[:,6];Theta=old.surface[:,3];Z=old.surface[:,4];eta=new.surface[:,8]
        age=Theta*np.exp(-V*dt/metadata['Dc'])-metadata['Dc']/V*np.expm1(-V*dt/metadata['Dc'])
        z=np.exp(-G*dt/eta)*Z-eta*np.expm1(-G*dt/eta)*V
        age_error=np.abs(age-new.surface[:,3]).max();z_error=np.abs(z-new.surface[:,4]).max()
        # Q1 remapping is not pointwise characteristic equality inside cells;
        # evaluate its interpolation defect separately from scalar sources.
        crossing=int(np.any(np.floor((X0+2)/new.h)!=np.floor((X1+2)/new.h),axis=1).sum())
        results.append(dict(step=step,time_s=float(row['time']),dt_s=dt,
          reconstruction_Linf_Pa=float(recon),curve_reference_Linf_m=float(curve_error),
          age_reaction_Linf_s=float(age_error),memory_reaction_Linf_Pa_m=float(z_error),
          crossed_surface_labels=crossing,
          direct_characteristic_L2_Pa=float(np.sqrt(np.sum(new.weights*error**2))),
          direct_characteristic_relative_L2=float(np.sqrt(np.sum(new.weights*error**2)/max(1e-300,np.sum(new.weights*scale**2)))),
          interior_characteristic_relative_L2=relative(interior),
          fault_center_characteristic_relative_L2=relative(center),
          omitted_R_relative_L2=float(np.sqrt(np.sum(new.weights*omitted_R**2)/max(1e-300,np.sum(new.weights*scale**2)))),
          direct_characteristic_Linf_Pa=float(error.max()),
          profile_remap_L2=float(np.sqrt(np.sum(new.weights*phase_one**2))),
          maximum_Eulerian_phase_decrease=float(np.max(old.values[:,0]-new.values[:,0])),
          history_publications=int(row['history_publications']),geometry_publications=int(row['geometry_publications']),retries=int(row['retries'])))
        assert curve_error<1e-12 and age_error<1e-12 and z_error<1e-12,(name,results[-1])
    inputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'command.json',*folder.glob('material-*.csv')]}
    return dict(case=name,metadata=metadata,command=command,steps=results,input_sha256=inputs),states

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cases',nargs='+');a=parser.parse_args()
    result=[];states={}
    for name in a.cases:
        r,s=assess(name);result.append(r);states[name]=s
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(cases=result,reference='Independent Q2 evaluation, geometric localization, characteristic pullback, closed-form rotation and exponential Maxwell/age/memory reactions.',
      scope='Curve/scalar/publication identities have arithmetic screens. Characteristic tensor and Q1 remapping errors are measured discretization errors; this is not a production or finite-domain Galilean qualification.')
    (OUT/'material-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for c in result:
        name=c['case'];rows=[r for r in c['steps'] if r['dt_s']>1e-10*c['command']['arguments']['dt']]
        s=states[name][rows[-1]['step']]
        axes[0,0].plot(s.surface[:,1],s.surface[:,2],'.-',label=name)
        axes[0,1].plot(s.surface[:,0],s.surface[:,3],'.-',label=name)
        axes[1,0].semilogy([r['time_s'] for r in rows],[r['direct_characteristic_relative_L2'] for r in rows],'.-',label=name)
        axes[1,1].semilogy([r['time_s'] for r in rows],[r['profile_remap_L2'] for r in rows],'.-',label=name)
    axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Accepted material curve');axes[0,0].set_aspect('equal')
    axes[0,1].set(xlabel='Material label',ylabel='Age (s)',title='Nonuniform physical surface history')
    axes[1,0].set(xlabel='Time (s)',ylabel='Relative L2',title='Tensor error against characteristic reference')
    axes[1,1].set(xlabel='Time (s)',ylabel='L2 profile error (m)',title='One-step Q1 remapping defect')
    for ax in axes.flat:ax.grid(alpha=.25)
    axes[0,1].legend(fontsize=7)
    fig.savefig(OUT/'material-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps([dict(case=c['case'],last=c['steps'][-1] if c['steps'] else None) for c in result]))

if __name__=='__main__':main()
