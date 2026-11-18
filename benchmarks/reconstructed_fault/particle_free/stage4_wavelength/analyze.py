#!/usr/bin/env python3
"""All responsive section ages, native response bases, frozen RSF spectra."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-wavelength-mpl')
from pathlib import Path
import sys,json,argparse,time,csv
import numpy as np
from scipy.linalg import eigh,eig
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'stage4_rsf'))
from stability import matrix_spectrum
OUT=HERE.parent/'evidence/stage4_wavelength';OUT.mkdir(exist_ok=True)
DC=3.2211126126299625e-7
data=np.load(HERE.parent/'evidence/stage4_rsf/reduced-operator.npz')
LAWS={'weakening':(DC,.02,.6),'matched_b_a':(DC,.015,float(data['mu_neutral'].flat[0])),
      'original_Dc':(.0005,.02,.6)}

def read(file):return np.genfromtxt(file,delimiter=',',names=True)

def spectrum(P,C,w,x,v,theta,law,profile=False,max_frequency=100.):
    # Whiten only coordinates, never drop weak/slow but responsive ages.
    gram=P.T@(w[:,None]*P)
    val,U=eigh(gram);assert val[0]>1e-8*val[-1],('degenerate force patterns',val)
    transform=U/np.sqrt(val)[None,:];P=P@transform;C=C@transform
    compliance=P.T@(w[:,None]*C)
    symmetry=np.linalg.norm(compliance-compliance.T)/np.linalg.norm(compliance)
    assert symmetry<2e-8
    K=np.linalg.inv(.5*(compliance+compliance.T));Psi=C@K;m=len(K)
    stiff=eigh(K,Psi.T@(w[:,None]*Psi),eigvals_only=True)
    dc,b,mu=LAWS[law]
    # Construct the same matrix as the retained analysis, but compute its
    # eigendecomposition once (matrix_spectrum also computes eigenvalues).
    from stability import friction
    _,fv,ft=friction(v,theta,dc,b,mu)
    M=Psi.T@((w*fv)[:,None]*Psi);N=Psi.T*(w*ft*theta)[None,:]
    L=-np.linalg.solve(M,K);Q=-np.linalg.solve(M,N)
    T=(np.sign(v)/dc)[:,None]*Psi;r=.01
    A=np.block([[L-r*np.eye(m),Q],[-T@L,-T@Q-np.diag(abs(v)/dc)]])
    ev,Y=eig(A,overwrite_a=False,check_finite=False)
    j=np.argmax(ev.real);z=Y[:,j];delta=np.asarray(Psi@(A[:m]@z+r*z[:m]),dtype=complex)
    residual=np.linalg.norm(A@z-ev[j]*z)/(np.linalg.norm(A)*np.linalg.norm(z));assert residual<1e-12
    # Phase alignment: make the largest in-patch complex rate real positive.
    k=np.argmax(abs(delta)*(x>2.5)*(x<5.5));delta*=np.exp(-1j*np.angle(delta[k]));delta/=np.max(abs(delta))
    result=dict(growth=float(ev[j].real),imag=float(ev[j].imag),minimum_stiffness=float(stiff[0]),
                gram_condition=float(val[-1]/val[0]),symmetry=float(symmetry),eigen_residual=float(residual),ages=len(x))
    if profile:
        # Signed complex rate carrier, not the modulus/envelope. The scan
        # retains real and imaginary parts (a global phase cannot change it).
        # Do not evaluate underintegrated Fourier factors at many oscillations
        # per cell: the six Gauss labels then produce spurious quadrature aliases.
        inside=(x>2.5)&(x<5.5);f=np.linspace(1/6,max_frequency,round(60*max_frequency))
        projection=np.exp(-2j*np.pi*f[:,None]*(x[inside]-2.5)[None,:])@(w[inside]*delta[inside])
        reverse=np.exp(-2j*np.pi*f[:,None]*(x[inside]-2.5)[None,:])@(w[inside]*delta[inside].conj())
        power=abs(projection)**2+abs(reverse)**2
        frequency=f[np.argmax(power)]
        result['signed_rate_wavelength']=float(1/frequency)
        reference=np.sum(w[inside]*delta[inside]*np.cos(2*np.pi*frequency*(x[inside]-4)))
        if abs(reference)>1e-14:delta*=np.exp(-1j*np.angle(reference))
        central=(x>3)&(x<5);xx=x[central];yy=delta.real[central]
        crossing=np.flatnonzero(yy[:-1]*yy[1:]<0)
        zeros=xx[crossing]-yy[crossing]*(xx[crossing+1]-xx[crossing])/(yy[crossing+1]-yy[crossing])
        result['central_signed_zero_crossings']=len(zeros)
        result['central_zero_crossing_wavelength']=float(2*np.median(np.diff(zeros))) if len(zeros)>3 else None
    return result,delta

def mesh(nx,suffix=''):
    started=time.monotonic();run=ROOT/f'build-particle-free/bulk-tip/runs/stage4-wavelength-{nx}{suffix}'
    directory=ROOT/f'build-particle-free/stage4-wavelength/input-{nx}'
    s=read(run/'output/restoring-sections.csv');raw=read(run/'output/restoring-modes.csv');checks=read(run/'output/restoring-checks.csv')
    waves=np.loadtxt(directory/'patterns.txt',skiprows=1,dtype=int);n=len(s);m=len(waves)
    assert len(raw)==m*n and len(checks)==m
    P=raw['force_basis_per_sqrt_m'].reshape(m,n).T;C=raw['slip_response_per_force_coefficient'].reshape(m,n).T
    operating=read(directory/'operating.csv');assert np.max(abs(operating['x']-s['x']))<1e-12
    active=s['cohesive_compliance_bound_m_Pa']>0
    assert np.max(abs(C[~active]))==0
    w=s['ds'][active];x=s['x'][active];v=operating['V'][active];theta=operating['theta'][active]
    P=P[active];C=C[active]
    rows=[];profiles=[]
    groups=[('smooth6',list(range(6)),6.)]
    for wave in sorted(set(waves[6:,0])):
        ids=list(np.flatnonzero(waves[:,0]==wave));groups += [(f'pair{wave}',ids,6/wave)]
        groups += [(f'smooth6_pair{wave}',list(range(6))+ids,6/wave)]
    for last in sorted(set(waves[6:,0])):
        ids=list(np.flatnonzero(waves[:,0]<=last));groups += [(f'coupled_through{last}',ids,6/last)]
    for name,ids,wavelength in groups:
        for law in LAWS:
            result,delta=spectrum(P[:,ids],C[:,ids],w,x,v,theta,law,profile=name.startswith('coupled_'),max_frequency=min(nx/8,1.5/wavelength))
            row=dict(nx=nx,hx=8/nx,space=name,law=law,shortest_trial_wavelength=wavelength,
                     cells_per_trial_wavelength=wavelength/(8/nx),patterns=len(ids),**result)
            rows.append(row)
            if name.startswith('coupled_') and law=='weakening':
                file=OUT/f'mode-{nx}-{name}.csv'
                np.savetxt(file,np.c_[x,delta.real,delta.imag],delimiter=',',header='x,rate_real,rate_imag',comments='')
                profiles.append(str(file.name))
            print(nx,name,law,round(result['growth'],6),flush=True)
    # Single-pattern labels never confused with the coupled pair spectrum.
    for j,(wave,shift) in enumerate(waves):
        for law in LAWS:
            result,_=spectrum(P[:,[j]],C[:,[j]],w,x,v,theta,law)
            rows.append(dict(nx=nx,hx=8/nx,space=f'single_{wave}_{shift}',law=law,
                shortest_trial_wavelength=6/wave,cells_per_trial_wavelength=6/wave/(8/nx),patterns=1,**result))
    record=dict(nx=nx,rows=rows,profiles=profiles,analysis_seconds=time.monotonic()-started,
         command=json.loads((run/'command.json').read_text()),transfer=json.loads((directory/'transfer.json').read_text()),
         checks={name:float(np.max(checks[name])) for name in ['relative_equilibrium_residual','relative_power_error','finite_difference_relative_error']},
         action_seconds=float(sum(checks['seconds'])),all_force_gram_eigenvalues=eigh(P.T@(w[:,None]*P),eigvals_only=True).tolist())
    if nx==128:
        old=ROOT/'build-particle-free/bulk-tip/runs/stage4-fixed-phase-stiffness-homogeneous/output'
        oldraw=read(old/'restoring-modes.csv');oldC=oldraw['slip_response_per_force_coefficient'].reshape(7,n).T[active]
        selected=list(range(6))+[int(np.flatnonzero((waves[:,0]==48)&(waves[:,1]==0))[0])]
        record['retained_action_relative_errors']=(np.sqrt(np.sum(w[:,None]*(C[:,selected]-oldC)**2,axis=0)/np.sum(w[:,None]*oldC**2,axis=0))).tolist()
    (OUT/f'spectrum-{nx}.json').write_text(json.dumps(record,indent=2)+'\n')
    with (OUT/f'spectrum-{nx}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('nx',type=int);p.add_argument('--suffix',default='');a=p.parse_args();mesh(a.nx,a.suffix)
