#!/usr/bin/env python3
"""Actual native slip-gradient data; no fitted length or continuum reference."""
from pathlib import Path
import csv,json,sys,time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import coulomb_localization_native as native

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/coulomb-gradient'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/coulomb_gradient'
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def col(rows,key):return np.array([float(r[key]) for r in rows])
def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def conditioning(case,step):
    run=WORK/'runs'/case;stem=f'localization-{step:04}-converged';start=time.monotonic()
    coords=read(run/(stem+'-coordinates.csv'));m=sum(r['kind']!='micro' for r in coords)
    J=native.matrix_file(run/(stem+'.bin'));S=J[:m,:m].toarray()
    if len(coords)>m:
        inverse=sla.splu(J[m:,m:].tocsc());right=J[m:,:m];left=J[:m,m:]
        for j in range(0,m,64):S[:,j:j+64]-=left@inverse.solve(right[:,j:j+64].toarray())
    original=native.read
    def filtered(path):
        rows=original(path)
        return [r for r in rows if r['kind']!='micro'] if str(path).endswith('-coordinates.csv') else rows
    native.read=filtered
    try:R,_=native.assemble(run,stem)
    finally:native.read=original
    L=la.cholesky(R,lower=True);A=la.solve_triangular(L,S,lower=True);A=la.solve_triangular(L,A.T,lower=True).T
    s=la.svdvals(A,check_finite=False)
    result=dict(case=case,step=step,mechanical_unknowns=m,micro_unknowns=len(coords)-m,
                Riesz_sigma_min=float(s[-1]),Riesz_sigma_max=float(s[0]),Riesz_condition=float(s[0]/s[-1]),seconds=time.monotonic()-start,
                scope='Actual mixed Schur complement after joint K elimination; H1/L2 norms. Singular values are not a stability certificate.')
    (OUT/f'native-condition-{case}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def summarize():
    rows=[];profiles=[]
    for case in ['local','n4','n4-checked','n8','n16-lifting','n16-retry','n16-final','n8-half','n8-l3','homogeneous','hold','resume']:
        run=WORK/'runs'/case
        if not (run/'bulk-accepted.csv').exists():continue
        ledger=read(run/'bulk-accepted.csv');grad=read(run/'slip-gradient.csv') if (run/'slip-gradient.csv').exists() else []
        for item in ledger:
            step=int(item['step']);file=run/f'coulomb-{step:04}.csv'
            if not file.exists():continue
            data=read(file);w=col(data,'JxW_m2');rate=2*np.hypot(col(data,'Wxx_s-1'),col(data,'Wxy_s-1'));mean=w@rate/w.sum()
            W=np.array([col(data,'Wxx_s-1'),col(data,'Wxy_s-1')]).T;fluct=2*la.norm(W-(w@W/w.sum()),axis=1)
            g=next((r for r in grad if int(r['step'])==step),{})
            row=dict(case=case,step=step,time_s=float(item['time']),dt_s=float(item['dt']),cells=len({r['cell'] for r in data}),points=len(data),
                     ell_s_m=float(g.get('length',0.)),mean_slip_rate_s1=mean,peak_slip_rate_s1=float(max(rate)),
                     participation_width_m=float((w@rate)**2/(w@(rate*rate))/.25) if w@(rate*rate)>0 else 0.,
                     fluctuation_width_m=float((w@fluct)**2/(w@(fluct*fluct))/.25) if w@(fluct*fluct)>0 else 0.,
                     gradient_energy_J_m=float(g.get('energy',0.)),gradient_BE_loss_J_m=float(g.get('BE_loss',0.)),
                     microforce_error=float(g.get('microforce_error',0.)),mechanical_residual=float(item['mechanical_residual']),
                     source_error=float(item['projection_error']),ledger_error_J_m=float(item['ledger_error']),
                     min_contact_Pa=float(min(col(data,'min_plane_normal_Pa'))),max_yield_margin_Pa=float(max(max(col(data,'f_plus_Pa')),max(col(data,'f_minus_Pa')))),
                     bulk_J_m=float(item['bulk']),cohesive_J_m=float(item['cohesive']),Wext_J_m=float(item['Wext']),
                     Db_J_m=float(item['Db']),Dc_J_m=float(item['Dc']),Df_J_m=float(item['Df']))
            rows.append(row)
            if abs(row['time_s']-.1125)<1e-12 or abs(row['time_s']-.125)<1e-12 or (case=='hold' and step>=40):
                position=np.array([col(data,'x_m'),col(data,'y_m')]).T
                # Same physical cross-band Gaussian kernel as the retained audit.
                theta=2.913739130895119;s=position@np.array([np.cos(theta),np.sin(theta)]);sigma=.046875/2
                for center in np.linspace(-.25,.25,151):
                    kernel=np.exp(-.5*((s-center)/sigma)**2);mass=w@kernel
                    profiles.append(dict(case=case,time_s=row['time_s'],normal_coordinate_m=center,kernel_sigma_m=sigma,wave_normal_rad=theta,
                                         kernel_mass_m2=mass,slip_rate_s1=float((w*kernel)@rate/mass),stress_xy_Pa=float((w*kernel)@col(data,'tauxy_Pa')/mass)))
    write(OUT/'native-history.csv',rows);write(OUT/'native-profiles.csv',profiles)
    print(json.dumps([r for r in rows if abs(r['time_s']-.125)<1e-12],indent=2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='condition':conditioning(sys.argv[2],int(sys.argv[3]))
    else:summarize()
