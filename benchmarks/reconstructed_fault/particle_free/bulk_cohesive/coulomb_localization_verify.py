#!/usr/bin/env python3
"""Independent single-plane formula, mixed null modes and coupled high-k limit."""
import hashlib,json
import numpy as np
from scipy.optimize import minimize_scalar
from coulomb_localization_analysis import ROOT,WORK,OUT,read,write,save,columns,rate_columns,symbol,minimum

rows=read(WORK/'point-states.csv');errors=[];controls=[]
for row in rows:
    h=float(row['h']);mu=float(row['mu']);active=int(row['active'])
    if h>0 and mu==.1 and active in [1,2]:
        H=1+1/h
        closed=lambda a:1-np.cos(2*a)*(np.cos(2*a)+mu*np.sin(2*a))/(H-mu*np.sin(2*a))
        exact=minimize_scalar(closed,bounds=(-.2,.2),method='bounded',options={'xatol':1e-14}).fun
        actual=minimum(rate_columns(row))[0];errors.append(abs(exact-actual))
    if mu==0:
        a=minimum(columns(row))[0];controls.append(a);assert a>0.
assert max(errors)<=1e-12,max(errors)
selected=next(r for r in rows if r['kind']=='path' and float(r['phi'])==.8 and float(r['normal_sign'])==1 and int(r['step'])==9)
root_rows=[]
for kind,C in [('actual finite step',columns(selected)),('continuous rate',rate_columns(selected))]:
    result=minimum(C);assert result[0]<0. and result[2]>0.
    for theta in result[3]:
        n=np.array([np.cos(theta),np.sin(theta)]);a=np.array([-n[1],n[0]])
        def traction(v):
            e=np.array([v[0]*n[0],v[1]*n[1],v[0]*n[1]+v[1]*n[0]]);s=C[:4,:3]@e
            return np.array([s[0]*n[0]+s[3]*n[1],s[3]*n[0]+s[1]*n[1]])
        Q=np.stack([traction(np.array([1.,0.])),traction(np.array([0.,1.]))],axis=1)
        b=np.array([[(C[0,3]-1),C[3,3]],[C[3,3],C[1,3]-1]])@n
        pressure=-float(n@traction(a))/(n@b)
        M=np.zeros((3,3));M[:2,:2]=Q;M[:2,2]=b;M[2,:2]=n
        z=np.array([*a,pressure]);defect=np.linalg.norm(M@z)/np.linalg.norm(z)
        assert defect<=1e-12
        root_rows.append({'operator':kind,'wave_normal_rad':theta,'band_tangent_rad':(theta+np.pi/2)%np.pi,
                          'u_x':a[0],'u_y':a[1],'pressure_over_G':pressure,'mixed_null_defect':defect,
                          'smallest_mixed_symbol_singular_value':float(np.linalg.svd(M,compute_uv=False)[-1])})
write('mixed-null-modes.csv',root_rows)

C=columns(selected);L0,theta,_,_,_=minimum(C);coupled=[]
for k in np.logspace(-2,5,141):
    L,_,_,_=symbol(C,theta,k);coupled.append({'k_times_ell':k,'mixed_transverse_over_G':float(L),'fixed_phi_over_G':L0,
        'phase_denominator_over_Kf':2*k*k-C[4,4]})
assert abs(coupled[-1]['mixed_transverse_over_G']-L0)<=1e-10
write('coupled-symbol.csv',coupled)
angle=[]
for theta in np.linspace(0,np.pi,1441):
    L,fixed,det,b=symbol(C,theta);rate=rate_columns(selected)
    angle.append({'wave_normal_rad':theta,'actual_mixed_over_G':float(L),'fixed_pressure_over_G':float(fixed),
                  'rate_mixed_over_G':float(symbol(rate,theta)[0]),'mixed_determinant':float(det)})
write('angular-symbol.csv',angle)

native=[]
for path in sorted(OUT.glob('*-conditioning.json')):
    r=json.loads(path.read_text());meta=json.loads((WORK/'runs'/r['case']/(r['stem']+'.json')).read_text())
    native.append({**r,'time_s':meta['time'],'dt_s':meta['dt'],'h_m':.25/int(r['case'].split('-n')[1].split('-')[0])})
write('conditioning.csv',native)
cutbacks={}
for name,record in json.loads((WORK/'cutback-inputs.json').read_text()).items():
    source=WORK/'runs'/name.replace('cutback','full')
    for filename,digest in record['source_files'].items():assert hashlib.sha256((source/'restart'/filename).read_bytes()).hexdigest()==digest
    run=WORK/'runs'/name;ledger=read(run/'bulk-accepted.csv')
    assert int(ledger[0]['publications'])==9 and abs(float(ledger[0]['time'])-.10625)<1e-12
    assert int(ledger[-1]['publications'])==12 and abs(float(ledger[-1]['time'])-.125)<1e-12
    assert all(float(r['dt'])==.00625 for r in ledger)
    cutbacks[name]={'unchanged_parent_checkpoint':True,'accepted_from_s':.1,'accepted_to_s':.125,'pending_dt_s':.00625}
save('independent-checks.json',{'single_plane_closed_formula_max_defect':max(errors),'single_plane_cases':len(errors),
    'constant_R_cases':len(controls),'minimum_constant_R_algorithmic_symbol':min(controls),
    'coupled_high_k_defect':abs(coupled[-1]['mixed_transverse_over_G']-L0),
    'phase_obstacle_residual_at_selected_state_Pa':8-float(selected['Y']),'cutbacks':cutbacks,
    'decision':'Mechanical prerequisite fails. No phase-active progression or tolerance change.'})
print('Mixed rank loss, independent rate formula, constant-R and cutback checks passed')
