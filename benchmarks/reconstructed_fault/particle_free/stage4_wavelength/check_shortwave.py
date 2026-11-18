#!/usr/bin/env python3
"""Independent matrix/root and polynomial-projection checks for the limit."""
from shortwave import *
from scipy.linalg import eigvals

z,w=np.polynomial.legendre.leggauss(32)
N=np.c_[np.ones_like(z),z];M=N.T@(w[:,None]*N)
projection={}
for name,f,expected in [('sine',np.sin(np.pi*(z+1)/2),8/np.pi**2),
                       ('cosine',np.cos(np.pi*(z+1)/2),96/np.pi**4)]:
    b=N.T@(w*f);rho=b@np.linalg.solve(M,b)/np.sum(w*f*f)
    assert abs(rho-expected)<1e-13
    projection[name]=float(rho)
z,w=np.polynomial.legendre.leggauss(10)
x=((np.arange(40,88)[:,None]+.5+.5*z)*.0625).ravel()
measure=np.tile(w*.0625/2,48)/np.cos(np.interp(x,sections['x'],sections['angle']))
C=np.array([limits(v)[2] for v in x]);I=sum(measure*C)
V=np.interp(x,operating['x'],operating['V']);theta=np.interp(x,operating['x'],operating['theta'])
checks=[]
for name,dc,b,mu in [('weakening',DC,.02,.6),('matched_b_a',DC,.015,.6080471895621705),('original_Dc',.0005,.02,.6)]:
    _,fv,ft=friction(V,theta,dc,b,mu)
    mass=sum(measure*C*C*fv);L=np.array([[-I/mass]])
    Q=(-C*measure*ft*theta/mass)[None,:];T=(np.sign(V)*C/dc)[:,None]
    A=np.block([[L-R,Q],[-T@L,-T@Q-np.diag(abs(V)/dc)]])
    ev=eigvals(A,check_finite=False);leading=ev[np.argmax(ev.real)]
    row=dict(law=name,all_age_matrix_growth=float(leading.real),imag=float(abs(leading.imag)),ages=len(x))
    if name=='weakening':
        reference=carrier_root(x,measure,C,C*C)
        row['rational_root_error']=float(abs(leading.real-reference['growth'])+abs(abs(leading.imag)-reference['imag']))
        assert row['rational_root_error']<1e-8
    else:assert leading.real<1e-8
    checks.append(row)
record=json.loads((OUT/'compatible-limit.json').read_text());rows=record['point_rows']
convergence=[]
for wave in sorted(set(r['wavelength'] for r in rows)):
    selected=[r for r in rows if r['wavelength']==wave and r['quadrature']==10]
    final=next(r for r in selected if r['ny']==512)
    convergence.append(dict(wavelength=wave,relative_compliance_256_to_512=abs(next(r for r in selected if r['ny']==256)['compliance']/final['compliance']-1)))
result=dict(projection_factors=projection,full_age_matrix_checks=checks,normal_convergence=convergence,
 maximum_work_error=max(r['work_error'] for r in rows),maximum_residual=max(r['residual'] for r in rows))
(OUT/'reference-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
