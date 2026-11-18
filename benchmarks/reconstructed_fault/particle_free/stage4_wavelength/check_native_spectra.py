#!/usr/bin/env python3
"""Check growing native roots in the small physical dispersion matrix.

Use original force coordinates, independently of the full-age matrix's Gram
whitening. This avoids relying solely on an eigensolver residual normalized
by a large, dimensionally scaled full-state matrix.
"""
from shortwave import *
def read(path):return np.genfromtxt(path,delimiter=',',names=True)
s=read(OUT/'native-512/restoring-sections.csv');raw=read(OUT/'native-512/restoring-modes.csv')
op=read(ROOT/'build-particle-free/stage4-wavelength/input-512/operating.csv')
active=s['cohesive_compliance_bound_m_Pa']>0;n=len(s)
P=raw['force_basis_per_sqrt_m'].reshape(10,n).T[active]
C=raw['slip_response_per_force_coefficient'].reshape(10,n).T[active]
w=s['ds'][active];V=op['V'][active];theta=op['theta'][active]
_,fv,ft=friction(V,theta,DC);c=ft*theta*np.sign(V)/DC;q=abs(V)/DC
groups={'smooth6':list(range(6)),'single_96_0':[6],'single_96_1':[7],'pair96':[6,7],
 'smooth6_pair96':list(range(8)),'single_192_0':[8],'single_192_1':[9],
 'pair192':[8,9],'smooth6_pair192':list(range(6))+[8,9],'coupled_through192':list(range(10))}
records=[]
for row in json.loads((OUT/'spectrum-512.json').read_text())['rows']:
 if row['law']!='weakening':continue
 ids=groups[row['space']];Cp=P[:,ids].T@(w[:,None]*C[:,ids]);K=np.linalg.inv(.5*(Cp+Cp.T));Psi=C[:,ids]@K
 lam=complex(row['growth'],row['imag']);elastic=K/(lam+R)
 direct=Psi.T@((w*fv)[:,None]*Psi);state=Psi.T@((w*c/(lam+q))[:,None]*Psi)
 D=elastic+direct-state
 relative=float(np.linalg.svd(D,compute_uv=False)[-1]/(np.linalg.norm(elastic)+np.linalg.norm(direct)+np.linalg.norm(state)))
 assert relative<1e-9,(row['space'],relative)
 records.append(dict(space=row['space'],relative_dispersion_singular_value=relative))
(OUT/'native-root-checks.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
