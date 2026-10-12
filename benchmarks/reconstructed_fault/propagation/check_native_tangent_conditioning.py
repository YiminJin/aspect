#!/usr/bin/env python3
"""Matching 1D free-phase preconditioned spectrum; not a native full spectrum."""
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import native_endpoint as native
import asymmetric_branch as old

E=Path(__file__).resolve().parent/'native_tangent_evidence'
p=native.SharedFE();r=p.r
f=np.load(native.E/'dt-1-1024/fields.npz');h=old.read(native.E/'dt-1-1024/history.json')
rows=[]
for label,j in [('growing',560),('asymmetric',608)]:
    p.incoming(f['phase'][j-1],f['R'][j-1],float(f['Z'][j-1]),h[j-2]['slip'],j/1024,1/1024)
    phi=f['phase'][j];H=p.hessian(phi)
    free=np.flatnonzero(phi>p.lower+1e-13)
    P=r.F/old.A+np.diag(r.mass)
    ev=eigh(H[np.ix_(free,free)],P[np.ix_(free,free)],eigvals_only=True)
    rows.append(dict(state=label,free=len(free),minimum=float(ev[0]),maximum=float(ev[-1]),
                     condition=float(ev[-1]/ev[0]),symmetry=float(max(abs((H-H.T).ravel())))))
old.write(E/'prototype-conditioning.json',rows)
print(old.json.dumps(rows))
