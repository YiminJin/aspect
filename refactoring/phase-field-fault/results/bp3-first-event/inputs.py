"""Check persistent IDs, locations and ages against the preserved historical startup."""
import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

if __name__=='__main__':
    stage=Path(sys.argv[1]).resolve()
    donor=Path(sys.argv[2])
    generated=subprocess.check_output([sys.executable,'make_historical_vertices.py'],cwd=stage)
    shipped=np.loadtxt(stage/'historical-vertices.txt',skiprows=2)
    current=np.loadtxt(io.BytesIO(generated),skiprows=2)
    assert len(current)==5776
    np.testing.assert_array_equal(current[:,0],np.arange(5776))
    np.testing.assert_allclose(current[:,:3],shipped[:,:3],rtol=0,atol=1e-8)
    np.testing.assert_allclose(current[:,3],shipped[:,3],rtol=1e-13,atol=0)
    rows=np.genfromtxt(donor,delimiter=',',names=True)[::-1]
    np.testing.assert_array_equal(rows['node'],5775-current[:,0])
    xy=np.column_stack((rows['x_m'],rows['y_m']))
    pos=np.linalg.norm(current[:,1:3]-xy,axis=1)
    state=abs(current[:,3]/rows['Theta_s']-1)
    assert max(pos)<1e-7 and max(state)<2e-13,(max(pos),max(state))
    ds=np.linalg.norm(np.diff(shipped[:,1:3],axis=0),axis=1)
    print(f"5776 reversed donor IDs; maximum position error {max(pos):.12g} m; relative age error {max(state):.12g}; spacing {min(ds):.12g}..{max(ds):.12g} m")
    print("donor time-zero surface/base rates, shear and friction compression",[(r['V_m_per_s'],r['q_weak_Pa'],r['sigma_n_weak_Pa']) for r in (rows[0],rows[-1])])
