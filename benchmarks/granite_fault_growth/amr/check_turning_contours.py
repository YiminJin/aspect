"""A bent connected excursion, hanging faces, a detached island and a turned material graph."""
from types import SimpleNamespace

import numpy as np

from contours import seed_front
from granite_history import graph


def check():
    origin=np.array([10.,20.])
    cells=[]
    def add(x,y,h,row):
        cells.append((origin+[x,y],np.full(2,h),np.tile(row,(3,1))))
    add(0,0,1.,[.1]*3)
    add(1,0,1.,[.1]*3)
    for dx in (0.,.5):
        for dy in (0.,.5):
            add(1+dx,1+dy,.5,[.1]*3)
    # Reach this taper by turning onto the upper row across a hanging face.
    add(2,1,1.,[.08,.04,0.])
    add(4,1,1.,[.1]*3)
    anchor=origin+[.5,.5]
    for n in (8,16):
        for level in (.01,.05):
            x,y,upper,diagonal=seed_front(cells,origin,anchor,level,n)
            np.testing.assert_allclose(x,origin[0]+3-level/.08,rtol=0,atol=3e-15)
            assert y>=21 and upper==15. and diagonal<=np.sqrt(2)*.5/n
            print(f"{n} intervals/finest edge, level {level}: connected x={x}, detached upper bound={upper} m")
    case=SimpleNamespace(seed=origin+[.75,.5],origin=origin,t=np.array([1.,0.]),
                         n=np.array([0.,1.]),ell=.25,seed_s=.75)
    strips=dict(xi=np.array([.5,1.25,1.75]),formed=np.ones(3),beta=np.arctan([0.,.5,.5]))
    tip,tracked,length,along,_=graph(case,strips,cells)
    np.testing.assert_allclose(tip,origin+[1.75,1.0],rtol=0,atol=1e-14)
    np.testing.assert_allclose(tracked,tip)
    np.testing.assert_allclose([length,along(1.75)],[np.hypot(1,.5)]*2,rtol=0,atol=1e-14)
    print(f"Formed and trace-supported graph advance 1 m in x and {length} m along fault")


if __name__=="__main__":
    check()
