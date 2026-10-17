#!/usr/bin/env python3
"""Independent all-nine-mode mature DG evaluator at explicit physical points.

Diagnostic only. Reads native accepted coefficients; does not solve mechanics,
change normalization, or interpolate reconstructed stress samples.
"""
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
from prepare_pf6_mature_resolution import mesh,index,locate
from analyze_pf6_mature_resolution import UPPER,TANGENT,NORMAL,L


def basis(x):
    return np.column_stack((2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)))


def sample(run,step,points):
    """Return tensor/phase/pressure from independent polynomial evaluation."""
    grid=mesh(run);idx=index(grid);owners=locate(points,idx)
    nodes=read(run/'dg-support.csv');coef=read(run/f'dg-coefficients-{step}.csv')
    coeff={}
    for cell,mode,xx,yy,xy in zip(coef['cell'],coef['mode'],coef['xx'],coef['yy'],coef['xy']):
        coeff.setdefault(cell,np.zeros((9,3)))[int(mode)]=[xx,yy,xy]
    fe=read(run/f'physical-fe-{step}.csv')
    physical={c:{(x,y):v for component,x,y,v in zip(fe['component'],fe['x'],fe['y'],fe['accepted']) if component==c} for c in (2,9)}
    groups={}
    for j,(cell,lo,h) in enumerate(owners):groups.setdefault(cell,[]).append(j)
    tensor=np.empty((len(points),3));phi=np.empty(len(points));pressure=np.empty(len(points))
    bounds={cell:(lo,h) for cell,level,lo,h in grid}
    for cell,indices in groups.items():
        ids=np.array(indices);lo,h=bounds[cell];a,b=((points[ids]-lo)/h).T
        bx,by=basis(a),basis(b)
        B=np.column_stack([bx[:,int(round(2*xi))]*by[:,int(round(2*eta))] for xi,eta in zip(nodes['xi'],nodes['eta'])])
        tensor[ids]=B@coeff[cell]
        Q=np.column_stack([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b])
        corners=[tuple(lo+h*np.array([i%2,i//2])) for i in range(4)]
        phi[ids]=Q@np.array([physical[9][x] for x in corners])
        pressure[ids]=Q@np.array([physical[2][x] for x in corners])
    surface=read(run/f'coupled-profile-{step}.csv');s=(points-UPPER)@TANGENT
    Z=np.interp(s,surface['s_m'],surface['Z_Pa_m']);I=np.interp(s,surface['s_m'],surface['I_h_m'])
    # Native AT1, p=1: m=Gc/(c0*ell*Hc), Hc=cohesion²/(2G).
    m=1e5/((8/3)*20*(1e12/(2*32038120320.)))
    h=m*phi*(1+phi)/(1-phi)**2
    # Native paired completion uses constant endpoint history/I continuation.
    # Outside its association enclosure phi is zero, so stress is independent
    # of the unused continuation values there.
    r=(points-UPPER)@NORMAL
    # Retained physical normal-profile cutoff in the open fault strip; native
    # paired boundary completion adds constant endpoint traces beyond its ends.
    active=(np.abs(r)<=39.529164020107309)|(s<0)|(s>L)
    chi=np.where(active,h/I,0.)
    S=np.array([-np.sqrt(3)/4,np.sqrt(3)/4,.25]);tau=tensor-2*(chi*Z)[:,None]*S
    return dict(phi=phi,chi=chi,Z=Z,T_xx=tensor[:,0],T_yy=tensor[:,1],T_xy=tensor[:,2],
                tau_xx=tau[:,0],tau_yy=tau[:,1],tau_zz=-tau[:,0]-tau[:,1],tau_xy=tau[:,2],p=pressure)
