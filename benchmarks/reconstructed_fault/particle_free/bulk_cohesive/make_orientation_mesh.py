#!/usr/bin/env python3
"""One conforming, fixed rotated-core mesh; no physical field is remapped."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def mesh():
    # Exactly the reference rectangular resolution in a rigid 45 degree core.
    x=np.arange(-10,11)/32;y=np.arange(-20,21)/64
    vertices=[];cells=[]
    for v in y:
        for u in x:vertices.append([.5+(u-v)/np.sqrt(2),.5+(u+v)/np.sqrt(2)])
    def index(i,j):return j*len(x)+i
    for j in range(len(y)-1):
        for i in range(len(x)-1):cells.append([index(i,j),index(i+1,j),index(i,j+1),index(i+1,j+1)])
    # Counterclockwise inner boundary, no duplicate corners.
    ring=([index(i,0) for i in range(len(x)-1)]+
          [index(len(x)-1,j) for j in range(len(y)-1)]+
          [index(i,len(y)-1) for i in range(len(x)-1,0,-1)]+
          [index(0,j) for j in range(len(y)-1,0,-1)])
    inner=np.array([vertices[i] for i in ring]);vec=inner-.5
    theta=np.arctan2(vec[:,1],vec[:,0]);outer_theta=theta.copy()
    # Attach each actual rectangle corner to its nearest inner perimeter node.
    # The small angular shifts replace a large twist across a narrow ring.
    for corner in np.array([[0,0],[1.5,0],[1.5,1],[0,1]]):
        angle=np.arctan2(corner[1]-.5,corner[0]-.5)
        diff=np.arctan2(np.sin(theta-angle),np.cos(theta-angle))
        outer_theta[np.argmin(abs(diff))]=angle
    direction=np.stack([np.cos(outer_theta),np.sin(outer_theta)],axis=-1)
    distances=np.minimum(np.divide(np.where(direction[:,0]>0,1.,.5),abs(direction[:,0]),out=np.full(len(ring),np.inf),where=abs(direction[:,0])>1e-14),
                         np.divide(.5,abs(direction[:,1]),out=np.full(len(ring),np.inf),where=abs(direction[:,1])>1e-14))
    outer=.5+direction*distances[:,None]
    # Snap outer faces exactly; IDs are assigned from physical boundary faces.
    for a in range(2):
        for b in ([0,1.5] if a==0 else [0,1]):outer[:,a][abs(outer[:,a]-b)<1e-12]=b
    for k in range(1,25):
        following=[]
        for point in (outer if k==24 else inner+(outer-inner)*k/24):
            following.append(len(vertices));vertices.append(list(point))
        for j in range(len(ring)):
            n=(j+1)%len(ring);cells.append([ring[j],following[j],ring[n],following[n]])
        ring=following
    return np.array(vertices),np.array(cells)


def main():
    vertices,cells=mesh();v=vertices[cells];minimum=np.inf;maximum=-np.inf
    # A bilinear quad determinant is affine in reference coordinates; positivity
    # at all four corners proves positivity throughout the cell, not just at QPs.
    for u in [0,1]:
        for w in [0,1]:
            a=(1-w)*(v[:,1]-v[:,0])+w*(v[:,3]-v[:,2])
            b=(1-u)*(v[:,2]-v[:,0])+u*(v[:,3]-v[:,1])
            det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
            minimum=min(minimum,float(det.min()));maximum=max(maximum,float(det.max()))
    assert minimum>0,(minimum,maximum)
    with (ROOT/'orientation.inp').open('w') as f:
        f.write(f'{len(vertices)} {len(cells)} 0 0 0\n')
        for i,p in enumerate(vertices):f.write(f'{i+1} {p[0]:.17g} {p[1]:.17g} 0\n')
        # AVS/UCD vertex order is counterclockwise, unlike deal.II tensor order.
        for i,c in enumerate(cells):f.write(f'{i+1} 0 quad {c[0]+1} {c[1]+1} {c[3]+1} {c[2]+1}\n')
    print(json.dumps(dict(vertices=len(vertices),cells=len(cells),core_cells=800,
                         corner_min_det=minimum,corner_max_det=maximum),indent=2))


if __name__=='__main__':main()
