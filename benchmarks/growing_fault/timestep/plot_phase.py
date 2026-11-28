"""Map the two accepted phase increments from their common 5 ms obstacle."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parameter_file import strip_material
from audit import Frame, shapes
from frames import aligned


def grid(nodes):
    x,y=np.meshgrid(nodes,nodes)
    return x,y,shapes(x.ravel(),y.ravel(),[1,1])[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    root=args.runs
    seed=Frame(root/'parent/output',0)
    parent=Frame(root/'parent/output',1)
    fine_parent=Frame(root/'half/output',1)
    aligned(parent.mesh,fine_parent.mesh,['cell','component','local','value'])
    coarse=Frame(root/'coarse/output',2)
    half=Frame(root/'half/output',3)
    for frame in (coarse,half):
        aligned(parent.mesh,frame.mesh,['cell','component','local','x0','x1','y0','y1'])
        aligned(parent.strip,frame.strip,['section','xi'])
    np.testing.assert_allclose([parent.time,coarse.time,half.time],[.005,.01,.01],atol=1e-15,rtol=0)
    formed=[frame.strip['formed']-parent.strip['formed'] for frame in (coarse,half)]
    for flags in formed:
        np.testing.assert_array_equal(flags,0)
    mask=parent.mesh['component']==parent.phase_component
    cells,indices,counts=np.unique(parent.mesh['cell'][mask],return_index=True,return_counts=True)
    np.testing.assert_array_equal(counts,9)
    np.testing.assert_array_equal(parent.mesh['local'][mask],np.tile(np.arange(9),len(cells)))
    bounds=np.array([parent.mesh[k][mask][indices] for k in ('x0','y0','x1','y1')]).T
    x,y,display=grid(np.linspace(0,1,19))
    nodes,weights=np.polynomial.legendre.leggauss(3)
    _,_,gauss=grid((nodes+1)/2)
    diff=(half.coeff-coarse.coeff).reshape(-1,9)
    measure=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    # Squaring a tensor Q2 field gives degree four per axis. Mapped Gauss(3)
    # integrates its physical-area norm exactly up to arithmetic roundoff.
    distance=np.sqrt(measure@((diff@gauss.T)**2@(np.outer(weights,weights).ravel()/4)))
    aligned(half.point,coarse.point,['cell','q','weight'])
    native=np.sqrt(half.point['weight']@(half.point['phi']-coarse.point['phi'])**2)
    np.testing.assert_allclose(distance,native,rtol=0,atol=1e-14)
    print(f'10 ms L2 distance: Gauss(3) {distance:.16g} m, native {native:.16g} m')
    increments=[(frame.coeff-parent.coeff).reshape(-1,9) for frame in (coarse,half)]
    values=[v@display.T for v in (*increments,diff)]
    band=(parent.coeff.reshape(-1,9)@display.T).reshape(-1,*x.shape)
    # Protect the zero end of the common increment scale. Roundoff in a
    # displayed sample may fall just below zero at an unchanged obstacle.
    if min(v.min() for v in values[:2]) < -2e-15:
        raise ValueError('Accepted phase decreased relative to the common obstacle')
    posmax=max(np.max(v) for v in values[:2])
    symmax=np.max(abs(values[2]))
    pos=np.linspace(0,np.ceil(posmax*100)/100,21)
    signed=np.linspace(-np.ceil(symmax*100)/100,np.ceil(symmax*100)/100,24)
    lo,hi=bounds[:,:2].min(axis=0),bounds[:,2:].max(axis=0)
    parameters=json.loads((root/'parent/output/parameters.json').read_text())
    angle=np.radians(float(strip_material(parameters)['Seed tangent angle']['value']))
    # Actual transverse strip boundaries are level lines of t dot (x-origin).
    edges=seed.strip['xi'][:-1]+seed.strip['ds'][:-1]/2
    np.testing.assert_allclose(edges,seed.strip['xi'][1:]-seed.strip['ds'][1:]/2,rtol=0,atol=1e-14)
    fig,axes=plt.subplots(1,3,figsize=(12.6,3.9),layout='constrained',sharey=True)
    figures=[]
    for ax,value,levels,cmap,title in zip(axes,values,[pos,pos,signed],['YlGnBu','YlGnBu','RdBu_r'],
            ['One 5 ms step','Two 2.5 ms steps','Two steps minus one'],strict=True):
        for (x0,y0,x1,y1),z,phi in zip(bounds,value,band,strict=True):
            px,py=x0+(x1-x0)*x,y0+(y1-y0)*y
            field=ax.contourf(px,py,np.maximum(levels[0],z).reshape(x.shape),levels=levels,cmap=cmap)
            if phi.min()<.05<phi.max():
                ax.contour(px,py,phi,levels=[.05],colors='#a82968',linewidths=1.4)
        figures.append(field)
        for edge in edges:
            yy=np.array([lo[1],hi[1]])
            xx=lo[0]+(edge-np.sin(angle)*(yy-lo[1]))/np.cos(angle)
            ax.plot(xx,yy,':',color='#555555',lw=.65,alpha=.68)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='x (m)',title=title)
    axes[0].set_ylabel('y (m)')
    fig.colorbar(figures[0],ax=axes[:2],location='bottom',shrink=.72,aspect=40,
                 label='Accepted phase increment, 5–10 ms',ticks=np.linspace(pos[0],pos[-1],5))
    fig.colorbar(figures[2],ax=axes[2],location='bottom',shrink=.9,aspect=22,
                 label='Signed phase difference at 10 ms',ticks=np.linspace(signed[0],signed[-1],5))
    fig.legend(handles=[Line2D([],[],color='#a82968',label='5 ms band, φ = 0.05'),
                        Line2D([],[],ls=':',color='#555555',label='Material strip boundaries')],
               loc='outside upper center',ncols=2,frameon=False)
    axes[0].text(.02,.94,f'{int(sum(formed[0]))} new registrations',transform=axes[0].transAxes,va='top',fontsize=9)
    axes[1].text(.02,.94,f'{int(sum(formed[1]))} new registrations',transform=axes[1].transAxes,va='top',fontsize=9)
    args.destination.mkdir(parents=True,exist_ok=True)
    for fmt in ('pdf','png'):
        fig.savefig(args.destination/f'phase-increments.{fmt}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    main()
