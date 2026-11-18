#!/usr/bin/env python3
"""Reduce three native static preload solves, never infer load from strength."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-preload-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BUILD=ROOT/'build-particle-free/stage4-preload';RUNS=ROOT/'build-particle-free/bulk-tip/runs'
OUT=HERE.parent/'evidence/stage4_preload';OUT.mkdir(exist_ok=True)
H=1/32
def scan(label,step=0):
    out=RUNS/('stage4-preload-'+label)/'output'
    files=sorted(out.glob(f'preload-scan-{step:04d}-r*.csv'));assert len(files)==4
    a=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1) for p in files])
    a=a[np.lexsort((a[:,2],a[:,0],a[:,1]))];assert len(a)==128*256*9
    return a
def direction(a,center,width=.0625):
    x=a[:,0]+(a[:,2]%3-1)*H/2;y=a[:,1]+(a[:,2]//3-1)*H/2
    return np.maximum(0,1-np.abs(x-center)/width)*np.maximum(0,1-np.abs(y-2)/.125)
def roots(coeff):
    # Polynomial is in gamma/1e-4. Find its first positive crossing.
    rows=np.atleast_2d(coeff);out=[]
    for c in rows:
        r=np.roots(c);r=[x.real for x in r if abs(x.imag)<1e-8 and x.real>0]
        out.append(min(r)*1e-4 if r else float('nan'))
    return np.array(out)
def main():
    status=json.loads((BUILD/'calibration-finished.json').read_text());assert status['complete']
    gammas=np.array([.0002,.0005,.0008]);a=[scan(f'cal-{i}') for i in range(3)]
    for b in a[1:]:assert np.max(abs(a[0][:,:6]-b[:,:6]))<1e-12
    x=a[0][:,0];y=a[0][:,1];mass=a[0][:,4];fracture=a[0][:,5]
    coords=np.arange(5.875,7.501,H)
    directions=np.array([direction(a[0],c) for c in coords]);measures=directions@mass
    curves=np.array([directions@(b[:,7]-fracture)/measures for b in a])
    probes={}
    for name,center in [('tip_cap',6.03125),('ahead',6.09375),('intact',6.375)]:
        chi=direction(a[0],center);area=chi@mass
        net=np.array([chi@(b[:,7]-fracture)/area for b in a])
        coef=np.polyfit(gammas/1e-4,net,2)
        probes[name]=dict(center=center,area=area,net=net.tolist(),polynomial=coef.tolist(),estimated_zero=roots(coef)[0])
    # Native row directions are independent nonnegative DG Bernstein bases.
    # Do not confuse their arbitrary small-amplitude activation with finite
    # extension of the absolute phi=.05 connected contour.
    cellB=a[0][:,3].reshape(-1,9)
    intact=np.repeat(np.max(abs(cellB),axis=1)==0,9)
    masks=dict(central_band=(abs(x-4)<1.5)&(abs(y-2)<.25),
               old_taper=(x>=5.5)&(x<6.3125)&(abs(y-2)<.25),
               axial_tip_cells=(x==6+H/2)&(abs(y-2)<H),
               connected_cap=(x>=6)&(x<6.15625)&(abs(y-2)<.125),
               exact_intact_ahead=intact&(x>=6.3125)&(x<6.5)&(abs(y-2)<.125),
               far_ahead=intact&(x>=6.5)&(x<7.5)&(abs(y-2)<.125),
               remote=(abs(x-4)>3)|(abs(y-2)>1))
    rows={}
    for name,mask in masks.items():
        rows[name]=[]
        for b,gamma in zip(a,gammas):
            net=(b[:,7]-fracture)/mass;index=np.flatnonzero(mask)[np.argmax(net[mask])]
            rows[name].append(dict(gamma=gamma,max_net=float(net[index]),location=b[index,:3].tolist(),positive_rows=int(np.sum(net[mask]>1e-8))))
    envelope=np.array([b[:,8]-fracture for b in a]);coef=np.linalg.solve(np.vander(gammas/1e-4,3),envelope).T
    selected=np.array([b[:,7]-fracture for b in a]);selected_coef=np.linalg.solve(np.vander(gammas/1e-4,3),selected).T
    chosen=intact&(x>=6.3125)&(abs(y-2)<.125)
    # Lowest positive root is a conservative all-orientation bound; verify the
    # chosen state's actual selected driving in its fresh native run.
    ids=np.flatnonzero(chosen);rr=roots(coef[chosen]);ii=int(np.nanargmin(rr))
    selected_roots=roots(selected_coef[chosen]);selected_index=int(np.nanargmin(selected_roots))
    tip_thresholds={}
    for name,center in [('right',6+H/2),('left',2-H/2)]:
        mask=(x==center)&(abs(y-2)<H);indices=np.flatnonzero(mask)
        zeros=roots(selected_coef[mask]);k=int(np.nanargmin(zeros))
        tip_thresholds[name]=dict(zero=float(zeros[k]),row=a[0][indices[k],:4].tolist(),
                                 polynomial=(selected_coef[indices[k]]/mass[indices[k]]).tolist())
    result=dict(gammas=gammas.tolist(),probes=probes,row_regions=rows,axial_tip_cell_thresholds=tip_thresholds,
                earliest_intact_envelope_zero=float(rr[ii]),envelope_zero_location=a[0][ids[ii],:3].tolist(),
                earliest_intact_selected_estimated_zero=float(selected_roots[selected_index]),selected_zero_location=a[0][ids[selected_index],:3].tolist(),
                selected_zero_same_row=float(roots(selected_coef[ids[ii]])[0]),
                note='Selected-plane quadratic interpolation is predictive only; virgin optimum can change. Envelope is quadratic in the native affine stress response. Fresh chosen preload verifies margins.')
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'calibration-arrays.npz',gammas=gammas,coordinates=coords,curves=curves,rows=np.stack(a))
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,g in enumerate(gammas):
        axs[0].plot(coords,curves[i],label=fr'$\gamma_e={g:g}$')
    axs[0].axhline(0,color='k',lw=.7);axs[0].axvline(6.00205438,color='gray',ls='--',label='Initial .05 end')
    axs[0].axvspan(6.3125,7.5,color='gray',alpha=.08);axs[0].set(xlabel='Probe center x (m)',ylabel='Mean net damage driving (Pa)',title='Fixed-width admissible axial increments');axs[0].legend(fontsize=8)
    gs=np.linspace(gammas[0],gammas[-1],250)
    for name,p in probes.items():
        line,=axs[1].plot(gs*1e4,np.polyval(p['polynomial'],gs/1e-4),label=name.replace('_',' '))
        axs[1].plot(gammas*1e4,p['net'],'o',ms=4,color=line.get_color())
    axs[1].plot(gs*1e4,np.polyval(tip_thresholds['right']['polynomial'],gs/1e-4),'k--',label='First axial-tip-cell row')
    axs[1].axvline(3.65,color='gray',ls=':',label='Selected preload')
    axs[1].axhline(0,color='k',lw=.7);axs[1].set(xlabel=r'Elastic preload amplitude $10^4\gamma_e$',ylabel='Mean net damage driving (Pa)',title='Native calibration, fixed phase');axs[1].legend()
    for ext in ('png','pdf'):fig.savefig(OUT/f'calibration.{ext}',dpi=170)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
