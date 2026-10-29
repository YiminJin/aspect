#!/usr/bin/env python3
"""One frozen curved-geometry pressure/work comparison; no physical evolution."""
import json
import csv
import shutil
import argparse
import numpy as np
from cohesive_interface_path import ROOT, WORK, OUT, kernel, save, table
from cohesive_interface_turn import Kernel as FittedKernel


def main():
    # Exactly the previously qualified +20-degree fixed curve, with identical
    # physical boundary/trace-memory prescription and Gauss points. This is a
    # mechanical approximation control, NOT transfer of physical bulk history.
    slope=float(np.tan(np.deg2rad(20)))
    arcs=[dict(x=3.,y=0.,end=6.,a=0.,b=0.,cells=32),
          dict(x=6.,y=0.,end=7.5,a=0.,b=slope,cells=16),
          dict(x=7.5,y=.75*slope,end=9.,a=slope,b=slope,cells=16)]
    fitpath=ROOT/'build-particle-free/cohesive-interface-turn/kernel-p20-n128-w125-a420'
    fitted=FittedKernel(fitpath,True,12.,.0021)
    state=json.loads((ROOT/'build-particle-free/cohesive-interface-turn/runs/half-p20/final.json').read_text())
    c=np.r_[state['wall_memory'],state['memory']]
    fs=np.loadtxt(fitpath/'shear.csv',delimiter=',')@fitted.field_transform
    w=fitted.w;interior=(fitted.s['x_m']>=3.75)&(fitted.s['x_m']<=8.25)
    def rms(v,mask=None):
        mask=np.ones(len(v),bool) if mask is None else mask
        return float(np.sqrt(w[mask]@(v[mask]**2)/sum(w[mask])))
    reference=dict(normal=fitted.N@c,shear=fs@c,weak=(fitted.K@c)[1:]/fitted.mass)
    z=np.polynomial.legendre.leggauss(3)[0];z=(z+1)/2;bubble=4*z*(1-z)
    mass=(w.reshape(-1,3)*bubble).sum(axis=1)
    projected=np.repeat((w.reshape(-1,3)*bubble*reference['normal'].reshape(-1,3)).sum(axis=1)/mass,3)
    results=[];profiles=[]
    for n in [63,64]:
        k=kernel(arcs,n,pressure_mode='work')
        assert max(abs(k.s['x_m']-fitted.s['x_m']))<1e-12 and max(abs(k.s['y_m']-fitted.s['y_m']))<1e-12
        assert len(c)==len(k.nodes)+1
        normal=k.N@c;rawnormal=.5*(k.raw_Nm+k.raw_Np)@c;shear=np.loadtxt(k.path/'shear.csv',delimiter=',')@c;weak=(k.K@c)[1:]/k.mass
        jump=(k.raw_Nm-k.raw_Np)@c;fitjump=(fitted.Nm-fitted.Np)@c
        xn=np.r_[3.,k.nodes['x_m'],9.];m=np.r_[0.,c[1:],0.]
        indices=np.minimum(len(xn)-2,np.searchsorted(xn,k.s['x_m'],side='right')-1)
        derivative=(m[indices+1]-m[indices])/(xn[indices+1]-xn[indices])
        flat=(k.s['x_m']>=3.75)&(k.s['x_m']<=5.25)
        # Plus-minus pn jump on a flat interface cutting a background cell:
        # [p']=0, [d_y r_y]=0, [div r]=m', full D3 gives
        # [tau_nn]=-2G m'/3, hence [pn]=+2G m'/3. This is an
        # approximation-space identity, not a nonlinear/solver tolerance issue.
        expected=2e6*derivative/3
        results.append(dict(mesh=n,kernel=k.name,normal_difference_RMS_Pa=rms(normal-projected),
            interior_normal_difference_RMS_Pa=rms(normal-projected,interior),
            raw_normal_difference_RMS_Pa=rms(rawnormal-reference['normal']),
            paired_normal_jump_RMS_Pa=rms(jump),interior_paired_normal_jump_RMS_Pa=rms(jump,interior),
            fitted_normal_jump_RMS_Pa=rms(fitjump),fitted_interior_normal_jump_RMS_Pa=rms(fitjump,interior),
            shear_difference_RMS_Pa=rms(shear-reference['shear']),
            weak_shear_difference_RMS_Pa=float(np.sqrt(np.mean((weak-reference['weak'])**2))),
            bulk_energy_J_m=float(.5*c@k.K@c),fitted_bulk_energy_J_m=float(.5*c@fitted.K@c),
            perturbation_normal_relative=rms(normal-projected)/rms(projected),
            flat_normal_jump_RMS_Pa=rms(jump,flat),
            cut_cell_identity_max_error_Pa=float(max(abs(-jump[flat]-expected[flat]))) if n==63 else 0.,
            flat_grid_alignment=n==64,raw_point_pressure_qualified=False))
        for q in range(len(w)):
            profiles.append(dict(mesh=n,x_m=float(k.s['x_m'][q]),y_m=float(k.s['y_m'][q]),weight_m=float(w[q]),
                normal_Pa=float(10000+normal[q]),raw_normal_Pa=float(10000+rawnormal[q]),fitted_normal_Pa=float(10000+projected[q]),
                normal_jump_Pa=float(jump[q]),fitted_normal_jump_Pa=float(fitjump[q]),
                shear_Pa=float(shear[q]),fitted_shear_Pa=float(reference['shear'][q])))
        k.close()
    table(OUT/'frozen-pressure-comparison.csv',results);table(OUT/'frozen-pressure-profiles.csv',profiles)
    save(OUT/'frozen-pressure-definition.json',dict(geometry=arcs,reference=str(fitpath.relative_to(ROOT)),
        frozen_trace_memory=c.tolist(),coordinates='Exactly matching native Gauss3 physical points',
        control='One fixed curved geometry, same affine strain and prescribed relaxed jump-memory field; bulk mechanics solved separately.',
        not_history_transfer=True,reference_is_independent_fitted_comparison=True,
        normal_convention='Current variational normal reaction, DG0 per cell; raw columns retain unfiltered sided/mean Cauchy point tractions.',
        reference_projection='Same bubble-weighted Gauss3 cell average of the independent fitted normal traction.',
        shifted_grid='63 rather than 64 cells puts the old horizontal interface inside cells at almost equal resolution',
        interior_definition='x in [3.75,8.25] m, excluding outer computational tip neighbourhoods'))
    fitted.close();print(json.dumps(results,indent=2))


def refine_existing_definition():
    path=OUT/'frozen-pressure-comparison.csv';diag=OUT/'diagnostics';diag.mkdir(exist_ok=True)
    if not (diag/'frozen-pressure-initial.csv').exists():shutil.copy2(path,diag/'frozen-pressure-initial.csv')
    with path.open() as f:results=list(csv.DictReader(f))
    profiles=np.genfromtxt(OUT/'frozen-pressure-profiles.csv',delimiter=',',names=True)
    b=4*((np.polynomial.legendre.leggauss(3)[0]+1)/2)*(1-(np.polynomial.legendre.leggauss(3)[0]+1)/2)
    for r in results:
        q=profiles[profiles['mesh']==int(r['mesh'])];w=q['weight_m'].reshape(-1,3)
        projected=np.repeat((w*b*q['raw_normal_Pa'].reshape(-1,3)).sum(axis=1)/(w*b).sum(axis=1),3)
        d=projected-q['fitted_normal_Pa'];r['raw_projected_difference_RMS_Pa']=float(np.sqrt(q['weight_m']@(d*d)/sum(q['weight_m'])))
        r.pop('cut_cell_normal_pressure_point_limit_qualified',None)
        r['flat_grid_alignment']=int(r['mesh'])==64;r['raw_point_pressure_qualified']=False
    table(path,results)
    definition=json.loads((OUT/'frozen-pressure-definition.json').read_text())
    definition.update(normal_convention='Current variational normal reaction DG0; raw columns are actual unfiltered Cauchy traces.',
        reference_projection='Same bubble-weighted Gauss3 cell average, not arbitrary smoothing.',
        raw_projected_difference='Raw point mean projected to the SAME DG0 bubble functional for an apples-to-apples comparison.')
    save(OUT/'frozen-pressure-definition.json',definition)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--existing',action='store_true');args=parser.parse_args()
    if not args.existing:main()
    refine_existing_definition()
