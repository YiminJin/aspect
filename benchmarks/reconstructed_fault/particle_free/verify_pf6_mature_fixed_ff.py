#!/usr/bin/env python3
"""Verify identical complete Q1 phase on the dyadic overlay, independently of mechanics."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
from prepare_pf6_mature_resolution import mesh,index,locate,BASE,ORIGIN
from verify_pf6_mature_resolution import q1


def nodal(run,grid):
    a=read(run/'physical-fe-0.csv')
    p={(x,y):v for c,x,y,v in zip(a['component'],a['x'],a['y'],a['accepted']) if c==9}
    return {cell:np.array([p[tuple(lo+h*np.array([i%2,i//2]))] for i in range(4)]) for cell,level,lo,h in grid}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('ff',type=Path);p.add_argument('fine',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    first,second=mesh(a.ff),mesh(a.fine);ci,fi=index(first),index(second);levels=range(max(k[0] for k in (ci|fi))+1)
    overlay=[]
    for key,(cell,lo,h) in (ci|fi).items():
        refined=any((level,*np.floor((lo+h/2-ORIGIN)/(BASE*2.**-level)).astype(int)) in idx
                    for idx in (ci,fi) for level in levels if level>key[0])
        if not refined:overlay.append((lo,h))
    assert sum(h*h for lo,h in overlay)==2000000.
    vertices=np.unique(np.concatenate([lo+h*np.array([[0,0],[1,0],[0,1],[1,1]]) for lo,h in overlay]),axis=0)
    expected=q1(vertices,locate(vertices,ci),nodal(a.ff,first));given=q1(vertices,locate(vertices,fi),nodal(a.fine,second))
    error=float(max(abs(expected-given)));assert error<2e-14
    I1=read(a.ff/'coupled-profile-0.csv')['I_h_m'];I2=read(a.fine/'coupled-profile-0.csv')['I_h_m']
    Ierror=float(max(abs(I1-I2)));Irel=Ierror/float(max(abs(I1)));assert Irel<2e-9
    support=read(a.fine/'dg-support.csv');assert len(support['mode'])==9
    result=dict(FF_cells=len(first),fine_cells=len(second),overlay_cells=len(overlay),overlay_vertices=len(vertices),
                complete_Q1_overlay_vertex_Linf=error,I_Linf_m=Ierror,I_relative_Linf=Irel,DGQ2_modes=9,
                exterior_reference_spacing_m=3.90625,mechanical_finest_m=1.953125)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
