#!/usr/bin/env python3
"""Make immutable Q1 phase and common physical-point inputs from retained meshes."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'build-particle-free/mature-dg/runs'
OUT=ROOT/'build-particle-free/mature-resolution/inputs-lf'
ORIGIN=np.array([-1234.5,137.25]);EXTENT=np.array([2000.,1000.]);BASE=500.


def mesh(path):
    m=read(path/'initial-mesh.csv')
    return [(str(cell),int(level),np.array([x-h/2,y-h/2]),float(h))
            for cell,level,x,y,h in zip(m['cell'],m['level'],m['x'],m['y'],m['edge'])]


def index(cells):
    return {(level,*np.rint((lo-ORIGIN)/h).astype(int)):(cell,lo,h) for cell,level,lo,h in cells}


def locate(points,idx):
    found=[None]*len(points)
    for level in sorted({k[0] for k in idx},reverse=True):
        h=BASE*2.**-level
        ij=np.floor((points-ORIGIN)/h).astype(int)
        ij=np.minimum(ij,np.rint(EXTENT/h).astype(int)-1)
        for n in range(len(points)):
            if found[n] is None:
                found[n]=idx.get((level,*ij[n]))
    assert all(v is not None for v in found)
    return found


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,default=OUT)
    OUT=parser.parse_args().output_directory
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(OLD/'operator-nonproportional/run/prescribed-profile.csv',OUT/'physical-profile.csv')
    coarse,fine=mesh(OLD/'dg-q6/run'),mesh(OLD/'dg-fine/run')
    ci,fi=index(coarse),index(fine)
    phase=read(OLD/'dg-q6/run/physical-fe-0.csv')
    pmap={(x,y):v for c,x,y,v in zip(phase['component'],phase['x'],phase['y'],phase['accepted']) if c==9}
    nodal={}
    with (OUT/'coarse-phase.csv').open('x') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['x','y','h','phi00','phi10','phi01','phi11'])
        for cell,level,lo,h in coarse:
            phi=[pmap[tuple(lo+h*np.array([i%2,i//2]))] for i in range(4)]
            w.writerow([*lo,h,*phi]);nodal[cell]=np.array(phi)
    # Form the exact dyadic overlay. The two retained graded meshes need not
    # be globally nested; each overlay square lies inside one cell of each.
    overlay=[]
    for key,(cell,lo,h) in {**ci,**fi}.items():
        level=key[0]
        refined=any((lev,*np.floor((lo+h/2-ORIGIN)/(BASE*2.**-lev)).astype(int)) in idx
                    for idx in (ci,fi) for lev in range(level+1,8))
        if not refined: overlay.append((cell,level,lo,h))
    assert sum(h*h for _,_,_,h in overlay)==float(np.prod(EXTENT))
    q,w=np.polynomial.legendre.leggauss(3);q=(q+1)/2;w=w/2
    points=np.concatenate([lo+h*np.array([[x,y] for y in q for x in q]) for _,_,lo,h in overlay])
    weights=np.concatenate([h*h*np.array([a*b for b in w for a in w]) for _,_,_,h in overlay])
    for name,idx in [('coarse',ci),('fine',fi)]:
        owners=locate(points,idx)
        with (OUT/(name+'-points.csv')).open('x') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['point','cell','x','y','JxW'])
            for n,(pt,weight,owner) in enumerate(zip(points,weights,owners)):
                writer.writerow([n,owner[0],*pt,weight])
    # A fine-cell Q1 prolongation is exact only if no coarse kink crosses it.
    crossed=[]
    for cell,level,lo,h in fine:
        inside=np.array([lo+h*np.array([x,y]) for x,y in [(.1,.1),(.9,.1),(.1,.9),(.9,.9)]])
        owners=locate(inside,ci)
        if len({o[0] for o in owners})>1:
            crossed.append(cell)
    assert not crossed, f'Fine cells cross coarse faces: {crossed[:10]}'
    report=dict(coarse_cells=len(coarse),fine_cells=len(fine),overlay_cells=len(overlay),
                common_points=len(points),coarse_boundary_spacing=min(h for _,_,_,h in coarse),
                fine_cells_crossing_coarse_faces=crossed,
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for path in [OLD/'dg-q6/run',OLD/'dg-fine/run']
                    for p in [path/'initial-mesh.csv',path/'physical-fe-0.csv']})
    (OUT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
