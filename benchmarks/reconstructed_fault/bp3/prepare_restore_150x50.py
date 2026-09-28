"""Generate fresh restored-BP3 inputs; never resize a checkpoint or launch ASPECT.

The outside completion uses the same virtual Cartesian Q1 continuation as the
qualified endpoint treatment. Profile and FE checks run again in the plugin.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from run_mechanical_width import stationary
from length_scale_study import render, parameters

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fixtures/bp3_150x50'
SN = math.sqrt(3)/2
HMIN = 2000/512
NORMAL = np.array([-SN, -.5])  # production geometric normal, not shear sense


def save(path, text):
    if path.exists():
        assert path.read_text() == text, f'Refusing to overwrite changed input {path}'
    else:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true', help='Plot an already exported actual mesh')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    r, phi, m = stationary(20.)
    nodes, weights = leggauss(8)
    z = (nodes+1)/2
    values = phi[:-1, None]+(phi[1:]-phi[:-1])[:, None]*z
    primitive = np.r_[0, np.cumsum(np.diff(r)*(m*values*(1+values)/(1-values)**2 @ (weights/2)))]
    save(OUT/'profile.txt', f'{len(r)} {m:.17g}\n'+''.join(
        f'{a:.17g} {b:.17g} {c:.17g}\n' for a,b,c in zip(r,phi,primitive)))

    # Supply only exact breakpoints. Match production segment-by-segment
    # resampling, including its ceil and floating-point endpoint arithmetic.
    down = np.array([50000/SN, 40000., 18000., 15000., 0.])
    corners = np.column_stack([.5*down, 50000-SN*down])
    corners[0,1] = 0.
    save(OUT/'fault.txt', ''.join(f'{x:.17g} {y:.17g} .6\n' for x,y in corners))
    vertices = []
    for j,(a,b) in enumerate(zip(corners[:-1],corners[1:])):
        length = math.sqrt(sum((b-a)**2))
        count = math.ceil(length/20.)
        for i in range(0 if j==0 else 1,count+1):
            s = length if i==count else i*(length/count)
            vertices.append(a+(s/length)*(b-a))
    vertices = np.array(vertices)
    save(OUT/'surface_vertices.csv', 'x,y\n'+''.join(f'{x:.17g},{y:.17g}\n' for x,y in vertices))
    # The normal support is enclosed in the fine band, including the virtual
    # outside cells. Cartesian origin matters: x=-60000 is not a multiple of h.
    extent = r[-1]+HMIN*(SN+.5)
    def q1(x,y):
        i,j=np.floor((x+60000)/HMIN),np.floor(y/HMIN)
        a,b=(x+60000)/HMIN-i,y/HMIN-j
        result=np.zeros(np.broadcast_shapes(np.shape(x),np.shape(y)))
        for u in (0,1):
            for v in (0,1):
                rr=SN*(-60000+(i+u)*HMIN)+.5*((j+v)*HMIN-50000)
                result+=(a if u else 1-a)*(b if v else 1-b)*np.interp(abs(rr),r,phi,right=0.)
        return result
    def integrate(origin,lo,hi,order):
        if lo>=hi: return 0.
        cuts=[lo,hi]
        for d,offset in [(0,-60000.),(1,0.)]:
            ends=origin[d]+NORMAL[d]*np.array([lo,hi])
            for k in range(math.floor((min(ends)-offset)/HMIN), math.ceil((max(ends)-offset)/HMIN)+1):
                t=(offset+k*HMIN-origin[d])/NORMAL[d]
                if lo<t<hi: cuts.append(t)
        cuts=np.unique(cuts);xx,ww=leggauss(order)
        def panel(a,b):
            t=(a+b)/2+(b-a)/2*xx
            p=origin[:,None]+NORMAL[:,None]*t
            f=q1(p[0],p[1])
            return (b-a)/2*np.dot(ww,m*f*(1+f)/(1-f)**2)
        def adaptive(a,b,depth=0):
            low=panel(a,b);mid=(a+b)/2;high=panel(a,mid)+panel(mid,b)
            if abs(high-low)<1e-12*max(1.,abs(high)): return high
            assert depth<24, 'Completion quadrature did not converge'
            return adaptive(a,mid,depth+1)+adaptive(mid,b,depth+1)
        return sum(adaptive(a,b) for a,b in zip(cuts[:-1],cuts[1:]))
    surface_z=(leggauss(3)[0]+1)/2
    rows=[];error=0.
    for a,b in zip(vertices[:-1],vertices[1:]):
        for panel in range(8):
            for q in surface_z:
                xi=(panel+q)/8
                p=(1-xi)*a+xi*b
                top=(50000-p[1])/NORMAL[1];bottom=-p[1]/NORMAL[1]
                intervals=[(-extent,min(top,extent)),(max(bottom,-extent),extent)]
                value=sum(integrate(p,lo,hi,8) for lo,hi in intervals)
                if value:
                    check=sum(integrate(p,lo,hi,16) for lo,hi in intervals)
                    error=max(error,abs(check-value))
                rows.append((len(rows),*p,value))
    assert error<1e-8
    save(OUT/'completion.txt',str(len(rows))+'\n'+''.join(
        f'{i} {x:.17g} {y:.17g} {v:.17g}\n' for i,x,y,v in rows))

    values=parameters((HERE/'bp3_modified_long_run.prm').read_text())
    root='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3'
    changes={
        ('Additional shared libraries',):root+'/build/libbp3_restore_150x50.release.so',
        ('Maximum time step',):'4e6', ('Maximum first time step',):'100',
        ('Geometry model','Box','X extent'):'150000',('Geometry model','Box','Y extent'):'50000',
        ('Geometry model','Box','Box origin X coordinate'):'-60000',
        ('Geometry model','Box','X repetitions'):'75',('Geometry model','Box','Y repetitions'):'25',
        ('Mesh refinement','BP3 saved mesh','Target cells file'):root+'/fixtures/bp3_150x50/target_cells.txt',
        ('Boundary velocity model','Prescribed velocity boundary indicators'):
            'left:reconstructed fault BP3, right:reconstructed fault BP3, bottom:reconstructed fault BP3',
        ('Boundary traction model','Prescribed traction boundary indicators'):'',
        ('Discretization','Composition polynomial degree'):'2',
        ('Discretization','Use discontinuous composition discretization'):'false',
        ('Phase field model','Length scale'):'20',
        ('Fault reconstruction','Structural point spacing'):'20',
        ('Fault reconstruction','Prescribed faults file'):root+'/fixtures/bp3_150x50/fault.txt',
        ('Material model','Phase field fault','I h integration backend'):'cell intervals',
        ('Material model','Phase field fault','I h surface quadrature subdivisions'):'8',
        ('Particles','Interpolation scheme'):'linear least squares',
        ('Particles','Interpolator','Linear least squares','Use linear least squares limiter'):'false',
        ('Particles','Interpolator','Linear least squares','Use boundary extrapolation'):'false',
        ('Time stepping','List of model names'):'convection time step, reconstructed fault time step',
        ('Postprocess','BP3','Bottom normalization completion file'):root+'/fixtures/bp3_150x50/completion.txt',
        ('Postprocess','BP3','Stop after first event'):'true',
        ('Postprocess','BP3','Last accepted step'):'10',
        ('Postprocess','BP3','Graceful wall seconds'):'3600',
        ('Postprocess','BP3','Profile time interval'):'1',
        ('Postprocess','BP3 restored monitor','Stationary profile file'):root+'/fixtures/bp3_150x50/profile.txt',
    }
    values.update(changes)
    values.pop(('Postprocess','BP3','Mature prestress file'), None)
    values['Postprocess','List of postprocessors']+=', BP3 restored monitor'
    for mode,length in [('raw',0),('filter20',20),('filter40',40)]:
        values['Output directory',]=root+'/restore-150x50-'+mode
        values['Postprocess','BP3 restored monitor','Friction normal input']='raw' if mode=='raw' else 'helmholtz'
        values['Postprocess','BP3 restored monitor','Normal filter length']=str(length)
        values['Postprocess','BP3 restored monitor','Write detailed diagnostics']='true'
        save(HERE/f'bp3_150x50_{mode}.prm','# Fresh modified BP3 development startup; seconds; no historical prestress.\n'+render(values))
    report=dict(ell=20.,hmin=HMIN,support_radius=float(r[-1]),fine_half_width=max(40.,float(r[-1])+2*HMIN),
                full_reference_integral=float(2*primitive[-1]),surface_nodes=len(vertices),
                spacing_range=np.array([np.linalg.norm(np.diff(vertices,axis=0),axis=1).min(),
                                        np.linalg.norm(np.diff(vertices,axis=0),axis=1).max()]).tolist(),
                completion_profiles=len(rows),nonzero_completion_profiles=int(sum(row[3]>0 for row in rows)),
                completion_order_difference=error,
                hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                        for name in ('completion.txt','profile.txt','fault.txt','surface_vertices.csv')})
    save(OUT/'preparation.json',json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if args.plots:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        cells=np.genfromtxt(OUT/'mesh.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,limits in zip(axes,[(-60000,90000,0,50000),(-100,100,49900,50000),
                                 (50000/SN/2-100,50000/SN/2+100,0,100)]):
            lines=[]
            for cell in cells:
                x,y,h=cell['x'],cell['y'],cell['h']
                if x+h<limits[0] or x-h>limits[1] or y+h<limits[2] or y-h>limits[3]: continue
                lines.append([(x-h/2,y-h/2),(x+h/2,y-h/2),(x+h/2,y+h/2),(x-h/2,y+h/2),(x-h/2,y-h/2)])
            ax.add_collection(LineCollection(lines,linewidths=.15,color='black'))
            ax.plot(vertices[:,0],vertices[:,1],color='red',lw=.7)
            ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x (m)',ylabel='y (m)',aspect='equal')
        fig.tight_layout();fig.savefig(OUT/'mesh.png',dpi=180)


if __name__=='__main__': main()
