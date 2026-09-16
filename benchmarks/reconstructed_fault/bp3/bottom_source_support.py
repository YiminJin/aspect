"""Frozen all-QP source audit: no timestep, history or production-policy change."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from analyze_uniform_sliding import read,write,records
import bottom_completion as continuation

HERE=Path(__file__).resolve().parent
BASE=continuation.BASE
RUN=continuation.OUT
OUT=HERE/'bottom-source-support-audit'
VP=1e-9
n=continuation.normal
s=np.array([.5,continuation.sn])
S=.5*(np.outer(s,n)+np.outer(n,s))
SNORM=np.linalg.norm(S)

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'all_qp.csv').exists(), 'Preserve completed audit inputs.'
    f=read(RUN/'fault_0.csv');vertices=np.column_stack([f['x'],f['y']])
    with (OUT/'geometry.txt').open('w') as out:
        out.write(f'{len(vertices)} {continuation.radius:.17g}\n');np.savetxt(out,vertices,fmt='%.17g')
    nodes,weights=leggauss(3);nodes=(nodes+1)/2;weights=weights/2
    # The previously unassociated wedge includes coarser cells. Use their
    # actual saved Q1 corner values, including hanging-node constraints, not
    # an assumed uniform 97.7-m lattice. VTU values have Float32 precision.
    fe_cells={}
    for path in sorted(RUN.glob('bulk_0_*.vtu')):
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
        grid=reader.GetOutput();field=grid.GetPointData().GetArray('component_9')
        for k in range(grid.GetNumberOfCells()):
            cell=grid.GetCell(k);bounds=cell.GetBounds()
            if bounds[2]>=2000 and not (59000-1000<continuation.sn*(100000-(bounds[2]+bounds[3])/2)+.5*(continuation.xt-(bounds[0]+bounds[1])/2)<61000+1000):continue
            points=[grid.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
            values=[field.GetTuple1(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
            fe_cells[((bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2,bounds[1]-bounds[0])]=(points,values)
    entries=[];seen=set()
    for path in sorted(RUN.glob('initial_mesh_*.csv')):
        rank=int(path.stem.split('_')[-1])
        for cell in csv.DictReader(path.open()):
            assert cell['cell'] not in seen;seen.add(cell['cell'])
            x,y,h=[float(cell[k]) for k in ('x','y','h')]
            xd=continuation.sn*(100000-y)+.5*(continuation.xt-x)
            # Select cells conservatively first; retain all their QPs in the
            # requested physical-height/interior windows, regardless of admission.
            if not (y-h/2<2000 or 59000-h<xd<61000+h):continue
            for j in range(3):
                for i in range(3):
                    p=np.array([x+h*(nodes[i]-.5),y+h*(nodes[j]-.5)])
                    foot=float(np.dot(p-vertices[0],s));r=float(np.dot(p-vertices[0],n))
                    xd=continuation.sn*(100000-p[1])+.5*(continuation.xt-p[0])
                    if not (p[1]<2000 or 59000<xd<61000):continue
                    # Radius plus a projected cell diameter is a conservative
                    # enclosure of a Q1 field initialized from compact nodal data.
                    if abs(r)>continuation.radius+h*(abs(n[0])+abs(n[1])):continue
                    corner,values=fe_cells[(x,y,h)]
                    phi=0.
                    for vertex,value in zip(corner,values):
                        u=(vertex[0]-(x-h/2))/h;v=(vertex[1]-(y-h/2))/h
                        phi+=value*(nodes[i] if u>.5 else 1-nodes[i])*(nodes[j] if v>.5 else 1-nodes[j])
                    virtual_phi=float(continuation.q1_phi(*p))
                    localization=continuation.m*phi*(1+phi)/(1-phi)**2
                    if localization<=0 and virtual_phi<=0:continue
                    entries.append(dict(id=len(entries),rank=rank,cell=cell['cell'],q=i+3*j,x=p[0],y=p[1],
                        weight=h*h*weights[i]*weights[j],hcell=h,xd=xd,foot=foot,r=r,phi=phi,h=localization,virtual_phi=virtual_phi))
    assert len(seen)==42880
    records(OUT/'all_qp.csv',entries)
    with (OUT/'points.txt').open('w') as out:
        for p in entries:out.write(f"{p['id']} {p['x']:.17g} {p['y']:.17g}\n")
    (OUT/'preparation.json').write_text(json.dumps(dict(cells=len(seen),points=len(entries),width=continuation.radius,
        quadrature='QGauss<2>(3), x-fast indexing; verified against saved production positions/JxW'),indent=2)+'\n')
    print((OUT/'preparation.json').read_text())

def analyze():
    rows=list(csv.DictReader((OUT/'all_qp.csv').open()))
    data=read(OUT/'all_qp.csv',('cell',));projection=read(OUT/'production_projection.csv')
    np.testing.assert_array_equal(data['id'],projection['id'])
    active=projection['active'].astype(bool)
    data['active']=active;data['segment']=projection['segment'];data['xi']=projection['xi']
    f=read(RUN/'fault_0.csv');base=read(BASE/'fault_0.csv')
    length=(f['x']-f['x'][0])*s[0]+(f['y']-f['y'][0])*s[1]
    # Constant endpoint continuation here is a COUNTERFACTUAL denominator for
    # unassociated QPs only. It is not installed in the production manager.
    Ih=np.interp(data['foot'],length,f['Ih'])
    data['Ih_reference_continued']=Ih
    ref=data['h']/Ih*VP
    data['reference_source']=ref
    data['assembled_source']=np.zeros(len(active))
    data['baseline_source']=np.zeros(len(active))
    regions=dict(bottom_0_200m=data['y']<200,bottom_200_1000m=(data['y']>=200)&(data['y']<1000),
        bottom_1000_2000m=(data['y']>=1000)&(data['y']<2000),interior_59_61km=(data['xd']>59000)&(data['xd']<61000))
    errors=dict(position=0.,weight=0.,phase=0.,chi=0.,source_tensor=0.)
    for case,key in [(RUN,'assembled_source'),(BASE,'baseline_source')]:
        saved={}
        for path in case.glob('uniform_bulk_0_rank*.csv'):
            for r in csv.DictReader(path.open()):saved[(r['cell'],int(r['qp']))]=r
        matches=0
        for i,p in enumerate(rows):
            r=saved.get((p['cell'],int(p['q'])))
            assert (r is not None)==bool(active[i]),'Production association differs from saved assembly.'
            if r is None:continue
            matches+=1;j=int(projection['segment'][i]);z=projection['xi'][i]
            assert j==int(r['segment']) and abs(z-float(r['xi']))<1e-9
            data[key][i]=float(r['chi'])*VP
            if case==RUN:
                errors['position']=max(errors['position'],abs(float(r['x'])-data['x'][i]),abs(float(r['y'])-data['y'][i]))
                errors['weight']=max(errors['weight'],abs(float(r['weight'])/data['weight'][i]-1))
                errors['phase']=max(errors['phase'],abs(float(r['phi'])-data['phi'][i]))
                # Retain full-precision production phi wherever it was saved;
                # only inactive QPs need the Float32 corner reconstruction.
                data['phi'][i]=float(r['phi'])
                data['h'][i]=continuation.m*data['phi'][i]*(1+data['phi'][i])/(1-data['phi'][i])**2
                ref[i]=data['h'][i]/Ih[i]*VP
                errors['chi']=max(errors['chi'],abs(data[key][i]-ref[i]))
                tensor=np.array([[float(r['crack_xx']),float(r['crack_xy'])],[float(r['crack_xy']),float(r['crack_yy'])]])
                errors['source_tensor']=max(errors['source_tensor'],np.linalg.norm(tensor-ref[i]*S))
        assert matches==np.count_nonzero(active)
    assert errors['position']<3e-11 and errors['weight']<1e-12 and errors['phase']<5e-8
    assert errors['chi']<1e-23 and errors['source_tensor']<1e-23
    # Separate missing tip wedge, missing normal-width tail, and represented
    # points. No missing-QP value is silently discarded from the denominator.
    tip=(~active)&(data['foot']<0)
    tail=(~active)&(~tip)&(abs(data['r'])>continuation.radius)
    unexplained=(~active)&(~tip)&(~tail)
    assert not np.any(unexplained)
    data['tip_excluded']=tip;data['width_excluded']=tail
    endpoint=active&((projection['xi']==0)|(projection['xi']==1))
    data['endpoint_coordinate']=endpoint
    Istar=json.loads((RUN/'analysis/completion_verification.json').read_text())['fixed_reference_full']
    same_resolution=data['h']/Istar*VP
    data['same_resolution_uniform_reference']=same_resolution
    vf=data['virtual_phi'];virtual_h=continuation.m*vf*(1+vf)/(1-vf)**2
    virtual_reference=virtual_h/Istar*VP
    data['uniform_grid_reference']=virtual_reference
    result=[]
    for region,mask in regions.items():
        w=data['weight'][mask];reference=ref[mask];actual=data['assembled_source'][mask]
        norm=lambda q:float(np.sqrt(np.sum(w*q*q)/np.sum(w))*SNORM)
        r=dict(region=region,points=int(mask.sum()),active=int(np.sum(mask&active)),tip_missing=int(np.sum(mask&tip)),
            width_missing=int(np.sum(mask&tail)),endpoint_coordinate=int(np.sum(mask&endpoint)),
            full_volume=float(w.sum()),missing_tip_volume=float(data['weight'][mask&tip].sum()),
            reference_rms=norm(reference),source_error_rms=norm(actual-reference),
            source_error_relative=norm(actual-reference)/norm(reference),
            baseline_source_error_relative=norm(data['baseline_source'][mask]-reference)/norm(reference),
            missing_integral_fraction=float(np.sum(w*(reference-actual))/np.sum(w*reference)),
            width_missing_integral_fraction=float(np.sum(data['weight'][mask&tail]*ref[mask&tail])/np.sum(w*reference)),
            same_resolution_reference_error_relative=norm(actual-same_resolution[mask])/norm(same_resolution[mask]),
            uniform_grid_reference_error_relative=norm(actual-virtual_reference[mask])/norm(virtual_reference[mask]))
        result.append(r)
    records(OUT/'region_summary.csv',result)
    # Individual tensor components and weighted weak-moment contributions can
    # be reconstructed without any new Maxwell or constitutive update.
    for name,tensor in [('actual',data['assembled_source']),('reference',ref),('difference',data['assembled_source']-ref)]:
        for comp,value in [('xx',S[0,0]),('yy',S[1,1]),('xy',S[0,1])]:data[name+'_'+comp]=tensor*value
    data['cell']=[row['cell'] for row in rows]
    write(OUT/'source_comparison.csv',data)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    mask=data['y']<500
    for ax,values,title in [(axes[0],data['assembled_source'],'Actual source amplitude'),(axes[1],ref-data['assembled_source'],'Missing continued source')]:
        plot=ax.scatter(data['x'][mask]-f['x'][0],data['y'][mask],c=values[mask],s=12)
        ax.plot([0,500*s[0]/s[1]],[0,500],'k-',lw=1)
        ax.set(xlabel='x - bottom fault intersection (m)',ylabel='Height above bottom (m)',title=title)
        fig.colorbar(plot,ax=ax,label='1/s')
    fig.tight_layout();fig.savefig(OUT/'source_coverage.png',dpi=160);plt.close(fig)
    summary=dict(validation_errors=errors,points=len(active),unexplained_inactive=int(unexplained.sum()),
        endpoint_QPs=int(endpoint.sum()),max_missing_source=float(ref[tip].max()),max_missing_height=float(data['y'][tip].max()),
        max_missing_foot_distance=float(-data['foot'][tip].min()),regions=result)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','analyze'])
    args=parser.parse_args();prepare() if args.action=='prepare' else analyze()
