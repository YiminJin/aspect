#!/usr/bin/env python3
"""Check first-assembly history against prior commits, not post-solve inference."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

start=time.monotonic()
base=Path(__file__).resolve().parent
run=base/'replay'
sys.path.insert(0,str(base.parent))
from measure_case import summarize
from audit_convergence import samples

def csv(path):
    return np.atleast_1d(np.genfromtxt(path,names=True,delimiter=',',dtype=None,encoding='utf8'))

def by_id(path):
    data=csv(path)
    return data[np.argsort(data['id'])]

def tensor(data,names):
    return np.column_stack([data[n] for n in names])

def vtk(path):
    reader=vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path)); reader.Update()
    data=reader.GetOutput()
    arrays={data.GetPointData().GetArrayName(i):vtk_to_numpy(data.GetPointData().GetArray(i))
            for i in range(data.GetPointData().GetNumberOfArrays())}
    return vtk_to_numpy(data.GetPoints().GetData()),arrays,data.GetFieldData().GetArray('TIME').GetTuple1(0)

logged=summarize(base/'replay.log')
assert [row['time'] for row in logged]==[0.,.5,1.]
for step in logged:
    last=step['nonlinear'][-1]
    assert last['bulk']<last['bulk target'] and last['surface']<1e-8*last['surface scale']
    assert all(r['fresh']<=r['target'] for r in step['linear'])

result=dict(representation='continuous Q2, incident-cell ADD/count',steps=[],comparison=[],
            surface_source='same immutable parent properties read directly in surface_system.cc before commit',
            nonlinear_acceptances=len(logged),fresh_linear_checks=sum(len(s['linear']) for s in logged))
previous=None
initial=None
new_slip=None
old_slip=None
old_base=base.parent/'domain-convergence/space32'
for k,t in enumerate((0.,.5,1.)):
    before=by_id(run/f'pre_advection_{k}.csv')
    first=by_id(run/f'first_assembly_particles_{k}.csv')
    committed=by_id(run/f'committed_{k}.csv')
    assert np.array_equal(before['id'],first['id']) and np.array_equal(first['id'],committed['id'])
    old=tensor(first,('xx','yy','xy'))
    new=tensor(committed,('new_xx','new_yy','new_xy'))
    assert np.array_equal(old,tensor(before,('xx','yy','xy')))
    assert np.array_equal(old,tensor(committed,('old_xx','old_yy','old_xy')))
    assert np.array_equal(tensor(first,('x','y')),tensor(committed,('x','y')))
    if k==0:
        assert np.array_equal(old,new)
        assert np.max(abs(new-np.array([0.,0.,1500.])))==0.
        initial=new.copy()
    else:
        assert np.array_equal(old,previous)
        assert np.max(abs(new-old))>1., 'History change must resolve a missed commit.'
    assert np.all(first['step']==k) and np.all(first['time']==t)

    # Reconstruct the transfer from the actual post-advection source particles.
    cells,ci=np.unique(first['cell'],return_inverse=True)
    counts=np.bincount(ci)
    means=np.column_stack([np.bincount(ci,weights=old[:,c])/counts for c in range(3)])
    nodes=csv(run/f'first_assembly_nodes_{k}.csv')
    nc=np.searchsorted(cells,nodes['cell'])
    assert np.array_equal(cells[nc],nodes['cell'])
    keys,ni=np.unique(np.column_stack((nodes['x'],nodes['y'],nodes['component'])),axis=0,return_inverse=True)
    nodal_counts=np.bincount(ni)
    expected=np.bincount(ni,weights=means[nc,nodes['component']])/nodal_counts
    published_error=float(max(abs(nodes['published']-expected[ni])))
    assert published_error<1e-9
    lookup={tuple(key):i for i,key in enumerate(keys)}
    constrained=expected.copy()
    # This fixed box constrains the right composition trace to the left;
    # there are no hanging nodes or prescribed composition boundary values.
    for i,key in enumerate(keys):
        if key[0]==.25: constrained[i]=expected[lookup[(0.,key[1],key[2])]]
    working_error=float(max(abs(nodes['working']-constrained[ni])))
    assert working_error<1e-9
    first_qp=csv(run/f'first_assembly_qp_{k}.csv')

    # Inspect the actual written arrays. No Maxwell evaluation or field refresh.
    position,viz,viz_time=vtk(run/f'particles/particles-{k:05}.0000.vtu')
    order=np.argsort(viz['id'])
    assert viz_time==t and np.array_equal(viz['id'][order],committed['id'])
    assert np.array_equal(position[order,:2],tensor(committed,('x','y')).astype(position.dtype))
    particle_output=np.column_stack([viz[f'maxwell stress_{c}'][order] for c in range(3)])
    assert np.array_equal(particle_output,new.astype(particle_output.dtype))
    particle_quantization=float(np.max(abs(particle_output-new)))
    position,viz,viz_time=vtk(run/f'solution/solution-{k:05}.0000.vtu')
    assert viz_time==t
    bulk_quantization=0.
    for c,name in enumerate(('tau_xx','tau_yy','tau_xy')):
        # Raw published nodal values, before the mechanical constraint lift.
        raw=np.zeros(len(keys))
        raw[ni]=nodes['published']
        expected_viz=np.array([raw[lookup[(float(p[0]),float(p[1]),float(c))]] for p in position])
        assert np.array_equal(viz[name],expected_viz.astype(viz[name].dtype))
        bulk_quantization=max(bulk_quantization,float(max(abs(viz[name]-expected_viz))))
    row=dict(step=k,time_s=t,physical_dt_s=float(first['dt'][0]),
             initialization_Maxwell_interval_s=2. if k==0 else None,
             particles=len(first),active_surface_parents=int(sum(first['active'])),
             cells=len(cells),first_bulk_qp_count=len(first_qp),
             transfer_error_Pa=published_error,working_error_Pa=working_error,
             committed_change_max_Pa=float(np.max(abs(new-old))),
             consumed_history_minus_initial_max_Pa=float(np.max(abs(old-initial))),
             bulk_VTU_quantization_Pa=bulk_quantization,particle_VTU_quantization_Pa=particle_quantization,
             first_working_xy_min_Pa=float(min(first_qp['xy'])),first_working_xy_max_Pa=float(max(first_qp['xy'])))
    result['steps'].append(row)
    previous=new

    # The saved same-mesh run is reused, not rerun. Compare actual weak traction
    # and retained histories; this is a short-run representation comparison.
    current=csv(run/f'surface_{k}.csv'); old_surface=csv(old_base/f'surface_{k}.csv')
    assert np.array_equal(current['x'],old_surface['x'])
    s=current['x']; x,w=samples(s)
    weak=csv(run/f'surface_weak_{k}.csv'); old_weak=csv(old_base/f'surface_weak_{k}.csv')
    def q(data):
        mass=np.diag(data['Mdiag'])+np.diag(data['Moff'][:-1],1)+np.diag(data['Moff'][:-1],-1)
        return np.linalg.solve(mass,data['q'])
    new_q=q(weak); old_q=q(old_weak)
    if k==0: new_slip=np.zeros(len(s)); old_slip=np.zeros(len(s))
    else:
        new_slip+=.5*current['V']; old_slip+=.5*old_surface['V']
    comparison=dict(step=k,time_s=t,fields={})
    for name,a,b in [('V',current['V'],old_surface['V']),('Theta',current['Theta'],old_surface['Theta']),
                     ('C',current['C'],old_surface['C']),('q',new_q,old_q),('slip',new_slip,old_slip)]:
        av=np.interp(x,s,a); bv=np.interp(x,s,b); delta=av-bv; mean=np.average(delta,weights=w)
        comparison['fields'][name]=dict(old_mean=float(np.average(bv,weights=w)),new_mean=float(np.average(av,weights=w)),
            difference_rms=float(np.sqrt(np.average(delta**2,weights=w))),
            anomaly_difference_rms=float(np.sqrt(np.average((delta-mean)**2,weights=w))))
    comparison['initial_phase_unchanged']=hashlib.sha256((run/'phase_0.csv').read_bytes()).digest()==hashlib.sha256((old_base/'phase_0.csv').read_bytes()).digest()
    result['comparison'].append(comparison)
result['wall_seconds']=time.monotonic()-start
(base/'timeline-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
