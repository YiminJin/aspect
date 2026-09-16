"""Compare native fault grids at common times without transferring histories."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

here=Path(__file__).resolve().parent
base=here/'junction-matched-qualified-local4/refined'
run=here/'fault-grid-50-local4'


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def col(data,key):return np.array([float(r[key]) for r in data])


def projected(root,k):
    path=root/f'analysis/step{k}'
    if not path.exists():
        subprocess.run([sys.executable,str(here/'analyze_normal_stress.py'),str(root),
            '--step',str(k),'--output',str(path)],check=True,stdout=subprocess.DEVNULL)
    return rows(path/'projected_full.csv')


def weak(root,k):
    parts=[rows(root/f'history_surface_step{k}_rank{rank}.csv') for rank in range(4)]
    result=[]
    for i,r in enumerate(parts[0]):
        record={key:float(r[key]) for key in list(r)[:5]}
        assert all(all(float(p[i][key])==record[key] for key in list(r)[:5]) for p in parts)
        record.update({key:sum(float(p[i][key]) for p in parts) for key in list(r)[5:]})
        result.append(record)
    return result


def depression(x,V):
    order=np.argsort(x);x=x[order];V=V[order]
    reference=float(np.interp(39700.,x,V));level=.5*reference
    indices=np.flatnonzero((x>=39700-1e-7)&(x<40000-1e-7))
    center=int(indices[np.argmin(V[indices])]);left=right=center
    while left>0 and V[left]<level:left-=1
    while right+1<len(V) and V[right]<level:right+=1
    if V[center]>=level:return dict(reference_rate=reference,threshold=level,minimum=float(V[center]),
                                   location=float(x[center]),width=0.)
    assert V[left]>=level and V[right]>=level
    crossing=lambda a,b:float(x[a]+(level-V[a])*(x[b]-x[a])/(V[b]-V[a]))
    lo=crossing(left,left+1);hi=crossing(right-1,right)
    return dict(reference_rate=reference,threshold=level,minimum=float(V[center]),location=float(x[center]),
                left=lo,right=hi,width=hi-lo)


def bulk_pressure(root):
    # Independent raw bulk pressure sample set on the identical mesh, not
    # surface-projected pressure or extrema selected by normal stress.
    points=[];pressure=[]
    for path in sorted(root.glob('bulk_13_*.vtu')):
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
        grid=reader.GetOutput();x=vtk_to_numpy(grid.GetPoints().GetData())
        p=vtk_to_numpy(grid.GetPointData().GetArray('delta_pressure'))
        s=(.5e5*(1+1/np.sqrt(3))-x[:,0])*.5+(1e5-x[:,1])*np.sqrt(3)/2
        z=(x[:,0]-.5e5*(1+1/np.sqrt(3)))*np.sqrt(3)/2-(x[:,1]-1e5)*.5
        mask=(s>=39000)&(s<=40500)&(abs(z)<=1500)
        points.extend(x[mask].tolist());pressure.extend(p[mask].tolist())
    assert pressure
    x=np.array(points);p=np.array(pressure)
    return dict(minimum=float(min(p)),maximum=float(max(p)),peak_to_peak=float(np.ptp(p)),
                minimum_point=x[p.argmin()].tolist(),maximum_point=x[p.argmax()].tolist(),
                samples=len(p),unique_points=len(set(map(tuple,points))),
                semantics='Bulk Q1 vertex samples in 39--40.5 km, |normal|<=1500 m; VTU precision, not exact global surface-pressure extrema.')


# A diagnostic width must not change merely because an identical piecewise
# linear profile is described with extra midpoint nodes.
assert depression(np.array([39700.,39900.,40000.]),np.ones(3))['width']==0.
for x in [np.array([39700.,39900.,40000.]),np.array([39700.,39800.,39900.,39950.,40000.])]:
    assert depression(x,np.interp(x,[39700.,39900.,40000.],[1.,0.,1.]))['width']==150.

execution=json.loads((run/'execution.json').read_text());assert execution['status']==0
plan=json.loads((run/'grid_plan.json').read_text());common=np.array(plan['common_node_map'])
old_states=rows(base/'accepted_steps.csv');new_states=rows(run/'accepted_steps.csv')
assert len(old_states)==len(new_states)==14
for key in ['time','dt']:np.testing.assert_allclose(col(new_states,key),col(old_states,key),rtol=1e-13,atol=1e-8)
old_mesh={r['cell']:r for f in base.glob('initial_mesh_*.csv') for r in rows(f)}
new_mesh={r['cell']:r for f in run.glob('initial_mesh_*.csv') for r in rows(f)}
assert len(old_mesh)==42880 and old_mesh==new_mesh
log=(run/'run.log').read_text();assert 'vertices=1236, coordinate error=0' in log
linear=[]
for line in log.splitlines():
    if 'Fault linear solve:' in line:
        get=lambda key:float(re.search(r'\b'+key+r'=([^,\s]+)',line)[1])
        linear.append(dict(iterations=int(get('iterations')),fresh=get('fresh'),target=get('target')))
assert linear and all(r['fresh']<=r['target'] for r in linear)
sections=re.split(r'\*\*\* Timestep (\d+):',log)
nonlinear=[]
for j in range(1,len(sections),2):
    text=sections[j+1];k=int(sections[j])
    relative=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',text)[-1]
    assert all(float(v)<1e-8 for v in relative)
    assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',text)
    nonlinear.append(dict(step=k,relative=list(map(float,relative))))
summary=dict(execution=execution,linear_returns=len(linear),krylov=sum(r['iterations'] for r in linear),
    worst_fresh_target=max(r['fresh']/r['target'] for r in linear),nonlinear=nonlinear,mesh_identical=True,
    initial={},states=[],common_coordinates=[],weak_R_max_relative_error=0.)
# Check the represented initial inputs, independently of the parameter/mesh
# guards. These exported arrays are FE fields, not a bytewise particle audit.
initial_fields=['component_3','tau_xx','tau_yy','tau_xy','theta_initial','strengthening','component_9']
for rank in range(4):
    grids=[]
    for root,k in [(base,0),(run,0),(run,13)]:
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(root/f'bulk_{k}_{rank}.vtu'));reader.Update()
        grid=reader.GetOutput()
        grids.append((vtk_to_numpy(grid.GetPoints().GetData()).copy(),
                      {key:vtk_to_numpy(grid.GetPointData().GetArray(key)).copy() for key in initial_fields}))
    np.testing.assert_array_equal(grids[0][0],grids[1][0])
    np.testing.assert_array_equal(grids[1][0],grids[2][0])
    for key in initial_fields:np.testing.assert_array_equal(grids[0][1][key],grids[1][1][key])
    np.testing.assert_array_equal(grids[1][1]['component_9'],grids[2][1]['component_9'])
summary['exported_initial_fields_identical']=initial_fields
summary['exported_final_phase_identical']=True
all_rows=[];data={};initial={}
for label,root,states in [('100m',base,old_states),('50m',run,new_states)]:
    previous=None
    for k,state in enumerate(states):
        f=rows(root/f'fault_{k}.csv');w=weak(root,k);p=rows(root/f'stress_projected_{k}.csv')
        x=col(f,'xd');V=col(f,'V');active=col(p,'lower_active').astype(bool)
        assert len(w)==len(f)==len(p)
        error=float(np.max(abs(col(f,'weak_residual')-np.array([r['particle_R'] for r in w]))))
        relative=error/max(abs(r['particle_q']) for r in w)
        assert relative<1e-12  # Same physical-term-scaled check as the preceding audit.
        summary['weak_R_max_relative_error']=max(summary['weak_R_max_relative_error'],relative)
        if k==0:initial[label]=f
        else:
            for key in ['x','y','tau_bg','sigma_n_bg']:np.testing.assert_array_equal(col(f,key),col(initial[label],key))
            np.testing.assert_allclose(col(f,'Ih'),col(initial[label],'Ih'),rtol=1e-12,atol=0)
            dt=float(state['dt']);z=V*dt/.008
            theta=col(previous,'Theta')*np.exp(-z)-(.008/V)*np.expm1(-z)
            np.testing.assert_allclose(col(f,'Theta'),theta,rtol=1e-12,atol=0)
            np.testing.assert_allclose(col(f,'slip'),col(previous,'slip')+dt*V,rtol=1e-13,atol=1e-15)
        assert float(rows(root/f'history_{k}.csv')[0]['Theta_reference_relative_error'])<1e-12
        assert np.all(V[col(f,'prescribed').astype(bool)]==1e-9)
        selected=(x>=39000-1e-7)&(x<=40500+1e-7)
        contacts=np.flatnonzero(active & selected)
        assert all(0<i<len(x)-1 and x[i+1]>=39000-1e-7 and x[i-1]<=40500+1e-7 for i in contacts)
        total=-sum(w[i]['particle_R'] for i in contacts)
        summary['states'].append(dict(case=label,step=k,time=float(state['time']),
            lower_active=int(sum(active)),window_bound_reaction=total,contacts=[dict(node=int(i),xd=x[i],
            reaction=-w[i]['particle_R']/w[i]['weight'],basis_support=x[i-1]-x[i+1]) for i in contacts],
            depression=depression(x,V),particle_tensile_weight=sum(r['particle_tensile_weight'] for i,r in enumerate(w) if selected[i])))
        for i in np.flatnonzero(selected|(abs(x-25000)<1e-7)):
            terms={key:w[i][key]/w[i]['weight'] for key in
                   ['particle_q','particle_C','particle_friction','particle_damping','particle_R','particle_sigma']}
            assert abs(terms['particle_q']-terms['particle_C']-terms['particle_friction']-terms['particle_damping']-terms['particle_R'])<1e-6
            all_rows.append(dict(case=label,step=k,time=float(state['time']),node=int(i),xd=x[i],V=V[i],V_over_Vp=V[i]/1e-9,
                Theta=float(f[i]['Theta']),Theta_used=float(previous[i]['Theta']) if k else None,
                slip=float(f[i]['slip']),C_committed=float(f[i]['C']),Ih=float(f[i]['Ih']),tau_bg=float(f[i]['tau_bg']),
                prescribed=int(f[i]['prescribed']),active=int(active[i]),weight=w[i]['weight'],
                reaction=-terms['particle_R'] if active[i] else 0.,sigma_projected=float(p[i]['sigma_n']),**terms))
        data[label,k]=(f,w,p);previous=f

for key in ['x','y']:np.testing.assert_array_equal(col(initial['100m'],key),col(initial['50m'],key)[common])
for key in ['Ih','C','C_evaluated','tau_bg','Theta','V']:
    a=col(initial['100m'],key);b=col(initial['50m'],key)[common];d=b-a;idx=int(np.argmax(abs(d)))
    summary['initial'][key]=dict(max_abs=float(max(abs(d))),xd=float(initial['100m'][idx]['xd']),
                               old=float(a[idx]),new=float(b[idx]))
summary['initial']['selected']=[dict(xd=float(initial['100m'][i]['xd']),
    **{label+'_'+key:float(initial[label][i if label=='100m' else common[i]][key])
       for label in ['100m','50m'] for key in ['Ih','C','C_evaluated','tau_bg','Theta','V']})
    for i in range(len(initial['100m'])) if 38999<float(initial['100m'][i]['xd'])<40501 or abs(float(initial['100m'][i]['xd'])-25000)<1e-7]
for k in range(14):
    a,b=data['100m',k][0],data['50m',k][0]
    for i,r in enumerate(a):
        if 38999<float(r['xd'])<40501 or abs(float(r['xd'])-25000)<1e-7:
            summary['common_coordinates'].append(dict(step=k,time=float(r['time']),xd=float(r['xd']),
                **{label+'_'+key:float((a[i] if label=='100m' else b[common[i]])[key])
                   for label in ['100m','50m'] for key in ['V','Theta','slip','C','Ih','tau_bg']}))

summary['traction_profiles']={}
for label,root in [('100m',base),('50m',run)]:
    p=projected(root,13);f,w,_=data[label,13];x=col(p,'xd');mask=(x>=39000-1e-7)&(x<=40500+1e-7)
    # No tensile measure touches the reporting-window boundary: summing the
    # selected test weights counts the local tensile domains in full.
    edges=((x>=38800)&(x<=39200))|((x>=40300)&(x<=40700))
    assert all(w[i][key]==0 for i in np.flatnonzero(edges)
               for key in ['particle_tensile_weight','fe_tensile_weight'])
    raw=rows(root/'analysis/step13/raw_selected.csv');samples=[r for r in raw if 39000<=float(r['xd'])<=40500]
    summary['traction_profiles'][label]=dict(projected_ptp={key:float(np.ptp(col(p,key)[mask]))
      for key in ['delta_p','minus_delta_tau_N','sigma_n','q']},
      raw_min=min(samples,key=lambda r:float(r['sigma_n'])),raw_max=max(samples,key=lambda r:float(r['sigma_n'])),
      raw_pressure_extrema_in_selected_samples=[min(float(r['delta_p']) for r in samples),max(float(r['delta_p']) for r in samples)],
      raw_tauN_extrema_in_selected_samples=[min(float(r['delta_tau_N']) for r in samples),max(float(r['delta_tau_N']) for r in samples)],
      particle_tensile_weight=sum(r['particle_tensile_weight'] for i,r in enumerate(w) if mask[i]),
      fe_tensile_weight=sum(r['fe_tensile_weight'] for i,r in enumerate(w) if mask[i]),
      bulk_pressure=bulk_pressure(root))

# Nested fault spaces allow exactly the same old Q1 test function to be
# applied to the fine weak vectors. This separates changed test footprints
# from the native-row comparison; it does not transfer constitutive history.
summary['common_test_final']=[]
old_f,old_w,_=data['100m',13];new_f,new_w,_=data['50m',13]
old_x=col(old_f,'xd');new_x=col(new_f,'xd')
for i,x in enumerate(old_x):
    if not (39000-1e-7<=x<=40000+1e-7 or abs(x-25000)<1e-7):continue
    a,b=old_x[i+1],old_x[i-1]
    test=np.maximum(0,np.minimum((new_x-a)/(x-a),(b-new_x)/(b-x)))
    fine={key:sum(test[j]*r[key] for j,r in enumerate(new_w)) for key in
          ['weight','particle_q','particle_C','particle_friction','particle_damping','particle_R']}
    summary['common_test_final'].append(dict(xd=float(x),
        **{label+'_'+key:(values[key]/values['weight'] if key!='weight' else values[key])
           for label,values in [('100m',old_w[i]),('50m',fine)] for key in fine}))

with (run/'junction_history.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
(run/'fault_grid_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(11,9),sharex=True)
for label,style in [('100m','o-'),('50m','.-')]:
    selected=sorted([r for r in all_rows if r['case']==label and r['step']==13 and r['xd']!=25000 and 39000-1e-7<=r['xd']<=40500+1e-7],key=lambda r:r['xd'])
    for ax,key,scale,title in zip(axes.flat,['V_over_Vp','Theta','slip','particle_C','particle_R','sigma_projected'],
          [1,31557600,1,1e6,1e6,1e6],['V / Vp','committed Theta (yr)','accumulated slip (m)',
          'mechanical cohesion / test weight (MPa)','physical R / test weight (MPa)','projected sigma_n (MPa)']):
        ax.plot(col(selected,'xd')/1000,col(selected,key)/scale,style,markersize=3,label=label)
        ax.set_ylabel(title);ax.grid(alpha=.25);ax.axvline(40,color='gray',ls=':')
for ax in axes.flat:ax.axvspan(40,40.5,color='gray',alpha=.08)
axes[0,0].legend();axes[-1,0].set_xlabel('down-dip distance (km)');axes[-1,1].set_xlabel('down-dip distance (km)')
fig.suptitle('Final common time: 2.232176379e9 s; gray region has prescribed V\nNative Q1 coordinates; weak densities use each grid\'s own test functions')
fig.tight_layout();fig.savefig(run/'fault_grid_profiles.png',dpi=160)
print(json.dumps({key:value for key,value in summary.items() if key not in ['common_coordinates','initial','states']},indent=2))
