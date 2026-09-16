"""Matched-clock force-isolation replay, with nodal Theta left unchanged."""
import csv
import json
from pathlib import Path
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_theta_interpolation import read, column, update, mu, velocity
from analyze_theta_history import weak, pressure
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

HERE=Path(__file__).resolve().parent
BASE=HERE/'fault-grid-50-local4'; RUN=HERE/'frozen-cohesion-50-local4'
OUT=RUN/'comparison'; YEAR=365.25*86400

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pressure_pattern(root,k):
    # Identical physical VTU vertices in the fixed junction window. This is
    # a pointwise pattern check, not a bulk-volume or weak-traction norm.
    values={}
    for path in sorted(root.glob(f'bulk_{k}_*.vtu')):
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
        grid=reader.GetOutput();xy=vtk_to_numpy(grid.GetPoints().GetData())
        p=vtk_to_numpy(grid.GetPointData().GetArray('delta_pressure'))
        xd=(.5e5*(1+1/np.sqrt(3))-xy[:,0])*.5+(1e5-xy[:,1])*np.sqrt(3)/2
        normal=(xy[:,0]-.5e5*(1+1/np.sqrt(3)))*np.sqrt(3)/2-(xy[:,1]-1e5)*.5
        for i in np.flatnonzero((xd>=39000)&(xd<=40500)&(abs(normal)<=1500)):
            key=tuple(float(v) for v in xy[i]);value=float(p[i])
            if key in values:assert values[key]==value
            values[key]=value
    if not values and (root/f'pressure_snapshot_{k}.csv').exists():
        values={tuple(float(r[key]) for key in ('x','y','z')):float(r['pressure'])
                for r in read(root/f'pressure_snapshot_{k}.csv')}
    assert values
    return values

def snapshot_pressure(case,step,export):
    """Evaluate the saved Q1 pressure polynomial on the existing VTU vertices."""
    meta=read(export/'checkpoint_bulk_metadata.csv')[0];accepted=read(case/f'fault_{step}.csv')[0]
    assert int(meta['step'])==step+1 and int(meta['cells'])==42880
    assert abs(float(meta['time'])-float(meta['dt'])-float(accepted['time']))<1e-6
    cells={}
    for path in export.glob('bulk_owned_rank*.csv'):
        for r in read(path):
            entries=cells.setdefault(r['cell'],{});i=int(r['local']);assert i not in entries
            entries[i]=float(r['value'])
    assert len(cells)==42880 and all(len(v)==22 for v in cells.values())
    template=pressure_pattern(case,0);values={}
    for path in case.glob('initial_mesh_*.csv'):
        for r in read(path):
            # This verified fixture has Q2 velocity and Q1 pressure. Among
            # Stokes entries, the first twelve are (ux,uy,p) at four vertices.
            entries=cells[r['cell']];ordered=[entries[i] for i in sorted(entries)]
            p=np.array([ordered[3*j+2] for j in range(4)])
            x,y,h=(float(r[key]) for key in ('x','y','h'))
            for xi in (0.,.5,1.):
                for eta in (0.,.5,1.):
                    key=(x+h*(xi-.5),y+h*(eta-.5),0.)
                    if key not in template:continue
                    value=float(p@np.array([(1-xi)*(1-eta),xi*(1-eta),(1-xi)*eta,xi*eta]))
                    if key in values:np.testing.assert_allclose(values[key],value,rtol=1e-12,atol=1e-7)
                    values[key]=value
    assert values.keys()==template.keys()
    validation=None
    if list(case.glob(f'bulk_{step}_*.vtu')):
        actual=pressure_pattern(case,step);assert actual.keys()==values.keys()
        keys=sorted(actual);a=np.array([actual[k] for k in keys]);b=np.array([values[k] for k in keys])
        np.testing.assert_allclose(b.astype(np.float32),a,rtol=0,atol=0)
        validation=dict(maximum_Pa=float(max(abs(a-b))),float32_identical=True)
    with (case/f'pressure_snapshot_{step}.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['x','y','z','pressure'])
        writer.writerows([*key,values[key]] for key in sorted(values))
    result=dict(step=step,vertices=len(values),validation_against_existing_vtu=validation)
    (export/'pressure_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def main():
    global RUN,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--pressure-export',type=Path)
    parser.add_argument('--pressure-step',type=int,default=13)
    parser.add_argument('--pressure-case',type=Path,default=BASE)
    args=parser.parse_args()
    if args.pressure_export:
        snapshot_pressure(args.pressure_case,args.pressure_step,args.pressure_export)
        return
    if args.adaptive:
        RUN=HERE/'frozen-cohesion-adaptive-50-local4';OUT=RUN/'comparison'
    OUT.mkdir(exist_ok=True)
    clocks={label:read(root/'accepted_steps.csv') for label,root in [('A',BASE),('C',RUN)]}
    count=len(clocks['C']);assert count>=1
    if not args.adaptive:
        clocks['A']=[r for r in clocks['A'] if float(r['time'])<=float(clocks['C'][-1]['time'])+1e-6]
    matches={};clock_rows=[]
    for k,a in enumerate(clocks['A']):
        if float(a['time'])>float(clocks['C'][-1]['time'])+1e-6:continue
        j=int(np.argmin(abs(column(clocks['C'],'time')-float(a['time']))));b=clocks['C'][j]
        np.testing.assert_allclose(float(a['time']),float(b['time']),rtol=1e-13,atol=1e-8)
        matches[k]=j
        clock_rows.append(dict(baseline_step=k,diagnostic_step=j,time_s=float(a['time']),
            baseline_dt=float(a['dt']),diagnostic_dt=float(b['dt'])))
    if not args.adaptive:
        for key in ('time','dt'):
            np.testing.assert_allclose(column(clocks['C'],key),column(clocks['A'],key),rtol=1e-13,atol=1e-8)
    for clock in clocks.values():
        np.testing.assert_allclose(np.diff(column(clock,'time')),column(clock,'dt')[1:],rtol=1e-13,atol=1e-8)
    log=(RUN/'run.log').read_text()
    linear=[(int(a),float(b),float(c)) for a,b,c in re.findall(
        r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)]
    assert linear and all(b<=c for _,b,c in linear)
    nonlinear=[];unpublished_mechanics=[]
    sections=re.split(r'\*\*\* Timestep (\d+):',log)
    for n in range(1,len(sections),2):
        k=int(sections[n]);section=sections[n+1]
        residuals=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)
        if not residuals:
            assert k>=count,'An accepted state must have convergence evidence.'
            unpublished_mechanics.append(dict(step=k,no_completed_iteration=True))
            continue
        last=residuals[-1]
        if k>=count:
            unpublished_mechanics.append(dict(step=k,bulk=float(last[0]),surface=float(last[1])))
            continue
        assert all(float(v)<1e-8 for v in last)
        assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',section)
        nonlinear.append(dict(step=k,bulk=float(last[0]),surface=float(last[1])))
    meshes=[{r['cell']:r for p in root.glob('initial_mesh_*.csv') for r in read(p)} for root in (BASE,RUN)]
    assert meshes[0]==meshes[1] and len(meshes[0])==42880
    provenance=json.loads((RUN/'provenance.json').read_text())
    assert 'ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC' not in provenance['environment']
    data={};series=[];budgets=[];profiles=[];extrema=[];friction_checks=[];cohesive_memory=[]
    for label,root in [('A',BASE),('C',RUN)]:
        previous=None
        for k,clock in enumerate(clocks[label]):
            f=read(root/f'fault_{k}.csv');data[label,k]=f;w=weak(root,k)
            active=column(read(root/f'stress_projected_{k}.csv'),'lower_active').astype(int)
            x=column(f,'xd');v=column(f,'V');slip=column(f,'slip');theta=column(f,'Theta')
            if previous is not None:
                np.testing.assert_allclose(theta,update(column(previous,'Theta'),v,float(clock['dt'])),rtol=1e-12)
                np.testing.assert_allclose(slip,column(previous,'slip')+float(clock['dt'])*v,rtol=1e-13,atol=1e-15)
            assert float(read(root/f'history_{k}.csv')[0]['Theta_reference_relative_error'])<1e-12
            for key in ('x','y','tau_bg','sigma_n_bg'):
                np.testing.assert_array_equal(column(f,key),column(data[label,0],key))
            np.testing.assert_allclose(column(f,'Ih'),column(data[label,0],'Ih'),rtol=1e-12)
            assert np.all(v[column(f,'prescribed')==1]==1e-9)
            np.testing.assert_allclose(w['particle_R'],column(f,'weak_residual'),rtol=0,atol=1e-12*max(abs(w['particle_q'])))
            np.testing.assert_allclose(w['particle_q']-w['particle_C']-w['particle_friction']-w['particle_damping'],
                                       w['particle_R'],rtol=0,atol=1e-12*max(abs(w['particle_q'])))
            raw=[r for rank in range(4) for r in read(root/f'stress_samples_{k}_rank{rank}.csv')]
            assert abs(min(float(r['sigma_n']) for r in raw)-float(clock['min_sigma_n']))<1e-6
            for support in (0,1,2):
                group=[r for r in raw if int(r['support'])==support]
                for selection,choose in [('minimum',min),('maximum',max)]:
                    r=choose(group,key=lambda r:float(r['sigma_n']));j=int(r['segment']);z=float(r['xi'])
                    extrema.append(dict(case=label,step=k,time_yr=float(clock['time'])/YEAR,support=support,selection=selection,
                        xd=(1-z)*x[j]+z*x[j+1],segment=j,xi=z,particle=int(r['particle']),
                        p_Pa=float(r['delta_p']),minus_tauN_Pa=-float(r['delta_tau_N']),sigma_Pa=float(r['sigma_n'])))
            subset=[r for r in raw if 18001<x[int(r['segment'])+1] and x[int(r['segment'])]<40001]
            if subset:
                jj=column(subset,'segment').astype(int);zz=column(subset,'xi')
                old=column(data[label,max(0,k-1)],'Theta');mechanical=(1-zz)*old[jj]+zz*old[jj+1]
                error=float(max(abs(mu(column(subset,'V'),mechanical)-column(subset,'mu'))))
                assert error<1e-12
                friction_checks.append(dict(case=label,step=k,mu_absolute_error=error,samples=len(subset)))
            gradient=np.diff(slip)/np.diff(x);mask=(x[:-1]>=39000)&(x[:-1]<=40500)
            series.append(dict(case=label,step=k,time_yr=float(clock['time'])/YEAR,
                V_last=v[796],V_neighbour=v[797],V_interior=v[985],last_over_Vp=v[796]/1e-9,
                active_last=int(active[796]),reaction_Pa=-w['particle_R'][796]/w['weight'][796],
                slip_last=slip[796],slip_junction=slip[795],slip_gradient_last=gradient[795],
                slip_gradient_penultimate=gradient[796],max_abs_junction_slip_gradient=float(max(abs(gradient[mask]))),
                sigma_min_Pa=min(float(r['sigma_n']) for r in raw),sigma_max_Pa=max(float(r['sigma_n']) for r in raw)))
            for i in (795,796,797,798,985):
                record=dict(case=label,step=k,time_yr=float(clock['time'])/YEAR,node=i,xd=x[i],V=v[i],active=int(active[i]),weight=w['weight'][i])
                for key in ('q','C','friction','damping','R','sigma'):
                    record[key+'_Pa']=w['particle_'+key][i]/w['weight'][i]
                record['C_shadow_nodal_Pa']=float(f[i]['C'])
                budgets.append(record)
            if label=='A' and k>0:
                # Frozen I_h makes the retained cohesive term exactly beta M C_old.
                # Keep its basis contributions separate from the measured
                # instantaneous V/I_h term, whose reciprocal I_h is not Q1.
                i=796;beta=np.exp(-float(clock['dt'])*32038120320/1e26)
                m=np.array([float(f[i-1]['mass_upper']),float(f[i]['mass_diagonal']),float(f[i]['mass_upper'])])
                np.testing.assert_allclose(sum(m),w['weight'][i],rtol=1e-11)
                history=beta*m*column(previous,'C')[i-1:i+2]/w['weight'][i]
                growth=beta*m*(column(previous,'C')[i-1:i+2]-column(data['A',0],'C')[i-1:i+2])/w['weight'][i]
                residual=w['particle_R'][i]/w['weight'][i]
                cohesive_memory.append(dict(step=k,time_yr=float(clock['time'])/YEAR,
                    prescribed_memory_Pa=history[0],self_memory_Pa=history[1],neighbour_memory_Pa=history[2],
                    prescribed_growth_Pa=growth[0],self_growth_Pa=growth[1],neighbour_growth_Pa=growth[2],
                    instantaneous_Pa=w['particle_C'][i]/w['weight'][i]-sum(history),
                    physical_R_Pa=residual,R_without_prescribed_memory_growth_Pa=residual+growth[0],
                    prescribed_mass_fraction=m[0]/w['weight'][i]))
            for i in np.flatnonzero((x>=39000-1e-7)&(x<=40500+1e-7)):
                profiles.append(dict(case=label,step=k,xd=x[i],V=v[i],slip=slip[i],Theta=theta[i],active=int(active[i]),
                    q_Pa=w['particle_q'][i]/w['weight'][i],C_force_Pa=w['particle_C'][i]/w['weight'][i],
                    C_shadow_Pa=float(f[i]['C']),p_Pa=w['p'][i]/w['weight'][i],sigma_Pa=w['particle_sigma'][i]/w['weight'][i]))
            previous=f
    initial={}
    for key in ('V','Theta','Ih','C','tau_bg','q','C_evaluated'):
        a=column(data['A',0],key);b=column(data['C',0],key)
        initial[key]=float(max(abs(a-b)));np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-20)
    prefix={}
    if args.adaptive:
        previous_run=HERE/'frozen-cohesion-50-local4'
        for k in range(min(10,count)):
            prior=read(previous_run/f'fault_{k}.csv')
            for key in ('V','Theta','C','Ih','slip','q'):
                a=column(prior,key);b=column(data['C',k],key)
                prefix[key]=max(prefix.get(key,0.),float(max(abs(a-b))))
    summary=dict(execution=json.loads((RUN/'execution.json').read_text()),accepted_states=count,
        initial_max_absolute_differences=initial,guarded_prefix_max_absolute_difference=prefix,
        mesh_identical=True,nonlinear=nonlinear,
        unpublished_mechanics=unpublished_mechanics,
        timestep_guard=[dict(next_step=int(k),requested=float(a),selected=float(b)) for k,a,b in re.findall(
            r'BP3 matched timestep: next=(\d+) requested=([^ ]+) selected=([^\n]+)',log) if float(a)!=float(b)],
        fresh_linear_checks=len(linear),worst_fresh_target=max(b/c for _,b,c in linear),
        derivative_checks=[s for s in log.splitlines() if 'Frozen cohesion' in s],
        friction_checks=friction_checks,onset={},final={},last_exported={},bulk_pressure={},common_times=clock_rows,
        final_support_extrema=[r for r in extrema if r['step']==(max(matches) if r['case']=='A' else matches[max(matches)])])
    if (RUN/'stop_record.json').exists():
        summary['intentional_stop']=json.loads((RUN/'stop_record.json').read_text())
    for label,root in [('A',BASE),('C',RUN)]:
        s=[r for r in series if r['case']==label]
        first=lambda condition:next((dict(step=r['step'],time_yr=r['time_yr']) for r in s if condition(r)),None)
        summary['onset'][label]={str(level):first(lambda r:r['last_over_Vp']<level) for level in (.9,.5,.1,.01)}
        summary['onset'][label]['bound']=first(lambda r:r['active_last']);summary['last_exported'][label]=s[-1]
        common_final=max(matches) if label=='A' else matches[max(matches)]
        summary['final'][label]=next(r for r in s if r['step']==common_final)
        summary['bulk_pressure'][label]={}
        for k in sorted(set((6,8,9,10,11,12,13,len(clocks['A'])-1))):
            if k not in matches:continue
            actual_step=k if label=='A' else matches[k]
            if not list(root.glob(f'bulk_{actual_step}_*.vtu')) and not (root/f'pressure_snapshot_{actual_step}.csv').exists():continue
            values=pressure_pattern(root,actual_step);lo=min(values,key=values.get);hi=max(values,key=values.get)
            summary['bulk_pressure'][label][str(k)]=dict(minimum_Pa=values[lo],maximum_Pa=values[hi],
                minimum_point=lo,maximum_point=hi,actual_step=actual_step,
                semantics='Q1 pressure on identical exported vertices in the junction window')
    mechanism=[]
    for label in ('A','C'):
        for baseline_step,values in summary['bulk_pressure'][label].items():
            k=values['actual_step'];r=next(r for r in series if r['case']==label and r['step']==k)
            amplitude=values['maximum_Pa']-values['minimum_Pa'];gradient=r['slip_gradient_last']
            mechanism.append(dict(case=label,baseline_step=int(baseline_step),actual_step=k,time_yr=r['time_yr'],
                pressure_range_Pa=amplitude,slip_gradient=gradient,pressure_per_gradient_Pa=amplitude/gradient))
    # A labelled quasi-steady scale estimate, not an exact weak friction law:
    # use the known VS slope sigma_ref(a-b), and retain measured weak driving.
    rate_scale=[]
    for k,j in matches.items():
        for node in (796,985):
            a=next(r for r in budgets if r['case']=='A' and r['step']==k and r['node']==node)
            b=next(r for r in budgets if r['case']=='C' and r['step']==j and r['node']==node)
            drive=(a['q_Pa']-a['C_Pa'])-(b['q_Pa']-b['C_Pa'])
            rate_scale.append(dict(baseline_step=k,diagnostic_step=j,node=node,time_yr=a['time_yr'],
                driving_difference_Pa=drive,observed_rate_ratio=a['V']/b['V'],
                steady_uniform_sigma_estimate=np.exp(drive/(50e6*(.025-.015)))))
    write('series.csv',series);write('force_budgets.csv',budgets);write('junction_profiles.csv',profiles);write('raw_support_extrema.csv',extrema)
    write('common_times.csv',clock_rows);write('pressure_slip_mechanism.csv',mechanism);write('steady_rate_scale.csv',rate_scale)
    write('cohesive_memory.csv',cohesive_memory)
    patterns=[]
    for k in sorted(set((9,10,max(matches)))):
        if str(k) not in summary['bulk_pressure']['C']:continue
        a=pressure_pattern(BASE,k);c=pressure_pattern(RUN,matches[k]);assert a.keys()==c.keys()
        keys=sorted(a);a=np.array([a[key] for key in keys]);c=np.array([c[key] for key in keys])
        row_a=next(r for r in series if r['case']=='A' and r['step']==k)
        row_c=next(r for r in series if r['case']=='C' and r['step']==matches[k])
        ratio=row_c['slip_gradient_last']/row_a['slip_gradient_last']
        patterns.append(dict(baseline_step=k,diagnostic_step=matches[k],vertices=len(a),
            gradient_ratio=ratio,pressure_best_scalar=float(a@c/(a@a)),
            pressure_pattern_correlation=float(np.corrcoef(a,c)[0,1]),
            relative_rms_using_gradient_ratio=float(np.linalg.norm(c-ratio*a)/np.linalg.norm(c))))
    write('pressure_pattern.csv',patterns)
    summary['pressure_pattern']=patterns
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for label,name in [('A','original cohesive force'),('C','frozen initial cohesive force')]:
        s=[r for r in series if r['case']==label];t=[r['time_yr'] for r in s]
        axes[0,0].plot(t,[r['last_over_Vp'] for r in s],'.-',label=name)
        axes[0,1].plot(t,[r['slip_gradient_last'] for r in s],'.-')
        for key,style in [('q_Pa','-'),('C_Pa','--'),('friction_Pa',':')]:
            b=[r for r in budgets if r['case']==label and r['node']==796]
            axes[1,0].plot(t,[(r[key]-b[0][key])/1e6 for r in b],style,label=label+' '+key)
        p=[r for r in profiles if r['case']==label and r['step']==summary['final'][label]['step']]
        axes[1,1].plot([r['xd']/1000 for r in p],[r['sigma_Pa']/1e6 for r in p],'.-',label=name)
    axes[0,0].set_ylabel('39.95 km V/Vp');axes[0,0].legend(fontsize=8)
    axes[0,1].set_ylabel('last-element accumulated-slip gradient')
    axes[1,0].set_ylabel('39.95 km weak-term change from t=0 (MPa)');axes[1,0].legend(fontsize=8)
    axes[1,1].set_ylabel('final weak normal traction / row mass (MPa)');axes[1,1].set_xlabel('down-dip distance (km)')
    axes[1,1].set_title(f"Common time {summary['final']['A']['time_yr']:.3f} yr; not raw extrema")
    for ax in axes.flat:ax.grid(alpha=.25)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('physical time (yr)')
    if args.adaptive:
        fig.suptitle('Identical timestep history through 29.242 yr; later histories use different timesteps')
        for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.axvline(29.241908941032182,color='grey',ls=':',lw=.8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170)
    print(json.dumps({k:summary[k] for k in ('execution','accepted_states','fresh_linear_checks','worst_fresh_target','derivative_checks','onset','final','bulk_pressure')},indent=2))

if __name__=='__main__':main()
