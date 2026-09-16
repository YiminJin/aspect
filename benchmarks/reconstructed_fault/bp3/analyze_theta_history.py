"""Matched-clock comparison; state-function replay is not a nodal reset."""
import csv
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from analyze_theta_interpolation import read, column, update, velocity, mu

HERE=Path(__file__).resolve().parent
BASE=HERE/'fault-grid-50-local4'
RUN=HERE/'theta-history-50-local4'
OUT=RUN/'comparison'
YEAR=365.25*86400


def write(name,records):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def weak(root,k):
    parts=[read(root/f'history_surface_step{k}_rank{rank}.csv') for rank in range(4)]
    return {key:sum(column(p,key) for p in parts) for key in list(parts[0][0])[5:]}


def pressure(root,k):
    values=[];points=[]
    for path in sorted(root.glob(f'bulk_{k}_*.vtu')):
        r=vtkXMLUnstructuredGridReader();r.SetFileName(str(path));r.Update();g=r.GetOutput()
        xy=vtk_to_numpy(g.GetPoints().GetData());p=vtk_to_numpy(g.GetPointData().GetArray('delta_pressure'))
        xd=(.5e5*(1+1/np.sqrt(3))-xy[:,0])*.5+(1e5-xy[:,1])*np.sqrt(3)/2
        normal=(xy[:,0]-.5e5*(1+1/np.sqrt(3)))*np.sqrt(3)/2-(xy[:,1]-1e5)*.5
        mask=(xd>=39000)&(xd<=40500)&(abs(normal)<=1500)
        values.extend(p[mask]);points.extend(xy[mask])
    a=np.array(values);p=np.array(points)
    return dict(minimum_Pa=float(a.min()),maximum_Pa=float(a.max()),
                minimum_point=p[a.argmin()].tolist(),maximum_point=p[a.argmax()].tolist(),
                semantics='raw bulk Q1 VTU vertex pressure, 39--40.5 km and |normal|<=1500 m')


def main():
    OUT.mkdir(exist_ok=True)
    execution=json.loads((RUN/'execution.json').read_text());assert execution['status']==0
    states=[read(root/'accepted_steps.csv') for root in (BASE,RUN)]
    assert len(states[0])==len(states[1])==14
    for key in ('time','dt'):
        np.testing.assert_allclose(column(states[0],key),column(states[1],key),rtol=1e-13,atol=1e-8)
    log=(RUN/'run.log').read_text()
    linear=[(int(a),float(b),float(c)) for a,b,c in re.findall(
        r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)]
    assert linear and all(b<=c for a,b,c in linear)
    sections=re.split(r'\*\*\* Timestep (\d+):',log);nonlinear=[]
    for n in range(1,len(sections),2):
        k=int(sections[n]);text=sections[n+1]
        last=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',text)[-1]
        assert all(float(v)<1e-8 for v in last)
        assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',text)
        nonlinear.append(dict(step=k,bulk=float(last[0]),surface=float(last[1])))
    meshes=[{r['cell']:r for p in root.glob('initial_mesh_*.csv') for r in read(p)} for root in (BASE,RUN)]
    assert meshes[0]==meshes[1] and len(meshes[0])==42880
    data={};series=[];nodes=[];gradients=[];raw_records=[];friction_checks=[];support_extrema=[]
    for label,root,clock in [('A',BASE,states[0]),('B',RUN,states[1])]:
        previous=None
        for k,state in enumerate(clock):
            f=read(root/f'fault_{k}.csv');w=weak(root,k);projected=read(root/f'stress_projected_{k}.csv')
            data[label,k]=f;x=column(f,'xd');v=column(f,'V');slip=column(f,'slip')
            assert len(f)==1236
            if previous is not None:
                np.testing.assert_allclose(column(f,'Theta'),update(column(previous,'Theta'),v,float(state['dt'])),rtol=1e-12)
                np.testing.assert_allclose(slip,column(previous,'slip')+float(state['dt'])*v,rtol=1e-13,atol=1e-15)
            assert float(read(root/f'history_{k}.csv')[0]['Theta_reference_relative_error'])<1e-12
            for key in ('x','y','tau_bg','sigma_n_bg'):
                np.testing.assert_array_equal(column(f,key),column(data[label,0],key))
            np.testing.assert_allclose(column(f,'Ih'),column(data[label,0],'Ih'),rtol=1e-12)
            assert np.all(v[column(f,'prescribed')==1]==1e-9)
            np.testing.assert_allclose(w['particle_R'],column(f,'weak_residual'),rtol=0,atol=1e-12*max(abs(w['particle_q'])))
            raw=[r for rank in range(4) for r in read(root/f'stress_samples_{k}_rank{rank}.csv')]
            # Exports are selected by sigma, so only sigma extrema are complete;
            # pressure at these points is a decomposition, not a pressure extremum.
            lo=min(raw,key=lambda r:float(r['sigma_n']));hi=max(raw,key=lambda r:float(r['sigma_n']))
            assert abs(float(lo['sigma_n'])-float(state['min_sigma_n']))<1e-6
            assert abs(float(hi['sigma_n'])-float(state['max_sigma_n']))<1e-6
            for selection,r in [('min_sigma',lo),('max_sigma',hi)]:
                raw_records.append(dict(r,case=label,step=k,selection=selection))
            for support in (0,1,2):
                group=[r for r in raw if int(r['support'])==support]
                for selection,choose in [('minimum',min),('maximum',max)]:
                    r=choose(group,key=lambda r:float(r['sigma_n']))
                    sj=int(r['segment']);xi=float(r['xi'])
                    support_extrema.append(dict(case=label,step=k,support=support,selection=selection,
                        xd=(1-xi)*x[sj]+xi*x[sj+1],segment=sj,xi=xi,particle=int(r['particle']),
                        parent_x=float(r['parent_x']),parent_y=float(r['parent_y']),
                        delta_p_Pa=float(r['delta_p']),minus_tauN_Pa=-float(r['delta_tau_N']),sigma_n_Pa=float(r['sigma_n'])))
            # Independent point-state check in the pure VS part of the RSF
            # region. Use the actual mechanical V, but only preceding updates.
            subset=[r for r in raw if 18001<float(f[int(r['segment'])+1]['xd'])
                    and float(f[int(r['segment'])]['xd'])<40001]
            if subset:
                jj=column(subset,'segment').astype(int);zz=column(subset,'xi')
                original=column(data[label,0],'Theta')
                theta=(1-zz)*original[jj]+zz*original[jj+1]
                if label=='A':
                    t=column(data[label,max(0,k-1)],'Theta');theta=(1-zz)*t[jj]+zz*t[jj+1]
                else:
                    for previous_step in range(1,k):
                        prior=data[label,previous_step]
                        theta=update(theta,velocity(column(prior,'V'),jj,zz),float(clock[previous_step]['dt']))
                error=float(np.max(abs(mu(column(subset,'V'),theta)-column(subset,'mu'))))
                assert error<1e-12
                friction_checks.append(dict(case=label,step=k,samples=len(subset),mu_absolute_error=error))
            g=np.diff(slip)/np.diff(x);mask=(x[:-1]>=39000)&(x[:-1]<=40500)
            s=dict(case=label,step=k,time_s=float(state['time']),time_yr=float(state['time'])/YEAR,
                V_last=v[796],V_neighbour=v[797],last_over_Vp=v[796]/1e-9,
                active_last=int(projected[796]['lower_active']),reaction_Pa=-w['particle_R'][796]/w['weight'][796],
                slip_last=slip[796],slip_junction=slip[795],slip_gradient_last=g[795],
                slip_gradient_penultimate=g[796],max_abs_junction_slip_gradient=float(max(abs(g[mask]))),
                sigma_min_Pa=float(lo['sigma_n']),sigma_max_Pa=float(hi['sigma_n']),
                sigma_min_xd=(1-float(lo['xi']))*x[int(lo['segment'])]+float(lo['xi'])*x[int(lo['segment'])+1],
                sigma_min_segment=int(lo['segment']),sigma_min_xi=float(lo['xi']),sigma_min_particle=int(lo['particle']),
                sigma_min_free_shape=float(lo['free_shape']),
                sigma_min_parent_x=float(lo['parent_x']),sigma_min_parent_y=float(lo['parent_y']),
                sigma_min_p_Pa=float(lo['delta_p']),sigma_min_minus_tauN_Pa=-float(lo['delta_tau_N']))
            series.append(s)
            for i in np.flatnonzero((x>=39000-1e-7)&(x<=40500+1e-7)):
                nodes.append(dict(case=label,step=k,xd=x[i],V=v[i],Theta_nodal=float(f[i]['Theta']),slip=slip[i],
                    active=int(projected[i]['lower_active']),p_weak_Pa=w['p'][i]/w['weight'][i],
                    sigma_weak_Pa=w['particle_sigma'][i]/w['weight'][i],R_Pa=w['particle_R'][i]/w['weight'][i]))
            for i in np.flatnonzero(mask):
                gradients.append(dict(case=label,step=k,segment=int(i),xd_mid=(x[i]+x[i+1])/2,slip_gradient=g[i]))
            previous=f
    initial={}
    for key in ('V','Theta','Ih','C','tau_bg'):
        a=column(data['A',0],key);b=column(data['B',0],key)
        initial[key]=float(max(abs(a-b)))
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-20)
    # Recover the accepted functional representation and independently check it
    # against this run's own accepted rates, never the baseline's later Theta.
    tokens=(RUN/'theta_function_history.txt').read_text().split();n=int(tokens[0]);position=1+3*n
    initial_function=np.array([float(tokens[1+3*i+2]) for i in range(n)])
    np.testing.assert_array_equal(initial_function,column(data['B',0],'Theta'))
    for k in range(1,14):
        assert int(tokens[position])==k
        assert float(tokens[position+1])==float(states[1][k]['dt'])
        np.testing.assert_array_equal(np.array(tokens[position+2:position+2+n],dtype=float),column(data['B',k],'V'))
        position+=2+n
    assert position==len(tokens)
    profiles=[];j=np.repeat(np.arange(795,1235),9);z=np.tile(np.linspace(0,1,9),440)
    theta=(1-z)*initial_function[j]+z*initial_function[j+1]
    for k in range(14):
        old=theta.copy();f=data['B',k];v=velocity(column(f,'V'),j,z)
        if k:theta=update(theta,v,float(states[1][k]['dt']))
        shadow=(1-z)*column(f,'Theta')[j]+z*column(f,'Theta')[j+1]
        assert np.max(abs(theta[z==0]/column(f,'Theta')[j[z==0]]-1))<1e-12
        for i in range(len(j)):
            profiles.append(dict(step=k,segment=j[i],xi=z[i],xd=(1-z[i])*float(f[j[i]]['xd'])+z[i]*float(f[j[i]+1]['xd']),
                V=v[i],Theta_mechanics=old[i],Theta_committed_function=theta[i],Theta_Q1_shadow=shadow[i]))
    write('series.csv',series);write('nodes.csv',nodes);write('slip_gradients.csv',gradients)
    write('raw_sigma_extrema.csv',raw_records);write('functional_state_profiles.csv',profiles)
    write('raw_support_extrema.csv',support_extrema)
    summary=dict(execution=execution,initial_max_absolute_differences=initial,mesh_identical=True,
        nonlinear=nonlinear,fresh_linear_checks=len(linear),worst_fresh_target=max(b/c for a,b,c in linear),
        onset={},final={},bulk_pressure={},pointwise_friction_checks=friction_checks,
        final_support_extrema=[r for r in support_extrema if r['step']==13])
    for label,root in [('A',BASE),('B',RUN)]:
        rows=[r for r in series if r['case']==label]
        first=lambda key:next((dict(step=r['step'],time_yr=r['time_yr']) for r in rows if key(r)),None)
        summary['onset'][label]={str(level):first(lambda r:r['last_over_Vp']<level) for level in (.9,.5,.1,.01)}
        summary['onset'][label]['bound']=first(lambda r:r['active_last'])
        summary['final'][label]=rows[-1]
        summary['bulk_pressure'][label]={str(k):pressure(root,k) for k in (11,12,13)}
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    for label in ('A','B'):
        s=[r for r in series if r['case']==label];t=[r['time_yr'] for r in s]
        ax[0,0].semilogy(t,[r['last_over_Vp'] for r in s],'-o',label=label)
        ax[0,1].plot(t,[r['slip_gradient_last'] for r in s],'-o',label=label)
        junction=[r for r in support_extrema if r['case']==label and r['support']==2 and r['selection']=='minimum']
        ax[1,0].plot(t,[r['sigma_n_Pa']/1e6 for r in junction],'-o',label=label+' junction')
        ax[1,0].plot(t,[r['sigma_min_Pa']/1e6 for r in s],'--',label=label+' global')
        f=data[label,13];x=column(f,'xd');m=(x>=39500)&(x<=40200)
        ax[1,1].semilogy(x[m]/1000,column(f,'V')[m],'-o',label=label)
    for a,title in zip(ax.flat,['39.95 km V/Vp','last-element accumulated-slip gradient','raw minimum sigma_n (MPa)','final V profile (m/s)']):
        a.set_title(title);a.grid();a.legend()
    for a in (ax[0,0],ax[0,1],ax[1,0]):a.set_xlabel('time (yr)')
    ax[1,1].set_xlabel('down-dip distance (km)')
    fig.suptitle('A: nodal aging; B: fixed-coordinate aging — same 50-m mesh and timesteps')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
