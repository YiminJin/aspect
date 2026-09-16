"""Bounded mature-law verification against the retained frozen-force reference."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from analyze_theta_interpolation import read,column,update
from analyze_theta_history import weak
from analyze_frozen_cohesion import pressure_pattern
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE=Path(__file__).resolve().parent
RUN=HERE/'mature-fault-50-local4'
REF=HERE/'frozen-cohesion-adaptive-50-local4'
YEAR=31557600.

def write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    out=RUN/'comparison';out.mkdir(exist_ok=True)
    clock=read(RUN/'accepted_steps.csv');refclock=read(REF/'accepted_steps.csv')[:len(clock)]
    assert len(clock)<=11
    for key in ('time','dt'):
        np.testing.assert_allclose(column(clock,key),column(refclock,key),rtol=1e-13,atol=1e-6)
    log=(RUN/'run.log').read_text()
    linear=[(float(a),float(b)) for a,b in re.findall(r'Fault linear solve: iterations=\d+, estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)]
    assert linear and all(a<=b for a,b in linear)
    sections=re.split(r'\*\*\* Timestep (\d+):',log);nonlinear=[]
    for n in range(1,len(sections),2):
        k=int(sections[n]);section=sections[n+1]
        if k>=len(clock):continue
        last=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)[-1]
        assert all(float(v)<1e-8 for v in last)
        assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',section)
        nonlinear.append(dict(step=k,bulk=float(last[0]),surface=float(last[1])))
    assert len(nonlinear)==len(clock)
    coefficients=np.loadtxt(RUN/'prestress.txt',skiprows=1)
    frozen=np.loadtxt(REF/'initial_cohesion.txt',skiprows=1)
    prep=json.loads((RUN/'preparation.json').read_text())
    beta,kappa=prep['beta0'],prep['kappa0']
    initial_bulk={}
    for rank in range(4):
        grids=[]
        for root in (RUN,REF):
            reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(root/f'bulk_0_{rank}.vtu'));reader.Update()
            grids.append(reader.GetOutput())
        np.testing.assert_array_equal(vtk_to_numpy(grids[0].GetPoints().GetData()),vtk_to_numpy(grids[1].GetPoints().GetData()))
        for name in ('velocity_0','velocity_1','delta_pressure','tau_xx','tau_yy','tau_xy','component_9'):
            a,b=[vtk_to_numpy(g.GetPointData().GetArray(name)) for g in grids]
            error=float(max(abs(a-b)));scale=float(max(abs(b)))
            initial_bulk[name]=max(initial_bulk.get(name,0.),error)
            if name.startswith('velocity') or name=='delta_pressure':
                assert error<=4*float(np.spacing(np.float32(scale)))
            else:np.testing.assert_array_equal(a,b)
    initial_H={};reference_geometry=None;previous=None;series=[];errors=[];raw_extrema=[]
    frozen_audit=[];pressure_comparison=[];initial_differences={}
    for k,c in enumerate(clock):
        current=read(RUN/f'fault_{k}.csv');reference=read(REF/f'fault_{k}.csv')
        w=weak(RUN,k);wr=weak(REF,k)
        x=column(current,'xd');v=column(current,'V');theta=column(current,'Theta');slip=column(current,'slip')
        assert len(v)==1236
        geometry=np.array([column(current,'x'),column(current,'y')])
        if reference_geometry is None:reference_geometry=geometry
        np.testing.assert_array_equal(geometry,reference_geometry)
        np.testing.assert_array_equal(geometry,np.array([column(reference,'x'),column(reference,'y')]))
        np.testing.assert_array_equal(column(current,'C'),np.zeros(len(v)))
        np.testing.assert_array_equal(column(current,'C_evaluated'),np.zeros(len(v)))
        np.testing.assert_array_equal(w['particle_C'],np.zeros(len(v)))
        if previous:
            np.testing.assert_allclose(theta,update(column(previous,'Theta'),v,float(c['dt'])),rtol=1e-12)
            np.testing.assert_allclose(slip,column(previous,'slip')+float(c['dt'])*v,rtol=1e-13,atol=1e-15)
        assert np.all(v[column(current,'prescribed')==1]==1e-9)
        for key,absolute in [('V',1e-20),('Theta',1e-8),('slip',1e-12),('Ih',1e-8)]:
            a=column(current,key);b=column(reference,key)
            # Data are full precision. This is a strict equivalence check,
            # not a replacement for either run's nonlinear convergence test.
            err=float(max(abs(a-b)));rel=float(max(abs(a-b)/np.maximum(abs(b),absolute)))
            errors.append(dict(step=k,field=key,max_absolute=err,max_relative=rel))
            np.testing.assert_allclose(a,b,rtol=1e-8,atol=absolute)
            if k==0:initial_differences[key]=err
        np.testing.assert_allclose(w['particle_q']-w['particle_friction']-w['particle_damping'],
            w['particle_R'],rtol=0,atol=1e-12*max(abs(w['particle_q'])))
        # All weak rows, including prescribed rows: q_new equals q_old-C_star.
        scale=np.maximum(w['weight'],wr['weight'])
        err=float(max(abs(w['particle_q']-(wr['particle_q']-wr['particle_C']))/scale))
        errors.append(dict(step=k,field='weak q_new versus q_old-C_star (Pa)',max_absolute=err,max_relative=err/50e6))
        assert err<.5
        for label,root,f,weights in [('mature',RUN,current,w),('frozen force',REF,reference,wr)]:
            rate=column(f,'V');s=column(f,'slip')
            series.append(dict(case=label,step=k,time_yr=float(c['time'])/YEAR,
                V_last=rate[796],V_neighbour=rate[797],V_control=rate[985],slip_last=s[796],
                slip_gradient_last=(s[795]-s[796])/50,
                q_weak_Pa=weights['particle_q'][796]/weights['weight'][796],
                C_weak_Pa=weights['particle_C'][796]/weights['weight'][796],
                R_weak_Pa=weights['particle_R'][796]/weights['weight'][796]))
            raw=[r for rank in range(4) for r in read(root/f'stress_samples_{k}_rank{rank}.csv')]
            for support in (0,1,2):
                group=[r for r in raw if int(r['support'])==support]
                for selection,fn in [('min',min),('max',max)]:
                    r=fn(group,key=lambda r:float(r['sigma_n']));j=int(r['segment']);z=float(r['xi'])
                    raw_extrema.append(dict(case=label,step=k,support=support,selection=selection,
                        xd=(1-z)*float(f[j]['xd'])+z*float(f[j+1]['xd']),sigma_Pa=float(r['sigma_n']),
                        p_Pa=float(r['delta_p']),minus_tauN_Pa=-float(r['delta_tau_N'])))
            if label=='frozen force':
                # Frozen-field algebraic check at every SAVED raw coordinate.
                # Independently compare captured C0,V0,I0 evaluation with the
                # generic rational prestress coefficients used in production.
                error=0.
                for r in raw:
                    j=int(r['segment']);z=float(r['xi']);V=float(r['V'])
                    a=(1-z)*coefficients[j,4]+z*coefficients[j+1,4]
                    b=(1-z)*coefficients[j,5]+z*coefficients[j+1,5]
                    d=(1-z)*coefficients[j,6]+z*coefficients[j+1,6]
                    old_C=(1-z)*frozen[j,2]+z*frozen[j+1,2]
                    old_V=frozen[j,3]+z*(frozen[j+1,3]-frozen[j,3])
                    old_I=(1-z)*frozen[j,4]+z*frozen[j+1,4]
                    Cstar=(kappa*old_V+beta*old_I*old_C)/old_I
                    # Initialization uses the old evaluated force at its solved
                    # V0, so the captured snapshot applies there as well.
                    original=float(r['q'])-Cstar-float(r['mu_sigma'])-4624440*V
                    reduced=(float(r['q'])-(a+b/d))-float(r['mu_sigma'])-4624440*V
                    error=max(error,abs(original-reduced))
                assert error<1e-6
                frozen_audit.append(dict(step=k,samples=len(raw),maximum_residual_difference_Pa=error))
        # Full stable-ID H retention, not a maximum or an advected point sample.
        seen=set()
        for rank in range(4):
            for r in read(RUN/f'mature_history_{k}_rank{rank}.csv'):
                i=int(r['id']);assert i not in seen;seen.add(i);H=float(r['H_inert'])
                if k==0:
                    initial_H[i]=H
                    assert all(float(r[t])==0 for t in ('tau_xx','tau_yy','tau_xy'))
                assert H==initial_H[i]
        assert len(seen)==len(initial_H)==385920
        if list(RUN.glob(f'bulk_{k}_*.vtu')) and list(REF.glob(f'bulk_{k}_*.vtu')):
            a=pressure_pattern(RUN,k);b=pressure_pattern(REF,k);assert a.keys()==b.keys()
            err=max(abs(a[p]-b[p]) for p in a)
            # Float32 VTU quantization is recorded separately from solve accuracy.
            output_ulp=float(np.spacing(np.float32(max(abs(v) for v in b.values()))))
            assert err<=max(1.,4*output_ulp)
            pressure_comparison.append(dict(step=k,vertices=len(a),max_difference_Pa=err,
                mature_minimum_Pa=min(a.values()),mature_maximum_Pa=max(a.values()),
                reference_minimum_Pa=min(b.values()),reference_maximum_Pa=max(b.values())))
        previous=current
    write(out/'trajectory.csv',series);write(out/'field_errors.csv',errors)
    write(out/'raw_normal_extrema.csv',raw_extrema);write(out/'saved_field_residual.csv',frozen_audit)
    write(out/'pressure_comparison.csv',pressure_comparison)
    raw_error=max(abs(a['sigma_Pa']-b['sigma_Pa']) for a,b in zip(
        [r for r in raw_extrema if r['case']=='mature'],[r for r in raw_extrema if r['case']=='frozen force']))
    assert raw_error<.5
    result=dict(execution=json.loads((RUN/'execution.json').read_text()),accepted_states=len(clock),
        last_time_yr=float(clock[-1]['time'])/YEAR,nonlinear=nonlinear,fresh_linear_checks=len(linear),
        worst_fresh_target=max(a/b for a,b in linear),initial_max_differences=initial_differences,
        H_retained_ids=len(initial_H),cohesive_force_and_energy_zero=True,
        initial_bulk_output_max_differences=initial_bulk,raw_normal_extrema_max_difference_Pa=raw_error,
        derivative_checks=[s for s in log.splitlines() if 'Frozen cohesion' in s],
        maximum_saved_residual_difference_Pa=max(r['maximum_residual_difference_Pa'] for r in frozen_audit))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
