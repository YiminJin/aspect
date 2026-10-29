#!/usr/bin/env python3
"""Compact response/physical-work evidence at actual accepted times."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from continuation_analysis import BASE,OUT,OLD,read,save_csv

def summarize(name):
    h=read(OUT/(name+'-history.csv'));window=(h['time_s']>.15875+1e-12)&(h['time_s']<=.17875+1e-12);a=h[window];assert len(a)
    hold=a[a['time_s']>.175+1e-12];loaded=a[a['time_s']<=.175+1e-12]
    R=a['trapezoidal_remainder_J_m'];W=a['trapezoidal_work_J_m'];den=float(np.sum(abs(W)))
    result=dict(case=name,dt_values_s=sorted(set(a['dt_s'].tolist())),window=[float(a['time_s'][0]),float(a['time_s'][-1])],maximum_loaded_step_phase_increment=float(loaded['max_step_phase_increment'].max()) if len(loaded) else None,first_hold_increment=float(hold['max_step_phase_increment'][0]) if len(hold) else None,max_abs_w_s_inv=float(a['max_abs_w_s_inv'].max()),max_hold_abs_w_s_inv=float(hold['max_abs_w_s_inv'].max()) if len(hold) else None,endpoint_work_J_m=float(a['endpoint_work_J_m'].sum()),trapezoidal_work_J_m=float(W.sum()),energy_change_J_m=float(a['energy_change_J_m'].sum()),physical_dissipation_J_m={k:float(a[k].sum()) for k in ['Db_J_m','Dc_J_m','friction_J_m']},physical_remainder_J_m=float(R.sum()),sum_absolute_step_remainders_J_m=float(abs(R).sum()),C_E_absolute=float(abs(R).sum()/den) if den else None,maximum_step_remainder_J_m=float(abs(R).max()),BE_terms_J_m={k:float(a[k].sum()) for k in ['stress_loss_J_m','traction_loss_J_m','phase_remainder_J_m','fracture_loss_J_m']},phase_residual_work_J_m=float(a['phase_work_J_m'].sum()),mechanical_residual_work_J_m=float(a['mechanical_work_J_m'].sum()),maximum_native_ledger_error_J_m=float(abs(a['ledger_error_J_m']).max()),hold_work_exactly_zero=bool(np.all(hold['trapezoidal_work_J_m']==0)) if len(hold) else None,maximum_reaction_work_difference_J_m=float(np.max(abs(a['shear_reaction_N_m']*a['native_prescribed_increment_m']-a['endpoint_work_J_m']))))
    assert result['maximum_reaction_work_difference_J_m']<1e-12 and result['maximum_native_ledger_error_J_m']<1e-10
    (OUT/(name+'-work-summary.json')).write_text(json.dumps(result,indent=2)+'\n');return result

def temporal_work_uncertainty():
    results=[]
    for coarse,fine in [('reference-continuation','half-continuation'),('half-continuation','quarter-continuation')]:
        data=[]
        for name in [coarse,fine]:
            h=read(OUT/(name+'-history.csv'));h=h[(h['time_s']>.15875+1e-12)&(h['time_s']<=.17875+1e-12)]
            cumulative={key:np.cumsum(h[key]) for key in ['trapezoidal_work_J_m','energy_change_J_m','Db_J_m','Dc_J_m','friction_J_m']}
            data.append((h,cumulative))
        (a,ca),(b,cb)=data;rows=[];scale=float(abs(b['trapezoidal_work_J_m']).sum())
        for i,t in enumerate(a['time_s']):
            j=np.argmin(abs(b['time_s']-t));assert abs(b['time_s'][j]-t)<1e-10
            differences=[abs(ca[key][i]-cb[key][j]) for key in ca]
            D=abs(sum(ca[key][i]-cb[key][j] for key in ['Db_J_m','Dc_J_m','friction_J_m']))
            rows.append([t,*differences,(differences[0]+differences[1]+D)/scale,sum(differences)/scale])
        save_csv(OUT/(coarse+'-vs-'+fine+'-work.csv'),['time_s','cumulative_work_difference_J_m','cumulative_storage_difference_J_m','cumulative_bulk_dissipation_difference_J_m','cumulative_cohesive_dissipation_difference_J_m','cumulative_friction_difference_J_m','C_T_total_dissipation','C_T_separate_dissipations'],rows)
        rows=np.asarray(rows);CE=float(abs(b['trapezoidal_remainder_J_m']).sum()/scale)
        results.append(dict(coarse=coarse,fine=fine,W_scale_J_m=scale,C_E=CE,C_T=float(rows[:,6].max()),B_E=CE+float(rows[:,6].max()),C_T_separate_dissipations=float(rows[:,7].max()),criterion=.01,physical_work_criterion_passed=bool(CE+rows[:,6].max()<=.01)))
    (OUT/'temporal-work-uncertainty.json').write_text(json.dumps(results,indent=2)+'\n');return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args();results=[summarize(n) for n in a.names]
    save_csv(OUT/'temporal-response-table.csv',['case','max_loaded_step_dphi','first_hold_dphi','max_abs_w_s_inv','endpoint_work_J_m','trapezoidal_work_J_m','energy_change_J_m','Db_J_m','Dc_J_m','friction_J_m','sum_abs_physical_remainders_J_m','C_E_absolute'],[[r['case'],r['maximum_loaded_step_phase_increment'],r['first_hold_increment'],r['max_abs_w_s_inv'],r['endpoint_work_J_m'],r['trapezoidal_work_J_m'],r['energy_change_J_m'],r['physical_dissipation_J_m']['Db_J_m'],r['physical_dissipation_J_m']['Dc_J_m'],r['physical_dissipation_J_m']['friction_J_m'],r['sum_absolute_step_remainders_J_m'],r['C_E_absolute']] for r in results]);print(json.dumps(results,indent=2))
    print(json.dumps(temporal_work_uncertainty(),indent=2))
