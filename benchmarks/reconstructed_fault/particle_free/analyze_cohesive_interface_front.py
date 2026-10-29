#!/usr/bin/env python3
"""Physical front/energy metrics, preserving onset and refinement limitations."""
import json,csv
from pathlib import Path
import numpy as np
from check_cohesive_interface_front import OUT,write,verify_native
import analyze_cohesive_interface_native as prior
CASES=['growth-32','growth-64','growth-64-half']

def table(p):
    names=p.open().readline().strip().split(',');return {k:v for k,v in zip(names,np.loadtxt(p,delimiter=',',skiprows=1,ndmin=2).T)}

def load(name):
    h=table(OUT/name/'history.csv');q=table(OUT/name/'states.csv');nt=len(h['time_s']);ng=len(q['time_s'])//nt;n=ng//3
    fields={k:v.reshape(nt,ng) for k,v in q.items()};Q,w,x,P=prior.surface_rule(n)
    return h,fields,w,x,n

def fronts(a,x,w,level):
    seed=(x>.375)&(x<.625);mask=a>=level;center=np.argmin(abs(x-.5));left=right=center
    while left>0 and mask[left-1]:left-=1
    while right<len(x)-1 and mask[right+1]:right+=1
    lo=.375 if seed[left] else x[left];hi=.625 if seed[right] else x[right]
    if not seed[left]:
        if left==0:lo=0.
        elif level<1:lo=x[left-1]+(level-a[left-1])/(a[left]-a[left-1])*(x[left]-x[left-1])
    if not seed[right]:
        if right==len(x)-1:hi=1.
        elif level<1:hi=x[right]+(level-a[right])/(a[right+1]-a[right])*(x[right+1]-x[right])
    connected=np.zeros(len(x),bool);connected[left:right+1]=True
    return float(lo),float(hi),float(w@(mask&~connected)),left,right

def summarize(name):
    h,q,w,x,n=load(name);initial=((x>.375)&(x<.625)).astype(float);previous=initial.copy();rows=[];oldE=0.;D=0.;W=0.;endpoint=0.;abs_error=0.;signed_error=0.;newF=0.;oldtime=0.;onset=None;full=None;remote=None;all_untied=None;all_full=None
    for k,t in enumerate(h['time_s']):
        a=q['damage'][k];increment=a-initial;levels={v:fronts(a,x,w,v) for v in [.01,.05,.1,.5,.9,1.]}
        newF+=h['fracture_increment_J_m'][k];D+=h['Maxwell_J_m'][k]+h['friction_J_m'][k]+h['radiation_J_m'][k];W+=h['trapezoid_work_J_m'][k];endpoint+=h['endpoint_work_J_m'][k]
        E=h['bulk_J_m'][k]+h['bond_J_m'][k];error=h['trapezoid_work_J_m'][k]-(E-oldE+h['fracture_increment_J_m'][k]+h['Maxwell_J_m'][k]+h['friction_J_m'][k]+h['radiation_J_m'][k]);abs_error+=abs(error);signed_error+=error
        if onset is None and np.any((initial==0)&(a>0)):onset=[oldtime,float(t)]
        if full is None and np.any((initial==0)&(a==1)):full=[oldtime,float(t)]
        if remote is None and levels[.01][2]>0:remote=[oldtime,float(t)]
        if all_untied is None and np.all(a>0):all_untied=[oldtime,float(t)]
        if all_full is None and np.all(a==1):all_full=[oldtime,float(t)]
        row=dict(case=name,time_s=float(t),U_m=.001*min(float(t),.84),delta_t_s=float(h['dt_s'][k]),delta_x_m=1/n,delta_y_m=.125/160,added_fracture_J_m=float(newF),integrated_growth_m=float(w@increment),peak_growth=float(max(increment)),max_step_damage=float(max(a-previous)),tied_length_m=float(w@(a==0)),cohesive_length_m=float(w@((a>0)&(a<1))),decohered_length_m=float(w@(a==1)),new_decohered_length_m=float(w@((initial==0)&(a==1))),mean_jump_m=float(w@q['jump_m'][k]),mean_rate_m_s=float(w@q['V_m_s'][k]),mean_absolute_slip_m=float(w@q['absolute_slip_m'][k]),reaction_Pa=float(w@q['shear_Pa'][k]),bulk_J_m=float(h['bulk_J_m'][k]),bond_J_m=float(h['bond_J_m'][k]),physical_loss_J_m=float(D),trapezoid_work_J_m=float(W),endpoint_work_J_m=float(endpoint),absolute_physical_remainder_J_m=float(abs_error),signed_physical_remainder_J_m=float(signed_error),min_curvature_Pa_m=float(h['min_curvature_Pa_m'][k]),release_curvature_Pa_m=float(h['release_curvature_Pa_m'][k]),weak_certificate_Pa=float(h['weak_certificate_Pa'][k]),certificate_iterations=float(h['certificate_iterations'][k]))
        for v,values in levels.items():
            label=str(v).replace('.','p');row['left_'+label+'_m']=values[0];row['right_'+label+'_m']=values[1];row['disconnected_'+label+'_m']=values[2]
        rows.append(row);oldE=E;oldtime=float(t);previous=a
    assert abs(newF-w@(q['damage'][-1]-initial))<1e-12
    sliding=float(sum(h['friction_J_m']+h['radiation_J_m']))
    result=dict(case=name,n=n,initial_fracture_J_m=float(w@initial),onset_bracket_s=onset,new_decohesion_bracket_s=full,disconnected_001_bracket_s=remote,all_admitted_bracket_s=all_untied,all_decohered_bracket_s=all_full,final_added_fracture_J_m=newF,sliding_loss_J_m=sliding,absolute_physical_remainder_J_m=abs_error,physical_remainder_over_sliding=abs_error/sliding,physical_remainder_over_fracture=abs_error/newF,max_step_damage=max(r['max_step_damage'] for r in rows),max_disconnected_001_m=max(r['disconnected_0p01_m'] for r in rows),minimum_curvature_Pa_m=min(h['min_curvature_Pa_m']),minimum_release_curvature_Pa_m=min(h['release_curvature_Pa_m']),max_certificate_iterations=max(h['certificate_iterations']),min_side_pressure_Pa=min(h['min_side_contact_pressure_Pa']),max_ledger_error_J_m=max(abs(h['ledger_error_J_m'])),final_mean_jump_m=rows[-1]['mean_jump_m'],max_reaction_Pa=max(r['reaction_Pa'] for r in rows))
    return rows,result

def comparisons(one,two,kind):
    h1,q1,w1,x1,n1=load(one);h2,q2,w2,x2,n2=load(two);grid=(np.arange(4096)+.5)/4096;rows=[]
    a01=((x1>.375)&(x1<.625));a02=((x2>.375)&(x2<.625))
    for t in [.22,.24,.3,.36,.4,.6,.8,.82,.822,.824,.826,.828,.83,.84,.86]:
        i=int(np.argmin(abs(h1['time_s']-t)));j=int(np.argmin(abs(h2['time_s']-t)));assert abs(h1['time_s'][i]-t)<1e-12 and abs(h2['time_s'][j]-t)<1e-12
        da=q1['damage'][i]-a01;db=q2['damage'][j]-a02
        # Same physical box kernel (1/32 m), including identical seed boundaries.
        # Do not interpolate across the pre-existing damage discontinuities.
        bins1=np.minimum(31,(32*x1).astype(int));bins2=np.minimum(31,(32*x2).astype(int))
        aa=np.bincount(bins1,weights=w1*da,minlength=32)*32;bb=np.bincount(bins2,weights=w2*db,minlength=32)*32
        norm=np.linalg.norm(bb);diff=np.linalg.norm(aa-bb)
        area1=w1@da;area2=w2@db;peak1=max(da);peak2=max(db)
        row=dict(comparison=kind,time_s=t,growth_L2_relative=float(diff/norm) if norm>0 else None,growth_L2_absolute=float(diff/np.sqrt(32)),integrated_growth_relative=float(abs(area1-area2)/area2) if area2>0 else None,peak_growth_relative=float(abs(peak1-peak2)/peak2) if peak2>0 else None,mean_jump_relative=float(abs(w1@q1['jump_m'][i]-w2@q2['jump_m'][j])/max(abs(w2@q2['jump_m'][j]),1e-30)),new_decohered_length_difference_m=float(abs(w1@((q1['damage'][i]==1)&~a01)-w2@((q2['damage'][j]==1)&~a02))),reaction_relative=float(abs(w1@q1['shear_Pa'][i]-w2@q2['shear_Pa'][j])/max(abs(w2@q2['shear_Pa'][j]),1e-30)))
        for level in [.05,.5,1.]:
            f1=fronts(q1['damage'][i],x1,w1,level);f2=fronts(q2['damage'][j],x2,w2,level);label=str(level).replace('.','p');row['front_'+label+'_difference_m']=max(abs(f1[0]-f2[0]),abs(f1[1]-f2[1]))
        rows.append(row)
    return rows

def main():
    histories=[];summaries=[];native=[];profiles=[]
    for name in CASES:
        rows,s=summarize(name);histories+=rows;summaries.append(s);native.append(verify_native(name,False))
        h,q,w,x,n=load(name)
        for t in [0.,.2,.224,.24,.3,.36,.4,.6,.8,.82,.822,.824,.825,.826,.828,.83,.84,.86]:
            if t==0:
                for i,xx in enumerate(x):profiles.append(dict(case=name,time_s=0.,x_m=float(xx),weight_m=float(w[i]),damage=float(.375<xx<.625),damage_increment=0.,jump_m=0.,rate_m_s=0.,shear_Pa=0.,cohesive_Pa=0.,normal_pressure_Pa=10000.,tied=int(not .375<xx<.625)))
                continue
            k=int(np.argmin(abs(h['time_s']-t)))
            if abs(h['time_s'][k]-t)>1e-12:continue
            for i,xx in enumerate(x):profiles.append(dict(case=name,time_s=float(h['time_s'][k]),x_m=float(xx),weight_m=float(w[i]),damage=float(q['damage'][k,i]),damage_increment=float(q['damage'][k,i]-(.375<xx<.625)),jump_m=float(q['jump_m'][k,i]),rate_m_s=float(q['V_m_s'][k,i]),shear_Pa=float(q['shear_Pa'][k,i]),cohesive_Pa=float(q['q_Pa'][k,i]),normal_pressure_Pa=float(q['pn_Pa'][k,i]),tied=int(q['tied'][k,i])))
    compare=comparisons('growth-32','growth-64','spatial')+comparisons('growth-64','growth-64-half','temporal')
    write('physical_histories.csv',histories);write('comparison.csv',compare);write('profiles.csv',profiles)
    result=dict(cases=summaries,native_checks=native,comparisons=compare)
    (OUT/'growth_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
