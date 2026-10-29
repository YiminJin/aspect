#!/usr/bin/env python3
"""Indexed physical front, buffer, transfer and energy evidence."""
import csv,json
from pathlib import Path
import numpy as np
from run_cohesive_interface_insertion import OUT,WORK

def read(p):return np.genfromtxt(p,delimiter=',',names=True,ndmin=1)
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(case):
    folder=OUT/case
    if not (folder/'states.csv').exists():folder=WORK/'runs'/case/'output'
    h=read(folder/'history.csv');data=read(folder/'states.csv');nq=len(data)//len(h)
    assert len(data)==len(h)*nq
    q={k:data[k].reshape(len(h),nq) for k in data.dtype.names};return h,q

def connected(a,x,lo,hi,level):
    mask=a>=level;seed=(x>lo)&(x<hi);left=np.flatnonzero(seed)[0];right=np.flatnonzero(seed)[-1]
    while left>0 and mask[left-1]:left-=1
    while right<len(x)-1 and mask[right+1]:right+=1
    linked=np.zeros(len(x),bool);linked[left:right+1]=True
    return (lo if seed[left] else float(x[left]),hi if seed[right] else float(x[right]),mask&~linked)

def summarize(case):
    h,q=load(case);x=q['x_m'][0];n=len(x)//3;w=np.tile(np.polynomial.legendre.leggauss(3)[1]/2*8/n,n);seed=(x>2.5)&(x<5.5)
    nodal=read(OUT/case/'nodal_history.csv');nodal_jump=nodal['jump_m'].reshape(len(h),n)
    nodal_rate=nodal['signed_rate_m_s'].reshape(len(h),n);nx=np.arange(n)*8/n
    initial=seed.astype(float);previous=initial.copy();prevtime=0.;onset=full=remote=None;rows=[];E0=0.;W=loss=sliding=fracture=absolute=signed=0.
    for k,t in enumerate(h['time_s']):
        a=q['damage'][k];added=a-initial
        E=h['bulk_J_m'][k]+h['bond_J_m'][k];diss=sum(h[key][k] for key in ['Maxwell_J_m','friction_J_m','radiation_J_m']);sliding+=h['friction_J_m'][k]+h['radiation_J_m'][k];loss+=diss;W+=h['trapezoid_work_J_m'][k];fracture+=h['fracture_increment_J_m'][k]
        error=h['trapezoid_work_J_m'][k]-(E-E0+diss+h['fracture_increment_J_m'][k]);absolute+=abs(error);signed+=error
        if onset is None and np.any((a>0)&~seed):onset=[prevtime,float(t)]
        if full is None and np.any((a==1)&~seed):full=[prevtime,float(t)]
        row=dict(case=case,time_s=float(t),U_m=.001*min(float(t),2.7),delta_t_s=float(h['dt_s'][k]),delta_x_m=8/n,delta_y_m=1/(40 if n==64 else 80),added_fracture_J_m=float(fracture),integrated_growth_m=float(w@added),peak_added_damage=float(max(added)),max_step_damage=float(max(a-previous)),integrated_jump_m2=float(w@q['jump_m'][k]),reaction_N_m=float(w@q['shear_Pa'][k]),bulk_J_m=float(h['bulk_J_m'][k]),bond_J_m=float(h['bond_J_m'][k]),sliding_loss_J_m=float(sliding),physical_loss_J_m=float(loss),trapezoid_work_J_m=float(W),absolute_physical_remainder_J_m=float(absolute),signed_physical_remainder_J_m=float(signed),step_physical_remainder_J_m=float(error),ledger_error_J_m=float(h['ledger_error_J_m'][k]),min_curvature_Pa_m=float(h['min_curvature_Pa_m'][k]),cohesive_length_m=float(w@((a>0)&(a<1))),resolved_cohesive_length_m=float(w@((a>=.01)&(a<1))),new_decohered_length_m=float(w@((a==1)&~seed)),installed_tied_length_m=float(w@((a==0)&(q['installed'][k]==1))),remote_peak_shear_Pa=float(max(q['shear_Pa'][k,(x<1)|(x>7)])))
        for level in [.01,.05,.5,1.]:
            l,r,detached=connected(a,x,2.5,5.5,level);tag=str(level).replace('.','p');row['left_'+tag+'_m']=l;row['right_'+tag+'_m']=r;row['detached_'+tag+'_m']=float(w@detached)
            if level==.01 and remote is None and np.any(detached):remote=[prevtime,float(t)]
        z=nodal_jump[k];threshold=2/3000;left=int(2.5*n/8);right=int(5.5*n/8)
        if z[left]>=threshold:
            while left>0 and z[left-1]>=threshold:left-=1
            root_left=nx[left] if left==0 else nx[left-1]+(threshold-z[left-1])/(z[left]-z[left-1])*8/n
        else:root_left=2.5
        if z[right]>=threshold:
            while right<n-1 and z[right+1]>=threshold:right+=1
            root_right=nx[right] if right==n-1 else nx[right]+(threshold-z[right])/(z[right+1]-z[right])*8/n
        else:root_right=5.5
        row['left_jump_threshold_m']=float(root_left);row['right_jump_threshold_m']=float(root_right)
        exact_fracture=exact_bond=exact_full=0.
        for cell in range(n):
            if 2.5<=nx[cell]<5.5:continue
            v0=z[cell];dv=z[(cell+1)%n]-v0;cuts=[0.,1.]
            if dv!=0:
                cuts += [u for u in [(0-v0)/dv,(threshold-v0)/dv] if 0<u<1]
            cuts=sorted(cuts)
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                value=v0+dv*(lo+hi)/2
                if value>=threshold:exact_fracture+=(hi-lo)*8/n;exact_full+=(hi-lo)*8/n
                elif value>0:
                    i1=v0*(hi-lo)+dv*(hi*hi-lo*lo)/2
                    i2=v0*v0*(hi-lo)+v0*dv*(hi*hi-lo*lo)+dv*dv*(hi**3-lo**3)/3
                    exact_fracture+=i1/threshold*8/n;exact_bond+=1500*(i1-i2/threshold)*8/n
        row['continuous_new_fracture_J_m']=float(exact_fracture);row['continuous_bond_J_m']=float(exact_bond);row['continuous_new_decohered_length_m']=float(exact_full)
        row['guaranteed_new_full_cell_length_m']=float(sum((np.minimum(z,np.roll(z,-1))>=threshold)&((nx<2.5)|(nx>=5.5)))*8/n)
        rows.append(row);previous=a;prevtime=float(t);E0=E
    assert abs(fracture-w@(q['damage'][-1]-initial))<1e-10
    # Independently reevaluate scalar physical history evolution at every point.
    olda=np.r_[initial[None,:],q['damage'][:-1]];oldjump=np.r_[np.zeros((1,len(x))),q['jump_m'][:-1]];oldage=np.r_[np.ones((1,len(x))),q['Theta'][:-1]];dt=h['dt_s'][:,None]
    age=(oldage+5e-6/.001*dt)/(1+abs(q['jump_m']-oldjump)/.001)
    damage=np.maximum(olda,np.minimum(abs(q['jump_m']-q['reference_m'])/(2/3000),1.))
    ae=float(np.max(abs(age-q['Theta'])));de=float(np.max(abs(damage-q['damage'])));assert ae<2e-14 and de<2e-14
    summary=dict(min_initially_intact_nodal_rate_m_s=float(np.min(nodal_rate[:,(nx<2.5)|(nx>5.5)])),case=case,steps=len(h),final_time_s=float(h['time_s'][-1]),onset_bracket_s=onset,new_decohesion_bracket_s=full,disconnected_bracket_s=remote,final=rows[-1],max_step_damage=max(r['max_step_damage'] for r in rows),max_ledger_error_J_m=float(max(abs(h['ledger_error_J_m']))),min_contact_Pa=float(min(h['min_side_contact_pressure_Pa'])),min_curvature_Pa_m=float(min(h['min_curvature_Pa_m'])),max_weak_error_Pa=float(max(h['weak_certificate_Pa'])),age_error=ae,damage_error=de,work_error_over_sliding=absolute/sliding,work_error_over_fracture=absolute/fracture if fracture else None)
    return rows,summary,h,q,w

def main():
    histories=[];summaries=[];snapshots=[];animation=[];loaded={}
    for case in ['growth-64','growth-128','growth-128-half']:
        if not (OUT/case/'history.csv').exists():continue
        rows,s,h,q,w=summarize(case);histories+=rows;summaries.append(s);loaded[case]=(h,q,w)
        indices=sorted(set([0,len(h)-1]+[int(np.argmin(abs(h['time_s']-t))) for t in [1,1.5,1.8,2.,2.2,2.3,2.38,2.4,2.42]]))
        for k in indices:
            for j,x in enumerate(q['x_m'][k]):snapshots.append(dict(case=case,time_s=float(h['time_s'][k]),x_m=float(x),weight_m=float(w[j]),damage=float(q['damage'][k,j]),initial_damage=float(2.5<x<5.5),jump_m=float(q['jump_m'][k,j]),V_m_s=float(q['V_m_s'][k,j]),shear_Pa=float(q['shear_Pa'][k,j]),normal_pressure_Pa=float(q['pn_Pa'][k,j]),cohesive_Pa=float(q['q_Pa'][k,j]),Theta=float(q['Theta'][k,j]),installed=int(q['installed'][k,j])))
        if case=='growth-128-half':
            for k in sorted(set(np.linspace(0,len(h)-1,100).astype(int))):
                for j,x in enumerate(q['x_m'][k]):animation.append(dict(time_s=float(h['time_s'][k]),U_m=.001*min(float(h['time_s'][k]),2.7),x_m=float(x),damage=float(q['damage'][k,j]),jump_m=float(q['jump_m'][k,j]),shear_Pa=float(q['shear_Pa'][k,j])))
    comparisons=[]
    for c1,c2,kind in [('growth-64','growth-128','spatial'),('growth-128','growth-128-half','temporal')]:
        if c1 not in loaded or c2 not in loaded:continue
        h1,q1,w1=loaded[c1];h2,q2,w2=loaded[c2]
        for t in [1.5,1.8,2,2.2,2.3,2.38,2.4,2.42]:
            i=np.argmin(abs(h1['time_s']-t));j=np.argmin(abs(h2['time_s']-t))
            if abs(h1['time_s'][i]-t)>1e-10 or abs(h2['time_s'][j]-t)>1e-10:continue
            x1=q1['x_m'][0];x2=q2['x_m'][0];a=q1['damage'][i]-((x1>2.5)&(x1<5.5));b=q2['damage'][j]-((x2>2.5)&(x2<5.5))
            aa=np.bincount((x1/.125).astype(int),weights=w1*a,minlength=64)/.125;bb=np.bincount((x2/.125).astype(int),weights=w2*b,minlength=64)/.125
            row=dict(comparison=kind,time_s=t,growth_L2_absolute=float(np.linalg.norm(aa-bb)*np.sqrt(.125)),growth_L2_relative=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb)) if np.linalg.norm(bb)>0 else None,integrated_growth_relative=float(abs(w1@a-w2@b)/(w2@b)) if w2@b>0 else None,peak_growth_relative=float(abs(max(a)-max(b))/max(b)) if max(b)>0 else None,integrated_slip_relative=float(abs(w1@q1['jump_m'][i]-w2@q2['jump_m'][j])/(w2@q2['jump_m'][j])),reaction_relative=float(abs(w1@q1['shear_Pa'][i]-w2@q2['shear_Pa'][j])/(w2@q2['shear_Pa'][j])))
            for level in [.05,1.]:
                f1=connected(q1['damage'][i],x1,2.5,5.5,level);f2=connected(q2['damage'][j],x2,2.5,5.5,level);row['front_'+str(level).replace('.','p')+'_difference_m']=max(abs(f1[0]-f2[0]),abs(f1[1]-f2[1]))
            r1=min((r for r in histories if r['case']==c1),key=lambda r:abs(r['time_s']-t));r2=min((r for r in histories if r['case']==c2),key=lambda r:abs(r['time_s']-t))
            row['jump_threshold_front_difference_m']=max(abs(r1['left_jump_threshold_m']-r2['left_jump_threshold_m']),abs(r1['right_jump_threshold_m']-r2['right_jump_threshold_m']))
            comparisons.append(row)
    table_rows=[]
    for case in loaded:
        row=min((r for r in histories if r['case']==case),key=lambda r:abs(r['time_s']-2.42))
        keys=['case','time_s','delta_x_m','delta_y_m','integrated_growth_m','integrated_jump_m2','left_0p05_m','left_jump_threshold_m','right_jump_threshold_m','continuous_new_decohered_length_m','guaranteed_new_full_cell_length_m','bulk_J_m','bond_J_m','added_fracture_J_m','sliding_loss_J_m','absolute_physical_remainder_J_m']
        table_rows.append({key:row[key] for key in keys})
    write('comparison_table.csv',table_rows)
    write('physical_histories.csv',histories);write('profiles.csv',snapshots);write('animation_frames.csv',animation);write('comparison.csv',comparisons)
    (OUT/'growth_summary.json').write_text(json.dumps(dict(cases=summaries,comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
