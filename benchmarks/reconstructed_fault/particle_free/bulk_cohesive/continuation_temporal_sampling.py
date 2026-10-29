#!/usr/bin/env python3
"""Time sampling of physical phase only, never constitutive-history interpolation."""
import argparse,json
import numpy as np
from continuation_analysis import BASE,OUT,LEVELS,displacement,crossing,save_csv

def norm2(p,dx,dy):
    a,b,c,d=p[:-1,:-1],p[:-1,1:],p[1:,:-1],p[1:,1:]
    return dx*dy/36*np.sum(4*(a*a+b*b+c*c+d*d)+4*(a*b+a*c+b*d+c*d)+2*(a*d+b*c))

def compare(coarse,fine,label):
    with np.load(BASE/(coarse+'-plot-data.npz')) as z:a={k:z[k] for k in ['x','y','phase','times','parent']}
    with np.load(BASE/(fine+'-plot-data.npz')) as z:b={k:z[k] for k in ['x','y','phase','times','parent']}
    if a['times'][0]>.15875+1e-12:
        a['times']=np.r_[.15875,a['times']];a['phase']=np.concatenate([a['parent'][None],a['phase']])
    assert np.array_equal(a['x'],b['x']) and np.array_equal(a['y'],b['y'])
    dx=a['x'][1]-a['x'][0];dy=a['y'][1]-a['y'][0];center=np.argmin(abs(a['y']-.5))
    parent=a['phase'][np.argmin(abs(a['times']-.15875))] if np.min(abs(a['times']-.15875))<1e-10 else a['parent']
    rows=[];fronts=[]
    for k,t in enumerate(b['times']):
        if not .15875+1e-12<t<=.17875+1e-12:continue
        j=np.searchsorted(a['times'],t);j=max(1,min(j,len(a['times'])-1));f=(t-a['times'][j-1])/(a['times'][j]-a['times'][j-1]);assert -1e-10<=f<=1+1e-10
        p=(1-f)*a['phase'][j-1]+f*a['phase'][j];diff=p-b['phase'][k];growth=b['phase'][k]-parent
        rows.append([t,displacement(t),a['times'][j-1],a['times'][j],np.sqrt(norm2(diff,dx,dy)/max(1e-100,norm2(growth,dx,dy))),np.max(abs(diff))/max(1e-100,np.max(abs(growth))),np.max(abs(diff)),np.sqrt(norm2(diff,dx,dy))])
        for level in LEVELS:
            ac=crossing(a['x'],p[center],level);bc=crossing(b['x'],b['phase'][k,center],level);fronts.append([t,displacement(t),level,ac,bc,ac-bc])
    rows=np.asarray(rows);fronts=np.asarray(fronts)
    save_csv(OUT/(label+'-sampling.csv'),['time_s','imposed_relative_displacement_m','coarse_bracket_lower_s','coarse_bracket_upper_s','interval_growth_relative_L2','interval_growth_relative_peak','max_absolute_phase_difference','absolute_Q1_L2_difference_m'],rows)
    save_csv(OUT/(label+'-sampling-fronts.csv'),['time_s','imposed_relative_displacement_m','phi_contour','coarse_linear_phase_front_x_m','fine_native_front_x_m','difference_m'],fronts)
    final_growth=b['phase'][-1]-parent;final_peak=float(np.max(abs(final_growth)));final_norm=float(np.sqrt(norm2(final_growth,dx,dy)))
    result=dict(coarse=coarse,fine=fine,definition='Linear interpolation of saved physical Q1 phase fields between native coarse accepted times; no interpolated constitutive state is evaluated.',max_growth_relative_L2=float(rows[:,4].max()),max_growth_relative_peak=float(rows[:,5].max()),max_phase_difference=float(rows[:,6].max()),max_absolute_difference_time_s=float(rows[rows[:,6].argmax(),0]),max_difference_over_final_interval_peak=float(rows[:,6].max()/final_peak),max_L2_difference_over_final_interval_growth=float(rows[:,7].max()/final_norm),supplementary_normalization='Final-interval normalization is reported for context; the original pointwise-growth failure is retained.',peak_difference_time=float(rows[rows[:,5].argmax(),0]),max_contour_position_difference_m=float(abs(fronts[:,5]).max()),primary_phi_0p1_difference_m=float(abs(fronts[fronts[:,2]==.1,5]).max()),predeclared_primary_tests_passed=bool(rows[:,4].max()<=.05 and rows[:,5].max()<=.05 and abs(fronts[fronts[:,2]==.1,5]).max()<=.015625),peak_target=.05,front_target_m=.015625,passed=bool(rows[:,5].max()<=.05 and abs(fronts[:,5]).max()<=.015625))
    (OUT/(label+'-sampling.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('coarse');p.add_argument('fine');p.add_argument('label');a=p.parse_args();compare(a.coarse,a.fine,a.label)
