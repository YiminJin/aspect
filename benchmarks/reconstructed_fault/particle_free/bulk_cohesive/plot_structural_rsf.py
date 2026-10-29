#!/usr/bin/env python3
"""Physical-unit figures and compact measured summary, no synthetic data."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from structural_rsf import *
from check_structural_rsf import csvwrite


def load(name):
    with (OUT/name).open() as f:return [{k:float(v) if v not in ['spatial','temporal','additional_temporal','matched_spatial','traction_Pa','slip_m','work_J_m2','rate_width_m'] else v for k,v in r.items()} for r in csv.DictReader(f)]

def vals(rows,key):return np.array([r[key] for r in rows])

def finish(fig,name):
    fig.savefig(OUT/name,dpi=160);plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    points=load('point_history.csv');ps=json.loads((OUT/'point_summary.json').read_text())
    normals=ps['formation']['candidates'];pref=np.degrees(np.arctan2(normals[1][1],normals[1][0]))
    a=[r for r in points if r['phi']==.8 and abs(r['preference_deg']-pref)<1e-6 and r['dt_s']==.003125 and r['nonproportional']==0]
    b=[r for r in points if r['phi']==.8 and abs(r['preference_deg']-pref)>1e-6 and r['dt_s']==.003125 and r['nonproportional']==0]
    fig,ax=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
    for n,color,label in zip(normals[::-1],['C0','C1'],['preference near horizontal plane','preference near vertical plane']):
        n=np.array(n);t=np.array([n[1],-n[0],0]);ax[0,0].plot([-t[0],t[0]],[-t[1],t[1]],color=color,lw=3,label=label)
        ax[0,0].arrow(0,0,n[0]*.7,n[1]*.7,color=color,head_width=.06,length_includes_head=True)
    ax[0,0].set(aspect='equal',xlim=(-1.1,1.1),ylim=(-1.1,1.1),xlabel='x (direction only)',ylabel='y (direction only)',title='Tied formation planes, τxy = 1100 Pa');ax[0,0].legend(fontsize=8,loc='lower left')
    for r,label in [(a,'plane near horizontal'),(b,'plane near vertical')]:
        ax[0,1].plot(vals(r,'time_s'),vals(r,'tau_xy_Pa'),label=label)
        ax[0,2].plot(vals(r,'time_s'),vals(r,'tau_xx_Pa'),label=label)
    ax[0,1].set(ylabel='Physical τxy (Pa)',xlabel='Time (s)',title='Same shear response');ax[0,1].legend(fontsize=8)
    ax[0,2].set(ylabel='Physical τxx (Pa)',xlabel='Time (s)',title='Distinct symmetry-broken tensor states')
    for phi in [.2,.8,.99,1.]:
        rows=[r for r in points if r['phi']==phi and abs(r['preference_deg']-pref)<1e-6 and r['dt_s']==.003125 and r['nonproportional']==0]
        ax[1,0].plot(vals(rows,'time_s'),1e6*vals(rows,'signed_V_m_s'),label=f'φ = {phi:g}')
    ax[1,0].set(xlabel='Time (s)',ylabel='Signed local slip speed (µm/s)',title='Reversal on the existing plane');ax[1,0].legend(fontsize=8)
    ax[1,1].plot(vals(a,'time_s'),vals(a,'other_plane_excess_Pa'));ax[1,1].axhline(0,color='.5',lw=.8)
    ax[1,1].set(xlabel='Time (s)',ylabel='Other-plane kinetic excess (Pa)',title='Competing fracture diagnostic; not slip rejection')
    ax[1,2].plot(vals(a,'time_s'),vals(a,'work_J_m3'),label='endpoint work');ax[1,2].plot(vals(a,'time_s'),vals(a,'dissipation_J_m3'),label='physical dissipation')
    ax[1,2].set(xlabel='Time (s)',ylabel='Cumulative energy density (J/m³)',title='No accumulated-K energy');ax[1,2].legend(fontsize=8)
    for axes in [ax[0,1],ax[0,2],*ax[1,:]]:axes.axvline(.15,color='.5',ls=':',lw=.8)
    fig.suptitle('Structural-plane candidate: fixed phase, full tensor memory; shear loading reverses at 0.15 s')
    finish(fig,'point_directions.png')

    history=load('band_history.csv');profiles=load('band_profiles.csv');dt=.000390625
    fig,ax=plt.subplots(2,3,figsize=(13,7.2),layout='constrained');summ=[]
    for cells,color in [(64,'C0'),(128,'C1'),(256,'C2')]:
        rows=[r for r in history if r['cells']==cells and r['dt_s']==dt];profile=[r for r in profiles if r['cells']==cells and r['dt_s']==dt and abs(r['time_s']-.3)<1e-10]
        label=f'{cells} cells'
        ax[0,0].plot(vals(rows,'time_s'),vals(rows,'traction_Pa'),color=color,label=label)
        ax[0,1].plot(1e6*vals(rows,'slip_m'),vals(rows,'traction_Pa'),color=color,label=label)
        ax[0,2].plot(vals(profile,'y_m'),1e6*vals(profile,'flow_rate_s_inv'),color=color,label=label)
        ax[1,0].plot(vals(rows,'time_s'),1000*vals(rows,'rate_width_m'),color=color,label=label)
        ax[1,1].plot(vals(rows,'time_s'),vals(rows,'work_J_m2'),color=color,label=label)
        end=rows[-1];summ.append({k:end[k] for k in ['cells','dt_s','time_s','traction_Pa','slip_m','interface_V_m_s','rate_width_m','work_J_m2','bulk_J_m2','cohesive_J_m2','friction_J_m2','radiation_J_m2','Maxwell_J_m2','BE_J_m2']})
    finest=[r for r in history if r['cells']==256 and r['dt_s']==dt]
    for key,label in [('friction_J_m2','friction'),('radiation_J_m2','radiation'),('Maxwell_J_m2','Maxwell'),('BE_J_m2','BE numerical remainder'),('cohesive_J_m2','cohesive storage')]:
        ax[1,2].plot(vals(finest,'time_s'),vals(finest,key),label=label)
    ax[0,0].set(xlabel='Time (s)',ylabel='Equilibrated shear traction (Pa)',title='Same Δt = 0.000390625 s');ax[0,0].legend()
    ax[0,1].set(xlabel='Integrated inelastic slip (µm)',ylabel='Shear traction (Pa)',title='Slip excludes elastic boundary motion')
    ax[0,2].set(xlabel='Cross-band coordinate y (m)',ylabel='Engineering flow rate (10⁻⁶ s⁻¹)',title='End-of-hold profile at 0.30 s')
    ax[1,0].set(xlabel='Time (s)',ylabel='Rate participation width (mm)',title='No shrinking mesh-width band')
    ax[1,1].set(xlabel='Time (s)',ylabel='Endpoint work (J/m²)',title='Bulk storage dominates this short test')
    ax[1,2].set(xlabel='Time (s)',ylabel='Energy per fault area (J/m²)',title='Separate physical and numerical terms');ax[1,2].set_yscale('symlog',linthresh=1e-9);ax[1,2].legend(fontsize=8)
    for axes in [ax[0,0],*ax[1,:]]:axes.axvline(.25,color='.5',ls=':',lw=.8)
    fig.suptitle('Fresh fixed-phase band: loading to 0.25 s, displacement hold to 0.30 s')
    finish(fig,'band_response.png');csvwrite('band_endpoints.csv',summ)

    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for cells in [64,128,256]:
        p=[r for r in profiles if r['cells']==cells and r['dt_s']==dt and abs(r['time_s']-.3)<1e-10]
        ax[0].plot(vals(p,'y_m'),vals(p,'phi'),label=f'{cells} cells')
        ax[1].plot(vals(p,'y_m'),vals(p,'chi_inv_m'),label=f'{cells} cells')
    for step in [.00078125,.000390625,.0001953125]:
        r=[r for r in history if r['cells']==256 and r['dt_s']==step]
        ax[2].plot(vals(r,'time_s'),1e6*vals(r,'slip_m'),label=f'Δt = {step:g} s')
    ax[0].set(xlabel='y (m)',ylabel='Fixed φ',title='Same freshly sampled continuum seed');ax[0].legend(fontsize=8)
    ax[1].set(xlabel='y (m)',ylabel='Normalization χ (1/m)',title='∫χ dy = 1 on each mesh')
    ax[2].set(xlabel='Time (s)',ylabel='Integrated slip (µm)',title='Temporal uncertainty measured separately');ax[2].legend(fontsize=8)
    finish(fig,'band_resolution.png')
    print(json.dumps(summ,indent=2))

if __name__=='__main__':main()
