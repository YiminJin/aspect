#!/usr/bin/env python3
"""Full incompressible branch symbols; no fixed-pressure surrogate or native run."""
import csv,json,sys
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_point import ROOT,OUT,point,initial
from coulomb_localization_analysis import rate_columns,symbol

def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def tau_columns(J):
    return np.stack([J[11],J[12],-J[11]-J[12],J[13]])/1e6
theta=np.linspace(0,np.pi,1441)
records=[];selected={};poles=[];minimum_high=2.
for phi in [.2,.6,.75,.8,.9,.95,.99]:
    old=initial(phi)
    for step in range(1,25):
        de=np.array([0,0,.02*.0125]) if step<=8 else (np.array([.02*.0125,-.02*.0125,0]) if step<=20 else np.zeros(3))
        r,J=point(old,de,.0125)
        if step in [8,9,16,24]:
            h=4*phi*(1+phi)/(1-phi)**2;C=tau_columns(J);U=.0125*J[14:16]
            eig=np.linalg.eigvals(U[:,4:6])
            for v in eig:
                if abs(v.imag)<1e-12 and v.real>0:poles.append({'phi':phi,'step':step,'k_times_ls':float(np.sqrt(1/(2*v.real)))})
            row={'phi':phi,'h':h,'mu':.1,'active':int(r[23]),'theta_plus':r[24],'theta_minus':r[25]}
            for k in np.r_[0.,np.logspace(-3,3,61)]:
                A=np.eye(2)-2*k*k*U[:,4:6]
                V=np.linalg.solve(A,U[:,:4]);D=C[:,:4]+C[:,4:6]@(2*k*k*V)
                full=np.zeros((5,5));full[:4,:4]=D
                rate=rate_columns({**row,'h':1/(1/h+k*k)})
                value=float(min(symbol(full,theta)[0]));reference=float(min(symbol(rate,theta)[0]))
                records.append({'phi':phi,'step':step,'active':int(r[23]),'k_times_ls':k,'algorithmic_mixed_min_over_G':value,
                    'continuous_rate_mixed_min_over_G':reference,'coupling_sigma_min':float(np.linalg.svd(A,compute_uv=False)[-1])})
                if k==1000.:minimum_high=min(minimum_high,value)
            if phi in [.8,.99] and step in [8,9]:selected[(phi,step)]=[v for v in records if v['phi']==phi and v['step']==step]
        old=r[:11].copy()
assert not poles,poles
assert minimum_high>.999,minimum_high
write('mixed-wavelength-symbols.csv',records)
hc=100.49813740694826;lengths=[]
for phi in [.75,.8,.9,.95,.99]:
    h=4*phi*(1+phi)/(1-phi)**2
    for ls in [.001,.003,.006]:
        critical=np.sqrt(1/hc-1/h) if h>hc else None
        lengths.append({'phi':phi,'h':h,'ell_s_m':ls,'k_c_per_m':critical/ls if critical else None,
            'single_system_cutoff_wavelength_m':2*np.pi*ls/critical if critical else None,
            'status':'Uncalibrated illustrative input; no identification with phase length or mesh.'})
write('exploratory-lengths.csv',lengths)
(OUT/'symbol-checks.json').write_text(json.dumps({'homogeneous_states':28,'phase_range':[.2,.6,.75,.8,.9,.95,.99],
 'minimum_high_k_algorithmic_mixed_coefficient_over_G':minimum_high,'sampled_high_k_times_ls':1000.,
 'positive_eigenvalue_point_elimination_poles':poles,'native_or_global_branch_qualification':False,
 'scope':'Strict homogeneous X=0 point branches, complete changing-direction pressure/X derivatives; not across the saved nonunique prescribed-X state.'},indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for (phi,step),data in selected.items():
    k=np.array([v['k_times_ls'] for v in data[1:]])
    ax[0].semilogx(k,[v['algorithmic_mixed_min_over_G'] for v in data[1:]],label=fr'$\phi={phi}$, '+('two systems' if step==8 else 'changing to single slip'))
ax[0].axhline(0,color='k',lw=.8);ax[0].set(xlabel=r'$k\ell_s$',ylabel=r'Minimum mixed $\Lambda/G$',title='Strict homogeneous branches: high-k approaches elasticity')
ax[0].legend(fontsize=8)
for phi in [.8,.9,.99]:
    data=[v for v in lengths if v['phi']==phi]
    ax[1].plot([1000*v['ell_s_m'] for v in data],[v['single_system_cutoff_wavelength_m'] for v in data],'o-',label=fr'$\phi={phi}$')
ax[1].set(xlabel=r'Illustrative slip length $\ell_s$ (mm)',ylabel=r'Single-system cutoff wavelength $\lambda_c$ (m)',title='Shorter wavelengths avoid the rate-level defect')
ax[1].legend();ax[1].text(.08,.9,'Uncalibrated lengths;\nnot the phase length or cell size.',transform=ax[1].transAxes,fontsize=9)
fig.savefig(OUT/'wavelengths.png',dpi=170);plt.close(fig)

roots=json.loads((OUT/'independent-ambiguous-roots.json').read_text())['roots'];table=[]
for r in roots:
    h=4*.99*1.99/.01**2;beta=1/(1+.0125/100);gamma=np.array(list(map(float,r['gamma_Pa'])));wi=h*gamma/(1e6*beta*.0125)
    table.append({'active':r['active'],'tauxx_Pa':float(r['tauxx_Pa']),'tauxy_Pa':float(r['tauxy_Pa']),
        'Qxx_Pa':float(r['Qxx_Pa']),'Qxy_Pa':float(r['Qxy_Pa']),'plus_system_speed_s-1':wi[0],'minus_system_speed_s-1':wi[1],
        'minimum_normal_pressure_Pa':float(r['min_normal_pressure_Pa']),'independent_equation_residual':float(r['residual'])})
write('ambiguous-point-roots.csv',table)
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained');labels=['Single +','Single −','Both']
for r,label in zip(table,labels):
    ax[0].plot(r['tauxx_Pa'],r['tauxy_Pa'],'o',ms=8,label=label)
    ax[0].annotate(label,(r['tauxx_Pa'],r['tauxy_Pa']),xytext=(4,-17 if r['active']==3 else 7),textcoords='offset points')
x=np.arange(3);ax[1].bar(x-.16,[r['plus_system_speed_s-1'] for r in table],width=.32,label='Plus system')
ax[1].bar(x+.16,[r['minus_system_speed_s-1'] for r in table],width=.32,label='Minus system')
ax[1].set_xticks(x,labels);ax[1].legend(fontsize=9)
ax[0].set(xlabel=r'Physical $\tau_{xx}$ (Pa)',ylabel=r'Physical $\tau_{xy}$ (Pa)',xlim=(-24,30),title='Same incoming state and increment')
ax[1].set(ylabel=r'System shear rate (s$^{-1}$)',title='Three admissible active-system choices')
fig.suptitle(r'$\phi=.99$, prescribed $X_{xy}=-.1$ Pa; 70-digit independent equations')
fig.savefig(OUT/'point-ambiguity.png',dpi=170);plt.close(fig)
print((OUT/'symbol-checks.json').read_text())
