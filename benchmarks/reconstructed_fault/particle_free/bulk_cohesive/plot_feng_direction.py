#!/usr/bin/env python3
"""Figures use exported checks and explicitly labeled analytic direction fields."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from feng_direction import *


def read(name):return list(csv.DictReader((OUT/name).open()))
def column(rows,key):return np.array([float(r[key]) for r in rows])
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


plt.rcParams.update({'font.size':10,'axes.titlesize':11})
tau=tensor(xy=MU*P0/np.sqrt(1+MU**2));zero=np.zeros((3,3));history=np.array([0.,1.,0.])
result=directions(tau,zero,P0);chosen,reason=select(result,history)
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
ax=axes[0,0]
for i,c in enumerate(result['candidates']):
    n=c['normal'][:2];t=J(n);selected=np.linalg.norm(c['N']-chosen['N'])<1e-12
    ax.plot([-t[0],t[0]],[-t[1],t[1]],color=f'C{i}',lw=3 if selected else 1.5,
            label=('Selected' if selected else 'Equal candidate')+f' plane, normal {np.rad2deg(c["theta"]):.2f}°')
    ax.arrow(0,0,.7*n[0],.7*n[1],color=f'C{i}',head_width=.05,length_includes_head=True)
ax.plot([-1,1],[0,0],'k--',lw=1,label='Supplied orientation preference')
ax.axhline(0,color='.7',lw=.5);ax.axvline(0,color='.7',lw=.5)
ax.set(aspect='equal',xlim=(-1.1,1.1),ylim=(-1.1,1.1),xlabel='x direction',ylabel='y direction',
       title='Mature pure shear: genuine conjugate tie')
ax.legend(fontsize=8,loc='lower left')
ax.text(.03,.95,'Arrows: plane normals\nLines: candidate slip planes\nNo phase-front geometry is calculated',transform=ax.transAxes,va='top',fontsize=8)

angles=np.linspace(0,90,721);curve=[]
for degree in angles:
    theta=np.deg2rad(degree);f,pa,pb,shear=objective(tau,zero,P0,theta)
    curve.append(dict(normal_angle_deg=degree,F_Pa=f,palpha_Pa=pa,pbeta_Pa=pb))
write('direction-objective.csv',curve)
axes[0,1].plot(angles,[r['F_Pa'] for r in curve],color='k')
for i,c in enumerate(result['candidates']):
    axes[0,1].plot(np.rad2deg(c['theta']),c['maximum'],'o',color=f'C{i}',ms=8)
axes[0,1].axhline(0,color='.5',ls=':')
axes[0,1].set(xlabel='Normal angle from +x [degrees]',ylabel='F(N) [Pa]',title='Use global maximizers of the actual direction rule')
axes[0,1].text(.5,.88,'Both peaks satisfy F = 0\nMixture direction has F = −4.9628 Pa',transform=axes[0,1].transAxes,ha='center')

spins=[]
for extra in np.linspace(-.02,.03,101):
    L=np.array([[0.,.02-extra],[extra,0.]])
    n,omega=legacy_normal(tau,L)
    spins.append(dict(added_spin_s1=extra,total_spin_s1=omega,normal_deg=np.rad2deg(np.arctan2(n[1],n[0])%np.pi),
                      continuity_normal_deg=np.rad2deg(chosen['theta'])))
write('observer-spin-curve.csv',spins)
axes[1,0].plot(column(spins,'added_spin_s1'),column(spins,'normal_deg'),label='Intended old spin selector')
axes[1,0].plot(column(spins,'added_spin_s1'),column(spins,'continuity_normal_deg'),'--',label='Maximizer + orientation continuity')
axes[1,0].axvline(.01,color='.6',ls=':')
axes[1,0].set(xlabel='Superposed observer spin [s⁻¹]',ylabel='Selected normal [degrees]',title='Same instantaneous stress and symmetric strain rate')
axes[1,0].legend(fontsize=8)
path=read('continuity-path.csv')
for i in [0,1]:
    rows=[r for r in path if int(r['candidate'])==i]
    axes[1,1].plot(column(rows,'stress_rotation_deg'),column(rows,'normal_deg'),color=f'C{i}',lw=1,label=f'Candidate {i+1}')
rows=[r for r in path if r['selected']=='True']
axes[1,1].scatter(column(rows,'stress_rotation_deg'),column(rows,'normal_deg'),s=12,color='k',label='Selected by previous axis')
axes[1,1].set(xlabel='Prescribed stress-axis rotation [degrees]',ylabel='Plane normal [degrees]',title='Continuity follows changing maxima; no plane lock')
axes[1,1].legend(fontsize=8)
fig.suptitle('Feng actual-increment direction and dissipation reference\nNormals, signed tensor flow and phase geometry are distinct; symmetry without a preference stays unresolved')
fig.savefig(OUT/'candidate-directions.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
angles=read('angular-symbol.csv');rows=[r for r in angles if 160<=float(r['wave_normal_deg'])<=174]
for key,label,style in [('algorithmic_mixed_over_G','Complete mixed finite-step tangent','-'),
                        ('algorithmic_fixed_pressure_over_G','Pressure held fixed (insufficient)','--'),
                        ('rate_mixed_over_G','Mixed continuous-rate limit',':')]:
    axes[0,0].plot(column(rows,'wave_normal_deg'),column(rows,key),style,label=label)
axes[0,0].axhline(0,color='k',lw=.7)
axes[0,0].set(xlabel='Perturbation wave normal [degrees]',ylabel='Transverse coefficient / G',title='φ = 0.8: the single-direction mechanical defect remains')
axes[0,0].legend(fontsize=8)
phase=[r for r in read('phase-range-symbol.csv') if float(r['slip_sign'])==1 and float(r['phi'])>=.6]
axes[0,1].plot(column(phase,'phi'),column(phase,'mixed_min_over_G'),'o-',label='Mixed rate limit')
axes[0,1].plot(column(phase,'phi'),column(phase,'fixed_pressure_min_over_G'),'s--',label='Fixed pressure')
critical=json.loads((OUT/'mechanical-checks.json').read_text())['critical_phi_m4']
axes[0,1].axvline(critical,color='.5',ls=':',label=f'φcrit = {critical:.5f}');axes[0,1].axhline(0,color='k',lw=.7)
axes[0,1].set(xlabel='Fixed phase φ',ylabel='Minimum coefficient / G',title='Both symmetry branches give the same threshold')
axes[0,1].legend(fontsize=8)
dt=[r for r in read('point-symbols.csv') if r['case'].startswith('forward-')]
dt=sorted(dt,key=lambda r:float(r['dt_s']))
axes[1,0].semilogx(column(dt,'dt_s'),column(dt,'mixed_min_over_G'),'o-',label='Changing-direction finite-step return')
axes[1,0].axhline(float(dt[0]['rate_limit_min_over_G']),color='.5',ls='--',label='Negative rate limit')
axes[1,0].set(xlabel='Δt [s]',ylabel='Minimum mixed coefficient / G',title='Identical incoming state; smaller Δt does not cure it')
axes[1,0].legend(fontsize=8)
powers=read('mature-power.csv');x=np.arange(3)
axes[1,1].bar(x-.18,column(powers,'physical_work_W_m3'),.35,label='Conjugate physical work')
axes[1,1].bar(x+.18,column(powers,'Feng_friction_W_m3'),.35,label='Feng actual-direction dissipation')
axes[1,1].set(xticks=x,xticklabels=['Single plane 1','Single plane 2','Two-system mixture'],ylim=(.0985,.1003),
              ylabel='Power density [W/m³]',title='Same equivalent strain rate 10⁻⁴ s⁻¹; expanded scale')
axes[1,1].legend(fontsize=8)
fig.suptitle('Removing conjugate redistribution is necessary, but does not establish well-posedness\nFixed phase, μ = 0.1, m = 4; full plane-strain tensor and solved pressure perturbation retained')
fig.savefig(OUT/'mechanical-comparison.png',dpi=170);plt.close(fig)
