"""Read-only comparison with the independent exact-accepted-time K3 reference."""
import json
import argparse
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',choices=('smoke','normal256','spatial0375_n128',
                                    'spatial0375_n256','spatial0375_n512',
                                    'spatial0375_n128_f32','spatial0375_n256_f32',
                                    'spatial0375_n128_f32_periodic',
                                    'spatial0375_n256_f32_periodic',
                                    'spatial0375_n512_f32_periodic',
                                    'spatial0375_n256_f32_periodic_floor',
                                    'spatial0375_n512_f32_periodic_floor'),default='smoke')
parser.add_argument('--expected-steps',type=int,default=2)
parser.add_argument('--reference',type=Path,help='Reuse an independent reference with the exact same accepted sequence.')
args=parser.parse_args()
OUTPUT=HERE/args.case
REFERENCE=args.reference if args.reference is not None else HERE/f'{args.case}-reference'


def read(name, step):
    return np.atleast_1d(np.genfromtxt(OUTPUT/f'{name}_{step}.csv',delimiter=',',names=True))


def main():
    reference=json.loads((REFERENCE/'report.json').read_text())
    assert reference['reference_kind']=='independent continuum initialization'
    completed_steps=reference['completed_real_steps']
    rows=[]
    previous_particles=None
    previous_surface=None
    previous_profile=None
    previous_reference_phi=None
    previous_reference_H=None
    initial_particles=None
    slip=0.
    reference_slip=0.
    for step in range(completed_steps+1):
        guard=json.loads((OUTPUT/f'guard_{step}.json').read_text())
        # Keep a failed support gate visible while still reporting its last
        # mechanically converged exported state. This never changes run gates.
        surface=read('surface',step)
        time=read('time',step)[0]
        phase=read('phase',step)
        particles=np.sort(read('particles',step),order='id')
        if step==0: initial_particles=particles.copy()
        bulk=read('bulk',step)
        history=read('history_transfer',step)
        x=surface['x']
        mean=lambda field: float(np.trapezoid(surface[field],x)/(x[-1]-x[0]))
        ref=reference['initial'] if step==0 else reference['steps'][step-1]
        ref_V=ref['evaluated']['V'] if step==0 else ref['V']
        ys=np.unique(phase['y'])
        phi=[]
        for y in ys:
            line=phase[phase['y']==y]
            _, index=np.unique(line['x'],return_index=True)
            line=np.sort(line[index],order='x')
            phi.append(np.trapezoid(line['phi'],line['x'])/(line['x'][-1]-line['x'][0]))
        phi=np.asarray(phi)
        reference_phi=np.loadtxt(REFERENCE/f'phase-{step}.csv',delimiter=',',skiprows=1)
        ref_at_nodes=np.interp(ys,reference_phi[:,0],reference_phi[:,1])
        dphi=np.zeros_like(phi) if step==0 else phi-previous_profile
        ref_dphi=np.zeros_like(phi) if step==0 else ref_at_nodes-previous_reference_phi
        np.savetxt(OUTPUT/f'comparison_phase_{step}.csv',
                   np.c_[ys,phi,ref_at_nodes,phi-ref_at_nodes,dphi,ref_dphi,dphi-ref_dphi],
                   delimiter=',',header='y,production_mean_phi,independent_phi,difference,increment,reference_increment,increment_error',comments='')
        reference_H=np.loadtxt(REFERENCE/f'history-{step}.csv',delimiter=',',skiprows=1)
        ref_at_particles=np.interp(particles['y'],reference_H[:,0],reference_H[:,1])
        H_rows=[]
        for y0 in np.unique(initial_particles['y']):
            ids=initial_particles['y']==y0
            weights=particles['volume'][ids]
            mean_y=float(np.average(particles['y'][ids],weights=weights))
            mean_H=float(np.average(particles['H'][ids],weights=weights))
            ref_H=float(np.interp(mean_y,reference_H[:,0],reference_H[:,1]))
            increment=0. if step==0 else float(np.average(particles['H'][ids]-previous_particles['H'][ids],weights=weights))
            ref_increment=0. if step==0 else ref_H-float(np.interp(mean_y,previous_reference_H[:,0],previous_reference_H[:,1]))
            H_rows.append((mean_y,y0,mean_H,ref_H,mean_H-ref_H,increment,ref_increment,increment-ref_increment))
        H_rows=np.asarray(H_rows)
        np.savetxt(OUTPUT/f'comparison_H_{step}.csv',H_rows,delimiter=',',
                   header='y,initial_y,production_mean_H,independent_H,difference,increment,reference_increment,increment_error',comments='')
        dt=float(time['dt'])
        kappa=-1e8*np.expm1(-dt/100)
        # Use the actual mechanically constrained old FE stress, not a newly
        # committed particle stress or the potentially distinct published FE field.
        evaluated_q=kappa*(bulk['ux_y']+bulk['uy_x']-bulk['chi']*bulk['V']-bulk['history'])
        evaluated_q+=np.exp(-dt/100)*history['assembly_xy']
        ref_q=ref['evaluated']['q'] if step==0 else ref['q']
        q_error=evaluated_q-ref_q
        weak=read('surface_weak',step)
        mass=np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
        surface_balance=float(np.sqrt(weak['F']@np.linalg.solve(mass,weak['F'])/np.sum(mass)))
        row=dict(step=step,time_s=float(time['time']),dt_s=dt,
                 surface_means={field:mean(field) for field in ('V','C','Theta','Ih')},
                 reference=dict(V=ref_V,C=ref['C'],Theta=200. if step==0 else ref['Theta'],Ih=ref['Ih']),
                 phi_mean_max=float(max(phi)),reference_phi_max=ref['phi_max'],
                 phi_max_absolute_error=float(max(abs(phi-ref_at_nodes))),
                 particle_H_max_absolute_error=float(max(abs(particles['H']-ref_at_particles))),
                 H_profile_max_absolute_error=float(max(abs(H_rows[:,4]))),
                 H_increment_profile_max_error=float(max(abs(H_rows[:,7]))),
                 phi_increment_profile_max_error=float(max(abs(dphi-ref_dphi))),
                 raw_stress_error_rms_Pa=float(np.sqrt(np.average(q_error*q_error,weights=bulk['weight']))),
                 raw_stress_error_max_Pa=float(max(abs(q_error))),
                 evaluated_stress_mean_Pa=float(np.average(evaluated_q,weights=bulk['weight'])),
                 surface_balance_rms_Pa=surface_balance,
                 free_nodes=int(np.count_nonzero(surface['V']>1e-12)),
                 active_nodes=int(np.count_nonzero(surface['V']<=1e-12)),guard=guard)
        if step:
            assert time['time']==ref['time_s'] and time['dt']==ref['dt_s'] and time['U']==ref['U']
            input_H=np.sort(np.atleast_1d(np.genfromtxt(OUTPUT/f'phase_input_{step}_rank0.csv',
                                                     delimiter=',',names=True)),order='id')
            assert np.array_equal(input_H['id'],previous_particles['id'])
            assert np.array_equal(input_H['H'],previous_particles['H'])
            decay=np.exp(-surface['V']*dt/.001)
            expected_theta=previous_surface['Theta']*decay+.001/surface['V']*(-np.expm1(-surface['V']*dt/.001))
            row['theta_update_max_error_s']=float(max(abs(surface['Theta']-expected_theta)))
            row['H_increment_max_Pa']=float(max(particles['H']-previous_particles['H']))
            row['H_increment_min_Pa']=float(min(particles['H']-previous_particles['H']))
            row['mean_phi_increment_max']=float(max(abs(phi-previous_profile)))
            row['reference_H_increment_max_Pa']=ref['H_change_max']
            row['reference_phi_increment_max']=ref['phi_change_max']
            assert row['H_increment_min_Pa']>=0
            assert row['theta_update_max_error_s']<1e-10
            slip+=dt*mean('V')
            reference_slip+=dt*ref_V
        row['accumulated_slip_m']=slip
        row['reference_slip_m']=reference_slip
        rows.append(row)
        previous_particles=particles
        previous_surface=surface
        previous_profile=phi
        previous_reference_phi=ref_at_nodes
        previous_reference_H=reference_H
    log=(HERE/f'{args.case}.log').read_text()
    linear=[]
    for match in re.finditer(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+),',log):
        count,estimated,fresh,target=match.groups()
        linear.append(dict(iterations=int(count),estimated=float(estimated),fresh=float(fresh),target=float(target)))
    assert linear and all(item['fresh']<=item['target'] for item in linear)
    report=dict(primary_reference='independent; exact exported time/dt/U; histories initialized once',
                states=rows,fresh_linear_checks=linear,
                all_fresh_linear_checks_pass=True,
                complete_smoke=(completed_steps==args.expected_steps
                                and all(row['guard']['passed'] for row in rows)))
    (HERE/f'{args.case}-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='fresh_linear_checks'},indent=2))


if __name__=='__main__':
    main()
