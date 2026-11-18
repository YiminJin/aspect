"""Compare accepted original and repaired 80 us trajectories on the step-68 fixture."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from phase import H,basis,coefficients,fronts,read

root=Path(__file__).resolve().parents[4]
old=root/'build-particle-free/bulk-tip/runs/stage4-propagation-pilot-resume51/output'
evidence=root/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
parser.add_argument('--through',type=int,choices=(69,70),default=70)
args=parser.parse_args()
new=args.run/'output'
prior=json.loads((evidence/'guard68/half68-comparison.json').read_text())['comparisons']

def row(folder,name,step):
    table=read(folder/name)
    match=table[table['step']==step]
    assert len(match)==1
    return match[0]

def sections(folder,step):
    return read(folder/f'pilot-sections-{step:04d}.csv')

def reaction(folder,step):
    return np.sort(np.concatenate([read(folder/f'tip-reactions-{step:04d}-r{rank}.csv') for rank in range(4)]),order='dof')

def norm(field):
    q,w=np.polynomial.legendre.leggauss(6)
    b=basis((q+1)/2)
    values=np.einsum('ai,yxij,bj->yxab',b,field,b,optimize=True)
    return H*np.sqrt(np.einsum('yxab,a,b->',values**2,w/2,w/2))

parent=coefficients(old,68)
results=[]
for step in range(69,args.through+1):
    ref_dt=prior[step-69]
    old_b,new_b=[coefficients(folder,step) for folder in (old,new)]
    old_f,new_f=[fronts(b,16) for b in (old_b,new_b)]
    old_s,new_s=[sections(folder,step) for folder in (old,new)]
    assert np.array_equal(old_s['x'],new_s['x'])
    old_p,new_p=[read(folder/f'tip-planes-{step:04d}.csv') for folder in (old,new)]
    e1,e2=[row(folder,'bulk-accepted.csv',step) for folder in (old,new)]
    old_r,new_r=[reaction(folder,step) for folder in (old,new)]
    assert np.array_equal(old_r['dof'],new_r['dof'])
    assert abs(e1['time']-e2['time'])<1e-14 and abs(e2['dt']-8e-5)<1e-14 and e2['publications']==step
    diff={key:float(np.max(abs(new_s[key]-old_s[key]))) for key in ('slip_endpoint','rate_stage','resistance_stage','age_endpoint','angle','ds')}
    slip=abs(new_s['ds']@new_s['slip_endpoint']-old_s['ds']@old_s['slip_endpoint'])
    pos=abs(old_f['right_x']-new_f['right_x'])
    phase=norm(new_b-old_b)
    differences={'phase_l2':phase,'integrated_slip':slip,'right_x':pos,'added_fracture':abs(e2['fracture']-e1['fracture'])}
    time_differences={'phase_l2':ref_dt['phase_l2_error'],'integrated_slip':abs(ref_dt['slip_integral_fine']-ref_dt['slip_integral_coarse']),
                      'right_x':abs(ref_dt['right_position_error']),'added_fracture':abs(ref_dt['fine_energy']['fracture_increment']-ref_dt['coarse_energy']['fracture_increment'])}
    ratios={k:float(differences[k]/v) for k,v in time_differences.items()}
    gates={'physical':bool(e2['phase_KKT']<=1e-7 and e2['mechanical_residual']<=1e-9),
           'same_formed':bool(np.array_equal(old_p['formed'],new_p['formed'])),
           'below_one_tenth_timestep_effect':max(ratios.values())<.1}
    def cost(folder):
        r=row(folder,'tip-profile.csv',step)
        s=row(folder,'tip-solver.csv',step)
        return {k:float(t[k]) for t,keys in [(r,('step_seconds','line_trials','tangent_assemblies')),(s,('nonlinear_iterations','krylov_iterations','active_releases','search_shifts'))] for k in keys}
    results.append(dict(step=step,time=e2['time'],phase_l2_relative_to_increment=phase/norm(old_b-parent),
                        phase_coefficient_max=float(abs(new_b-old_b).max()),section_max_differences=diff,
                        formed_mismatches=int(np.sum(old_p['formed']!=new_p['formed'])),
                        left_position_difference=new_f['left_x']-old_f['left_x'],
                        boundary_reaction_relative_l2=float(np.linalg.norm(new_r['force']-old_r['force'])/np.linalg.norm(old_r['force'])),
                        energy_differences={k:e2[k]-e1[k] for k in ('bulk','cohesive','fracture','Wext','Db','Dc','Df')},
                        differences=differences,ratio_to_80_40_difference=ratios,gates=gates,
                        old_cost=cost(old),new_cost=cost(new)))
destination=evidence/'guard80'/('same-dt.json' if args.through==70 else 'first-interval.json')
record=dict(steps=results,pass_comparison=all(all(r['gates'].values()) for r in results))
destination.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
