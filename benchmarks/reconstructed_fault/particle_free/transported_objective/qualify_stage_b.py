#!/usr/bin/env python3
"""Consolidate Stage B results and select the smallest tested native candidate.

No native configuration is enabled. P1 figures are a best-projection screen,
not a transported P1 solve. All initial failed candidates remain in evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import moving_model as m

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve()
    result=json.loads((out/'results.json').read_text());screen=json.loads((out/'space_screen.json').read_text())
    surface=json.loads((out/'surface_screen.json').read_text())
    fixed_path=DEFAULT.parent/'stage_b_fixed/results.json';fixed=json.loads(fixed_path.read_text())
    passed=lambda r:all(r[k]<1e-3 for k in ('tensor_scaled','normal_scaled','shear_scaled','energy_relative_error'))
    candidate=[]
    for kind in ('direct','separated'):
        r=next(r for r in result['spatial'] if r['kind']==kind and passed(r))
        candidate.append(dict(representation=kind,n=r['n'],tensor_scaled=r['tensor_scaled'],
                              scalar_tensor_coefficients=(27 if kind=='direct' else 54)*r['n']**2,
                              independent_storage_bytes=(27 if kind=='direct' else 54)*r['n']**2*8))
    p1=next(r for r in screen['rows'] if r['n']==8 and r['residual_degree']==1)
    p2=next(r for r in screen['rows'] if r['n']==8 and r['residual_degree']==2)
    # Independent material profile: pull back an advected reference scalar;
    # compare against the normalized current geometric profile, not its max.
    x,y=np.meshgrid(np.linspace(-.3,.3,31),np.linspace(-.3,.3,31),indexing='ij')
    material=np.column_stack([x.ravel(),y.ravel()]);points=m.position(material,m.END,.24)
    old=np.exp(-((material@m.N0)/m.ELL)**2)
    f=m.kinematics(points,m.END,.24);carried=np.exp(-(f['z0']/m.ELL)**2)
    profile_error=float(np.max(np.abs(carried-old)))
    fixed_points=material
    initial_fixed=np.exp(-((fixed_points@m.N0)/m.ELL)**2)
    final_fixed=np.exp(-(m.kinematics(fixed_points,m.END,.24)['z0']/m.ELL)**2)
    trail=float(np.max(np.maximum(initial_fixed,final_fixed)-final_fixed))
    checks=[dict(name='B1_selected_configuration',passed=all(c['passed'] for c in fixed['acceptance'] if c['required_for_selected_q8'])),
            dict(name='B2_all_declared_gates',passed=result['status']=='passed'),
            dict(name='coarse_P1_residual_rejected',passed=p1['full_tensor_projection_scaled']>1e-3),
            dict(name='coarse_complete_Q2_capacity',passed=p2['full_tensor_projection_scaled']<1e-3),
            dict(name='separated_minimum_tested_coefficient_count',passed=candidate[1]['scalar_tensor_coefficients']<candidate[0]['scalar_tensor_coefficients']),
            dict(name='material_profile_pullback',passed=profile_error<1e-12),
            dict(name='bad_Eulerian_damage_max_detected',passed=trail>1e-3),
            dict(name='surface_memory_capacity',passed=surface['status']=='passed' and surface['rows'][-1]['B_extension_scaled']<1e-3)]
    answer=dict(scope='Stage B completed; Stage C native integration not performed',
        selection='Existing trial T DGQ2 plus independent complete residual R DGQ2 and surface scalar memory',
        reason='At the predeclared1e-3 target, actual separated Q2 meets accuracy at N8 with3456 tensor coefficients; direct Q2 first passes at N16 with6912. Best P1 residual already misses the N8 target.',
        refinement_limit='Selection is conditional on measured affine Cartesian, constant-G controls. No globally minimal basis, native pressure/work/lifecycle, AMR or physical event qualification is claimed.',
        candidates_at_first_passing_tested_mesh=candidate,
        surface_memory_capacity=surface,
        P1_screen=dict(n8_full_tensor_best_projection=p1['full_tensor_projection_scaled'],n8_Q2_full_tensor_best_projection=p2['full_tensor_projection_scaled'],transported_P1_implemented=False),
        profile=dict(material_pullback_max_defect=profile_error,wrong_Eulerian_max_trailing_damage=trail,native_obstacle_implementation=False),
        checks=checks,status='passed' if all(c['passed'] for c in checks) else 'failed',
        provenance=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        moving_results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),
                        space_screen_sha256=hashlib.sha256((out/'space_screen.json').read_bytes()).hexdigest(),
                        surface_screen_sha256=hashlib.sha256((out/'surface_screen.json').read_bytes()).hexdigest(),
                        B1_results_sha256=hashlib.sha256(fixed_path.read_bytes()).hexdigest()))
    (out/'qualification.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))
    if answer['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
