#!/usr/bin/env python3
"""Compact review tables from the independently reduced coupled comparisons."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile'

def main():
    rows=json.loads((OUT/'comparison.json').read_text());by={r['case']:r for r in rows}
    def get(c,d,m,res):
        n=f'bpf-{c}-{d}-{m}-{res}';return by[n+'-live' if n=='bpf-c-p-m-base' else n]
    pair=lambda a,b,key:f"{100*a['fine_reference'][key]:.3f} → {100*b['fine_reference'][key]:.3f}"
    sections=['# Corrected full-normal-support coupled measurements','',
              'Q → M denotes Q1 remap → bounded MacCormack. Errors are percentages relative to the same 64², dt=.025 bounded reference for each coefficient/direction. These are numerical references, not exact solutions.','',
              '| Viscosity, direction | Cells, dt | Profile L2 | I L2 | Stress L2 | Point traction L2 | Work traction L2 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for c in ['c','l']:
        for d in ['p','n']:
            for res in ['base','joint']:
                a,b=get(c,d,'q',res),get(c,d,'m',res)
                vals=[pair(a,b,k) for k in ['profile_relative_L2','normalization_relative_L2','stress_relative_L2','point_traction_relative_L2','work_traction_relative_L2']]
                sections.append(f"| {'Constant' if c=='c' else 'Layered'}, {'+' if d=='p' else '−'} | {2**a['arguments']['level']}², {a['arguments']['dt']} | "+' | '.join(vals)+' |')
    sections+=['','## Separate spatial/time controls, positive direction','',
               '| Viscosity | Cells, dt | Profile Q → M (%) | Stress Q → M (%) |','|---|---|---:|---:|']
    for c in ['c','l']:
        for res in ['base','space','time','joint']:
            a,b=get(c,'p','q',res),get(c,'p','m',res)
            sections.append(f"| {c} | {2**a['arguments']['level']}², {a['arguments']['dt']} | {pair(a,b,'profile_relative_L2')} | {pair(a,b,'stress_relative_L2')} |")
    sections+=['','## Shape, state, normalization and cost','',
               '| Positive, 32² dt=.05 | Peak | Mass drift (%) | Normal variance (m²) | Cumulative slip error (%) | V error (%) | Age error (%) | Z error (%) | Wall (s) | Linear solves / iterations |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in ['c','l']:
        for m in ['q','m']:
            r=get(c,'p',m,'joint');f=r['fine_reference']
            sections.append(f"| {c}, {m} | {r['peak']:.6f} | {100*(r['profile_mass']/r['initial_mass']-1):.5f} | {r['normal_variance']:.7f} | {100*f['cumulative_slip_relative_L2']:.5f} | {100*f['slip_rate_relative_L2']:.5f} | {100*f['age_relative_L2']:.6f} | {100*f['memory_relative_L2']:.5f} | {r['wall_s']:.2f} | {r['linear_solves']} / {r['linear_iterations']} |")
    sections+=['',f"Maximum measured normal support loss: {max(r['normal_support_fraction_lost'] for r in rows):.3e}. Maximum absolute integral normalization error |∫χ/LΓ−1|: {max(abs(r['normalization_integral_relative_error']) for r in rows):.3e}.",
               f"Native profile replay maximum: {max(s.get('profile_replay_Linf',0) for r in rows for s in r['steps']):.3e}; donor-bound violation: {max(s.get('donor_bound_violation',0) for r in rows for s in r['steps']):.3e}. Age/memory source maxima: {max(s.get('age_source_Linf',0) for r in rows for s in r['steps']):.3e} / {max(s.get('memory_source_Linf',0) for r in rows for s in r['steps']):.3e}.",
               f"Projection orthogonality defect maximum: {max(abs(r['projection_orthogonality_defect']) for r in rows):.3e} J/m. Cell crossings summed over labels/steps: {sum(s.get('crossed_labels',0) for r in rows for s in r['steps'])}.",
               '', '## Energy and material flux, positive direction','',
               'All signed and absolute sums are retained separately; no cancellation-based pass criterion. Power scale is the time-integral of the absolute active bulk terms, not the much larger cancelling background/friction work.','',
               '| Case | Endpoint signed / absolute (J/m) | Trapezoid signed / absolute (J/m) | Endpoint / trapezoid relative (%) | Net material flux integral (J/m) | Flux error / reference active scale (%) |',
               '|---|---:|---:|---:|---:|---:|']
    for c in ['c','l']:
        for res in ['space','joint','ref']:
            for m in (['m'] if res=='ref' else ['q','m']):
                r=get(c,'p',m,res);f=r['fine_reference']
                sections.append(f"| {c}-{m}-{res} | {r['endpoint_signed']:.4e} / {r['endpoint_absolute']:.4e} | {r['trapezoid_signed']:.4e} / {r['trapezoid_absolute']:.4e} | {100*r['endpoint_relative']:.3f} / {100*r['trapezoid_relative']:.3f} | {r['trapezoid_integrals']['flux']:.4e} | {100*f['flux_error_over_active_scale']:.4f} |")
    sections+=['','## Independent coupled work checks','',
               '| Case | Coupled signed / absolute (J/m) | Relative active / gross (%) | Max physical-step surface work residual (W/m) | Final raw face / weak reaction power (W/m) |',
               '|---|---:|---:|---:|---:|']
    work=json.loads((OUT/'coupled-work.json').read_text())
    for r in work:
        e=r['steps'][-1]
        sections.append(f"| {r['case']} | {r['signed']:.4e} / {r['absolute']:.4e} | {100*r['relative_active']:.3f} / {100*r['relative_gross']:.6f} | {max(abs(s['surface_work_residual']) for s in r['steps'][1:]):.3e} | {e['face_mechanical_power']:.3e} / {e['weak_mechanical_power']:.3e} |")
    (OUT/'MEASUREMENTS.md').write_text('\n'.join(sections)+'\n')
    flat=[]
    for r in rows:
        record={k:v for k,v in r.items() if isinstance(v,(str,int,float))};record.update(r['fine_reference'])
        record.update(cells_per_axis=2**r['arguments']['level'],dt=r['arguments']['dt'],eta=r['arguments']['eta'],direction=r['arguments']['speed'],method=r['arguments']['profile_method'])
        flat.append(record)
    with (OUT/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('Wrote compact measured tables and all-case CSV.')

if __name__=='__main__':main()
