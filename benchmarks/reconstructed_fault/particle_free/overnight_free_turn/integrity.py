#!/usr/bin/env python3
"""Read-only acceptance, phase, plane and virgin-age audits of closed runs."""
from observations import *
import argparse

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('labels', nargs='+')
p.add_argument('--hy', type=float, default=.03125)
p.add_argument('--name', required=True)
a=p.parse_args()
report=[]
previous=None
previous_plane=None
for label in a.labels:
    path=RUNS/label/'output'
    accepted=read(path/'bulk-accepted.csv')
    steps=[]
    for r in accepted:
        step=int(r.step)
        rows=native(path,step)
        phi=RectBernstein(rows,.0625,a.hy)
        increment=None if previous is None else float(np.min(phi.coefficients-previous))
        previous=phi.coefficients
        section=read(path/f'tip-sections-{step:04d}.csv')
        row=dict(step=step,time=float(r.time),dt=float(r.dt),
                 coefficient_minimum=float(phi.coefficients.min()),
                 coefficient_maximum=float(phi.coefficients.max()),
                 minimum_increment_from_physical_obstacle=float(min(rows.B-rows.lower)),
                 minimum_increment_from_previous_export=increment,
                 mechanics=float(r.mechanical_residual),phase_KKT=float(r.phase_KKT),
                 publication_count=int(r.publications),minimum_section_age=float(min(section.age)))
        plane_path=path/f'tip-planes-{step:04d}.csv'
        if plane_path.exists():
            plane=read(plane_path)
            parent=plane.parent_formed>0
            # Formed planes and their measures are frozen in this model.
            row['formed']=int(sum(plane.formed>0))
            row['lost_parent_formed']=int(sum(parent & (plane.formed<=0)))
            row['maximum_measure_error']=float(max(abs(plane.ds-plane.reference_dx/np.cos(plane.angle))))
            virgin=plane.formed<=0
            row['virgin_age_error']=float(max(abs(plane.age[virgin]-(.5+r.time)))) if any(virgin) else None
            row['section_age_error']=float(max(abs(section.age-plane.age)))
            row['section_slip_error']=float(max(abs(section.slip-plane.slip)))
            if previous_plane is not None:
                old=previous_plane.formed>0
                row['old_formed_angle_change']=float(max(abs(plane.angle[old]-previous_plane.angle[old]))) if any(old) else 0.
                row['old_formed_measure_change']=float(max(abs(plane.ds[old]-previous_plane.ds[old]))) if any(old) else 0.
            previous_plane=plane
        assert row['coefficient_minimum']>=-1e-12 and row['coefficient_maximum']<=1+1e-12
        assert row['minimum_increment_from_physical_obstacle']>=-1e-12
        assert increment is None or increment>=-1e-12
        assert row['minimum_section_age']>0 and row['publication_count']==step
        assert row['mechanics']<=1e-9 and row['phase_KKT']<=1e-7
        if plane_path.exists():
            assert row['lost_parent_formed']==0
            for name in ['maximum_measure_error','section_age_error','section_slip_error','old_formed_angle_change','old_formed_measure_change']:
                assert row.get(name,0.)<=1e-12,(label,step,name,row.get(name))
            assert row['virgin_age_error'] is None or row['virgin_age_error']<=1e-12
        steps.append(row)
    report.append(dict(label=label,accepted_steps=steps,
                       last_good_slot=(path/'restart/last_good_checkpoint.txt').read_text().strip()))
(OUT/(a.name+'-integrity.json')).write_text(json.dumps(report,indent=2)+'\n')
print(a.name,[(r['label'],r['accepted_steps'][-1]['time']) for r in report])
