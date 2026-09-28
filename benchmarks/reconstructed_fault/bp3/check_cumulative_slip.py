"""Check saved profiles, or a legacy dense slip table, against accepted states."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from plot_cumulative_slip import read_scheduled_profiles


def check(root):
    with (root/'accepted_steps.csv').open() as f:
        accepted=list(csv.DictReader(f))
    if (root/'profiles.csv').is_file() and not (root/'cumulative_slip.csv').exists():
        states={int(row['step']):row for row in accepted}
        previous=None
        count=0
        adjacent=0
        for profile in read_scheduled_profiles(root/'profiles.csv'):
            row=states[profile.step]
            assert profile.time==float(row['time'])
            np.testing.assert_allclose(np.max(np.abs(profile.velocity)),float(row['max_V']),rtol=1e-14)
            if profile.step==0:
                np.testing.assert_array_equal(profile.slip,0.)
            if previous is not None and profile.step==previous.step+1:
                np.testing.assert_allclose(profile.slip-previous.slip,
                    float(row['dt'])*profile.velocity,rtol=1e-12,atol=1e-18)
                adjacent+=1
            previous=profile
            count+=1
        assert count>0
        report=dict(passed=True,accepted_states=len(accepted),saved_profiles=count,
                    vertices=len(previous.nodes),adjacent_slip_updates_checked=adjacent,
                    unsaved_slip_updates_checked=False)
        (root/'profile_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return
    with (root/'cumulative_slip.csv').open() as f:
        records=csv.DictReader(f)
        previous=None
        count=0
        for accepted_row,group in itertools.zip_longest(
                accepted,itertools.groupby(records,key=lambda r:int(r['step']))):
            assert accepted_row is not None and group is not None
            step,rows=group
            rows=list(rows)
            assert step==int(accepted_row['step'])
            current=np.array([[float(r[k]) for k in ('node','s_m','xd_m','slip_m')] for r in rows])
            assert np.all(np.diff(current[:,1])>0)
            np.testing.assert_array_equal(current[:,0],np.arange(len(rows)))
            assert all(float(r['time_s'])==float(accepted_row['time']) for r in rows)
            profile=root/f'profiles/fault_{step}.csv'
            if profile.exists():
                with profile.open() as p: fields=list(csv.DictReader(p))
                # The plotting table uses ten significant slip digits; the
                # retained profile/checkpoint is still full precision.
                np.testing.assert_allclose(current[:,3],[float(r['slip_m']) for r in fields],
                                           rtol=5e-10,atol=0.)
                if previous is not None:
                    increment=float(accepted_row['dt'])*np.array([float(r['V_m_per_s']) for r in fields])
                    rounding=5e-10*(np.abs(current[:,3])+np.abs(previous[:,3]))
                    assert np.all(np.abs(current[:,3]-previous[:,3]-increment)
                                  <=rounding+1e-12*np.abs(increment)+1e-18)
            if step==0: np.testing.assert_array_equal(current[:,3],0.)
            previous=current
            count+=1
    assert count==len(accepted)
    for vtu in (root/'reconstructed_faults').glob('*.vtu'):
        tree=ET.parse(vtu)
        names={a.get('Name') for a in tree.findall('.//PointData/DataArray')}
        assert {'slip_state','cohesive_traction','previous_I_h','composition_strengthening',
                'background_tractions','cumulative_slip','fault_id','slip_rate'}<=names
        assert all(' ' not in name for name in names)
        assert not {'vertex_id','BP3_fixed_shear_correction','mature_fault_reference_geometry'} & names
        assert not tree.findall('.//CellData/DataArray[@Name="cell_id"]')
    report={'passed':True,'accepted_states':count,'vertices':len(previous),
            's_range_m':[float(previous[0,1]),float(previous[-1,1])],
            'every_step_file_bytes':(root/'cumulative_slip.csv').stat().st_size}
    (root/'cumulative_slip_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    check(p.parse_args().run)
