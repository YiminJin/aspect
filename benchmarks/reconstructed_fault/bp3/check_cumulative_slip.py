"""Check every accepted slip row against the clock and retained profile outputs."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def check(root):
    with (root/'accepted_steps.csv').open() as f:
        accepted=list(csv.DictReader(f))
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
                np.testing.assert_array_equal(current[:,3],[float(r['slip_m']) for r in fields])
                if previous is not None:
                    np.testing.assert_allclose(current[:,3]-previous[:,3],
                        float(accepted_row['dt'])*np.array([float(r['V_m_per_s']) for r in fields]),
                        rtol=1e-12,atol=1e-18)
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
