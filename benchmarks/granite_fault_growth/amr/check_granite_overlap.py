"""Match the geometry-based analysis to the frozen first 134 driven intervals."""
import argparse
import csv
from pathlib import Path

import numpy as np


def check(history,reference):
    with reference.open() as f: old=[r for r in csv.DictReader(f) if r["case"]=="drive"]
    with history.open() as f: new=list(csv.DictReader(f))
    if len(new)<len(old) or len(old)!=135:
        raise ValueError("All 135 original driven endpoints must be present")
    for old_col,new_col,atol in (("time_s","time_s",0),("step","step",0),
                                 ("tip_001_m","x001_m",1e-11),("tip_005_m","x005_m",1e-11),
                                 ("max_abs_phase_coefficient_ahead","ahead_phi_upper",1e-13),
                                 ("tip_V_m_s","fixed_V_m_s",0),("peak_abs_V_m_s","peak_m_s",0),
                                 ("tip_slip_m","fixed_slip_m",0),("peak_V_id","peak_id",0)):
        pairs=[(float(a[old_col]),float(b[new_col])) for a,b in zip(old,new) if a[old_col]]
        a,b=np.array(pairs).T
        np.testing.assert_allclose(a,b,atol=atol,rtol=0)
        print(f"{new_col}: {len(a)} clocks, maximum difference {max(abs(a-b)):.12g}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory",type=Path)
    p.add_argument("reference",type=Path)
    args=p.parse_args()
    check(args.trajectory,args.reference)
