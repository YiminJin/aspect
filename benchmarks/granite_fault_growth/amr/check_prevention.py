"""Verify that a moving, supported tip consumed a previously sufficient buffer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from trajectory import Case

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_remesh_history import geometry, phase_coefficients, read_snapshot


def clearance(cells, tip, resolution):
    distances = [np.linalg.norm(np.maximum(0., np.maximum(c.lo-tip,tip-c.hi)))
                 for c in cells if max(c.size)>resolution*(1+1e-12)]
    return min(distances, default=np.inf)


def check(folder, ranks):
    case=Case(folder/"output")
    h=.134
    logs=(folder/"timing/solver.log").read_text()
    assert logs.count("Growing fault preventive refinement: refine and advance")==1
    assert logs.count("Midpoint iteration 0,")==2 and "missed extension" not in logs
    first=case.load(1)
    metrics=case.metrics(case.phase_patches(first[0],first[1]),first[2],None,64)
    new=np.array([metrics["tracked_x_m"],metrics["tracked_y_m"]])
    last_formed=np.array([metrics["formed_tip_x_m"],metrics["formed_tip_y_m"]])
    assert new[0]>case.seed[0]
    result={}
    previous_phase=metadata=None
    for when in ("before","after"):
        data,energy,meta,ledger=read_snapshot(folder/"output",f"remesh-1-1-{when}",ranks)
        cells=[geometry(*p) for p in data.values()]
        candidates=[c for c in cells if all(c.lo<=new+1e-12) and all(new<=c.hi+1e-12)]
        phase=max((c.shape(new[None,:])[0]@phase_coefficients(data[c.name][0],c))[0,0] for c in candidates)
        assert phase>=.05
        d=clearance(cells,new,h)
        if when=="before":
            assert clearance(cells,case.seed,h)>=3*case.ell and d<3*case.ell
            # With these straight accepted segments all underresolved cells
            # lie ahead of the last formed station. The band alone would pass.
            assert all(c.lo[0]-last_formed[0]>2*case.ell for c in cells if max(c.size)>h)
            metadata=(meta,ledger)
            previous_phase=phase
        else:
            assert d>4*case.ell
            assert (meta,ledger)==metadata
            np.testing.assert_allclose(phase,previous_phase,rtol=0,atol=2e-13)
        result[when]=dict(cells=len(cells),clearance_new_m=float(d) if np.isfinite(d) else "whole domain fine",
                          clearance_old_m=clearance(cells,case.seed,h) if np.isfinite(clearance(cells,case.seed,h)) else "whole domain fine",
                          tip_x_m=new[0],tip_y_m=new[1],physical_phase_at_tip=phase,
                          bulk_J_m=energy[0],cohesive_J_m=energy[1],fracture_J_m=energy[2])
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--ranks",type=int,default=1)
    args=parser.parse_args()
    print(json.dumps(check(args.folder,args.ranks),indent=2,allow_nan=False))
