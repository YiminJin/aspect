#!/usr/bin/env python3
"""Native nonuniform first-append restart and final-state comparison."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'bulk_tip'))
from analyze_resolution_matched import read,gather
ROOT=HERE.parents[3]
RUN=ROOT/'build-particle-free/bulk-tip/runs'
a=RUN/'first-turn-append-admissible/output';b=RUN/'first-turn-append-recovery/output'
def compare(a,b,names=None):
    assert len(a)==len(b)
    errors={}
    for n in names or [n for n in a.dtype.names if n in b.dtype.names]:
        if a[n].dtype.kind in 'SU':
            assert np.array_equal(a[n],b[n]);continue
        assert np.array_equal(np.isnan(a[n]),np.isnan(b[n]))
        f=np.isfinite(a[n]);e=float(np.max(abs(a[n][f]-b[n][f]))) if f.any() else 0.
        errors[n]=e
        assert e<=2e-13*max(1.,float(np.max(abs(a[n][f]))) if f.any() else 0),(n,e)
    return errors
report={}
report['restore_points']=compare(gather(a,'tip-points-0014-r',['cell','q']),gather(b,'restore-points-r',['cell','q']))
x=gather(b,'restore-phase-r',['cell','i']);assert np.max(abs(x.B-x.solution_B))==0
report['restore_phase']=compare(gather(a,'tip-balance-0014-r',['cell','i']),x,['cell','i','B'])
report['restore_sections']=compare(read(a/'tip-sections-0014.csv'),read(b/'restore-sections.csv'),['s','ds','age','slip','V','force'])
report['restore_planes']=compare(read(a/'tip-planes-0014.csv'),read(b/'restore-planes.csv'),['x','ds','angle','formed'])
for stem,keys in [('points',['cell','q']),('balance',['cell','i'])]:
    report['final_'+stem]=compare(gather(a,'tip-'+stem+'-0016-r',keys),gather(b,'tip-'+stem+'-0016-r',keys))
for stem in ['sections','planes']:
    report['final_'+stem]=compare(read(a/f'tip-{stem}-0016.csv'),read(b/f'tip-{stem}-0016.csv'))
report['criterion']='Field maxima <=2e-13 max(1,reference maximum); exact material ordering.'
(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/recovery-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: restored parent14 and final16, maximum field difference',max(max(v.values()) for v in report.values() if isinstance(v,dict)))
