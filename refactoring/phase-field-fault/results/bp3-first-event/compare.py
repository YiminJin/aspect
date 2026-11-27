"""Compare every accepted BP3 material ID and the keyed native fields, including restart."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'benchmarks/bp3_qd_thrust'))
from check_historical import profile
from native_frames import frames,read
from history import read_statistics


def compare(a,b):
    def profiles(d):
        return {int(p.stem.rsplit('-',1)[-1]):p for p in d.glob('fault-profile-*.txt')}
    pa,pb=profiles(a),profiles(b)
    assert pa.keys()==pb.keys() and pa,"Missing accepted frames"
    sa=list(read_statistics(a/'statistics'));sb=list(read_statistics(b/'statistics'))
    assert len(sa)==len(sb)==len(pa)
    keys=['BP3 event onset (s)','BP3 event below-threshold streak','BP3 event completion (s)']
    for fa,fb in zip(sa,sb):
        assert all(fa[k]==fb[k] for k in keys)
        assert fa['Time step number']==fb['Time step number']
    errors={}
    failures=[]
    for step in sorted(pa):
        ta,va=profile(pa[step]);tb,vb=profile(pb[step])
        assert abs(ta-tb)<1e-6,(ta,tb)
        for name,tol in [('id',0),('s_m',0),('x_m',0),('y_m',0),('rate_m_s',2e-16),('age_s',1e-5),
                         ('slip_m',2e-13),('shear_Pa',2e-4),('raw_compression_Pa',2e-4),('friction_compression_Pa',2e-4),
                         ('normal_integral_m',1e-7)]:
            err=max(abs(va[name]-vb[name]));assert err<=tol,(step,name,err,tol)
            errors[name]=max(errors.get(name,0),err)
    aa,bb=frames(a),frames(b)
    assert len(aa)==len(bb)==len(pa)
    for ta,tb in zip(sorted(aa),sorted(bb)):
        assert abs(ta-tb)<1e-6
        ka,va,_=read(aa[ta]);kb,vb,_=read(bb[tb])
        np.testing.assert_array_equal(ka,kb)
        for name in va:
            # VTU is Float32; allow one output rounding at shared coordinates.
            tol={'velocity':1e-16,'phase_field':0.}.get(name,1e-5)
            mask=abs(va[name]-vb[name])>tol+2e-6*abs(vb[name])
            if np.any(mask):
                i=np.unravel_index(np.argmax(abs(va[name]-vb[name])),va[name].shape)
                failures.append((ta,name,int(mask.sum()),ka[i[0]].tolist(),va[name][i],vb[name][i]))

            errors['native_'+name]=max(errors.get('native_'+name,0),np.max(abs(va[name]-vb[name])))
    print(len(pa),'accepted frames,',len(va['p']),'native patches/frame; maximum absolute differences')
    for k,v in errors.items(): print(k,format(v,'.9g'))
    if failures:
        for row in failures: print('native tolerance exceeded',row)
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    args=p.parse_args()
    compare(args.first,args.second)
