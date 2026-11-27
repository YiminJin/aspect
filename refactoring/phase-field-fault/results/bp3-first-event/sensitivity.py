"""Compare physical quantities at shared accepted clocks and report each trajectory's schedule."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'benchmarks/bp3_qd_thrust'))
from check_historical import profile
from native_frames import read,frames
from history import read_statistics


def tables(directory):
    return {t:data for t,data in (profile(path) for path in directory.glob('fault-profile-*.txt'))}


def measures(directory):
    profiles=tables(directory)
    print(directory.name,'accepted clocks (s)',list(sorted(profiles)))
    for time,native in frames(directory).items():
        if time not in (0.,400.): continue
        _,fields,_=read(native)
        print('clock',time,'native max absolute (ux,uy,p,xx,yy,xy)',*(np.max(abs(fields['velocity'][:,:,d])) for d in (0,1)),
              *(np.max(abs(fields[k])) for k in ('p','fault_tau_xx','fault_tau_yy','fault_tau_xy')))
        data=profiles[time]
        print('clock',time,'range (rate, slip, age, shear, raw, friction compression)')
        for name in ('rate_m_s','slip_m','age_s','shear_Pa','raw_compression_Pa','friction_compression_Pa'):
            print(name, min(data[name]),max(data[name]))


def difference(first,second):
    a,b=tables(first),tables(second)
    print(first.name,'versus',second.name,'accepted clocks (s)',sorted(a),sorted(b))
    for t in sorted(a.keys()&b.keys()):
        va,vb=a[t],b[t]
        assert np.array_equal(va['id'],vb['id'])
        print('clock',t,'maximum absolute difference and distance in metres')
        for name in ('rate_m_s','age_s','slip_m','raw_compression_Pa','friction_compression_Pa','shear_Pa'):
            d=abs(va[name]-vb[name]);i=np.argmax(d)
            print(name,format(d[i],'.10g'),format(va['s_m'][i],'.8g'))


if __name__=='__main__':
    a,b=map(Path,sys.argv[1:3])
    if '--fields' in sys.argv: measures(a);measures(b)
    difference(a,b)
