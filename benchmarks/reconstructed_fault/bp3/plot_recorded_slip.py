"""Plot recorded cumulative slip; never integrate sparse rate samples."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-bp3-slip-plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--event-time',type=float,help='Physical seconds of the event reference profile')
p.add_argument('--interpolate-reference',action='store_true',help='Explicitly permit an interpolated, not solved, event reference')
p.add_argument('--deficit',action='store_true',help='Plot Vp*t minus recorded slip')
p.add_argument('--output',type=Path)
args=p.parse_args()
with (args.run/'profiles.csv').open() as stream: rows=list(csv.DictReader(stream))
times=np.array([float(r['time_s']) for r in rows])
assert np.all(np.diff(times)>0),'Profile index contains duplicate or unordered times'
fields=[np.genfromtxt(args.run/r['file'],delimiter=',',names=True) for r in rows]
for f in fields: np.testing.assert_array_equal(f['xd_m'],fields[0]['xd_m'])
slip=np.array([f['slip_m'] for f in fields]);reference=np.zeros(slip.shape[1]);label=''
if args.event_time is not None:
    match=np.flatnonzero(np.isclose(times,args.event_time,rtol=1e-13,atol=0.))
    if len(match): reference=slip[match[0]];label=f'; event reference t={times[match[0]]:.6g} s (recorded)'
    else:
        assert args.interpolate_reference,'No recorded profile at event time; interpolation requires explicit permission'
        i=np.searchsorted(times,args.event_time)
        assert 0<i<len(times),'No extrapolation outside recorded profiles'
        a=(args.event_time-times[i-1])/(times[i]-times[i-1])
        reference=(1-a)*slip[i-1]+a*slip[i]
        label=f'; INTERPOLATED reference t={args.event_time:.6g} s, not a solved state'
fig,ax=plt.subplots(figsize=(9,5))
for i,t in enumerate(times):
    values=1e-9*t-slip[i] if args.deficit else slip[i]-reference
    ax.plot(fields[0]['xd_m']/1000,values,label=f'{t/31557600:.4g} yr')
ax.set(xlabel='Down-dip distance [km]',ylabel='Slip deficit [m]' if args.deficit else 'Cumulative signed slip [m]',
       title='Recorded accepted-state slip'+label)
if len(times)<=20: ax.legend()
fig.tight_layout();fig.savefig(args.output or args.run/'recorded_slip.png',dpi=150)
