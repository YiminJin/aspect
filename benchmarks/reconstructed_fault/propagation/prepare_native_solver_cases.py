#!/usr/bin/env python3
"""Explicit fixed-entry diagnostics and one economical fresh onset fixture."""
import json
from pathlib import Path

import numpy as np

import native_endpoint as native
from native_economical_schedule import schedule

E = Path(__file__).resolve().parent/'native_solver_evidence'


def write_entry(directory, name, p, start, expected, Va):
    directory.mkdir(exist_ok=True)
    p.evaluate(expected)
    Zb, Vb = p.last[5], p.last[3]['V']
    expected_u = p.r.velocity.copy()
    p.evaluate(start)
    initial_u, Vstart = p.r.velocity.copy(), p.last[3]['V']
    # All arrays represent the same fixed Q1 phase/Q2 velocity spaces.
    meta = [512, p.time, p.dt, p.Za, p.slip_a, Va, Zb, Vb, Vstart]
    data = np.r_[meta, p.lower, p.Ra, start, expected, initial_u, expected_u]
    np.savetxt(directory/name, data[None], fmt='%.17g')


def fixture(name, directory=None, coarse=False, dt=1/1024, economical=False):
    text = (native.E/'native-onset.prm').read_text()
    text += '\nsubsection Solver parameters\n subsection Phase field solver parameters\n'
    text += '  set Adaptive endpoint linear accuracy = true\n'
    text += f'  set Endpoint translation preconditioner = {str(coarse).lower()}\n end\nend\n'
    if directory is not None:
        text += f'set End time = 0\nsubsection Postprocess\n subsection P2 native endpoint\n  set Fixed entry input directory = {directory.resolve()}\n end\nend\n'
        text += f'subsection Boundary velocity model\n subsection Function\n  set Function expression = 0.256*max(0,t-{dt/2:.17g})*(y-0.5); 0\n end\nend\n'
    if economical:
        # Same continuous ramp, interval means and selected prototype windows.
        # The specified jump from 1/1024 to 1/64 requires a 1500% increase.
        # ASPECT's default 91% cap otherwise changes both schedule and midpoint.
        text += 'set Maximum time step = 0.015625\nset Maximum relative increase in time step = 1500\nsubsection Time stepping\n subsection Function\n'
        text += '  set Function expression = if(time<0.078125,0.015625,if(time<0.09375,0.0009765625,if(time<0.5,0.015625,if(time<0.5234375,0.0078125,0.0009765625))))\n end\nend\n'
        text += 'subsection Boundary velocity model\n subsection Function\n'
        text += '  set Function expression = 0.256*max(0,t-if(t<=0.078125,0.0078125,if(t<=0.09375,0.00048828125,if(t<=0.5,0.0078125,if(t<=0.5234375,0.00390625,0.00048828125)))))*(y-0.5); 0\n end\nend\n'
        text += 'subsection Postprocess\n subsection P2 native endpoint\n  set Record all step energies = true\n end\nend\n'
        text += f'subsection Postprocess\n subsection P2 native endpoint\n  set Expected timestep schedule file = {(E/"onset-schedule.txt").resolve()}\n end\nend\n'
    (E/f'{name}.prm').write_text(text)


if __name__ == '__main__':
    E.mkdir(exist_ok=True)
    f = dict(np.load(native.E/'prefix-2-fine-1024-onset/fields.npz'))
    history = json.loads((native.E/'prefix-2-fine-1024-onset/history.json').read_text())
    for group, times in [('core-last',[.53125]),('core-all',[.529296875,.5302734375,.53125])]:
        directory = E/group
        directory.mkdir(exist_ok=True)
        names = []
        for t in times:
            j = int(np.flatnonzero(f['time'] == t)[0])
            p = native.SharedFE()
            p.incoming(f['phase'][j-1], f['R'][j-1], f['Z'][j-1], history[j-2]['slip'],t,t-f['time'][j-1])
            name = f'core-{j}.txt'
            write_entry(directory,name,p,p.lower.copy(),f['phase'][j],history[j-2]['V'])
            names.append(name)
        (directory/'files.txt').write_text('\n'.join(names)+'\n')
        fixture(group+'-adaptive',directory)
        fixture(group+'-coarse',directory,True)
    data = dict(np.load(E/'saddle-entry.npz'))
    p = native.SharedFE()
    p.incoming(data['lower'],data['Ra'],float(data['Za']),float(data['slip_a']),float(data['time']),float(data['dt']))
    full = json.loads((native.E/'dt-1-1024/history.json').read_text())
    directory = E/'saddle';directory.mkdir(exist_ok=True)
    write_entry(directory,'symmetric-saddle.txt',p,data['symmetric'],data['stable'],full[574]['V'])
    (directory/'files.txt').write_text('symmetric-saddle.txt\n')
    fixture('saddle-adaptive',directory,False,float(data['dt']))
    fixture('saddle-coarse',directory,True,float(data['dt']))
    fixture('economical-onset-adaptive',economical=True)
    fixture('economical-onset-coarse',coarse=True,economical=True)
    times = schedule(1);times = times[times<=.53125]
    (E/'onset-schedule.json').write_text(json.dumps(times.tolist())+'\n')
    np.savetxt(E/'onset-schedule.txt',times,fmt='%.17g')
    print(f'Economical onset: {len(times)} steps; final {times[-1]} s; failure/retry at step55.')
