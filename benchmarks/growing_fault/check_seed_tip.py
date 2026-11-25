"""Locate a tapered initial FE seed across an MPI face and reject normally displaced tips."""
import argparse
import csv
from math import cos, sin, pi
import os
from pathlib import Path
import subprocess

from check_repairs import function, subsection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugin','destination'):
        parser.add_argument(name,type=lambda p: Path(p).resolve())
    parser.add_argument('--launcher',default='/opt/openmpi/5.0.6/bin/mpirun')
    args = parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=False)
    base = Path(__file__).with_name('onecut.prm').read_text()
    env = os.environ | dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    t = (cos(pi/6),sin(pi/6))
    normal = (-t[1],t[0])
    failures = []
    for name, shift in [('taper',0.),('outside',1.),('detached',-.6),('partition',0.)]:
        run = args.destination/name
        run.mkdir()
        tip = tuple(c+.5*d+shift*n for c,d,n in zip((1.,.5),t,normal))
        extra = ''
        if name=='partition':
            # Taper ends exactly at x=1, on the two-rank partition. Only the
            # upstream owner sees positive phase and the tip itself is zero.
            tip = (1.,.5)
            extra = 'set Seed tangent angle = 0\n'
        prm = base + subsection('Boundary velocity model',function('0;0'))
        if name=='partition':
            prm += subsection('Initial composition model',function('0;0;0;.2*min(1,max(0,4*(1-x)))*max(0,1-((y-.5)/.25)^2)'))
        prm += subsection('Material model',subsection('Growing fault',extra+f'set Forward seed tip = {tip[0]:.17g},{tip[1]:.17g}'))
        prm += subsection('Postprocess','set List of postprocessors = midpoint publication')
        prm += f'\nset End time = .005\nset Additional shared libraries = {args.plugin}\nset Output directory = {run/"output"}\n'
        (run/'input.prm').write_text(prm)
        with (run/'log').open('w') as log:
            result = subprocess.run([args.launcher,'-np','2',args.aspect,run/'input.prm'],
                                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        accepted = run/'output/sections-0001.csv'
        flags = 0
        if accepted.exists():
            with accepted.open() as stream:
                flags = sum(int(row['formed']) for row in csv.DictReader(stream))
        print(f'{name}: tip={tip}, exit={result.returncode}, accepted formed sections={flags}',flush=True)
        diagnostic = {'outside':'Forward seed tip lies outside the mesh',
                      'detached':'No local phase support immediately behind the forward seed tip'}.get(name)
        if diagnostic:
            with (run/'output/sections-0000.csv').open() as stream:
                if any(int(row['formed']) for row in csv.DictReader(stream)):
                    raise ValueError('Virgin seed already has formed sections')
            if result.returncode==0 or diagnostic not in ' '.join((run/'log').read_text(errors='replace').split()) or accepted.exists():
                failures.append(name)
        else:
            result.check_returncode()
            if not flags:
                failures.append(name)
        if name=='partition':
            for rank in (0,1):
                with (run/f'output/points-0000-r{rank}.csv').open() as stream:
                    data = list(csv.DictReader(stream))
                side = [float(r['x']) for r in data]
                peak = max(float(r['phi']) for r in data)
                upstream = max(side)<1 and peak>.1
                downstream = min(side)>1 and peak==0
                if not (upstream if rank==0 else downstream):
                    raise ValueError('The taper must cross the ownership face with support on one rank')
    if failures:
        raise ValueError(f'Initial seed checks failed: {failures}')


if __name__=='__main__':
    main()
