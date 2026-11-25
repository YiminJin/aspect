"""Resume deformed material radii and full normal rays, including legacy checkpoint writers."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check import read, same


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugin','destination'):
        parser.add_argument(name,type=lambda p: Path(p).resolve())
    parser.add_argument('--seeds',type=Path,help='Reuse full-seed and finite-seed checkpoints from a previous writer')
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True,exist_ok=False)
    here = Path(__file__).resolve().parent
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(here.parents[1]), OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    base = (here/'translated_horizontal.prm').read_text()
    base += '\nsubsection Boundary velocity model\n subsection Function\n  set Function expression = -.1*x;1+.1*y\n end\nend\n'

    def run(name,width,checkpoint=None,seed=False):
        directory = root/name
        directory.mkdir()
        out = directory/'output'
        prm = base + f'\nsubsection Material model\n subsection Phase field fault\n  set Profile half width = {width}\n end\nend\n'
        if checkpoint:
            shutil.copytree(checkpoint/'restart',out/'restart')
            prm += '\nset Resume computation = true\n'
        if seed:
            prm += '\nsubsection Termination criteria\n set Checkpoint on termination = true\nend\n'
        else:
            prm += '\nset End time = .025\n'
        (directory/'input.prm').write_text(prm+f'\nset Output directory = {out}\nset Additional shared libraries = {args.plugin}\n')
        with (directory/'log').open('w') as log:
            result = subprocess.run([args.aspect,directory/'input.prm'],env=env,stdout=log,stderr=subprocess.STDOUT)
        return result,out

    failures=[]
    for name,width,other in [('finite',2.1,0.),('full',0.,2.1)]:
        if args.seeds:
            seed=args.seeds.resolve()/(name+'-seed/output')
        else:
            result,seed=run(name+'-seed',width,seed=True)
            result.check_returncode()
        result,clean=run(name+'-clean',width)
        result.check_returncode()
        result,resumed=run(name+'-resume',width,seed)
        result.check_returncode()
        saved=read(seed/'material-1-surface-0.csv')['width_m']
        if name=='finite' and np.min(abs(saved-width))<1e-4:
            raise ValueError('The checkpoint must contain measurably stretched radii')
        worst=0.
        for suffix in ('surface-0','rank0'):
            filename=f'material-2-{suffix}.csv'
            a,b=(read(p/filename) for p in (clean,resumed))
            for k in a.dtype.names:
                if a[k].dtype.kind in 'fiu':
                    worst=max(worst,same(a[k],b[k],0.,k))
                else:
                    np.testing.assert_array_equal(a[k],b[k])
        result,changed=run(name+'-changed',other,seed)
        log=(changed.parent/'log').read_text()
        print(f'{name}: saved widths {saved}, unchanged continuation matched, '
              f'maximum difference {worst}, changed width exit {result.returncode}',flush=True)
        if (result.returncode==0 or 'Checkpoint normal support convention' not in log or
                (changed/'material-2-surface-0.csv').exists()):
            failures.append(name)
    if failures:
        raise ValueError(f'Incompatible restart checks failed: {failures}')


if __name__=='__main__':
    main()
