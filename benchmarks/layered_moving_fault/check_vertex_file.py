"""Compare ordered file/inline histories before and after a partition crossing."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from check import Frame, h, origin, same
from run import crossing_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugin','destination'):
        parser.add_argument(name,type=lambda s:Path(s).resolve())
    parser.add_argument('--mpiexec',default='mpirun')
    args=parser.parse_args()
    root=args.destination
    root.mkdir(parents=True,exist_ok=False)
    inline=crossing_input(Path(__file__).with_name('layered.prm').read_text())
    sources={}
    lines=[]
    for line in inline.splitlines():
        entry=line.strip().split(' = ',1)
        if entry[0] in ('set Fault vertices','set Initial label ages','set Initial label shear tractions'):
            sources[entry[0]]=entry[1]
        else:
            lines.append(line)
    vertices=sources['set Fault vertices'].split(';')
    ages=sources['set Initial label ages'].split(',')
    shear=sources['set Initial label shear tractions'].split(',')
    table=root/'vertices.txt'
    table.write_text('id x_m y_m initial_age_s background_shear_Pa\n'+''.join(
        f'{i} {xy.replace(",", " ")} {age} {load}\n' for i,(xy,age,load) in enumerate(zip(vertices,ages,shear,strict=True))))
    from_file='\n'.join(lines)+f'\nsubsection Material model\n subsection Phase field fault\n  set Fault vertex file = {table}\n end\nend\n'

    def run(name,source,ranks,seed=None):
        directory=root/name
        directory.mkdir()
        output=directory/'output'
        if seed:
            shutil.copytree(seed/'restart/02',output/'restart/02')
            source+='\nset Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = 2\nend\n'
        else:
            source+='\nsubsection Checkpointing\n set Steps between checkpoint = 1\n set Number of checkpoints to keep = 4\nend\n'
        (directory/'input.prm').write_text(source+f'\nset Additional shared libraries = {args.plugin}\nset Output directory = output\n')
        launch=[] if ranks==1 else [args.mpiexec,'-np',str(ranks)]
        with (directory/'log').open('w') as log:
            subprocess.run([*launch,args.aspect,'input.prm'],cwd=directory,stdout=log,stderr=subprocess.STDOUT,
                           env=os.environ|dict(OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),check=True)
        return output

    serial=run('inline1',inline,1)
    for name,source,ranks in [('file1',from_file,1),('inline2',inline,2),('file2',from_file,2)]:
        out=run(name,source,ranks)
        for step in range(3):
            check_frame(out,serial,step,1e-8 if ranks==2 else 0.)
            if name=='file2':
                inline2=root/'inline2/output'
                check_frame(out,inline2,step,0.)
                for rank in range(2):
                    filename=f'material-{step}-surface-{rank}.csv'
                    if (out/filename).read_bytes()!=(inline2/filename).read_bytes():
                        raise ValueError(f'Different per-ID history on rank {rank} at step {step}')
    start,end=(Frame(root/'file2/output',step) for step in (0,1))
    owners=[f.owner_grid[tuple(np.floor((f.X[9]-origin)/h).astype(int))] for f in (start,end)]
    if owners[0]==owners[1]:
        raise ValueError('Vertex 9 did not cross a partition')
    print('Material ID 9 crossed from rank',owners[0],'to',owners[1])
    resumed=run('resume2',from_file,2,seed=serial)
    check_frame(resumed,serial,2,1e-8)


def check_frame(out,control,step,tol):
    a,b=Frame(out,step),Frame(control,step)
    errors=[]
    for left,right in [(a.surface,b.surface),(a.rows,b.rows)]:
        if left.dtype.names!=right.dtype.names:
            raise ValueError('Field identity mismatch')
        for key in left.dtype.names:
            if left[key].dtype.kind in 'fiu':
                errors.append(same(left[key],right[key],tol,key))
            else:
                np.testing.assert_array_equal(left[key],right[key])
    print(out.parent.name,step,'max label and native sample difference',max(errors),flush=True)


if __name__=='__main__':
    main()
