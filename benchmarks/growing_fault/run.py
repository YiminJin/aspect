"""Run the nine-strip control and its two rollback and checkpoint comparisons."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from compare import continuation
from parameter_file import public_nine_strip_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugin','destination'):
        parser.add_argument(name,type=lambda p:Path(p).resolve())
    parser.add_argument('--ranks',type=int,default=2)
    parser.add_argument('--mpiexec',default='mpirun')
    parser.add_argument('--principal-only',action='store_true')
    args=parser.parse_args()
    root=args.destination
    root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    base=public_nine_strip_input()
    for name in (['principal'] if args.principal_only else ['principal','retry','restart-first','restart']):
        run=root/name
        run.mkdir()
        output=run/'output'
        prm=base+f'''
set Output directory = {output}
set Additional shared libraries = {args.plugin}
subsection Time stepping
 set List of model names = convection time step, check returned midpoint
end
subsection Postprocess
 set List of postprocessors = growing fault samples
end
'''
        if name=='principal':
            prm+='\nsubsection Checkpointing\n set Steps between checkpoint = 1\n set Number of checkpoints to keep = 4\nend\n'
        if name=='retry':
            prm+='''
set Maximum time step = .01
set End time = .05
subsection Termination criteria
 set Termination criteria = end step
 set End step = 3
end
subsection Time stepping
 set List of model names = retry growing fault, check returned midpoint
end
'''
        restart=name.startswith('restart')
        if restart:
            # Generation 01 resumes before the step-1 operator observer runs.
            generation=1 if name=='restart-first' else 3
            slot=f'{generation:02}'
            shutil.copytree(root/'principal/output/restart'/slot,output/'restart'/slot)
            prm+=f'\nset Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = {generation}\n set Number of checkpoints to keep = 4\nend\n'
        input_file=run/'input.prm'
        input_file.write_text(prm)
        cmd=[str(args.aspect),str(input_file)]
        if args.ranks>1:
            cmd=[args.mpiexec,'--bind-to','core','--map-by','core','-np',str(args.ranks),*cmd]
        started=time.monotonic()
        with (run/'log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        print(f'{name}: {time.monotonic()-started:.3f} s',flush=True)
        log_text = (run/'log').read_text()
        if log_text.count('Growing fault native section and point history') != (4-generation if restart else 4):
            raise ValueError('Publication count differs from the expected accepted clocks')
        if name=='retry':
            if log_text.count('Repeating the current time step')!=2:
                raise ValueError('Expected two completed and rejected solves')
            continuation(root/'principal/output',output,[1,2,3])
        if restart:
            continuation(root/'principal/output',output,range(generation,4))


if __name__=='__main__':
    main()
