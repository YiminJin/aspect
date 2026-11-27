"""Run one ordinary historical qualification trajectory in a fresh output directory."""
import argparse
import os
from pathlib import Path
import subprocess

SCENARIOS={
    'control':'',
    'raw': 'subsection Material model\n subsection Phase field fault\n set Normal traction filter length = 0\n end\nend',
    'uncompleted': 'subsection Material model\n subsection Phase field fault\n set Complete prescribed contact profiles = false\n end\nend',
    'linear11':'subsection Solver parameters\n subsection Stokes solver parameters\n set Linear solver tolerance = 1e-11\n end\nend',
    'tolerance7':'set Nonlinear solver tolerance = 1e-7',
    'growth91':'set Maximum relative increase in time step = 91',
    'age02':'subsection Time stepping\n subsection Fault RSF\n set Maximum logarithmic state change = 0.2\n end\nend',
    'maxwell4e6':'subsection Material model\n subsection Phase field fault\n set Initialization interval = 4e6\n end\nend',
}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',type=Path)
    p.add_argument('case',choices=('contact','intermediate','production'))
    p.add_argument('scenario',choices=SCENARIOS)
    p.add_argument('output',type=Path)
    p.add_argument('--ranks',type=int,choices=(1,2),default=1)
    p.add_argument('--first',type=float)
    p.add_argument('--end',type=float)
    p.add_argument('--end-step',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--diagnostic-iterations',type=int)
    args=p.parse_args()
    stage=args.stage.resolve()
    dest=args.output.resolve()
    if args.resume != (dest/'restart/last_good_checkpoint.txt').is_file() or (dest.exists() and not args.resume):
        raise SystemExit('Fresh and restart output must be separate; resume needs this run’s checkpoint.')
    cmd=[(stage/'executable.path').read_text().strip()]
    if args.ranks>1: cmd=['mpiexec','-n',str(args.ranks)]+cmd
    prm=dest.with_name(dest.name+('-resume' if args.resume else '-start')+'.prm')
    text=f'include {args.case}.prm\nset Output directory = {dest}\nset Resume computation = {str(args.resume).lower()}\n'+SCENARIOS[args.scenario]+'\n'
    for value,key in [(args.first,'Maximum first time step'),(args.end,'End time'),(args.diagnostic_iterations,'Max nonlinear iterations')]:
        if value is not None: text+=f'set {key} = {value}\n'
    if args.end_step is not None:
        text+=f'subsection Termination criteria\n set Termination criteria = end time, end step, user request, BP3 first event complete\n set End step = {args.end_step}\nend\n'
    # ASPECT resolves includes relative to the current input file. Stage a private
    # copy beside its immutable includes, and retain the exact invocation next to the output.
    staged=stage/prm.name
    with prm.open('x') as f: f.write(text)
    with staged.open('x') as f: f.write(text)
    env=dict(os.environ,OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with prm.with_suffix('.log').open('x') as log:
        subprocess.run(['sha256sum','--check','--quiet','inputs.sha256'],cwd=stage,check=True)
        subprocess.run(cmd+[staged.name],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
