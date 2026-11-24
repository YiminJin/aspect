"""Run one original schedule on two ranks in a new directory; preserve failure output."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', choices=['parent', 'coarse', 'half'])
parser.add_argument('destination', type=Path)
parser.add_argument('observer', type=Path)
parser.add_argument('--reference', type=Path, help='Replay the indexed donor with an extra FE observer')
parser.add_argument('--aspect', type=Path)
parser.add_argument('--plugin', type=Path)
parser.add_argument('--parent', type=Path, help='Target parent output with generation 02')
parser.add_argument('--launcher', default='/opt/openmpi/5.0.6/bin/mpirun')
args = parser.parse_args()
if not args.reference and (args.aspect is None or args.plugin is None):
    parser.error('Target runs require --aspect and --plugin')
if not args.reference and args.case == 'coarse' and args.parent is None:
    parser.error('The coarse continuation requires the verified target --parent')
root = args.destination.resolve()
root.mkdir(parents=True, exist_ok=False)
output = root/'output'
observer = args.observer.resolve(strict=True)
case = args.case
env = {k:v for k,v in os.environ.items() if not k.startswith(('ASPECT_TIP_', 'ASPECT_SECTION_', 'ASPECT_BULK_'))}
env.update(OMP_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
if args.reference:
    ref = args.reference.resolve()
    indexed = ref/'runs'/dict(parent='localized-oneo1', coarse='localized-oneo2', half='strip18-midpoint-half')[case]
    record = json.loads((indexed/'command.json').read_text())
    text = (indexed/'input.prm').read_text()
    exe = ref/'tip/runtime/aspect-release'
    plugin = ref/'tip/runtime/libbulk_tip_checks.release.so'
    plugins = 'bulk tip checks, midpoint mesh fields'
    env.update(record['environment'])
    mpi = record['command'][1:-2]
    if case != 'parent':
        src = ref/'runs/localized-oneo1/output/restart/02'
        shutil.copytree(src, output/'restart/02')
        (output/'restart/last_good_checkpoint.txt').write_text('2')
else:
    exe, plugin = args.aspect.resolve(strict=True), args.plugin.resolve(strict=True)
    mpi = ['--map-by','core','--bind-to','core','--report-bindings','-np','2']
    plugins = 'midpoint publication, midpoint mesh fields'
    base = Path(__file__).resolve().parents[1]/'onecut.prm'
    text = base.read_text() + '''
subsection Geometry model
 subsection Box
  set X repetitions = 16
  set Y repetitions = 8
 end
end
set End time = 0.05
subsection Checkpointing
 set Steps between checkpoint = 1
 set Number of checkpoints to keep = 4
end
'''
    if case == 'coarse':
        shutil.copytree(args.parent.resolve()/'restart/02', output/'restart/02')
        text += '\nset Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = 2\nend\n'
    duration = 'if(time<0.004,0.005,0.0025)' if case=='half' else '0.005'
    steps = dict(parent=1, coarse=2, half=3)[case]
    text += f'''
subsection Termination criteria
 set Termination criteria = end step
 set End step = {steps}
end
subsection Time stepping
 set List of model names = function
 subsection Function
  set Function expression = {duration}
 end
end
'''
text += f'''
set Output directory = {output}
set Additional shared libraries = {plugin}, {observer}
subsection Postprocess
 set List of postprocessors = {plugins}
end
'''
prm = root/'input.prm'
prm.write_text(text)
start = time.monotonic()
command = [args.launcher,*mpi,str(exe),str(prm)]
with (root/'log').open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360)
(root/'run.json').write_text(json.dumps(dict(command=command,seconds=time.monotonic()-start,
                                             exit_code=result.returncode),indent=2)+'\n')
result.check_returncode()
