"""Prepare a distinct bounded adaptive modified-BP3 run; execute only on request.

Restart branches copy prior evidence into a new directory and restore metadata
from the selected checkpoint. Old runs are never modified. Old-layout
checkpoints are not accepted. This can require substantial disk space.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time
from slip_history import restore_prefix

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--purpose',choices=('first-event','recurrence'),default='recurrence')
    parser.add_argument('--end-years',type=float,default=1500.)
    parser.add_argument('--wall-hours',type=float,required=True)
    parser.add_argument('--ranks',type=int,default=4)
    parser.add_argument('--velocity-preconditioner',choices=('gmg','amg'),default='gmg')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--checkpoint',type=int,choices=(1,2,3))
    parser.add_argument('--verify-through-step',type=int)
    parser.add_argument('--verify-output',action='store_true')
    parser.add_argument('--verify-200km',action='store_true',help='Verification only: preserve the previous unrefined 200-km long-run mesh')
    args=parser.parse_args()
    assert 0<args.end_years<1e6 and 0<args.wall_hours<1e6 and args.ranks>0
    assert not (args.verify_200km or args.verify_output) or args.verify_through_step is not None
    out=args.output.resolve();assert not out.exists(),'Refuse to overwrite evidence'
    inputs=HERE/('fixtures/modified_bp3_long_run' if args.verify_200km else 'fixtures/modified_bp3_long_run_300km')
    mesh=properties=inputs
    paths=[mesh/'target_cells.txt',properties/'fault.txt',properties/'prestress.txt',properties/'completion.txt']
    paths += [HERE/'bp3_modified_long_run.prm',HERE/'environment.sh']
    for directory in {mesh,properties}:
        for name,record in json.loads((directory/'manifest.json').read_text()).items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==record['sha256'],name
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    binary=REPO/'build-pf-cpdi/aspect-release';plugin=HERE/'build/libbp3.release.so'
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (binary,plugin)})
    if args.resume_from:
        prior=args.resume_from.resolve()
        provenance=json.loads((prior/'launch.json').read_text())
        assert provenance['ranks']==args.ranks,'Only same-rank restart is qualified'
        assert provenance['hashes']==hashes,'Binary/plugin or physical fixture changed on resume'
        checkpoint=args.checkpoint or int((prior/'restart/last_good_checkpoint.txt').read_text())
        chosen=prior/f'restart/{checkpoint:02d}'
        step,accepted_time=(chosen/'bp3_accepted_state.txt').read_text().split()
        assert (chosen/'bp3_output_metadata').is_dir(),'Not a qualified new-layout checkpoint'
        shutil.copytree(prior,out)
        # This is a disposable branch, not an in-place rollback of prior evidence.
        (out/'restart/last_good_checkpoint.txt').write_text(str(checkpoint)+'\n')
        for source in (chosen/'bp3_output_metadata').iterdir(): shutil.copy2(source,out/source.name)
        restore_prefix(out/'cumulative_slip.csv',int(step))
        phase='resume'
    else:
        out.mkdir(parents=True);phase='fresh';step=None;accepted_time=None
    prm=f'''include {HERE/'bp3_modified_long_run.prm'}
set Output directory = {out}
set Resume computation = {str(bool(args.resume_from)).lower()}
set End time = {args.end_years*31557600.:.17g}
subsection Fault reconstruction
  set Prescribed faults file = {properties/'fault.txt'}
end
subsection Mesh refinement
  subsection BP3 saved mesh
    set Target cells file = {mesh/'target_cells.txt'}
  end
end
subsection Postprocess
  subsection BP3
    set Mature prestress file = {properties/'prestress.txt'}
    set Bottom normalization completion file = {properties/'completion.txt'}
    set Stop after first event = {str(args.purpose=='first-event').lower()}
    set Graceful wall seconds = {args.wall_hours*3600:.17g}
    set Last accepted step = {args.verify_through_step if args.verify_through_step is not None else 2147483647}
  end
end
'''
    if args.verify_output:
        prm+='''subsection Postprocess
  subsection BP3
    set Heavy output slip interval = 0.004
    set Profile slip interval = 0.0001
    set Audit full state every step = true
  end
end
subsection Checkpointing
  set Steps between checkpoint = 1
  set Time between checkpoint = 0
end
'''
    if args.verify_200km:
        prm+='''subsection Geometry model
  subsection Box
    set Box origin X coordinate = -50000
    set X extent = 200000
    set X repetitions = 4
  end
end
'''
    # Resolve the one maintained PRM plus launch overrides into one standalone
    # file. No inherited investigation parameters or include chain is emitted.
    def parameters(text):
        values={};section=[]
        for line in text.splitlines():
            line=line.split('#')[0].strip()
            if line.startswith('subsection '): section.append(line[11:])
            elif line=='end': section.pop()
            elif line.startswith('set '):
                key,value=line[4:].split('=',1)
                values[tuple(section+[key.strip()])]=value.strip()
        assert not section
        return values
    values=parameters((HERE/'bp3_modified_long_run.prm').read_text())
    values.update(parameters(prm))
    def render(prefix=()):
        lines=[];children=[];indent='  '*len(prefix)
        for path,value in values.items():
            if path[:len(prefix)]!=prefix: continue
            if len(path)==len(prefix)+1: lines.append(f'{indent}set {path[-1]} = {value}\n')
            elif path[len(prefix)] not in children: children.append(path[len(prefix)])
        for name in children:
            lines.extend([f'\n{indent}subsection {name}\n',render(prefix+(name,)),f'{indent}end\n'])
        return ''.join(lines)
    (out/f'{phase}.prm').write_text('# Resolved modified BP3: no parameter includes.\n'+render())
    # The direct mpirun recipe and this launcher share one environment authority.
    raw=subprocess.check_output(['bash','-c','source "$1" "$2" && env -0',
                                 'bp3',str(HERE/'environment.sh'),args.velocity_preconditioner])
    env=dict(item.split('=',1) for item in raw.decode().split('\0') if item)
    command=['mpirun','-np',str(args.ranks),'--bind-to','core','--map-by','core',str(binary),str(out/f'{phase}.prm')]
    record=dict(command=command,ranks=args.ranks,hashes=hashes,purpose=args.purpose,
                checkpoint_step=step,checkpoint_time=accepted_time,
                environment={k:v for k,v in env.items() if k.startswith('ASPECT_')})
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/f'{phase}_source.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=REPO))
    selected={k:v for k,v in env.items() if k.startswith('ASPECT_')
              or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS')}
    # A prepared launch must be as clean as --execute, even from a shell left
    # over from an investigation. Preserve MPI/library environment, not probes.
    (out/'launch.sh').write_text('#!/bin/sh\nexec python3 - <<\'PY\'\n'
        'import os, hashlib\nfrom pathlib import Path\n'
        f'expected={hashes!r}\n'
        'for name,digest in expected.items():\n'
        '    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name\n'
        'env={k:v for k,v in os.environ.items() if not k.startswith("ASPECT_")}\n'
        f'env.update({selected!r})\ncommand={command!r}\n'
        f'os.chdir({str(HERE)!r})\n'
        'os.execvpe(command[0],command,env)\nPY\n')
    if not args.execute:
        print(f'Prepared {out}; run: sh {out}/launch.sh');return
    start=time.monotonic()
    with (out/f'{phase}.log').open('w') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    elapsed=time.monotonic()-start
    result=dict(status=result.returncode,seconds=elapsed,peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/f'{phase}_execution.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
    assert result['status']==0,'Preserve failure, no automatic retry'

if __name__=='__main__':main()
