"""Replay a frozen closeout input into a new directory using its recorded runtime."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def replay(reference, case, target, launcher):
    mature = case == 'mature'
    fixture = reference/'mature' if mature else reference/'runs'/case
    runtime = reference/('mature' if mature else 'tip')/'runtime'
    text = (fixture/'input.prm').read_text()
    target.mkdir(parents=True,exist_ok=False)
    output = target/'output'

    def replace_setting(match):
        key, value = match.groups()
        if key == 'Output directory':
            value = str(output)
        elif key == 'Prescribed faults file':
            value = str(reference/'mature'/'fault.txt')
        else:
            value = ', '.join(str(runtime/Path(p.strip()).name) for p in value.split(','))
        return f'set {key} = {value}'

    text = re.sub(r'^\s*set (Output directory|Additional shared libraries|Prescribed faults file)\s*=\s*(.*)$',
                  replace_setting,text,flags=re.MULTILINE)
    prm = target/'input.prm'
    prm.write_text(text)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('ASPECT_TIP_','ASPECT_SECTION_','ASPECT_BULK_'))}
    env.update(OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if mature:
        args = ['-np','1']
    else:
        run = json.loads((fixture/'command.json').read_text())
        args = run['command'][1:-2]
        env.update(run['environment'])
        parent = run['arguments']['resume_from']
        if parent:
            source = reference/'runs'/Path(parent).parent.name/'output'/'restart'
            idx = run['arguments']['snapshot_id'] or int((source/'last_good_checkpoint.txt').read_text())
            dest = output/'restart'
            shutil.copytree(source/f'{idx:02}',dest/f'{idx:02}')
            (dest/'last_good_checkpoint.txt').write_text(str(idx))
    command = [launcher,*args,str(runtime/'aspect-release'),str(prm)]
    with (target/'run.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('case',choices=['mature','localized-onecut','localized-oneresume','localized-oneo1','localized-oneo2',
                                   'strip18-midpoint-half','strip18-audit-full','strip18-audit-half1'])
    p.add_argument('new_directory',type=Path)
    p.add_argument('--launcher',default='/opt/openmpi/5.0.6/bin/mpirun')
    a = p.parse_args()
    replay(a.reference.resolve(),a.case,a.new_directory.resolve(),a.launcher)
