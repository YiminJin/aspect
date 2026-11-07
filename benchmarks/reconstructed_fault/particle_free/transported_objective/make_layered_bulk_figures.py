#!/usr/bin/env python3
"""Reproduce the completed corrected-spin campaign's compact figure set."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign'


def main():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LAYERED_EVIDENCE=str(OUT))
    commands=[
        ['plot_layered_campaign.py','layered-bulk-zero','layered-bulk-plus',
         'layered-bulk-minus','layered-bulk-plus-wide','--prefix','layered-directions',
         '--animate','layered-bulk-plus-fine-cartesian'],
        ['plot_layered_campaign.py','layered-bulk-plus','layered-bulk-plus-half',
         'layered-bulk-adaptive','--prefix','layered-timestep'],
        ['plot_layered_spin.py','layered-bulk-plus-fine-cartesian',
         'layered-bulk-total-plus-fine-cartesian'],
        ['plot_layered_accuracy.py','layered-bulk-plus','layered-bulk-plus-half',
         'layered-bulk-plus-fine-cartesian','layered-bulk-adaptive','layered-bulk-plus-wide']]
    records=[]
    for arguments in commands:
        command=[sys.executable,str(HERE/arguments[0]),*arguments[1:]]
        start=time.perf_counter()
        with (OUT/(arguments[0]+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(command=command,exit_code=result.returncode,wall_s=time.perf_counter()-start)
        records.append(record)
        (OUT/'figure-commands.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        result.check_returncode()


if __name__=='__main__':
    main()
