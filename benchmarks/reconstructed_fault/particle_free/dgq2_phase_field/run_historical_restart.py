#!/usr/bin/env python3
"""Short current-core continuation from an actual preserved pre-DG Q1 checkpoint.

The main runner copies checkpoint files into the new output directory. The
original campaign checkpoint, output and binary remain read-only controls.
Run sequentially after the four selected coupled comparisons.
"""
import os
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[4]
support=Path(__file__).resolve().parents[1]/'transported_objective'
env=os.environ.copy()
env['LAYERED_EVIDENCE']=str(root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/lifecycle')
label='dgphase-q1-original-restart01'
command=[sys.executable,str(support/'run_layered_campaign.py'),label,
 '--candidate',str(root/'build-particle-free/dgq2-phase/candidate'),
 '--phase-discretization','Q1','--spin','bulk slip subtracted',
 '--translation','2','--background-shear','3.7','--halfwidth','6','--level','5',
 '--dt','.0125','--end','.6125','--ranks','2','--seconds','300',
 '--resume-from','layered-bulk-plus','--checkpoint-slot','oldest']
subprocess.run(command,cwd=root,env=env,check=True)
subprocess.run([sys.executable,str(support/'check_layered_lifecycle.py'),
 'layered-bulk-plus','layered-'+label,'--restart'],cwd=root,env=env,check=True)
subprocess.run([sys.executable,str(support/'check_layered_reference.py'),
 'layered-'+label],cwd=root,env=env,check=True)
