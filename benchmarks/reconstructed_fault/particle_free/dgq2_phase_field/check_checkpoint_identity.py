#!/usr/bin/env python3
"""Exercise native phase-family rejection before vector/history deserialization."""
from pathlib import Path
import json,os,sys,shutil,subprocess,time,signal
root=Path(__file__).resolve().parents[4]
runs=root/'build-particle-free/transported-objective/stage-c/material-runs'
out=root/'build-particle-free/dgq2-phase/checkpoint-identity01';out.mkdir(exist_ok=False)
records=[]
for label,parent,dg,omit_metadata in [('dg-to-q1','layered-dgphase-life-control01',False,False),
                                      ('old-q1-to-dg','layered-dgphase-q1-native01',True,True)]:
 case=out/label;case.mkdir();(case/'output').mkdir()
 source=runs/parent
 shutil.copytree(source/'output/restart',case/'output/restart',ignore=shutil.ignore_patterns('phase-space.txt') if omit_metadata else None)
 text=(source/'input.prm').read_text().replace(str(source/'output'),str(case/'output'))
 text=text.replace('set Resume computation = false','set Resume computation = true')
 text+=f'\nsubsection Discretization\n set Use discontinuous phase discretization = {str(dg).lower()}\n set Phase polynomial degree = {2 if dg else 1}\nend\n'
 (case/'input.prm').write_text(text)
 command=['/opt/openmpi/5.0.6/bin/mpirun','-np','2',str(root/'build-particle-free/dgq2-phase/candidate/aspect-release'),str(case/'input.prm')]
 start=time.monotonic()
 with (case/'run.log').open('w') as stream:
  process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=process.wait(timeout=60)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
 message='Checkpoint phase family/degree/representation mismatch. Q1/DGQ2 conversion is not supported.'
 log=(case/'run.log').read_text(errors='replace')
 record=dict(label=label,parent=parent,omitted_metadata=omit_metadata,command=command,exit_code=code,
             expected_rejection=message in log,seconds=time.monotonic()-start)
 records.append(record);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
 assert code not in (0,124) and record['expected_rejection'],record
print(json.dumps(records),flush=True)
