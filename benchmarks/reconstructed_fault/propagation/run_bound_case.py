#!/usr/bin/env python3
"""Run exactly one selected P1 reference/bound case, preserving all evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('case',choices=['reference32','reference64','bounds_straight','bounds_padded'])
parser.add_argument('--build',type=Path,required=True)
parser.add_argument('--executable',type=Path,required=True)
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--label',help='Distinct evidence/output label for a diagnosed repeat')
a=parser.parse_args()
here=Path(__file__).resolve().parent
repo=here.parents[2]
build=a.build.resolve(); evidence=a.evidence.resolve(); executable=a.executable.resolve()
label=a.label or a.case
assert label.replace('_','').isalnum()
plugin=build/'libbound_diagnostic.release.so'; output=build/('output-'+label)
assert plugin.is_file() and executable.is_file()
assert not output.exists() and not list(evidence.glob(label+'.*')), 'Refusing to overwrite evidence'
evidence.mkdir(parents=True,exist_ok=True)
overlay=build/(label+'.prm')
overlay.write_text(f'include {here / (a.case+".prm")}\nset Additional shared libraries = {plugin}\nset Output directory = {output}\n')
env=os.environ.copy()
env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
command=[str(executable),str(overlay)]
start=time.monotonic()
with (evidence/(label+'.log')).open('w') as log:
    try:
        code=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    except subprocess.TimeoutExpired:
        code=124
record=dict(case=a.case,label=label,command=command,exit_code=code,seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            environment={k:env[k] for k in ['ASPECT_SOURCE_DIR','OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS','PATH']},
            hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [plugin,executable,here/'bound_diagnostic.cc',here/'reference_1d.h',repo/'tests/phase_field_test_access.h',overlay]})
(evidence/(label+'.run.json')).write_text(json.dumps(record,indent=2)+'\n')
for p in output.glob('*.csv'):
    shutil.copyfile(p,evidence/(label+'.'+p.name))
print(json.dumps(record,indent=2))
assert record['peak_rss_kib']<2*1024*1024
assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<10*1024*1024
raise SystemExit(code)
