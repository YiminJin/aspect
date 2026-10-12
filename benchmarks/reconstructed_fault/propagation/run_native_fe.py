#!/usr/bin/env python3
"""Flatten native fixture includes, replacing unavailable inherited observer names.

Only the final observer is installed. All physical parameter overrides retain
native ParameterHandler order. No constitutive/history mutation occurs here.
"""
import argparse, os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'benchmarks/reconstructed_fault/propagation'
def flatten(path):
    lines=[]
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith('include '):
            name=line.strip()[8:].replace('$ASPECT_SOURCE_DIR',str(ROOT))
            lines.extend(flatten(name))
        elif re.match(r'\s*set List of postprocessors\s*=',line):
            lines.append(' set List of postprocessors = particles, P2 native FE strain')
        else: lines.append(line.replace('$ASPECT_SOURCE_DIR',str(ROOT)))
    return lines
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    generated=P/'native_fe_evidence'/f'{a.case}-native.prm'
    generated.write_text('\n'.join(flatten(P/'native_fe_evidence'/f'{a.case}.prm'))+'\n')
    if not a.prepare_only:
        env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=ROOT/'build-propagation-integration'/f'native-fe-{a.case}.log'
        with log.open('w') as out:
            result=subprocess.run([str(ROOT/'build-propagation-integration/aspect'),str(generated)],env=env,stdout=out,stderr=subprocess.STDOUT)
        print(f'{a.case}: exit {result.returncode}; log {log}')
        raise SystemExit(result.returncode)
