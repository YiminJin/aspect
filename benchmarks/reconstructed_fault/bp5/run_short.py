"""Bounded 2-D BP5-friction research diagnostic; never an official BP5 run.

Setup and simulations are separate. All simulations share a hard 3600-s budget;
failed output is preserved and a label can never be run twice.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
BP3 = HERE.parent/'bp3'
sys.path.insert(0, str(BP3))
from length_scale_study import ROOT, BIN, LIB, OLD, parameters, render, digest
from run_mechanical_width import VirtualProfile, NORMAL

OUT = HERE/'dc010-ell100'
GENERATOR = ROOT/'benchmarks/reconstructed_fault/performance/build-gmg/bp3_length_scale_mesh'


def setup():
    start = time.monotonic()
    for name, flag, h in [('candidate', 0, 24.4140625), ('reference', 3, 12.20703125)]:
        dest = OUT/'fixtures'/name
        dest.mkdir(parents=True)
        with (dest/'mesh.log').open('x') as log:
            subprocess.run([str(GENERATOR), '100', '24.4140625', str(flag), str(dest/'target_cells.txt')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for file in ('fault.txt', 'prestress.txt'):
            shutil.copy2(OLD/file, dest/file)
        profile = VirtualProfile(100, h)
        fault = np.loadtxt(dest/'fault.txt')[:, :2]
        rows, error = [], 0.
        for j, (a, b) in enumerate(zip(fault[:-1], fault[1:])):
            for q, z in enumerate((leggauss(3)[0]+1)/2):
                p = (1-z)*a+z*b
                e = profile.extent
                intervals = [(-e, min(-p[1]/NORMAL[1], e)),
                             (max((100000-p[1])/NORMAL[1], -e), e)]
                value = sum(profile.integrate(p, lo, hi) for lo, hi in intervals)
                check = sum(profile.integrate(p, lo, hi, 16) for lo, hi in intervals) if value else 0.
                error = max(error, abs(value-check))
                rows.append([3*j+q, *p, value])
        assert error < 1e-6
        with (dest/'completion.txt').open('x') as stream:
            stream.write(str(len(rows))+'\n')
            np.savetxt(stream, rows, fmt=['%d','%.17g','%.17g','%.17g'])
        values = parameters((BP3/'bp3_modified_long_run.prm').read_text())
        material = ('Material model','Phase field fault')
        values[material+('Direct effect parameters',)] = '0.004, 0.04'
        values[material+('Evolution effect parameters',)] = '0.03'
        values[material+('Characteristic slip distance',)] = '0.1'
        values['Phase field model','Length scale'] = '100'
        values['Maximum time step',] = '4e6'
        values['Solver parameters','Stokes solver parameters','Stokes solver type'] = 'block AMG'
        values['Fault reconstruction','Prescribed faults file'] = str(dest/'fault.txt')
        values['Mesh refinement','BP3 saved mesh','Target cells file'] = str(dest/'target_cells.txt')
        level = max(len(s.split(':')[1]) for s in (dest/'target_cells.txt').read_text().split())
        values['Mesh refinement','Initial adaptive refinement'] = str(level-1)
        values['Postprocess','BP3','Mature prestress file'] = str(dest/'prestress.txt')
        values['Postprocess','BP3','Bottom normalization completion file'] = str(dest/'completion.txt')
        values['Postprocess','BP3','Last accepted step'] = '6'
        values['Postprocess','BP3','Profile time interval'] = '1'
        values['Postprocess','BP3','Graceful wall seconds'] = '1750'
        values['Checkpointing','Time between checkpoint'] = '0'
        # ASPECT checkpoints the pending next step: modulo 3 first captures
        # accepted step 2. Final checkpoint is separately requested on termination.
        values['Checkpointing','Steps between checkpoint'] = '0'
        values['Resume computation',] = 'false'
        values['Output directory',] = str(OUT/name)
        values['Additional shared libraries',] = str(BP3/'build/libbp3.release.so')+', '+str(LIB)
        (dest/'case.prm').write_text('# Modified 2-D BP5-friction diagnostic, NOT official 3-D BP5.\n'+render(values))
        files = [dest/p for p in ('case.prm','fault.txt','prestress.txt','target_cells.txt','completion.txt')]
        record = dict(ell=100, h=h, patch_mode=flag, support=2*profile.r[-1], completion_error=error,
                      prestress_unchanged=digest(dest/'prestress.txt')==digest(OLD/'prestress.txt'),
                      hashes={str(p):digest(p) for p in files})
        (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    (OUT/'setup.json').write_text(json.dumps(dict(seconds=time.monotonic()-start),indent=2)+'\n')


def prepare(label, mesh, probe, end=None):
    out = OUT/label
    out.mkdir()
    values = parameters((OUT/'fixtures'/mesh/'case.prm').read_text())
    values['Output directory',] = str(out)
    values['Postprocess','BP3','Graceful wall seconds'] = '1750'
    if probe:
        values['Termination criteria','Checkpoint on termination'] = 'false'
    else:
        assert json.loads((OUT/'probe_comparison.json').read_text())['gate']
        # Save accepted step 2, plus the final state; avoid intermediate full
        # checkpoints by disabling the periodic rule after the first callback.
        values['Checkpointing','Steps between checkpoint'] = '3' if mesh=='candidate' else '0'
        if end is not None:
            values['End time',] = str(end)
            values['Postprocess','BP3','Last accepted step'] = '2147483647'
    (out/'run.prm').write_text(render(values))
    env = dict(ASPECT_SOURCE_DIR=str(ROOT), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_MECHANICAL_WIDTH_PROBE='1',
               ASPECT_MECHANICAL_PROBE_STEP='0', ASPECT_MECHANICAL_PROBE_NEWTON='0',
               ASPECT_BP3_LENGTH_STUDY='1', ASPECT_BP5_SHORT_TEST='1')
    if not probe:
        env.update(ASPECT_BP3_LENGTH_QUALIFICATION='1', ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC='1')
    paths=[out/'run.prm', BIN, LIB, BP3/'build/libbp3.release.so', Path(__file__)]
    paths += list((OUT/'fixtures'/mesh).glob('*.txt'))
    record=dict(probe=probe, mesh=mesh, cap_seconds=1800, environment=env,
                hashes={str(p):digest(p) for p in paths},
                command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(BIN),str(out/'run.prm')])
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


def run(label):
    out=OUT/label
    record=json.loads((out/'launch.json').read_text())
    used=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('*/execution.json'))
    cap=min(record['cap_seconds'], int(3600-used)-20)
    assert cap>0, 'Aggregate simulation budget exhausted'
    for p, sha in record['hashes'].items(): assert digest(Path(p))==sha, p
    assert shutil.disk_usage(OUT).free>3*1024**3, 'Insufficient disk space for an intact checkpoint'
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',LD_BIND_NOW='1')
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(['timeout','--kill-after=15',str(cap)]+record['command'],cwd=BP3,env=env,
                              stdout=log,stderr=subprocess.STDOUT)
    text=(out/'run.log').read_text()
    rows=list(csv.DictReader((out/'accepted_steps.csv').open())) if (out/'accepted_steps.csv').exists() else []
    success=('MECHANICAL MODES VERIFIED' in text and not rows) if record['probe'] else (
        result.returncode==0 and bool(rows) and all(int(r['fresh_linear_checks_passed'])==1 for r in rows))
    info=dict(seconds=time.monotonic()-start, status=result.returncode, expected_completion=success,
              cap_seconds=cap, previous_simulation_seconds=used, accepted_steps=[int(r['step']) for r in rows],
              peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
    assert success, 'Preserved failure; no automatic retry or changed tolerance'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','prepare','run'])
    parser.add_argument('--label',default='probe-candidate')
    parser.add_argument('--mesh',choices=['candidate','reference'],default='candidate')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--end',type=float)
    args=parser.parse_args()
    if args.action=='setup': setup()
    elif args.action=='prepare': prepare(args.label,args.mesh,args.probe,args.end)
    else: run(args.label)
