"""Bounded paired top completion/source experiment, with the bottom retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import bottom_completion as g
from analyze_uniform_sliding import read, cat

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'bottom-source-complete-wedge-50-local4'

def prepare(out, paired):
    out.mkdir() # Preserve failed as well as accepted evidence.
    (out/'clock.csv').write_bytes((BASE/'clock.csv').read_bytes())
    completion=HERE/'bottom-completion-50-local4/completion.txt'
    if paired:
        saved=cat(BASE.glob('uniform_bulk_0_rank*.csv'),('cell',))
        mask=saved['y']>98000
        error=float(np.max(abs(g.q1_phi(saved['x'][mask],saved['y'][mask])-saved['phi'][mask])))
        assert error<2e-12,('Virtual Q1 differs from physical top FE',error)
        rows=np.loadtxt(completion,skiprows=1)
        extent=g.radius+g.h*np.sum(abs(g.normal))
        top=[];order_error=0.
        for row in rows:
            origin=row[1:3];limit=(100000-origin[1])/g.normal[1]
            missing=g.integrate(origin,max(limit,-extent),extent) if limit<extent else 0.
            if missing:
                check=g.integrate(origin,max(limit,-extent),extent,16)
                order_error=max(order_error,abs(missing-check))
            top.append(missing)
        assert order_error<1e-6
        top=np.array(top);rows[:,3]+=top
        completion=out/'completion.txt'
        with completion.open('w') as f:
            f.write(str(len(rows))+'\n')
            np.savetxt(f,rows,fmt=['%d','%.17g','%.17g','%.17g'])
        (out/'completion_preflight.json').write_text(json.dumps(dict(
            q1_match_max_phi_error=error,order_check_max_integral_error=order_error,
            added_top_profiles=int(np.count_nonzero(top)),spacing=g.h,enclosure=extent),indent=2)+'\n')
    (out/'run.prm').write_text(f'include {BASE}/run.prm\nset Output directory = {out}\n'
        f'subsection Postprocess\n  subsection BP3\n    set Bottom normalization completion file = {completion}\n  end\nend\n')
    return completion

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['control','paired'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    args=parser.parse_args()
    out=HERE/f'top-source-{args.case}-50-local4'
    completion=(out/'completion.txt' if args.case=='paired'
                else HERE/'bottom-completion-50-local4/completion.txt')
    if not args.run_prepared:completion=prepare(out,args.case=='paired')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(json.loads((BASE/'provenance.json').read_text())['environment'])
    env.update(ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'clock.csv'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    if args.case=='paired':env['ASPECT_BP3_TOP_SOURCE_EXPERIMENT']='1'
    command=['timeout','--signal=TERM','--kill-after=15','600','mpirun','-np','4',
             '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(out/'run.prm')]
    paths=[HERE/'bp3.cc',HERE/'uniform_sliding.h',Path(__file__),HERE/'build/libbp3.release.so',
           REPO/'build-pf-cpdi/aspect-release',out/'run.prm',out/'clock.csv',completion,
           REPO/'source/material_model/phase_field_fault.cc',REPO/'source/reconstructed_fault/manager.cc']
    (out/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    if args.prepare_only:
        print(f'Prepared only: {out}',flush=True)
        return
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status=result.returncode,seconds=time.monotonic()-start)
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert result.returncode==0,'Preserve failure; no automatic retry.'

if __name__=='__main__':main()
