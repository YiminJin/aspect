"""Four tiny actual-transfer tests: DWA/least-squares on one/two ranks."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    template=Path(__file__).with_name('stress_transfer_test.prm').read_text()
    results={}; fields={}
    for ranks in (1,2):
        for scheme in ('distance weighted average','linear least squares'):
            case=root/(('dwa' if scheme.startswith('distance') else 'ls')+str(ranks));case.mkdir()
            prm=template.replace('./libbp5_stress_cycle.release.so',str(args.library.resolve()))
            prm=prm.replace('set Interpolation scheme = distance weighted average','set Interpolation scheme = '+scheme)
            (case/'run.prm').write_text(prm)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
            env['ASPECT_STRESS_CYCLE_TRACE']='1'
            env['ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION']='1'
            if scheme.startswith('linear'):env['STRESS_TEST_EXPECT_AFFINE']='1'
            else:env.pop('STRESS_TEST_EXPECT_AFFINE',None)
            with (case/'run.log').open('x') as log:
                proc=subprocess.run(['timeout','120','mpirun','-np',str(ranks),str(args.binary.resolve()),'run.prm'],
                                    cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:raise SystemExit(f'Failed {case}: {proc.returncode}')
            results[case.name]=next(csv.DictReader((case/'output/transfer_test.csv').open()))
            assert int(results[case.name]['step'])==1 and float(results[case.name]['dt'])==1
            values={}
            for file in (case/'output').glob('stress_transfer_*_rank*.csv'):
                for r in csv.reader(file.open()):
                    if r[0]=='published_FE':
                        key=tuple(r[i] for i in (1,4,5,6))
                        assert key not in values
                        values[key]=float(r[-1])
            assert values
            fields[case.name]=values
            print(case.name,results[case.name],flush=True)
    for scheme in ('dwa','ls'):
        a,b=fields[scheme+'1'],fields[scheme+'2'];assert a.keys()==b.keys()
        difference=max(abs(a[k]-b[k]) for k in a)
        assert difference<1e-11
        results[scheme+'_MPI_max_difference']=difference
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
