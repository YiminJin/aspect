"""Small real checkpoint tests; never loads BP5 data or launches BP5 mechanics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('build',type=Path)
    p.add_argument('--ranks',type=int,default=1)
    args=p.parse_args();binary=args.binary.resolve();build=args.build.resolve()
    root=Path(tempfile.mkdtemp(prefix=f'bp5-restart-clock-{args.ranks}-'))
    print(root,flush=True)
    base=Path(__file__).with_suffix('.prm').read_text()
    libraries=f'set Additional shared libraries = {build}/libtest_restart_clock.release.so, {build}/libbp5_normal_stress_diagnostic.release.so\n'
    records=[]

    def run(name,extra,ok=True,needle=None):
        directory=root/name;directory.mkdir(exist_ok=True)
        path=root/f'{name}.prm';path.write_text(base+libraries+f'set Output directory = {directory}\n'+extra)
        result=subprocess.run(['mpirun','-np',str(args.ranks),str(binary),str(path)],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        (root/f'{name}.log').write_text(result.stdout)
        if (result.returncode==0)!=ok or (needle and needle not in result.stdout):
            raise RuntimeError(f'{name}: unexpected result; see {root}/{name}.log')
        records.append(dict(name=name,status=result.returncode,expected_success=ok))
        print(f'{name}: expected result',flush=True)
        return directory

    fresh=run('fresh','')
    checkpoint=fresh/'restart'
    hashes={str(f.relative_to(checkpoint)):hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoint.rglob('*') if f.is_file()}

    def branch(name,mode,end,extra='',ok=True,needle=None):
        directory=root/name;directory.mkdir()
        shutil.copytree(checkpoint,directory/'restart')
        return run(name,f'set Resume computation = true\nset End time = {end}\nsubsection Postprocess\n subsection Restart clock test\n  set Mode = {mode}\n end\nend\n'+extra,ok,needle)

    message='incoming vectors, old dt and step unchanged; pending clock verified'
    branch('keep','keep',2,needle=message)
    branch('half','half',1.5,needle=message)
    for mode in ('zero','increase','nonfinite'):
        branch(mode,mode,2,ok=False,needle='A restart timestep override must be finite, positive')
    if args.ranks==2:
        branch('rank-disagreement','rank disagreement',2,ok=False,needle='Restart timestep overrides disagree')

    # Exercise B's actual C++ CSV parser, runtime hook, controller and guard on
    # the tiny bulk checkpoint, without instantiating the BP5 fault observer.
    reference=root/'A.csv'
    reference.write_text('step,time_s,dt\n2,2,1\n3,3,1\n4,4,1\n5,5,1\n')
    extra=f'''subsection Time stepping
 set List of model names = convection time step, BP5 recorded half steps
 subsection BP5 recorded half steps
  set Reference trajectory file = {reference}
 end
end
subsection Postprocess
 subsection BP5 normal diagnostic
  set Checkpoint accepted step = 1
  set Checkpoint physical time = 1
  set New accepted steps = 8
 end
end
'''
    b=branch('recorded','external',5,extra,needle=message)
    rows=list(csv.reader((b/'clock_test.csv').open()))
    assert len(rows)==8
    for i,row in enumerate(rows):assert (int(row[0]),float(row[1]),float(row[2]))==(i+2,1.5+.5*i,.5)
    shorter=extra.replace('convection time step, BP5 recorded half steps','convection time step, BP5 recorded half steps, function')
    shorter+='subsection Time stepping\n subsection Function\n  set Function expression = .25\n end\nend\n'
    failed=branch('safety-shortened','external',5,shorter,ok=False,needle='Half-step clock changed before mechanics')
    assert len(list(csv.reader((failed/'clock_test.csv').open())))==1
    for name,digest in hashes.items():assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==digest
    (root/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    print('All expected outcomes passed; original checkpoint hashes unchanged.',flush=True)


if __name__=='__main__':main()
