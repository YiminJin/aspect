#!/usr/bin/env python3
"""Verify/recompute bounded pilot evidence without ASPECT, MPI or evolution."""
import argparse,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    provenance=json.loads((a.evidence/'provenance.json').read_text());archive=a.evidence/provenance['archive']
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==provenance['sha256']
    manifest=json.loads((a.evidence/'archive-manifest.json').read_text());a.destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames())==set(manifest)
        for entry in tar:
            assert entry.isfile() and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
            expected=manifest[entry.name];data=tar.extractfile(entry).read()
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
            path=a.destination/entry.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    root=a.destination/'snapshot';study=root/'build-particle-free/mature-pilot';scripts=root/'benchmarks/reconstructed_fault/particle_free'
    # Reconstitute pinned artifacts as hardlinks, never execute them.
    for directory in (study/'runs',study/'current/runs'):
        for record in directory.glob('*/record.json'):
            for relative,sha in json.loads(record.read_text())['artifacts'].items():
                name=Path(relative).name;os.link(a.destination/'artifacts'/sha/name,record.parent/name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(script,*args):subprocess.run([sys.executable,str(scripts/script),*map(str,args)],cwd=root,env=env,check=True)
    result=a.destination/'recomputed-summary.json'
    run('reduce_pf6_mature_pilot.py',study,'--output',result)
    assert json.loads(result.read_text())==json.loads((study/'measured-summary.json').read_text())
    comparisons=[('mpi-2-vs-1','dg-life-r2-owned',3,'dg-life-r1-final',3,False),
      ('mpi-4-vs-1','dg-life-r4-owned',3,'dg-life-r1-final',3,False),
      ('restart-serial','dg-r1-restart-second-leg',5,'dg-r1-restart-reference',5,False),
      ('restart-1-to-4','dg-r1-to-r4-restart',5,'dg-r1-restart-reference',5,False),
      ('restart-4-to-2','dg-r4-to-r2-restart',6,'dg-r4-restart-reference',6,False),
      ('rejection-outer','dg-outer-cutback-r2-final',4,'dg-cutback-matching-time',4,False),
      ('rejection-nonlinear','dg-nonlinear-cutback-r4',4,'dg-cutback-matching-time',4,False),
      ('fine-time-half','dg-fixed-ff-h8-dt100',6,'dg-fixed-ff-h8-final',3,False),
      ('fixed-ff-dt100-final','dg-fixed-ff-h8-dt100',6,'dg-ff-dt100-final',6,True)]
    for label,first,n,second,m,independent in comparisons:
        output=a.destination/(label+'.json')
        run('analyze_pf6_mature_pilot.py',study/'current/runs'/first/'run',study/'current/runs'/second/'run',
            '--steps',n,m,'--output',output,*(['--independent'] if independent else []))
        assert json.loads(output.read_text())==json.loads((study/(label+'.json')).read_text()),label
    output=a.destination/'fixed-ff-field-verification.json'
    run('verify_pf6_mature_fixed_ff.py',root/'build-particle-free/mature-resolution/runs/dg-ff/run',
      study/'current/runs/dg-fixed-ff-h8-final/run','--output',output)
    assert json.loads(output.read_text())==json.loads((study/output.name).read_text())
    output=a.destination/'surface-independent.json'
    # Relative paths retain the original result keys as well as measured values.
    run('verify_pf6_mature_surface.py','build-particle-free/mature-pilot/surface-full-final/run',
      'build-particle-free/mature-pilot/surface-full-intended/run','--output',output)
    assert json.loads(output.read_text())==json.loads((study/output.name).read_text())
    result=dict(archive_sha256=provenance['sha256'],verified_members=len(manifest),
      exact_summary_replay=True,exact_independent_comparisons=len(comparisons),exact_Q1_overlay_replay=True,
      exact_independent_surface_initialization_replay=True,
      ASPECT_launched=False,MPI_launched=False,strict_reference_match_passed=False)
    (a.destination/'replay-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
