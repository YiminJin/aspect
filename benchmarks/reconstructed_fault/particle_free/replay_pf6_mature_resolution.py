#!/usr/bin/env python3
"""Verify and replay the bounded mature resolution packet without ASPECT/MPI."""
import argparse,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    provenance=json.loads((a.evidence/'provenance.json').read_text());archive=a.evidence/provenance['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==provenance['sha256']
    manifest=json.loads((a.evidence/'archive-manifest.json').read_text());a.destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as t:
        assert set(t.getnames())==set(manifest)
        for entry in t:
            assert entry.isfile() and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
            data=t.extractfile(entry).read();expected=manifest[entry.name]
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
            dest=a.destination/entry.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    scripts=a.destination/'source_snapshot/benchmarks/reconstructed_fault/particle_free'
    study=a.destination/'study';env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    checks=[]
    for script,directory,reference in [('analyze_pf6_mature_resolution.py',study/'runs','measured-summary.json'),
                                      ('verify_pf6_mature_resolution.py',study,'independent-verification.json')]:
        output=a.destination/('replayed-'+reference)
        subprocess.run([sys.executable,str(scripts/script),str(directory),'--output',str(output)],env=env,check=True)
        assert json.loads(output.read_text())==json.loads((a.evidence/reference).read_text()),reference
        checks.append(reference)
    baseline_files=0
    for label in ['dg-cc','dg-ff']:
        current=json.loads((study/'runs'/label/'record.json').read_text())
        retained=json.loads((a.destination/'retained'/(label+'-record.json')).read_text())
        for prefix in ['coupled-profile','authoritative-stress','dg-coefficients','physical-fe']:
            for step in range(4):
                name=f'{prefix}-{step}.csv'
                assert current['csv_hashes'][name]==retained['csv_hashes'][name],(label,name)
                baseline_files+=1
    default=json.loads((a.destination/'retained/particle-default-regression-record.json').read_text())
    for step in [0,1]:
        name=f'coupled-profile-{step}.csv'
        assert hashlib.sha256((study/'runs/particle-default-regression/run'/name).read_bytes()).hexdigest()==default['csv_hashes'][name]
    for label in ['dg-cc','dg-fc-final','dg-ff','particle-cc','particle-fc','particle-ff']:
        output=a.destination/(label+'-replayed-tractions.csv')
        subprocess.run([sys.executable,str(scripts/'export_pf6_mature_tractions.py'),
                        str(study/'runs'/label/'run/common-stress-3.csv'),str(output)],env=env,check=True)
        assert output.read_bytes()==(study/'derived'/(label+'-raw-tractions.csv')).read_bytes()
    print(json.dumps(dict(archive_members=len(manifest),exact_replayed_results=checks,
                         retained_field_hashes=baseline_files,raw_traction_exports=6,ASPECT_launched=False)))
if __name__=='__main__':main()
