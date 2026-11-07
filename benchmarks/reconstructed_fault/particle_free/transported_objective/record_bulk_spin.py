#!/usr/bin/env python3
"""Record preserved baselines and exact local source/binary/input identities."""
from pathlib import Path
import hashlib,json,subprocess,datetime,os
ROOT=Path(__file__).resolve().parents[4];B=ROOT/'build-particle-free/transported-objective/stage-c';OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    entry=json.loads((OUT/'entry.json').read_text());preserved={}
    for name,expected in entry['binaries'].items():
        actual=sha(ROOT/name);assert actual==expected,(name,actual,expected);preserved[name]=actual
    for name,expected in entry['source_hashes'].items():
        copy=B/'bulk-spin/baseline-source'/name
        assert copy.is_file() and sha(copy)==expected,name
    paths={ROOT/p for p in entry['source_hashes']}
    paths.update((ROOT/'doc/reconstructed_fault/particle_free/mature_moving_fault').glob('*.md'))
    paths.update(Path(__file__).parent.glob('*bulk_spin*.py'))
    paths.update((Path(__file__).parent/'native').glob('bulk_spin*.h'))
    paths.update((Path(__file__).parent/'bulk_spin_server').glob('*'))
    paths.add(ROOT/'doc/reconstructed_fault/particle_free/BULK_SPIN_CORRECTION_REPORT.md')
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}
    names=set(json.loads((OUT/'coupled.json').read_text())['cases'])
    names.update(['bulk-old-total-control','bulk-cs-restart','bulk-cd','bulk-cd2'])
    for file in OUT.glob('lifecycle-*.json'):
        if file.name=='lifecycle-comparison.json':continue
        names.update(r['case'] for r in json.loads(file.read_text()))
    names.update(p.name for p in (B/'material-runs').glob('bulk-prescribed*'))
    runs={}
    for name in sorted(names):
        folder=B/'material-runs'/name
        if not (folder/'command.json').exists():continue
        command=json.loads((folder/'command.json').read_text())
        files=[folder/'command.json',folder/'input.prm',folder/'fault.txt',folder/'run.log',*sorted((folder/'output').glob('bulk-spin*.csv')),*sorted((folder/'output').glob('material-*.csv'))]
        runs[name]=dict(exit_code=command['exit_code'],command=command['command'],core_sha256=command['core_sha256'],checker_sha256=command['plugin_sha256'],files={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()})
    executable=B/'bulk-spin/aspect-release';checker=B/'bulk-spin/checks/libtransported_bulk_spin.release.so'
    manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(ROOT),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),preserved=preserved,baseline_source_verified=True,source=source,final_binary={str(p.relative_to(ROOT)):sha(p) for p in [executable,checker]},runs=runs)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'final.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/opt/openmpi/5.0.6/lib'
    (OUT/'runtime-libraries.txt').write_text(subprocess.check_output(['ldd',str(executable)],env=env,text=True))
    print(json.dumps(dict(preserved=len(preserved),source_files=len(source),recorded_runs=len(runs),final_binary=manifest['final_binary'])))
