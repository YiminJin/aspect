#!/usr/bin/env python3
"""Losslessly index completed P5c raw exports; leave native checkpoints untouched."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(work,names):
    work=work.resolve();directory=work/'archives';directory.mkdir(exist_ok=True);archive=directory/('raw-'+ '-'.join(names)+'.zip');assert not archive.exists()
    files=[]
    for name in names:
        run=work/'runs'/name
        # Only the task's completed, explicitly named runs; no checkpoints.
        assert (run/'bulk-accepted.csv').exists()
        for pattern in ['record-[0-9][0-9][0-9][0-9].bin','phase-[0-9][0-9][0-9][0-9].csv','means-[0-9][0-9][0-9][0-9].csv','phase-forces-[0-9][0-9][0-9][0-9].csv','boundary-reactions-[0-9][0-9][0-9][0-9].csv']:
            files+=sorted(run.glob(pattern))
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(manifest)
        for name,m in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
    record={'archive':str(archive.relative_to(ROOT)),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':manifest,'restore':'unzip ARCHIVE -d WORKTREE_ROOT (missing files only; inspect before overwriting)'}
    archive.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    for p in files:
        assert sha(p)==manifest[str(p.relative_to(ROOT))]['sha256'];p.unlink()
    print(json.dumps({'archive':str(archive),'files':len(files),'original_bytes':sum(m['bytes'] for m in manifest.values()),'archive_bytes':archive.stat().st_size},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('names',nargs='+');a=p.parse_args();main(a.work,a.names)
