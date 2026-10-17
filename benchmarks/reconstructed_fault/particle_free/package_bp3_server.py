#!/usr/bin/env python3
"""Assemble a reviewable source distribution. Packaging only; never builds or runs jobs."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[3]
TEMPLATE=ROOT/'benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,data): p.write_text(json.dumps(data,indent=2)+'\n')


def select_sources():
    roots=['source','include','cmake','unit_tests','tests','data','contrib','doc',
           'benchmarks/reconstructed_fault/bp3_native','benchmarks/reconstructed_fault/bp3/plugin',
           'benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event']
    paths=set(subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard','-z',*roots],cwd=ROOT).decode().split('\0'))
    paths.update(p.name for p in ROOT.iterdir() if p.is_file() and not p.name.startswith('.'))
    # Include the focused regression's local helper imports recursively.
    helpers=ROOT/'benchmarks/reconstructed_fault/particle_free'; pending=['run_pf6_mature','analyze_pf6_mature_pilot']
    seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);p=helpers/(name+'.py')
        if not p.is_file():continue
        paths.add(str(p.relative_to(ROOT)))
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module and (helpers/(node.module+'.py')).is_file():pending.append(node.module)
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    selected=[];omitted=[]
    for name in sorted(paths-{''}):
        p=ROOT/name
        skip=p.is_symlink() or not p.is_file() or any(x in p.parts for x in ('__pycache__','evidence','results'))
        # Preserve legitimate compressed standard data and bundled CMake input.
        if name.startswith('benchmarks/') and (p.suffix in ('.so','.gz','.xz','.pdf','.png') or 'build' in p.parts):skip=True
        if skip:omitted.append(name);continue
        assert p.read_bytes()[:4]!=b'\x7fELF',name
        selected.append(name)
    return selected,omitted


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--qualification',type=Path,required=True,help='Prepared compact review evidence; copied without launching anything')
    a.add_argument('--name',default='bp3-dg-server-startup-20261017-r2')
    args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    dest=out/'distribution';dest.mkdir(exist_ok=False)
    selected,omitted=select_sources();source=dest/'aspect-source'
    for name in selected:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copytree(TEMPLATE,dest/'bp3_dg',ignore=shutil.ignore_patterns('build','*.so','aspect-release','__pycache__'))
    shutil.copytree(args.qualification,dest/'qualification')
    for name in ['README.md','ENVIRONMENT.md','VERIFICATION.md','RESOURCES.md']:shutil.copy2(TEMPLATE/name,dest/name)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    dump(dest/'source-selection.json',dict(head=head,branch=branch,source_files=len(selected),source_bytes=sum((ROOT/p).stat().st_size for p in selected),selected=selected,omitted=omitted))
    (dest/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
    dump(dest/'PROVENANCE.json',dict(base_head=head,branch=branch,source_state='Actual full current snapshot plus preserved incoming work; not HEAD alone.',
        user_selected='Collective-stop repair and prepared bounded server startup; no server launch.',
        verification=json.loads((dest/'qualification/verification.json').read_text()),strict_reference_match_passed=False))
    (dest/'PROVENANCE.md').write_text('Full current source snapshot. source.patch is an informational diff against base HEAD, already applied. Do not apply it again.\nNo workstation binary, build cache, old output/checkpoint or runtime Python controller is shipped.\nLocal qualification, prepared full-grid startup and the unqualified first-event experiment are distinct.\nSee VERIFICATION.md, RESOURCES.md, qualification/ and per-file checksums.\n')
    manifest={}
    for p in sorted(dest.rglob('*')):
        if p.is_file():manifest[str(p.relative_to(dest))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    dump(dest/'FILE_MANIFEST.json',manifest)
    (dest/'SHA256SUMS').write_text(''.join(f"{entry['sha256']}  {name}\n" for name,entry in manifest.items())+f"{sha(dest/'FILE_MANIFEST.json')}  FILE_MANIFEST.json\n")
    archive=out/(args.name+'.tar.gz')
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tar:
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            info=tarfile.TarInfo(args.name+'/'+str(p.relative_to(dest)));info.size=p.stat().st_size;info.mode=0o755 if p.stat().st_mode & 0o111 else 0o644;info.mtime=0
            with p.open('rb') as stream:tar.addfile(info,stream)
    digest=sha(archive);(out/(archive.name+'.sha256')).write_text(digest+'  '+archive.name+'\n')
    actual={};elf=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            name=str(Path(member.name).relative_to(args.name));data=tar.extractfile(member).read()
            if data[:4]==b'\x7fELF':elf.append(name)
            actual[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    expected=dict(manifest)
    for name in ['FILE_MANIFEST.json','SHA256SUMS']:expected[name]=dict(bytes=(dest/name).stat().st_size,sha256=sha(dest/name))
    assert actual==expected and not elf
    result=dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,regular_members=len(actual),hashed_payload_files=len(manifest),source_files=len(selected),all_archive_bytes_verified=True,no_elf=True)
    dump(out/'archive-verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
