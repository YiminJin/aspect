#!/usr/bin/env python3
"""Apply ASPECT's actual indent functions to an explicit file list, at most four at once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='Explicit source paths, or @path to a JSON list')
    parser.add_argument('--workers', type=int, choices=range(1,5), default=4)
    parser.add_argument('--astyle', default='/opt/astyle/2.04/bin/astyle')
    args = parser.parse_args()
    version = subprocess.check_output([args.astyle,'--version'],stderr=subprocess.STDOUT,text=True).strip()
    assert version == 'Artistic Style Version 2.04', version
    script = (ROOT/'contrib/utilities/indent').read_text()
    names = ['indent_file','dos_to_unix','tab_to_space','remove_trailing_whitespace','ensure_single_trailing_newline']
    # Reuse the functions verbatim, including the nested-template preprocessing.
    functions = '\n'.join(re.search(r'^'+name+r'\(\)\s*\{.*?^\}',script,re.M|re.S)[0] for name in names)
    command = functions+'\n'+'\n'.join(name+' "$1"' for name in names)
    files = []
    for name in args.files:
        files.extend(json.loads(Path(name[1:]).read_text()) if name.startswith('@') else [name])
    files = sorted(set(files))
    env = os.environ.copy()
    env.update(astyle=str(Path(args.astyle).resolve()),TMPDIR='/tmp')
    def format_file(name):
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and path.suffix in ('.cc','.h'), path
        before = path.read_bytes()
        metadata = path.stat()
        subprocess.run(['bash','-e','-c',command,'indent-subset',str(path)],cwd=ROOT,env=env,check=True)
        path.chmod(path.stat().st_mode & ~0o111)
        if path.read_bytes() == before:
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(format_file,files))
    print(f'{version}: formatted {len(files)} explicit files with {args.workers} workers.')


if __name__ == '__main__':
    main()
