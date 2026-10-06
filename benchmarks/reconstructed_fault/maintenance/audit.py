#!/usr/bin/env python3
"""Check retained closeout dependencies and exact protected/archived bytes."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

here=Path(__file__).resolve().parent
root=here.parent
repo=root.parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
records=list(csv.DictReader((here/'classification.tsv').open(),delimiter='\t'))
tracked=set(subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines())
changed={'benchmarks/reconstructed_fault/.gitignore','benchmarks/reconstructed_fault/bp3/README.md','benchmarks/reconstructed_fault/refactoring_r1/.gitignore'}
checks={}
checks['retained_original_bytes']=all(hashlib.sha256((repo/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in records if x['action']=='keep' and x['path'] not in changed)
retired=[x for x in records if x['action']=='untrack-keep-local']
checks['retired_local_bytes']=all(hashlib.sha256((repo/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in retired)
checks['retired_not_in_index']=all(x['path'] not in tracked for x in retired)
checks['retired_ignored']=subprocess.run(['git','check-ignore','--stdin'],cwd=repo,input='\n'.join(x['path'] for x in retired)+'\n',text=True,stdout=subprocess.PIPE,check=True).stdout.splitlines()==[x['path'] for x in retired]
checks['restored_cmake_exact']= (repo/'CMakeLists.txt').read_bytes()==subprocess.check_output(['git','show','0f66d9869:CMakeLists.txt'],cwd=repo)
checks['production_source_unchanged']=not subprocess.check_output(['git','diff','f277a53ae','--','source','include'],cwd=repo)
missing=[]
for edge in json.loads((here/'dependencies.json').read_text())['edges']:
    for relative in edge['dependencies']:
        if not (root/relative).exists():missing.append((edge['consumer'],relative))
checks['reviewed_dependencies_present']=not missing
# Every quoted local include resolvable before cleanup remains byte-preserved;
# system/deal.II includes are the compiler's responsibility.
includes=[]
syntax=[]
prm_includes=[]
for row in records:
    p=repo/row['path']
    if row['action']!='keep':continue
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p));syntax.append(row['path'])
    if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True);syntax.append(row['path'])
    if p.suffix in ['.cc','.h']:
        for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"',p.read_text(),re.M):
            q=p.parent/inc
            if q.exists():includes.append(str(q.resolve().relative_to(repo)))
    if p.suffix=='.prm':
        for inc in re.findall(r'^\s*include\s+([^\n#]+)',p.read_text(),re.M):
            expanded=inc.strip().replace('$ASPECT_SOURCE_DIR',str(repo));q=Path(expanded)
            if not q.is_absolute():
                # ASPECT launchers documented here run from the repository root.
                q=repo/q
            prm_includes.append({'consumer':row['path'],'include':expanded,'available':q.is_file()})
# This records historical deployment/generated-input gaps instead of claiming
# that grep/AST analysis proves every constructed runtime path is available.
report={'checks':checks,'local_include_edges':len(includes),'syntax_files':len(syntax),'missing_reviewed_dependencies':missing,
        'prm_include_edges':len(prm_includes),'unavailable_historical_prm_includes':[x for x in prm_includes if not x['available']],
        'retired_files':len(retired),'retired_bytes':sum(int(x['bytes']) for x in retired)}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='unavailable_historical_prm_includes'},indent=2))
assert all(checks.values())
