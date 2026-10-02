#!/usr/bin/env python3
"""Verify byte-exact movement, unchanged dependencies and unity membership."""
from pathlib import Path
import hashlib,json,re
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
a=(e/'manager-before.cc').read_text();b=(repo/'source/reconstructed_fault/manager.cc').read_text();m=(repo/'source/reconstructed_fault/manager_particle_projection.cc').read_text()
x=json.loads((e/'move-ranges.json').read_text());h0,h1,b0,b1=(x[k] for k in ('helper_start','helper_end','member_start','member_end'))
helpers=a[h0:h1];block=a[b0:b1];names=re.findall(r'ReconstructedFaultManager<dim>::(\w+)\(',block)
checks={'10-members':len(names)==10,'two-exclusive-helpers':all(n in helpers for n in ('factor_tridiagonal','solve_tridiagonal_factors')),
 'byte-exact-members':block in m,'byte-exact-helpers':helpers in m,'remaining-manager-exact':b==a[:h0]+a[h1:b0]+a[b1:],
 'no-old-member-definitions':all(f'ReconstructedFaultManager<dim>::{n}(' not in b for n in names),
 'one-member-instantiation-each':all(len(re.findall(r'template [^\n]*::'+n+r'\(',m))==1 for n in names)}
manifest=json.loads((e/'reference-candidate-source-hashes.json').read_text())
checks['unchanged-headers-other-source-unit-tests']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest.items() if p!='source/reconstructed_fault/manager.cc')
def unity(build):
 return {p.name:p.read_text() for p in (repo/build/'CMakeFiles/aspect.exe.release.dir/Unity').glob('*.cxx')}
checks['baseline-unity-groups-preserved']=unity('build-refactor-r5a1')==unity('build-refactor-r5a2')
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');(e/'moved-methods.json').write_text(json.dumps(names,indent=2)+'\n')
print(json.dumps(checks,indent=2));assert all(checks.values())
