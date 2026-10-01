#!/usr/bin/env python3
"""Verify exact definition movement and all untouched production/header files."""
from pathlib import Path
import hashlib,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
a=(e/'manager-before.cc').read_text();b=(repo/'source/reconstructed_fault/manager.cc').read_text();m=(repo/'source/reconstructed_fault/manager_slip_rate.cc').read_text()
start=a.index('  // -----------------------------------------------------------------------------\n  // Slip-rate nonlinear state')
end=a.index('  template <int dim>\n  void\n  ReconstructedFaultManager<dim>::rebuild_after_deserialization()',start)
marker='  // -----------------------------------------------------------------------------\n  // Restart reconstruction and serialization support\n  // -----------------------------------------------------------------------------\n\n'
block=a[start:end].replace(marker,'')
names=re.findall(r'ReconstructedFaultManager<dim>::(\w+)\(',block)
checks={'16-complete-definitions':len(names)==16, 'byte-exact-moved-block':block in m,
        'remaining-manager-exact':b==a[:start]+marker+a[end:],
        'moved-definitions-absent-from-manager':all(f'ReconstructedFaultManager<dim>::{n}(' not in b for n in names),
        'one-member-instantiation-each':all(len(re.findall(r'template [^\n]*::'+n+r'\(',m))==1 for n in names)}
manifest=json.loads((e/'reference-source-hashes.json').read_text())
checks['unchanged-headers-callers-tests-and-other-source']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest.items() if p!='source/reconstructed_fault/manager.cc') and subprocess.check_output(['git','diff','HEAD','--','tests'],cwd=repo)==b''
checks['unchanged-CMake']=subprocess.check_output(['git','diff','HEAD','--','CMakeLists.txt'],cwd=repo)==b''
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');(e/'moved-methods.json').write_text(json.dumps(names,indent=2)+'\n')
print(json.dumps(checks,indent=2));assert all(checks.values())
