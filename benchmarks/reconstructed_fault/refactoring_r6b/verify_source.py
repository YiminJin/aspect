#!/usr/bin/env python3
"""Verify the bounded substitution and protection of the accepted source tree."""
from pathlib import Path
import hashlib,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';checks={}
ref='1a3b57eda';driver='source/simulator/solver/reconstructed_fault_stokes.cc'
old=subprocess.check_output(['git','show',ref+':'+driver],cwd=repo,text=True)
new=(repo/driver).read_text();restored=new.replace('#include "reconstructed_fault_bound_diagnostics.h"\n','')
oldstarts=['                  {\n                    out.open(', '                        out << nonlinear_iteration', '                pcout << "      Fault bound audit:']
newstarts=['                  internal::open_fault_bound_diagnostic(', '                        internal::write_fault_bound_diagnostic(', '                internal::report_fault_bound_diagnostic(']
ends=['                for (unsigned int f = 0; f < slip_rate.size(); ++f)', '                    }\n', '              }\n']
blocks=[]
for a,b,end in zip(oldstarts,newstarts,ends):
 original=old[old.index(a):old.index(end,old.index(a))]
 replacement=restored[restored.index(b):restored.index(end,restored.index(b))]
 restored=restored.replace(replacement,original,1);blocks.append(original)
checks['surrounding-driver-byte-exact']=restored==old
helper=(repo/'source/simulator/solver/reconstructed_fault_bound_diagnostics.cc').read_text()
def compact(s):return re.sub(r'\s+','',s)
# Compare actual stream bodies after substituting only the explicitly passed inputs.
a=helper.index('      out.open(');b=helper.index('\n    }',a)
checks['open-expressions']=compact(helper[a:b].replace('dealii::',''))==compact(blocks[0].strip()[1:-1].replace('parameters.output_directory','output_directory'))
a=helper.index('      out << nonlinear_iteration');b=helper.index('\n    }',a)
s=blocks[1]
for x,y in [('slip_rate_direction[f][v]','slip_rate_direction'),('(active_set[f][v] && !prescribed[f][v])','lower_active'),('slip_rate[f][v]','slip_rate'),('prescribed[f][v]','prescribed'),("<< f <<",'<< fault <<'),("<< v <<",'<< vertex <<')]:s=s.replace(x,y)
checks['row-expressions']=compact(helper[a:b])==compact(s)
a=helper.index('      pcout <<');b=helper.index('\n    }',a)
checks['summary-expressions']=compact(helper[a:b])==compact(blocks[2])
checks['no-helper-MPI-or-numerical-access']=not any(x in helper for x in ('MPI','getenv','evaluate_','Manager','Simulator','Timer','setstate','exceptions('))
allowed={driver,'CMakeLists.txt'}
hashes=json.loads((e/'reference-hashes.json').read_text())
checks['accepted-source-and-local-edits-preserved']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in hashes.items() if p not in allowed)
cmake=(repo/'CMakeLists.txt').read_text().replace('  source/simulator/solver/reconstructed_fault_bound_diagnostics.cc\n','')
checks['only-focused-build-exclusion']=cmake==subprocess.check_output(['git','show',ref+':CMakeLists.txt'],cwd=repo,text=True)
a=repo/'build-refactor-r6a/CMakeFiles/aspect.exe.release.dir/Unity';b=repo/'build-refactor-r6b/CMakeFiles/aspect.exe.release.dir/Unity'
checks['unity-groups-preserved']={p.name:p.read_bytes() for p in a.glob('*.cxx')}=={p.name:p.read_bytes() for p in b.glob('*.cxx')}
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
