#!/usr/bin/env python3
"""Check only the authorized diagnostic boundary changed; preserve baseline artifacts."""
from pathlib import Path
import hashlib,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
a=(e/'history-before.cc').read_text();b=(repo/'source/material_model/phase_field_fault/history.cc').read_text();c=(repo/'source/material_model/phase_field_fault/history_diagnostics.cc').read_text();checks={}
def region(s,start,end):return s[s.index(start):s.index(end,s.index(start))]
old_setup=region(a,'      if (std::getenv("ASPECT_STRESS_CYCLE_TRACE"))','      // First evaluate')
new_setup=region(b,'      if (std::getenv("ASPECT_STRESS_CYCLE_TRACE"))','      // First evaluate')
old_rows=region(a,'              if (stress_cycle_audit.is_open()','              ++particle_index;')
new_rows=region(b,'              if (diagnostics.selects_stress_cycle','              ++particle_index;')
restored=b.replace(new_setup,old_setup).replace(new_rows,old_rows).replace('#include "history_diagnostics.h"\n','')
restored=restored.replace('#include <chrono>\n','#include <fstream>\n#include <iomanip>\n#include <chrono>\n#include <set>\n')
restored=restored.replace('      aspect::internal::FaultHistoryDiagnostics<dim> diagnostics;','      std::ofstream source_history_audit;\n      std::set<std::string> trace_cells;\n      std::ofstream stress_cycle_audit;')
restored=restored.replace('      auto &diagnostics = candidates.diagnostics;','      auto &source_history_audit = candidates.source_history_audit;\n      auto &trace_cells = candidates.trace_cells;\n      auto &stress_cycle_audit = candidates.stress_cycle_audit;')
checks['all-surrounding-history-byte-exact']=restored==a
mapping={'this->get_output_directory()':'output_directory','this->get_mpi_communicator()':'communicator','this->get_timestep_number()':'step','this->get_time()':'time','bulk_coefficients.beta':'beta','bulk_coefficients.eta_ve':'eta_ve','candidate.stress':'candidate_stress','points[particle_index]':'sample_point','velocity_gradients[particle_index]':'gradient','phase_fields[particle_index]':'phase_field'}
for method,original,begin,end in [('open_stress_cycle',old_setup,'          const auto rank=','\n        }'),('open_source_history',old_setup,'          source_history_audit.open','\n        }'),('record_stress_cycle',old_rows,'                  const auto x=','\n                }'),('record_source_history',old_rows,'                  const auto position=','\n                }')]:
 old=region(original,begin,end).replace('                  const auto &gradient=velocity_gradients[particle_index];\n','')
 for x,y in mapping.items():old=old.replace(x,y)
 start=c.index('    {\n',c.index('::'+method+'('))+6;new=c[start:c.index('\n    }',start)]
 checks[method+'/unchanged-expressions']=re.sub(r'\s+','',old)==re.sub(r'\s+','',new)
 checks[method+'/exact-string-literals']=re.findall(r'"(?:\\.|[^"\\])*"',old)==re.findall(r'"(?:\\.|[^"\\])*"',new)
checks['lazy-guards']=('return stress_cycle_audit.is_open()\n             && trace_cells.count(particle.get_surrounding_cell()->id().to_string());' in c and 'if (diagnostics.source_history_is_open() && association.active && !associations[particle_index].active)' in new_rows and new_setup.count('std::getenv(')==2)
checks['same-fields-and-no-numerical-owner']=all(x in (repo/'source/material_model/phase_field_fault/history_diagnostics.h').read_text() for x in ('std::ofstream source_history_audit;','std::set<std::string> trace_cells;','std::ofstream stress_cycle_audit;')) and not any(x in c for x in ('MPI::sum','MPI::min','exceptions(','evaluate_','getenv','prepare_'))
allowed={'source/material_model/phase_field_fault/history.cc','CMakeLists.txt'}
for name in ('reference-hashes','r5b2-artifacts'):
 items=json.loads((e/(name+'.json')).read_text());checks[name+'/preserved']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==v for p,v in items.items() if p not in allowed)
before=subprocess.check_output(['git','show','HEAD:CMakeLists.txt'],text=True,cwd=repo)
checks['only-focused-build-exclusion']=(repo/'CMakeLists.txt').read_text().replace('  source/material_model/phase_field_fault/history_diagnostics.cc\n','')==before
# Verify the established unity grouping was not disturbed by source discovery.
for build in ('build-refactor-r5b2','build-refactor-r6a'):
 paths=sorted((repo/build/'CMakeFiles/aspect.exe.release.dir/Unity').glob('unity_*_cxx.cxx'))
 groups=[re.findall(r'#include "([^"]+)"',p.read_text()) for p in paths]
 if build.endswith('r5b2'):old_groups=groups
 else:checks['unity-groups-preserved']=groups==old_groups
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
