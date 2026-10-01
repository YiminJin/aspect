#!/usr/bin/env python3
"""Check extraction tokens/order and all protected reference artifacts."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent;e=root/'evidence'
old=(e/'driver-before.cc').read_text();new=Path('source/simulator/solver/reconstructed_fault_stokes.cc').read_text()
compact=lambda s:re.sub(r'\s+','',s)
a=old.index('        // Bound the continuity');b=old.index('        auto solve_condensed_system =',a);end=old.index('\n        for (nonlinear_iteration',b)
start=old.index('          direction = 0.0;',b);finish=old.rindex('        };',b,end)
pressure=old[a:b];body=old[start:finish]
fields={'initial_residual.bulk_norm':'scales.initial_bulk_residual','aspect_bulk_reference':'scales.aspect_bulk_reference','bulk_scale':'scales.bulk_scale','bulk_precision':'scales.bulk_precision','bulk_convergence_scale':'scales.bulk_convergence_scale'}
for x,y in fields.items():body=re.sub(r'\b'+re.escape(x)+r'\b',y,body)
na=new.index('    // Bound the continuity');nb=new.index('    direction = 0.0;',na);ne=new.index('\n  }\n',nb)
checks={'pressure-assembly-unchanged':compact(pressure)==compact(new[na:nb]),'linear-solve-unchanged-except-explicit-scales':compact(body)==compact(new[nb:ne])}
method='  template <int dim>\n  void\n  Simulator<dim>::solve_reconstructed_fault_stokes ()'
old_driver=old[old.index(method):old.index('\n}\n',old.index(method))].replace(old[a:end],'')
new_driver=new[new.index(method):new.index('\n}\n',new.index(method))]
callstart=new_driver.index('                const ReconstructedFaultLinearSolveScales linear_solve_scales{')
callend=new_driver.index('                  total_fault_krylov_iterations);',callstart)+len('                  total_fault_krylov_iterations);')
new_driver=new_driver[:callstart]+'                solve_condensed_system(*linearization, active_set, bulk_rhs, bulk_direction, already_converged);'+new_driver[callend:]
checks['driver-otherwise-identical']=old_driver==new_driver
header=Path('include/aspect/simulator.h').read_text()
header=header.replace('#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>\n','')
ha=header.index('      /** Related residual scales supplied to one condensed solve. */')
hb=header.index('        unsigned int &total_fault_krylov_iterations);',ha)+len('        unsigned int &total_fault_krylov_iterations);\n\n')
header=header[:ha]+header[hb:]
checks['header-only-private-declarations-and-include']=header==(e/'simulator-before.h').read_text()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((e/'entry-source-hashes.json').read_text());changed=[p for p,v in m.items() if h(Path(p))!=v]
checks['only-selected-production-files']=set(changed)=={'include/aspect/simulator.h','source/simulator/solver/reconstructed_fault_stokes.cc'}
p=json.loads((e/'protected-hashes.json').read_text());checks['reference-and-local-files-preserved']=all(Path(f).exists() and h(Path(f))==v for f,v in p.items())
(e/'source-verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2));assert all(checks.values())
