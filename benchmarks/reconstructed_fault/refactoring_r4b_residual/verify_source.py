#!/usr/bin/env python3
"""Check the second extraction against the accepted first-subpass snapshots."""
from pathlib import Path
import difflib, hashlib, json, re

root = Path(__file__).resolve().parent
repo = root.parents[2]
e = root / 'evidence'
old = (e / 'before-reconstructed_fault_stokes.cc').read_text()
new = (repo / 'source/simulator/solver/reconstructed_fault_stokes.cc').read_text()
compact = lambda s: re.sub(r'\s+', '', s)
start = old.index('        auto evaluate_coupled_residual =')
body_start = old.index('          // Use the very same absolute V', start)
body_end = old.index('\n        };', body_start)
end = body_end + len('\n        };\n\n')
new_start = new.index('    // Use the very same absolute V')
new_end = new.index('\n  }\n', new_start)
checks = {
    'residual-body-and-operation-order-identical':
        compact(old[body_start:body_end].replace('CoupledResidual', 'ReconstructedFaultCoupledResidual'))
        == compact(new[new_start:new_end])
}
struct_start = old.index('    struct CoupledResidual')
struct_end = old.index('    };\n\n', struct_start) + len('    };\n\n')
new_struct_start = new.index('  struct Simulator<dim>::ReconstructedFaultCoupledResidual')
new_struct_end = new.index('  };', new_struct_start)
checks['result-fields-identical'] = (
    compact(old[old.index('{', struct_start):struct_end])
    == compact(new[new.index('{', new_struct_start):new_struct_end + len('  };')]))
method = '  template <int dim>\n  void\n  Simulator<dim>::solve_reconstructed_fault_stokes ()'
driver_start = old.index(method)
driver = old[driver_start:old.index('\n}\n', driver_start)]
driver = driver.replace(old[start:end], '').replace(old[struct_start:struct_end], '')
driver = driver.replace('CoupledResidual', 'ReconstructedFaultCoupledResidual')
driver = driver.replace('evaluate_coupled_residual(', 'evaluate_reconstructed_fault_coupled_residual(')
new_driver_start = new.index(method)
new_driver = new[new_driver_start:new.index('\n}\n', new_driver_start)]
checks['driver-otherwise-byte-identical'] = driver == new_driver
checks['all-five-calls-retained'] = (
    driver.count('evaluate_reconstructed_fault_coupled_residual(') == 5
    and new_driver.count('evaluate_reconstructed_fault_coupled_residual(') == 5)
linear_start = '  template <int dim>\n  struct Simulator<dim>::ReconstructedFaultLinearSolveScales'
checks['accepted-condensed-solve-byte-identical'] = (
    old[old.index(linear_start):driver_start] == new[new.index(linear_start):new_driver_start])
header = (repo / 'include/aspect/simulator.h').read_text()
ha = header.index('      /** Bulk norm and surface residual returned by one coupled evaluation. */')
hb = header.index('      /** Related residual scales supplied to one condensed solve. */', ha)
checks['header-only-private-declarations'] = (
    header[:ha] + header[hb:] == (e / 'before-simulator.h').read_text())
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m = json.loads((e / 'entry-source-hashes.json').read_text())
changed = [p for p, v in m.items() if h(repo / p) != v]
checks['only-selected-production-files'] = set(changed) == {
    'include/aspect/simulator.h', 'source/simulator/solver/reconstructed_fault_stokes.cc'}
p = json.loads((e / 'protected-hashes.json').read_text())
checks['accepted-reference-and-local-files-preserved'] = all(
    (repo / f).exists() and h(repo / f) == v for f, v in p.items())
patch = ''
for path, before in [('source/simulator/solver/reconstructed_fault_stokes.cc', old),
                     ('include/aspect/simulator.h', (e / 'before-simulator.h').read_text())]:
    patch += ''.join(difflib.unified_diff(before.splitlines(True), (repo / path).read_text().splitlines(True),
                                        fromfile='accepted/' + path, tofile='candidate/' + path))
(e / 'second-subpass-source.patch').write_text(patch)
(e / 'source-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
assert all(checks.values())
