#!/usr/bin/env python3
"""Prove the candidate block and all retained source remain exact."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
a=(e/'surface_system.cc').read_text();b=(repo/'source/reconstructed_fault/surface_system.cc').read_text()
start='    auto candidate = std::make_unique<SurfaceLinearization>();\n'
end='    normal_diagnostic = assembled.normal_diagnostic;\n'
block=a[a.index(start):a.index(end)]
helper_start='#ifdef DEAL_II_WITH_UMFPACK\n  template <int dim>\n  std::unique_ptr<typename ReconstructedFaultSurfaceSystem<dim>::SurfaceLinearization>\n  ReconstructedFaultSurfaceSystem<dim>::prepare_surface_linearization(\n    const SurfaceAssembly &assembled)\n  {\n'
helper=helper_start+block+'    return candidate;\n  }\n#endif\n\n\n'
checks={'candidate-block-byte-exact':helper in b,
 'caller-and-rest-exact':b.replace(helper,'').replace('    auto candidate = prepare_surface_linearization(assembled);\n',block)==a}
h=(repo/'include/aspect/reconstructed_fault/surface_system.h').read_text();before=(e/'surface_system.h').read_text()
x=h.index('#ifdef DEAL_II_WITH_UMFPACK\n',h.index('    private:'))
y=h.index('      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault;',x)
checks['only-private-declaration']=h[:x]+h[y:]==before and 'prepare_surface_linearization(const SurfaceAssembly &assembled);' in h[x:y]
allowed={'source/reconstructed_fault/surface_system.cc','include/aspect/reconstructed_fault/surface_system.h'}
manifest=json.loads((e/'reference-source-hashes.json').read_text())
checks['all-other-source-tests-build-local-unchanged']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==v for p,v in manifest.items() if p not in allowed)
manifest=json.loads((e/'r5b1-executed-artifacts.json').read_text())
checks['accepted-artifacts-preserved']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==v for p,v in manifest.items())
checks['single-definition-and-call']=b.count('prepare_surface_linearization(')==2
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
