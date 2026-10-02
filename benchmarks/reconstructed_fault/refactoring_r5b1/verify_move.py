#!/usr/bin/env python3
"""Check exact backend/record movement and unchanged lifecycle/ownership."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
a=(e/'surface_system.cc').read_text();ranges=json.loads((e/'move-ranges.json').read_text())
a0,a1=ranges['record'];b0,b1=ranges['particle'];c0,c1=ranges['bulk']
record=a[a0:a1];original=a[b0:b1];branch='    if (bulk_work_measure)\n      return assemble_bulk_work_system(bulk_state, slip_rate, assemble_jacobian);\n'
particle=original.replace('::assemble_surface_system(','::assemble_particle_system(',1).replace(branch,'',1)
dispatch=original[:original.index(branch)]+branch+'    return assemble_particle_system(bulk_state, slip_rate, assemble_jacobian);\n  }\n\n\n'
remaining=(a[:a0]+dispatch+a[b1:c0]+a[c1:]).replace('#include "surface_direct_internal.h"','#include "surface_system_internal.h"\n#include "surface_direct_internal.h"',1)
def src(name):return (repo/'source/reconstructed_fault'/name).read_text()
checks={'particle-body-byte-exact':particle in src('surface_system_particle.cc'),'bulk-method-byte-exact':a[c0:c1] in src('surface_system_bulk_work.cc'),
 'record-byte-exact-private':record in src('surface_system_internal.h'),'remaining-lifecycle-exact':remaining==src('surface_system.cc')}
h=(e/'surface_system.h').read_text();i=h.index('      SurfaceAssembly\n      assemble_bulk_work_system')
addition='      SurfaceAssembly\n      assemble_particle_system(const LinearAlgebra::BlockVector &bulk_state,\n                               const FaultVector &slip_rate,\n                               const bool assemble_jacobian) const;\n\n'
expected_header=(h[:i]+addition+h[i:]).replace('#include <deal.II/base/timer.h>','#include <deal.II/base/timer.h>\n#include <deal.II/particles/property_pool.h>',1)
checks['header-private-declaration-and-direct-type-include']=(repo/'include/aspect/reconstructed_fault/surface_system.h').read_text()==expected_header
manifest=json.loads((e/'r5a2-candidate-source-hashes.json').read_text());exceptions={'source/reconstructed_fault/surface_system.cc','include/aspect/reconstructed_fault/surface_system.h'}
checks['other-source-header-units-unchanged']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==v for p,v in manifest.items() if p not in exceptions)
def unity(build):return {p.name:p.read_text() for p in (repo/build/'CMakeFiles/aspect.exe.release.dir/Unity').glob('*.cxx')}
checks['unity-groups-unchanged']=unity('build-refactor-r5a2')==unity('build-refactor-r5b1')
checks['single-assembly-record-definition']=sum('struct ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly' in p.read_text() for p in (repo/'source/reconstructed_fault').glob('*') if p.is_file())==1
(e/'source-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
