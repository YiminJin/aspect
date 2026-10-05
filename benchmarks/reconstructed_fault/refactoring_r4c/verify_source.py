#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';checks={}
compact=lambda s:re.sub(r'\s+','',s)
old=(e/'before-solver.cc').read_text();new=(repo/'source/simulator/solver.cc').read_text()
a=new.index('    std::unique_ptr<SchurComplementOperator>');b=new.index('    void StokesBlock::vmult',a)
helper=new[a:b]
body=helper[helper.index('      if (use_bfbt)'):helper.index('\n    }')]
expected="""if (use_bfbt)
 return std::make_unique<WeightedBFBT<LinearAlgebra::PreconditionBase>>(
 pressure_matrix, pressure_preconditioner, solver_tolerance,
 inverse_lumped_mass_matrix.block(velocity_block_index), system_matrix);
 else
 return std::make_unique<InverseWeightedMassMatrix<LinearAlgebra::PreconditionBase>>(
 pressure_matrix, pressure_preconditioner, solver_tolerance);"""
checks['constructor-expressions-and-lazy-bfbt-access']=compact(body)==compact(expected)
new=new[:a]+new[b:]
for name,current,before,indent,pressure,velocity,terminator in [
 ('ordinary',new,old,'        ','pressure_block_index,pressure_block_index','velocity_block_index','        // create a cheap preconditioner'),
 ('coupled',(repo/'source/simulator/solver/reconstructed_fault_stokes.cc').read_text(),(e/'before-reconstructed_fault_stokes.cc').read_text(),'    ','1,1','0','    const auto solve_with_velocity_preconditioner')]:
 a=before.index(indent+'std::unique_ptr<internal::SchurComplementOperator> schur;');b=before.index(terminator,a)
 c=current.index(indent+'const auto schur = internal::make_stokes_schur_preconditioner(');d=current.index(terminator,c)
 call=f"""const auto schur = internal::make_stokes_schur_preconditioner(
 parameters.use_bfbt, system_preconditioner_matrix.block({pressure}),
 *Mp_preconditioner, parameters.linear_solver_S_block_tolerance,
 inverse_lumped_mass_matrix, {velocity}, system_matrix);"""
 checks[name+'-arguments-unchanged']=compact(current[c:d])==compact(call)
 checks[name+'-otherwise-byte-identical']=current[:c]+before[a:b]+current[d:]==before
header=(repo/'include/aspect/simulator/solver/stokes_operators.h').read_text()
a=header.index('    /**\n     * Construct the selected Schur wrapper');b=header.index('    /**\n     * This class approximates',a)
header=(header[:a]+header[b:]).replace('\n#include <memory>\n','')
checks['header-only-construction-declaration']=header==(e/'before-stokes_operators.h').read_text()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((e/'entry-source-hashes.json').read_text());changed={p for p,v in m.items() if h(repo/p)!=v}
checks['only-three-selected-production-files']=changed=={'source/simulator/solver.cc','source/simulator/solver/reconstructed_fault_stokes.cc','source/simulator/solver/stokes_operators.h'}
checks['simulator-header-and-melt-restriction-unchanged']=all(h(repo/p)==m[p] for p in ['include/aspect/simulator.h','source/simulator/solver/reconstructed_fault_condensed_system.cc','source/simulator/core.cc'])
m=json.loads((e/'protected-hashes.json').read_text());checks['reference-and-local-edits-preserved']=all(h(repo/p)==v for p,v in m.items())
(e/'source-verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2));assert all(checks.values())
