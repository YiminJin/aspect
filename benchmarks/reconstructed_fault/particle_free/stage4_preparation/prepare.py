#!/usr/bin/env python3
"""Separate checker for one zero-clock loaded-seed preparation."""
from pathlib import Path
import hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'build-particle-free/stage4-near-tip';BUILD=ROOT/'build-particle-free/stage4-preparation'
BUILD.mkdir(exist_ok=True)
source=OLD/'tested/checks.cc';text=source.read_text()
text=text.replace('#include <aspect/phase_field.h>', '#include <aspect/phase_field.h>\n#include <aspect/phase_field/obstacle.h>\n#include <aspect/phase_field/box_constraints.h>')
old_header=str(OLD/'preload.h');assert old_header in text
text=text.replace('#include "'+old_header+'"','#include "'+str(HERE/'static_equilibrium.h')+'"\n#include "'+str(BUILD/'preload.h')+'"')
text=text.replace('#include "'+str(HERE/'static_equilibrium.h')+'"', '#include "'+str(HERE/'prepared_observation.h')+'"\n#include "'+str(HERE/'static_equilibrium.h')+'"')
text=text.replace('if(this->get_timestep_number()==0)prepare_slipping_background();','if(this->get_timestep_number()==0){prepare_slipping_background();export_prepared_history();}')
original_observer=str(HERE.parent/'stage4_release/observe.h')
text=text.replace(original_observer,str(BUILD/'observe.h'))
observer=(HERE.parent/'stage4_release/observe.h').read_text()
needle='if(s.step==0)\n    {\n      release_initial_field=std::make_unique<Field>(this->get_solution());\n      release_initial_fracture_gradient=endpoint_gradient;\n      release_initial_fracture=fracture;\n    }'
assert needle in observer
observer=observer.replace(needle,'if(!release_initial_field)\n    {\n      release_initial_field=std::make_unique<Field>(s.step==0 ? Field(this->get_solution()) : prepared_phase_reference());\n      release_initial_fracture=this->get_phase_field_handler().assemble_dg_fracture(*release_initial_field,release_initial_fracture_gradient);\n    }')
(BUILD/'observe.h').write_text(observer)
(BUILD/'checks.cc').write_text(text)
preload=(OLD/'tested/preload.h').read_text()
needle='  x+=answer;\n  const auto final='
assert needle in preload
preload=preload.replace(needle,'  x+=answer;\n  settle_loaded_seed(op,x,force);\n  const auto final=')
preload=preload.replace('  std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);','  std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);\n  std::vector<double> prepared_phi(values.n_quadrature_points);\n  std::vector<unsigned int> prepared_formed(ns,0);')
needle='      values[this->introspection().extractors.velocities].get_function_symmetric_gradients(readable,strains);'
assert preload.count(needle)==1
preload=preload.replace(needle,needle+'\n      values[FEValuesExtractors::Scalar(op.pc)].get_function_values(readable,prepared_phi);')
preload=preload.replace('const double phi=op.parent_phase_values.at(entry.first)[q];','const double phi=prepared_phi[q];\n          if(phi>0.)prepared_formed[k]=1;')
needle='  double boundary_work=0.;';assert preload.count(needle)==1
preload=preload.replace(needle,'  Utilities::MPI::max(prepared_formed,comm,prepared_formed);\n  for(unsigned int k=0;k<ns;++k)state.sections[k].formed=state.sections[k].formed || prepared_formed[k];\n  state.tip_phase=x.block(op.pb);\n'+needle)
needle='owned.block(1)*=op.G;*field=owned;';assert preload.count(needle)==1
preload=preload.replace(needle,'owned.block(1)*=op.G;owned.block(op.pb)=x.block(op.pb);*field=owned;')
preload=preload.replace('model().mechanical_quadrature(),update_gradients|update_JxW_values','model().mechanical_quadrature(),update_values|update_gradients|update_JxW_values')
(BUILD/'preload.h').write_text(preload)
for law in ('weakening','neutral'):shutil.copy2(OLD/f'256-{law}.prm',BUILD/f'256-{law}.prm')
target=BUILD/'input-inherited';target.mkdir(exist_ok=True)
for p in (OLD/'input-256').iterdir():shutil.copy2(p,target/p.name)
previous=json.loads((OLD/'build-command.json').read_text())
commands={key:[v.replace(str(OLD),str(BUILD)) if v in [str(OLD/'checks.cc'),str(OLD/'checks.o'),str(OLD/'libbulk_tip_checks.release.so')] else v for v in previous[key]] for key in ('compile','link')}
dependencies=[source,BUILD/'checks.cc',BUILD/'preload.h',HERE/'static_equilibrium.h',HERE/'prepared_observation.h',BUILD/'observe.h',HERE.parent/'stage4_preload/scan.h',HERE.parent/'stage4_release/observe.h',ROOT/'build-particle-free/stage4-pilot/import.h',*target.iterdir()]
commands['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
(BUILD/'build-command.json').write_text(json.dumps(commands,indent=2)+'\n')
with (BUILD/'build.log').open('w') as log:
    for key in ('compile','link'):subprocess.run(commands[key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Built independent zero-clock preparation checker; protected runtimes unchanged.')
