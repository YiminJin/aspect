#!/usr/bin/env python3
"""Perform the selected one-time method relocation from the saved R2b source."""
from pathlib import Path
import json,re
root=Path(__file__).resolve().parent
repo=root.parents[2]
source=repo/'source/material_model/phase_field_fault.cc'
before=(root/'evidence/phase_field_fault.cc.before').read_text()
assert source.read_text()==before, 'Source is not the captured R3a entry state'
constitutive=['compute_maxwell_coefficients','compute_maxwell_stress','evaluate_frozen_maxwell_stress',
 'evaluate_reconstructed_fault_point','evaluate_reconstructed_fault_bulk_point',
 'compute_crack_driving_force_candidate','evaluate_reconstructed_fault_localization',
 'compute_cohesive_response','compute_creep_viscosity']
history=['initialize','prepare_reconstructed_fault_mechanical_solve','compute_fault_surface_temperatures',
 'validate_reconstructed_fault_constitutive_state','commit_reconstructed_fault_mechanical_history',
 'compute_reconstructed_fault_time_step','validate_cohesive_state_commit','commit_cohesive_state',
 'initialize_cohesive_state_from_initial_fields','evaluate_initial_cohesive_particle_values']
spans=[]
for dest,names in [('constitutive',constitutive),('history',history)]:
 for name in names:
  match=re.search(r'PhaseFieldFault<dim>::\s*'+name+r'\(',before)
  assert match,name
  start=before.rfind('    template <int dim>\n',0,match.start())
  end=before.index('\n    }',match.end())+len('\n    }')
  spans.append(dict(name=name,destination=dest,start=start,end=end))
header=before[:before.index('namespace aspect\n')]
helper_start=before.index('  // -----------------------------------------------------------------------------')
helper_end=before.index('  namespace MaterialModel\n')
helper=before[helper_start:helper_end]
internal=helper[:helper.index('  namespace\n  {')]
anon=helper[helper.index('    template <int dim>'):helper.rindex('\n  }')]
scalar_start=anon.rfind('    template <int dim>\n')
helpers={'history':internal+'  namespace\n  {\n'+anon[:scalar_start]+'  }\n\n',
         'constitutive':'  namespace\n  {\n'+anon[scalar_start:]+'\n  }\n\n'}
instances={
'constitutive':'''    template PhaseFieldFault<dim>::MaxwellCoefficients PhaseFieldFault<dim>::compute_maxwell_coefficients(const double, const double, const double);
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::compute_maxwell_stress(const MaxwellCoefficients &, const SymmetricTensor<2,dim> &, const SymmetricTensor<2,dim> &);
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_frozen_maxwell_stress(const double, const std::vector<double> &, const SymmetricTensor<2,dim> &) const;
    template PhaseFieldFault<dim>::ReconstructedFaultPointResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_point(const ReconstructedFaultPointInputs &) const;
    template PhaseFieldFault<dim>::ReconstructedFaultBulkPointResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_bulk_point(const ReconstructedFaultBulkPointInputs &) const;
    template double PhaseFieldFault<dim>::compute_crack_driving_force_candidate(const double, const MaxwellCoefficients &, const double, const double, const double, const double);
    template PhaseFieldFault<dim>::LocalizationResponse PhaseFieldFault<dim>::evaluate_reconstructed_fault_localization(const unsigned int, const unsigned int, const double, const double, const double, const std::string &) const;
    template PhaseFieldFault<dim>::CohesiveResponse PhaseFieldFault<dim>::compute_cohesive_response(const MaxwellCoefficients &, const double, const double, const double, const double, const double, const double, const bool);
    template double PhaseFieldFault<dim>::compute_creep_viscosity(const std::vector<double> &, const double) const;''',
'history':'''    template void PhaseFieldFault<dim>::initialize();
    template void PhaseFieldFault<dim>::prepare_reconstructed_fault_mechanical_solve();
    template void PhaseFieldFault<dim>::compute_fault_surface_temperatures();
    template void PhaseFieldFault<dim>::validate_reconstructed_fault_constitutive_state() const;
    template void PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history(const LinearAlgebra::BlockVector &);
    template double PhaseFieldFault<dim>::compute_reconstructed_fault_time_step(const double) const;
    template void PhaseFieldFault<dim>::validate_cohesive_state_commit(const std::vector<std::vector<double>> &) const;
    template void PhaseFieldFault<dim>::commit_cohesive_state(const std::vector<std::vector<double>> &) noexcept;
    template void PhaseFieldFault<dim>::initialize_cohesive_state_from_initial_fields();
    template std::map<types::particle_index,double> PhaseFieldFault<dim>::evaluate_initial_cohesive_particle_values();'''}
for dest in ('constitutive','history'):
 text=header+'namespace aspect\n{\n'+helpers[dest]+'  namespace MaterialModel\n  {\n'
 if dest=='history':text+='    using aspect::internal::throw_if_history_error;\n'
 text+='\n\n'.join(before[s['start']:s['end']] for s in sorted(spans,key=lambda s:s['start']) if s['destination']==dest)
 text+='\n\n\n    // Registration remains in phase_field_fault.cc; instantiate only moved members.\n#define INSTANTIATE(dim) \\\n'
 text+=' \\\n'.join(instances[dest].splitlines())+'\n\n    ASPECT_INSTANTIATE(INSTANTIATE)\n#undef INSTANTIATE\n  }\n}\n'
 (source.parent/'phase_field_fault'/f'{dest}.cc').write_text(text)
# Remove only captured methods/helper scaffolding; preserve all remaining bodies.
removals=[(s['start'],s['end']) for s in spans]+[(helper_start,helper_end)]
after=before
for start,end in sorted(removals,reverse=True):after=after[:start]+after[end:]
after=after.replace('    using aspect::internal::throw_if_history_error;\n','')
for title in ('Maxwell constitutive law','Cohesive constitutive law','Initial cohesive-state initialization'):
 after=after.replace('    // -----------------------------------------------------------------------------\n    // '+title+'\n    // -----------------------------------------------------------------------------\n','')
# Collapse only empty inter-method space left by the move.
after=re.sub(r'\n{5,}', '\n\n\n\n',after)
source.write_text(after)
(root/'evidence/moved-methods.json').write_text(json.dumps(spans,indent=2)+'\n')
print('Moved',len(constitutive),'constitutive and',len(history),'history methods.')
