#!/usr/bin/env python3
"""Verify moved numerical bodies, unchanged surroundings and the R3a review."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
root = Path(__file__).resolve().parent
repo = root.parents[2]
e = root/'evidence'
before = (e/'history.cc.before').read_text()
after = (repo/'source/material_model/phase_field_fault/history.cc').read_text()
blocks = json.loads((e/'extraction-blocks.json').read_text())

def body(name):
    start = after.index(f'PhaseFieldFault<dim>::{name}(')
    start = after.index('    {\n', start) + len('    {\n')
    end = after.index('\n    }', start)
    return after[start:end]

sampling = blocks['sampling'].replace('      std::vector<Point<dim>> points;', '      auto &points = samples.points;')
sampling = sampling.replace('      Utilities::MPI::RemotePointEvaluation<dim> point_cache;', '      auto &point_cache = samples.point_cache;')
for declaration, name in [('const auto','velocity_gradients'), ('const std::vector<double>','temperatures'), ('const std::vector<double>','phase_fields'), ('const std::vector<double>','previous_phase_fields')]:
    sampling = sampling.replace(f'      {declaration} {name} =', f'      samples.{name} =')
actual = body('sample_accepted_history')
assert actual[actual.index('      const auto &associations ='):].strip() == sampling.strip()
computation = blocks['computation'].replace('      const auto cohesive_projection =', '      cohesive_projection =').replace('      std::vector<std::vector<double>> state_candidates;\n','')
actual = body('compute_history_candidates')
assert actual[actual.index('      if (std::getenv('):].strip() == computation.strip()
actual = body('validate_history_candidates')
marker = '      const unsigned int local_valid ='
assert actual[actual.index(marker):].strip() == blocks['validation'][blocks['validation'].index(marker):].strip()
actual = body('publish_history_candidates')
assert actual[actual.index('      commit_cohesive_state('):].strip() == blocks['publication'].strip()
start = before.index('    template <int dim>\n    void\n    PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history(')
end = before.index('    template <int dim>\n    double\n    PhaseFieldFault<dim>::compute_reconstructed_fault_time_step(',start)
assert after[:after.index('    // Per-call scratch storage')] == before[:start]
assert after[after.index('    template <int dim>\n    double\n    PhaseFieldFault<dim>::compute_reconstructed_fault_time_step('):] == before[end:]
setup = before[start:before.index('      const auto &associations =', start)]
setup = setup.replace('      ReconstructedFaultManager<dim> &fault_manager =\n        this->get_reconstructed_fault_manager();\n      const auto &faults = fault_manager.get_faults();\n','').replace('      auto &particle_handler = particle_manager.get_particle_handler();\n','')
assert setup in after
calls = re.findall(r'\b(sample_accepted_history|compute_history_candidates|validate_history_candidates|publish_history_candidates)\(', body('commit_reconstructed_fault_mechanical_history'))
assert calls == ['sample_accepted_history','compute_history_candidates','validate_history_candidates','publish_history_candidates']
header = (repo/'include/aspect/material_model/phase_field_fault.h').read_text()
a = header.index('        /** Scratch buffers confined')
b = header.index('        /**\n         * @name Initial cohesive-state setup',a)
assert header[:a]+header[b:] == (e/'phase_field_fault.h.before').read_text()
entry = json.loads((e/'entry-source-hashes.json').read_text())
changed = sorted(name for name,h in entry.items() if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=h)
assert changed == ['include/aspect/material_model/phase_field_fault.h','source/material_model/phase_field_fault/history.cc'],changed
review = (repo/'doc/reconstructed_fault/refactor_review.md').read_text()
assert (e/'R3a-review-verified.md').read_text() in review
binary = repo/'build-restart-fix/aspect-r3a-corrected-qualified'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == '2bd6dca29898f79eff45ca616e9db6f86e9d3c6611ee867b57d289c85798436b'
record = dict(exact_sampling_expressions=True, exact_candidate_expressions_and_error_boundaries=True,
              exact_validation_and_diagnostics=True, exact_publication=True,
              entry_checks_and_timestep_zero_preserved=True, other_history_methods_unchanged=True,
              only_private_header_additions=True, unchanged_entry_files=len(entry)-2,
              R3a_review_intact=True, corrected_reference_binary_unchanged=True)
(e/'source-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
