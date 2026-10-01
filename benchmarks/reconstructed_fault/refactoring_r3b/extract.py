#!/usr/bin/env python3
"""Extract the four existing history stages without rewriting numerical bodies."""
from pathlib import Path
import json
root = Path(__file__).resolve().parent
repo = root.parents[2]
source = repo/'source/material_model/phase_field_fault/history.cc'
header = repo/'include/aspect/material_model/phase_field_fault.h'
s = (root/'evidence/history.cc.before').read_text()
a = s.index('    template <int dim>\n    void\n    PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history(')
b = s.index('    template <int dim>\n    double\n    PhaseFieldFault<dim>::compute_reconstructed_fault_time_step(', a)
method = s[a:b]
start_sample = method.index('      const auto &associations =')
start_candidates = method.index('      struct ParticleCandidate')
start_compute = method.index('      if (std::getenv(')
start_validate = method.index('      // Every rank validates')
start_publish = method.index('      commit_cohesive_state(cohesive_projection.nodal_values);')
sampling = method[start_sample:start_candidates]
compute = method[start_compute:start_validate]
validation = method[start_validate:start_publish]
publication = method[start_publish:method.rindex('    }')]
# Keep the original names as local references to the per-call buffers.
sampling = sampling.replace('      std::vector<Point<dim>> points;','      auto &points = samples.points;')
sampling = sampling.replace('      Utilities::MPI::RemotePointEvaluation<dim> point_cache;', '      auto &point_cache = samples.point_cache;')
for declaration,name in [('const auto','velocity_gradients'),('const std::vector<double>','temperatures'),('const std::vector<double>','phase_fields'),('const std::vector<double>','previous_phase_fields')]:
 sampling = sampling.replace(f'      {declaration} {name} =', f'      samples.{name} =')
compute = compute.replace('      const auto cohesive_projection =', '      cohesive_projection =')
compute = compute.replace('      std::vector<std::vector<double>> state_candidates;\n', '')
structs = '''    // Per-call scratch storage, never a second owner of committed history.
    // Keep the sampling cache and audit streams alive through publication.
    template <int dim>
    struct PhaseFieldFault<dim>::HistorySamples
    {
      std::vector<Point<dim>> points;
      Utilities::MPI::RemotePointEvaluation<dim> point_cache;
      std::vector<Tensor<2,dim>> velocity_gradients;
      std::vector<double> temperatures;
      std::vector<double> phase_fields;
      std::vector<double> previous_phase_fields;
    };

    template <int dim>
    struct PhaseFieldFault<dim>::HistoryCandidates
    {
      struct ParticleCandidate
      {
        SymmetricTensor<2,dim> stress;
        double crack_driving_force;
      };
      std::map<types::particle_index, double> cohesive_samples;
      std::map<types::particle_index, ParticleCandidate> particle_candidates;
      std::ofstream source_history_audit;
      std::set<std::string> trace_cells;
      std::ofstream stress_cycle_audit;
      typename ReconstructedFaultManager<dim>::ParticleScalarProjectionResult cohesive_projection;
      std::vector<std::vector<double>> state_candidates;
    };

'''
driver = method[:start_sample]
# These aliases are now private to each operation that uses them.
driver = driver.replace('      ReconstructedFaultManager<dim> &fault_manager =\n        this->get_reconstructed_fault_manager();\n      const auto &faults = fault_manager.get_faults();\n','')
driver = driver.replace('      auto &particle_handler = particle_manager.get_particle_handler();\n','')
driver += '''      HistorySamples samples;
      sample_accepted_history(accepted_bulk_state, samples);

      HistoryCandidates candidates;
      compute_history_candidates(samples, time_step, stress_position, H_position,
                                 chemical_positions, candidates);
      validate_history_candidates(candidates);
      publish_history_candidates(candidates, stress_position, H_position);
    }

'''
manager_alias = '''      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
'''
particle_alias = '''      auto &particle_handler = this->get_phase_field_handler()
                               .get_associated_particle_manager().get_particle_handler();
'''
sample_aliases = ''.join(f'      const auto &{name} = samples.{name};\n' for name in ['points','velocity_gradients','temperatures','phase_fields','previous_phase_fields'])
candidate_aliases = ''.join(f'      auto &{name} = candidates.{name};\n' for name in ['cohesive_samples','particle_candidates','source_history_audit','trace_cells','stress_cycle_audit','cohesive_projection','state_candidates'])
helpers = '''    template <int dim>
    void
    PhaseFieldFault<dim>::sample_accepted_history(
      const LinearAlgebra::BlockVector &accepted_bulk_state,
      HistorySamples &samples)
    {
''' + manager_alias + sampling + '''    }

    template <int dim>
    void
    PhaseFieldFault<dim>::compute_history_candidates(
      const HistorySamples &samples,
      const double time_step,
      const unsigned int stress_position,
      const unsigned int H_position,
      const std::vector<unsigned int> &chemical_positions,
      HistoryCandidates &candidates)
    {
''' + manager_alias + '''      const auto &faults = fault_manager.get_faults();
''' + particle_alias + '''      const auto &associations =
        fault_manager.get_locally_owned_particle_fault_associations();
      using ParticleCandidate = typename HistoryCandidates::ParticleCandidate;
''' + sample_aliases + candidate_aliases + '\n' + compute + '''    }

    template <int dim>
    void
    PhaseFieldFault<dim>::validate_history_candidates(
      const HistoryCandidates &candidates)
    {
''' + particle_alias + '''      const auto &particle_candidates = candidates.particle_candidates;
      const auto &cohesive_projection = candidates.cohesive_projection;
''' + validation + '''    }

    template <int dim>
    void
    PhaseFieldFault<dim>::publish_history_candidates(
      const HistoryCandidates &candidates,
      const unsigned int stress_position,
      const unsigned int H_position)
    {
''' + manager_alias + '''      const auto &faults = fault_manager.get_faults();
''' + particle_alias + '''      using ParticleCandidate = typename HistoryCandidates::ParticleCandidate;
      const auto &particle_candidates = candidates.particle_candidates;
      const auto &cohesive_projection = candidates.cohesive_projection;
      const auto &state_candidates = candidates.state_candidates;
''' + publication + '''    }

'''
result = s[:a] + structs + driver + helpers + s[b:]
result = result.replace('      // Every rank validates before the first persistent write. The terminal\n      // block below performs only fixed-size scalar assignments.', '      // Finish the existing collective and local checks before publication.')
result = result.replace('        VectorTools::EvaluationFlags::avg, phase_field_component);\n\n    }', '        VectorTools::EvaluationFlags::avg, phase_field_component);\n    }')
result = result.replace('          << \" Pa\" << std::endl;\n\n    }', '          << \" Pa\" << std::endl;\n    }')
source.write_text(result)
h = (root/'evidence/phase_field_fault.h.before').read_text()
needle = '        /**\n         * @name Initial cohesive-state setup'
new = '''        /** Scratch buffers confined to one accepted-history call. */
        struct HistorySamples;
        struct HistoryCandidates;

        /** Sample accepted bulk and previous phase fields in association order. */
        void sample_accepted_history(const LinearAlgebra::BlockVector &accepted_bulk_state,
                                     HistorySamples &samples);

        /** Project cohesive samples before computing Theta and particle candidates.
         * Intermediate collective error checks stay at their existing boundaries. */
        void compute_history_candidates(const HistorySamples &samples,
                                        const double time_step,
                                        const unsigned int stress_position,
                                        const unsigned int H_position,
                                        const std::vector<unsigned int> &chemical_positions,
                                        HistoryCandidates &candidates);

        /** Complete existing collective/local checks and projection diagnostics. */
        void validate_history_candidates(const HistoryCandidates &candidates);

        /** Publish the already validated histories in their existing order.
         * Timestep acceptance and slip-rate publication remain with the caller. */
        void publish_history_candidates(const HistoryCandidates &candidates,
                                        const unsigned int stress_position,
                                        const unsigned int H_position);

'''
assert needle in h
header.write_text(h.replace(needle,new+needle,1))
(root/'evidence/extraction-blocks.json').write_text(json.dumps(dict(
 sampling=method[start_sample:start_candidates],
 computation=method[start_compute:start_validate],
 validation=validation, publication=publication),indent=2)+'\n')
