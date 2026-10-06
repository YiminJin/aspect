/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "phase_field_fault_ih.cc"
#include <aspect/particle/particle_domain.h>

#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyParticleProjectionCache : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute(TableHandler &) override
        {
          AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 2,
                      ExcMessage("The cold/warm particle-projection fixture requires two ranks."));
          auto &manager = this->get_reconstructed_fault_manager();
          const auto property = manager.get_property_index("phase field fault cohesive traction");
#ifdef R7B_EMPTY_OWNER
          // This manufactured cache-only probe has no constitutive postprocessor.
          // Set finite, varying nodal data on the replicated prescribed geometry.
          const auto position = manager.get_property_information()[property].position;
          for (unsigned int f = 0; f < manager.get_faults().size(); ++f)
            for (unsigned int v = 0; v < manager.get_fault(f).n_vertices(); ++v)
              manager.get_fault(f).get_properties(v)[position] = 2.0 + v;
#endif
          const auto expected = manager.interpolate_property_at_particle_projections(property);
          AssertThrow(Utilities::MPI::sum(expected.size(), this->get_mpi_communicator()) > 0,
                      ExcMessage("The fixture must have admitted particles globally."));
          const auto diagnostics = manager.get_particle_projection_diagnostics();
          const auto build_count = [this]()
          {
            const auto counts = this->get_computing_timer().get_summary_data(TimerOutput::n_calls);
            const auto entry = counts.find("Fault: Cache build");
            return entry == counts.end() ? 0.0 : entry->second;
          };
          const double before = build_count();

          // Every rank deliberately invalidates and enters preparation together.
          // The production local validity predicate does not establish agreement.
          manager.invalidate_particle_projection_cache();
          AssertThrow(manager.get_particle_projection_diagnostics().empty(),
                      ExcMessage("Invalidation did not clear projection diagnostics."));
          const auto cold = manager.interpolate_property_at_particle_projections(property);
          const double after_cold = build_count();
          const auto warm = manager.interpolate_property_at_particle_projections(property);
          const double after_warm = build_count();
          AssertThrow(after_cold == before + 1.0 && after_warm == after_cold,
                      ExcMessage("Cold interpolation must rebuild once; warm interpolation must reuse."));
          AssertThrow(cold == expected && warm == expected,
                      ExcMessage("Cold/warm reverse interpolation changed particle values."));
          const auto &rebuilt_diagnostics = manager.get_particle_projection_diagnostics();
          AssertThrow(diagnostics.size() == rebuilt_diagnostics.size(), ExcInternalError());
          for (unsigned int f = 0; f < diagnostics.size(); ++f)
            AssertThrow(diagnostics[f].weighted_support == rebuilt_diagnostics[f].weighted_support
                        && diagnostics[f].n_contributing_particles
                        == rebuilt_diagnostics[f].n_contributing_particles,
                        ExcMessage("Rebuilt projection support differs from the prepared cache."));

          // A supported all-rank domain rebuild changes the geometry version,
          // even on an empty owner and even if all domain volumes are unchanged.
          auto &particles = this->get_phase_field_handler().get_associated_particle_manager();
          auto &domains = const_cast<Particle::ParticleDomainHandler<dim> &>(
            particles.get_particle_domain_handler());
          std::vector<double> volumes;
          for (const auto &particle : particles.get_particle_handler())
            volumes.push_back(domains.get_particle_domain(particle.get_local_index()).volume());
          const auto generation = domains.geometry_version();
          domains.generate_particle_domains();
          AssertThrow(domains.geometry_version() == generation+1, ExcInternalError());
          unsigned int index = 0;
          for (const auto &particle : particles.get_particle_handler())
            AssertThrow(volumes.at(index++) == domains.get_particle_domain(particle.get_local_index()).volume(),
                        ExcMessage("Regeneration changed the fixture domain volumes."));
          const auto regenerated = manager.interpolate_property_at_particle_projections(property);
          const double after_regeneration = build_count();
          const auto reused = manager.interpolate_property_at_particle_projections(property);
          AssertThrow(regenerated == warm && reused == warm, ExcMessage("Regeneration changed reverse values."));
          AssertThrow(after_regeneration == after_warm+1 && build_count() == after_regeneration,
                      ExcMessage("Domain generation must invalidate once, then reuse."));

          // Full local values allow exact matched-executable comparisons.
          const unsigned int rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream out(this->get_output_directory() + "particle-cache-rank-"
                            + std::to_string(rank) + ".txt");
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out << "cold rebuilds " << after_cold-before
              << " warm rebuilds " << after_warm-after_cold
              << " regeneration rebuilds " << after_regeneration-after_warm
              << " local particles " << expected.size() << '\n' << std::hexfloat;
          for (const auto &entry : warm)
            {
              out << entry.first;
              for (const double value : entry.second)
                out << ' ' << value;
              out << '\n';
            }
          for (const auto &diagnostic : rebuilt_diagnostics)
            {
              out << "support " << diagnostic.n_contributing_particles;
              for (const double value : diagnostic.weighted_support)
                out << ' ' << value;
              out << '\n';
            }
          return {"Particle projection cold/warm cache:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyParticleProjectionCache,
                                  "verify particle projection cold warm cache",
                                  "Check collective lazy preparation followed by local cache reuse "
                                  "on two ranks, using existing timer call counts and projection values.")
  }
}
