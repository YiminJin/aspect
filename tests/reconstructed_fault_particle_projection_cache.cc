/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "phase_field_fault_ih.cc"

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
          const auto expected = manager.interpolate_property_at_particle_projections(property);
          AssertThrow(!expected.empty(), ExcMessage("Both fixture ranks must own admitted particles."));
          const auto diagnostics = manager.get_particle_projection_diagnostics();
          const auto build_count = [this]()
          {
            const auto counts = this->get_computing_timer().get_summary_data(TimerOutput::n_calls);
            const auto entry = counts.find("Fault: Cache build");
            return entry == counts.end() ? 0.0 : entry->second;
          };
          const auto factor_count = [this]()
          {
            const auto counts = this->get_computing_timer().get_summary_data(TimerOutput::n_calls);
            const auto entry = counts.find("Fault: Projection preparation");
            return entry == counts.end() ? 0.0 : entry->second;
          };
          const double factors_before = factor_count();
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
          AssertThrow(factor_count() == factors_before,
                      ExcMessage("Association interpolation unnecessarily prepared projection factors."));
          std::map<types::particle_index,double> constant_samples;
          for (const auto &a:manager.get_locally_owned_particle_fault_associations())
            if (a.active)
              constant_samples.emplace(a.particle_id,1.);
          const auto first_projection=manager.project_particle_scalar(constant_samples);
          AssertThrow(factor_count() == factors_before+1.,ExcMessage("First projection must prepare its factors once."));
          const auto repeat_projection=manager.project_particle_scalar(constant_samples);
          AssertThrow(factor_count() == factors_before+1.
                      && first_projection.nodal_values == repeat_projection.nodal_values,
                      ExcMessage("Repeated projection did not reuse its factors/results."));
          for (const auto &fault:first_projection.nodal_values)
            for (const double value:fault)
              AssertThrow(std::abs(value-1.)<1e-12,ExcMessage("Lazy projection lost constant reproduction."));
          manager.invalidate_particle_projection_cache();
          manager.get_locally_owned_particle_fault_associations();
          AssertThrow(factor_count() == factors_before+1.,ExcMessage("Association rebuilding unexpectedly prepared factors."));
          const auto rebuilt_projection=manager.project_particle_scalar(constant_samples);
          AssertThrow(factor_count() == factors_before+2.
                      && rebuilt_projection.nodal_values == first_projection.nodal_values,
                      ExcMessage("Projection factors did not follow association invalidation."));
          auto incomplete=constant_samples;
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==1)
            incomplete.clear();
          bool rejected=false;
          try
            {
              manager.project_particle_scalar(incomplete);
            }
          catch (const std::exception &error)
            {
              rejected=std::string(error.what()).find("No scalar projection value")!=std::string::npos;
            }
          AssertThrow(rejected,ExcMessage("Missing owner values did not reject collectively."));
          AssertThrow(manager.project_particle_scalar(constant_samples).nodal_values==first_projection.nodal_values,
                      ExcMessage("Failed projection changed valid factors or subsequent results."));
          AssertThrow(cold == expected && warm == expected,
                      ExcMessage("Cold/warm reverse interpolation changed particle values."));
          const auto &rebuilt_diagnostics = manager.get_particle_projection_diagnostics();
          AssertThrow(diagnostics.size() == rebuilt_diagnostics.size(), ExcInternalError());
          for (unsigned int f = 0; f < diagnostics.size(); ++f)
            AssertThrow(diagnostics[f].weighted_support == rebuilt_diagnostics[f].weighted_support
                        && diagnostics[f].n_contributing_particles
                        == rebuilt_diagnostics[f].n_contributing_particles,
                        ExcMessage("Rebuilt projection support differs from the prepared cache."));

          // Full local values allow exact matched-executable comparisons.
          const unsigned int rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream out(this->get_output_directory() + "particle-cache-rank-"
                            + std::to_string(rank) + ".txt");
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out << "cold rebuilds " << after_cold-before
              << " warm rebuilds " << after_warm-after_cold << '\n' << std::hexfloat;
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
