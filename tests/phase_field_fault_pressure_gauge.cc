/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifdef GAUGE_ROLLBACK
#include "phase_field_fault_stage_i_rollback.cc"
#else
#include "phase_field_fault_changed_loading.cc"
#endif

#include <deal.II/fe/fe_values.h>
#include <aspect/geometry_model/interface.h>
#include <fstream>
#include <iomanip>

#ifndef GAUGE_OFFSET
#define GAUGE_OFFSET 0.0
#endif

namespace aspect
{
  namespace
  {
    template <int dim>
    void shift_initial_gauge(const SimulatorAccess<dim> &sim)
    {
      // Deliberately perturb only the initial pressure null mode. In rollback
      // variants this runs before the existing complete-state snapshot.
      LinearAlgebra::BlockVector owned(sim.introspection().index_sets.system_partitioning,
                                       sim.get_mpi_communicator());
      owned = sim.get_solution();
      owned.block(sim.introspection().block_indices.pressure).add(GAUGE_OFFSET);
      const_cast<LinearAlgebra::BlockVector &>(sim.get_solution()) = owned;
    }
  }

  template <int dim>
  void connect_pressure_gauge(SimulatorSignals<dim> &signals)
  {
    signals.post_set_initial_state.connect(&shift_initial_gauge<dim>, boost::signals2::at_front);
  }
  namespace pressure_gauge_signals
  {
    ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_pressure_gauge<2>, connect_pressure_gauge<3>)
  }

#ifndef GAUGE_ROLLBACK
  namespace Postprocess
  {
    template <int dim>
    class VerifyPressureGauge : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &intro = this->introspection();
          const auto &parameters = this->get_parameters();
          const auto comm = this->get_mpi_communicator();
          const bool surface = parameters.pressure_normalization == "surface";
          const double target = surface ? parameters.surface_pressure : 0.;
          FEValues<dim> volume_values(this->get_mapping(),this->get_fe(),intro.quadratures.pressure,
                                     update_values | update_JxW_values);
          FEFaceValues<dim> face_values(this->get_mapping(),this->get_fe(),intro.face_quadratures.pressure,
                                       update_values | update_JxW_values);
          const auto top = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
          for (const auto *state : {&this->get_solution(), &this->get_current_linearization_point()})
            {
              double integral = 0., measure = 0.;
              const auto integrate = [&](const FEValuesBase<dim> &values)
              {
                std::vector<double> p(values.n_quadrature_points);
                values[intro.extractors.pressure].get_function_values(*state,p);
                for (unsigned int q=0; q<p.size(); ++q)
                  {
                    integral += p[q]*values.JxW(q);
                    measure += values.JxW(q);
                  }
              };
              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    if (!surface)
                      {
                        volume_values.reinit(cell);
                        integrate(volume_values);
                      }
                    else
                      for (const auto face : cell->face_indices())
                        if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == top)
                          {
                            face_values.reinit(cell,face);
                            integrate(face_values);
                          }
                  }
              const double mean = Utilities::MPI::sum(integral,comm)/Utilities::MPI::sum(measure,comm);
              if (parameters.pressure_normalization != "no")
                AssertThrow(std::abs(mean-target) < 1e-8,
                            ExcMessage("Coupled publication did not honor the configured pressure gauge."));
              else
                AssertThrow(std::abs(mean) > 1.,
                            ExcMessage("Pressure normalization=no unexpectedly removed the test offset."));
            }

          // Compare gauge-equivalent runs using physical velocity, surface
          // histories and particle stress, not the arbitrary pressure offset.
          std::vector<double> fingerprint;
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto position = [&](const std::string &name)
          { return manager.get_property_information()[manager.get_property_index(name)].position; };
          const auto theta = position("phase field fault state");
          const auto C = position("phase field fault cohesive traction");
          const auto Ih = position("phase field fault previous I h");
          for (unsigned int f=0; f<manager.get_faults().size(); ++f)
            for (unsigned int v=0; v<manager.get_fault(f).n_vertices(); ++v)
              {
                const auto values = manager.get_fault(f).get_properties(v);
                fingerprint.insert(fingerprint.end(), {manager.get_slip_rate(f)[v]/1e-4,
                                                       values[theta]/200., values[C]/1500., values[Ih]/100.});
              }
          LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
          owned = this->get_solution();
          fingerprint.push_back(owned.block(intro.block_indices.velocities).l2_norm()/1e-4);
          const auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
          const auto stress = pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          double stress_sum=0., count=0.;
          for (const auto &p : pm.get_particle_handler())
            {
              stress_sum += p.get_properties()[stress+2];
              count += 1.;
            }
          fingerprint.push_back(Utilities::MPI::sum(stress_sum,comm)/Utilities::MPI::sum(count,comm)/1500.);
          const std::string filename = "gauge-state-"+std::to_string(this->get_timestep_number())+".txt";
#ifdef GAUGE_REFERENCE
          std::vector<double> expected;
          if (Utilities::MPI::this_mpi_process(comm)==0)
            {
              std::ifstream input(std::string(GAUGE_REFERENCE)+"/"+filename);
              double value;
              while (input >> value)
                expected.push_back(value);
            }
          expected = Utilities::MPI::broadcast(comm,expected,0);
          AssertThrow(expected.size()==fingerprint.size(),ExcMessage("Missing gauge-equivalent reference run."));
          for (unsigned int i=0; i<expected.size(); ++i)
            AssertThrow(std::abs(fingerprint[i]-expected[i]) < 1e-9*std::max(1.,std::abs(expected[i])),
                        ExcMessage("Pressure gauge changed velocity or committed fault/particle history."));
#endif
          if (Utilities::MPI::this_mpi_process(comm)==0)
            {
              std::ofstream output(this->get_output_directory()+filename);
              output << std::setprecision(17);
              for (const auto value : fingerprint)
                output << value << '\n';
            }
          return {"Coupled pressure gauge:", "verified step "+std::to_string(this->get_timestep_number())};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyPressureGauge,"verify coupled pressure gauge",
                                 "Check physical pressure normalization and gauge-independent histories.")
  }
#endif
}
