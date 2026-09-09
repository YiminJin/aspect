/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>

namespace aspect
{
  namespace
  {
    std::map<types::global_dof_index,double> frozen_phi;
    unsigned int converged_solves = 0;
    void record_convergence(const SolverControl &control)
    {
      AssertThrow(control.last_check() == SolverControl::success
                  && control.last_value() < 1e-8 && control.last_step() > 0,
                  ExcMessage("Changed loading did not converge both coupled blocks."));
      ++converged_solves;
    }

    template <int dim>
    void freeze_profile(const SimulatorAccess<dim> &simulator,
                        AffineConstraints<double> &constraints)
    {
      if (simulator.get_timestep_number() == 0)
        return;
      for (const auto &value : frozen_phi)
        if (simulator.introspection().index_sets.system_relevant_set.is_element(value.first)
            && !constraints.is_constrained(value.first))
          {
            constraints.add_line(value.first);
            constraints.set_inhomogeneity(value.first, value.second);
          }
    }
  }

  template <int dim>
  void connect_changed_loading(SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect(&freeze_profile<dim>);
    signals.post_nonlinear_solver.connect(&record_convergence);
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_changed_loading<2>, connect_changed_loading<3>)

  namespace Postprocess
  {
    template <int dim>
    class VerifyChangedLoading : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const unsigned int step = this->get_timestep_number();
          AssertThrow(converged_solves == step+1, ExcInternalError());
          const auto &intro = this->introspection();
          const auto &fe = this->get_fe();
          const unsigned int phase = intro.variable("phase_field").first_component_index;
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                for (unsigned int i=0; i<indices.size(); ++i)
                  if (fe.system_to_component_index(i).first == phase)
                    {
                      if (step == 0)
                        frozen_phi[indices[i]] = this->get_solution()[indices[i]];
                      else
                        AssertThrow(this->get_solution()[indices[i]] == frozen_phi.at(indices[i]),
                                    ExcMessage("Changed-loading fixture did not freeze phi."));
                    }
              }
          if (step == 0)
            {
              // The test's fixed profile must constrain imported periodic and
              // shared DoFs identically on both ranks, not just local-cell DoFs.
              const auto snapshots = Utilities::MPI::all_gather(
                this->get_mpi_communicator(), frozen_phi);
              for (const auto &snapshot : snapshots)
                frozen_phi.insert(snapshot.begin(),snapshot.end());
            }
          const auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const unsigned int theta = manager.get_property_information()[
            manager.get_property_index("phase field fault state")].position;
          const auto &V = manager.get_slip_rate(0);
          for (unsigned int i=0; i<V.size(); ++i)
            {
              const double state = fault.get_properties(i)[theta];
              if (step == 0)
                AssertThrow(std::abs(state-200.) < 1e-10,
                            ExcMessage("Initial Theta was evolved."));
              else
                {
                  AssertThrow(V[i]-initial_V[i] > 1e-6,
                              ExcMessage("Fixture did not exercise changed mechanical loading."));
                  const double steady = .001/V[i];
                  const double expected = steady+(200.-steady)*std::exp(-this->get_timestep()/steady);
                  AssertThrow(std::abs(state-expected) < 1e-8 && 200.-expected > 1.,
                              ExcMessage("Accepted real-step state was not published correctly."));
                }
            }
          if (step == 0)
            {
              initial_V = V;
              const auto &particles = this->get_phase_field_handler().get_associated_particle_manager();
              const unsigned int stress = particles.get_property_manager().get_data_info()
                .get_position_by_field_name("maxwell stress");
              for (const auto &particle : particles.get_particle_handler())
                AssertThrow(particle.get_properties()[stress+2] == 1500.,
                            ExcMessage("Initial Maxwell history was evolved."));
            }
          return {"Changed-loading coupled regression:", "verified step "+std::to_string(step)};
        }
      private:
        std::vector<double> initial_V;
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyChangedLoading, "uniform shear pilot",
                                  "Verify a balanced surface survives a real bulk loading change.")
  }
}
