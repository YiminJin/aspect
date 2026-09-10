/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>

#include "phase_field_fault_test_access.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace
  {
    bool coupled_solve_converged = false;
  }

  namespace StageIConvergence
  {
    template <int dim>
    void connect_stage_i_convergence(SimulatorSignals<dim> &signals)
    {
      signals.start_timestep.connect([](const SimulatorAccess<dim> &)
      {
        coupled_solve_converged = false;
      });
      signals.post_nonlinear_solver.connect([](const SolverControl &control)
      {
        coupled_solve_converged = control.last_check() == SolverControl::success
          && std::isfinite(control.last_value())
          && control.last_value() < control.tolerance();
      });
    }
    // Included lifecycle fixtures register their own connectors. The macro
    // needs a distinct namespace for this shared positive-convergence guard.
    ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_stage_i_convergence<2>,
                                      connect_stage_i_convergence<3>)
  }

  namespace Postprocess
  {
    template <int dim>
    class VerifyPhaseFieldFaultStageI : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          AssertThrow(coupled_solve_converged,
                      ExcMessage("The positive Stage-I/lifecycle test requires both "
                                 "final nonlinear convergence criteria; continued "
                                 "execution after solver failure is not a pass."));
          const auto &model =
            Plugins::get_plugin_as_type<
              const MaterialModel::PhaseFieldFault<dim>>(
                this->get_material_model());
          ReconstructedFaultManager<dim> &fault_manager =
            this->get_reconstructed_fault_manager();
          AssertThrow(fault_manager.slip_rates_are_initialized(),
                      ExcMessage("The Stage-I solve did not commit slip rate."));

          bool found_exact_bound = false;
          for (unsigned int fault = 0;
               fault < fault_manager.get_faults().size(); ++fault)
            {
              const auto &values =
                fault_manager.get_timestep_committed_slip_rate(fault);
              for (const double value : values)
                {
                  AssertThrow(std::isfinite(value)
                              && value >= model.minimum_fault_slip_rate(),
                              ExcMessage("The Stage-I solve committed an inadmissible "
                                         "slip rate."));
                  found_exact_bound =
                    found_exact_bound
                    || value == model.minimum_fault_slip_rate();
                }
            }
          AssertThrow(found_exact_bound,
                      ExcMessage("The Stage-I active-set fixture did not permit exact "
                                 "contact with V_min."));

          if (!fault_manager.has_property("phase field fault state"))
            {
              ReconstructedFaultVector slip_rate(
                fault_manager.get_faults().size());
              for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
                slip_rate[fault] = fault_manager.get_slip_rate(fault);
              const auto residual =
                this->get_reconstructed_fault_surface_system()
                .evaluate_surface_residual(this->get_solution(), slip_rate);
              bool found_bound_with_outward_residual = false;
              for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
                for (unsigned int vertex = 0;
                     vertex < slip_rate[fault].size(); ++vertex)
                  found_bound_with_outward_residual =
                    found_bound_with_outward_residual
                    || (slip_rate[fault][vertex]
                        == model.minimum_fault_slip_rate()
                        && std::abs(residual.values[fault][vertex]) > 1.e-6);
              AssertThrow(found_bound_with_outward_residual,
                          ExcMessage("The Stage-I fixture did not exercise a "
                                     "nonzero bound-active surface residual."));
            }

          if (fault_manager.has_property("phase field fault state"))
            {
              std::vector<std::vector<double>> expected_initial_state;
              if (this->get_timestep_number() == 0)
                {
                  const auto &particle_manager =
                    this->get_phase_field_handler().get_associated_particle_manager();
                  const auto &particle_data =
                    particle_manager.get_property_manager().get_data_info();
                  const unsigned int particle_state_position =
                    particle_data.get_position_by_field_name(
                      "phase field fault state");
                  std::map<types::particle_index, double> particle_state;
                  for (const auto &particle : particle_manager.get_particle_handler())
                    particle_state.emplace(
                      particle.get_id(),
                      particle.get_properties()[particle_state_position]);
                  expected_initial_state = fault_manager
                                           .project_particle_scalar(particle_state)
                                           .nodal_values;
                }

              const unsigned int property = fault_manager.get_property_index(
                "phase field fault state");
              const unsigned int position =
                fault_manager.get_property_information()[property].position;
              double minimum_state = std::numeric_limits<double>::max();
              double maximum_state = 0.0;
              for (unsigned int fault_index = 0;
                   fault_index < fault_manager.get_faults().size(); ++fault_index)
                {
                  const auto &fault = fault_manager.get_fault(fault_index);
                for (unsigned int vertex = 0; vertex < fault.n_vertices(); ++vertex)
                  {
                    const double state = fault.get_properties(vertex)[position];
                    AssertThrow(std::isfinite(state) && state > 0.0,
                                ExcMessage("The projected initial Theta is inadmissible."));
                    minimum_state = std::min(minimum_state, state);
                    maximum_state = std::max(maximum_state, state);
                    if (this->get_timestep_number() == 0)
                      {
                        AssertThrow(state == expected_initial_state[fault_index][vertex],
                                    ExcMessage("Theta_0 was changed by the artificial "
                                               "initial timestep."));
                      }
                    else if (this->get_timestep_number() == 1)
                      {
                        const double expected =
                          MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
                          ::fault_friction(model).update_state(
                            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
                            ::surface_material_fractions_at_vertex(
                              model, fault_manager, fault_index, vertex),
                            fault_manager.get_timestep_committed_slip_rate(
                              fault_index)[vertex],
                            theta_zero[fault_index][vertex],
                            this->get_timestep());
                        AssertThrow(std::abs(state-expected)
                                    <= 1.e-10*std::max(state, expected),
                                    ExcMessage("Theta_1 does not equal the exact "
                                               "accepted-state aging update."));
                      }
                  }
                }
              if (this->get_timestep_number() == 0)
                theta_zero = expected_initial_state;
              AssertThrow(maximum_state-minimum_state > 1.e3,
                          ExcMessage("The Stage-I rate-and-state fixture did not "
                                     "preserve the spatial variation in Theta_0."));
            }

          if (this->get_timestep_number() == 1)
            {
              bool maxwell_stress_evolved = false;
              const auto &particle_manager =
                this->get_phase_field_handler().get_associated_particle_manager();
              const auto &particle_data =
                particle_manager.get_property_manager().get_data_info();
              const unsigned int stress_position =
                particle_data.get_position_by_plugin_index(
                  particle_manager.get_property_manager()
                  .get_plugin_index_by_name("maxwell stress"));
              const unsigned int H_position =
                particle_data.get_position_by_field_name("crack_driving_force");
              for (const auto &particle : particle_manager.get_particle_handler())
                {
                  const ArrayView<const double> properties = particle.get_properties();
                  maxwell_stress_evolved = maxwell_stress_evolved
                    || std::abs(properties[stress_position]-4.e6) > 1.e-6
                    || std::abs(properties[stress_position+1]
                                -(-4.e6+8.e6*particle.get_location()[0])) > 1.e-6
                    || std::abs(properties[stress_position+2]-1.e6) > 1.e-6;
                  AssertThrow(std::isfinite(properties[H_position])
                              && properties[H_position]
                                 >= H_zero.at(particle.get_id()),
                              ExcMessage("The irreversible H history decreased."));
                }
              AssertThrow(maxwell_stress_evolved,
                          ExcMessage("The first real timestep did not evolve Maxwell stress."));
            }


          if (this->get_timestep_number() == 0)
            {
              // The initial mechanical solve commits V_0, but all particle
              // histories retain their explicit initialization semantics.
              const auto &particle_manager =
                this->get_phase_field_handler().get_associated_particle_manager();
              const auto &particle_data =
                particle_manager.get_property_manager().get_data_info();
              const unsigned int stress_position =
                particle_data.get_position_by_plugin_index(
                  particle_manager.get_property_manager()
                  .get_plugin_index_by_name("maxwell stress"));
              const unsigned int H_position =
                particle_data.get_position_by_field_name("crack_driving_force");
              for (const auto &particle : particle_manager.get_particle_handler())
                {
                  const ArrayView<const double> properties = particle.get_properties();
                  AssertThrow(properties[stress_position] == 4.e6
                              && properties[stress_position+2] == 1.e6,
                              ExcMessage("Timestep zero evolved initialized Maxwell stress."));
                  AssertThrow(std::abs(properties[stress_position+1]
                                       -(-4.e6+8.e6*particle.get_location()[0]))
                              <= 1.e-8*8.e6,
                              ExcMessage("Timestep zero evolved initialized Maxwell stress."));
                  AssertThrow(std::isfinite(properties[H_position])
                              && properties[H_position] >= 0.0,
                              ExcMessage("Initialized H_0 is inadmissible."));
                  H_zero.emplace(particle.get_id(), properties[H_position]);
                }

              const auto &current_I_h =
                MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
                ::current_normalization_integrals(model);
              const unsigned int previous_I_h_property =
                fault_manager.get_property_index("phase field fault previous I h");
              const unsigned int previous_I_h_position =
                fault_manager.get_property_information()[previous_I_h_property].position;
              for (unsigned int fault = 0; fault < current_I_h.size(); ++fault)
                for (unsigned int vertex = 0;
                     vertex < current_I_h[fault].size(); ++vertex)
                  AssertThrow(
                    fault_manager.get_fault(fault).get_properties(vertex)
                      [previous_I_h_position] == current_I_h[fault][vertex],
                    ExcMessage("I_h,0 was not retained as the previous-I_h snapshot."));
            }
          model.validate_reconstructed_fault_constitutive_state();
          return {"Reconstructed-fault Stage-I solve:", "verified"};
        }

      private:
        std::vector<std::vector<double>> theta_zero;
        std::map<types::particle_index, double> H_zero;
    };


    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhaseFieldFaultStageI,
                                  "verify phase field fault stage i",
                                  "Verify the committed lower-bound state after the "
                                  "Stage-I coupled mechanical solve.")
  }
}
