/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/particle/manager.h>

#include <iostream>
#include <memory>

namespace aspect
{
  namespace Postprocess
  {
    /** Satisfy the postprocessor name inherited from the common Stage-I fixture. */
    template <int dim>
    class RollbackFixtureStageIPostprocessor : public Interface<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          return {"", ""};
        }
    };


    ASPECT_REGISTER_POSTPROCESSOR(RollbackFixtureStageIPostprocessor,
                                  "verify phase field fault stage i",
                                  "Placeholder for the inherited Stage-I fixture; "
                                  "the rollback signal performs this test's checks.")
  }


  namespace
  {
    template <int dim>
    struct StageIRollbackState
    {
      std::unique_ptr<SimulatorAccess<dim>> simulator;
      std::unique_ptr<LinearAlgebra::BlockVector> production_solution;
      std::vector<std::vector<double>> surface_history;
      std::map<types::particle_index, std::vector<double>> particle_history;
    };


    template <int dim>
    StageIRollbackState<dim> &
    rollback_state()
    {
      static StageIRollbackState<dim> state;
      return state;
    }


    template <int dim>
    LinearAlgebra::BlockVector
    owned_copy(const SimulatorAccess<dim> &simulator,
               const LinearAlgebra::BlockVector &source)
    {
      LinearAlgebra::BlockVector result(
        simulator.introspection().index_sets.system_partitioning,
        simulator.get_mpi_communicator());
      result = source;
      return result;
    }


    template <int dim>
    void
    capture_initial_bulk_state(const SimulatorAccess<dim> &simulator)
    {
      auto &state = rollback_state<dim>();
      state = StageIRollbackState<dim>();
      state.simulator = std::make_unique<SimulatorAccess<dim>>(simulator);
      state.production_solution =
        std::make_unique<LinearAlgebra::BlockVector>(
          owned_copy(simulator, simulator.get_solution()));
    }


    template <int dim>
    void capture_prepared_histories(const SimulatorAccess<dim> &simulator,
                                   AffineConstraints<double> &)
    {
      auto &state = rollback_state<dim>();
      const auto &manager = simulator.get_reconstructed_fault_manager();
      if (!manager.slip_rates_are_initialized() || !state.surface_history.empty())
        return;
      for (const auto &fault : manager.get_faults())
        for (unsigned int vertex=0; vertex<fault.n_vertices(); ++vertex)
          {
            const auto values = fault.get_properties(vertex);
            state.surface_history.emplace_back(values.begin(), values.end());
          }
      for (const auto &particle : simulator.get_phase_field_handler()
           .get_associated_particle_manager().get_particle_handler())
        {
          const auto values = particle.get_properties();
          state.particle_history.emplace(particle.get_id(),
            std::vector<double>(values.begin(), values.end()));
        }
    }


    template <int dim>
    void
    verify_failed_solve_rollback(const SolverControl &control)
    {
      if (control.last_check() != SolverControl::failure)
        return;

      auto &state = rollback_state<dim>();
      AssertThrow(state.simulator != nullptr && state.production_solution != nullptr,
                  ExcMessage("The Stage-I rollback test did not capture the "
                             "pre-solve bulk state."));

      LinearAlgebra::BlockVector solution_difference(*state.production_solution);
      solution_difference -= owned_copy(*state.simulator,
                                        state.simulator->get_solution());
      const auto &blocks = state.simulator->introspection().block_indices;
      AssertThrow(solution_difference.block(blocks.velocities).l2_norm() == 0.0
                  && solution_difference.block(blocks.pressure).l2_norm() == 0.0,
                  ExcMessage("A failed Stage-I solve did not restore the bulk state."));

      const ReconstructedFaultManager<dim> &manager =
        state.simulator->get_reconstructed_fault_manager();
      const auto &model = Plugins::get_plugin_as_type<
        const MaterialModel::PhaseFieldFault<dim>>(
          state.simulator->get_material_model());
      AssertThrow(!state.surface_history.empty() && !state.particle_history.empty(),
                  ExcMessage("The rollback fixture did not snapshot prepared histories."));
      unsigned int entry = 0;
      for (const auto &fault : manager.get_faults())
        for (unsigned int vertex=0; vertex<fault.n_vertices(); ++vertex)
          {
            const auto values = fault.get_properties(vertex);
            AssertThrow(std::equal(values.begin(), values.end(), state.surface_history[entry++].begin()),
                        ExcMessage("Failed Newton changed committed surface histories."));
          }
      for (const auto &particle : state.simulator->get_phase_field_handler()
           .get_associated_particle_manager().get_particle_handler())
        {
          const auto values = particle.get_properties();
          const auto &saved = state.particle_history.at(particle.get_id());
          AssertThrow(std::equal(values.begin(), values.end(), saved.begin()),
                      ExcMessage("Failed Newton changed particle Maxwell/H histories."));
        }
      for (unsigned int fault = 0; fault < manager.get_faults().size(); ++fault)
        for (unsigned int vertex = 0;
             vertex < manager.get_faults()[fault].n_vertices(); ++vertex)
          AssertThrow(
            manager.get_slip_rate(fault)[vertex]
              == model.minimum_fault_slip_rate()
            && manager.get_timestep_committed_slip_rate(fault)[vertex]
               == model.minimum_fault_slip_rate(),
            ExcMessage("A failed Stage-I solve did not restore current and "
                       "timestep-committed V."));

      std::cout << "Stage-I rollback after an accepted Newton update: verified"
                << std::endl;
    }
  }


  template <int dim>
  void
  connect_stage_i_rollback_signals(SimulatorSignals<dim> &signals)
  {
    signals.post_set_initial_state.connect(
      &capture_initial_bulk_state<dim>);
    signals.post_nonlinear_solver.connect(
      &verify_failed_solve_rollback<dim>);
    signals.post_constraints_creation.connect(&capture_prepared_histories<dim>);
  }


  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_stage_i_rollback_signals<2>,
                                    connect_stage_i_rollback_signals<3>)
}
