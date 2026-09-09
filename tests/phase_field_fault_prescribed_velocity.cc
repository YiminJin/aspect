/* Copyright (C) 2026 - by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>

namespace aspect
{
  namespace
  {
    struct FirstIterateErrors
    {
      double prescribed_velocity = 0.0;
      double stokes_constraint = 0.0;
    };

    template <int dim>
    std::map<unsigned int, FirstIterateErrors> &first_iterate_errors()
    {
      static std::map<unsigned int, FirstIterateErrors> errors;
      return errors;
    }

    template <int dim>
    void check_first_iterate(const SimulatorAccess<dim> &simulator)
    {
      auto &errors = first_iterate_errors<dim>();
      if (!simulator.get_reconstructed_fault_manager().slip_rates_are_initialized()
          || errors.count(simulator.get_timestep_number()))
        return;

      // Inspect the actual first residual input, independently of the solver's
      // working constraints (which must be homogeneous for Newton directions).
      std::map<types::global_dof_index, double> prescribed;
      const auto &manager = simulator.get_boundary_velocity_manager();
      for (const auto boundary : manager.get_prescribed_boundary_velocity_indicators())
        {
          Utilities::VectorFunctionFromVelocityFunctionObject<dim> velocity(
            simulator.introspection().n_components,
            [&](const Point<dim> &point)
            { return manager.boundary_velocity(boundary, point); });
          VectorTools::interpolate_boundary_values(
            simulator.get_mapping(), simulator.get_dof_handler(), boundary,
            velocity, prescribed, manager.get_component_mask(boundary));
        }
      double error = 0.0;
      for (const auto &entry : prescribed)
        if (simulator.get_dof_handler().locally_owned_dofs().is_element(entry.first))
          error = std::max(error, std::abs(
            simulator.get_current_linearization_point()[entry.first] - entry.second));
      FirstIterateErrors snapshot;
      snapshot.prescribed_velocity = error;
      const auto &blocks = simulator.introspection().block_indices;
      const auto &solution = simulator.get_current_linearization_point();
      const auto n_stokes_dofs = solution.block(blocks.velocities).size()
                                 + solution.block(blocks.pressure).size();
      for (const auto &line : simulator.get_current_constraints().get_lines())
        if (line.index < n_stokes_dofs)
          snapshot.stokes_constraint = std::max(snapshot.stokes_constraint,
                                                 std::abs(line.inhomogeneity));
      errors.emplace(simulator.get_timestep_number(), snapshot);
      simulator.get_pcout() << "First coupled iterate boundary error at step "
                           << simulator.get_timestep_number() << ": " << error << " m/s" << std::endl;
    }

    // The coupled path does not emit the ordinary Newton pre-assembly signal.
    // This no-op test assembler samples its first local residual input instead.
    // The fixtures use ASPECT's single-threaded assembly. No MPI collective or
    // exception is introduced inside the worker; validation follows the solve.
    template <int dim>
    class FirstIterateProbe : public Assemblers::Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        explicit FirstIterateProbe(const SimulatorAccess<dim> &simulator)
          : SimulatorAccess<dim>(simulator) {}

        void execute(internal::Assembly::Scratch::ScratchBase<dim> &,
                     internal::Assembly::CopyData::CopyDataBase<dim> &) const override
        { check_first_iterate<dim>(*this); }
    };

    template <int dim>
    void add_iterate_probe(const SimulatorAccess<dim> &simulator,
                           Assemblers::Manager<dim> &assemblers)
    {
      assemblers.stokes_system.insert(assemblers.stokes_system.begin(),
        std::make_unique<FirstIterateProbe<dim>>(simulator));
    }
  }

  template <int dim>
  void connect_prescribed_velocity_check(SimulatorSignals<dim> &signals)
  {
    signals.set_assemblers.connect(&add_iterate_probe<dim>);
  }

  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_prescribed_velocity_check<2>,
                                    connect_prescribed_velocity_check<3>)

  namespace Postprocess
  {
    template <int dim>
    class VerifyFaultPrescribedVelocity : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &errors = first_iterate_errors<dim>();
          AssertThrow(errors.count(this->get_timestep_number()),
                      ExcMessage("First coupled residual was not inspected."));
          AssertThrow(Utilities::MPI::max(errors.at(this->get_timestep_number()).prescribed_velocity,
                                         this->get_mpi_communicator()) < 1e-13,
                      ExcMessage("First coupled residual violates current prescribed velocities."));
          AssertThrow(Utilities::MPI::max(errors.at(this->get_timestep_number()).stokes_constraint,
                                         this->get_mpi_communicator()) == 0.0,
                      ExcMessage("Newton assembly must not subtract a second physical lift."));
          // The cohesive initialization can drive interior flow even with
          // zero Maxwell history. Only the prescribed boundary is constant.
          const auto expected = this->get_boundary_velocity_manager().boundary_velocity(0, Point<dim>());
          const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(
            this->get_dof_handler(), this->introspection().component_masks.velocities);
          const auto &fe = this->get_fe();
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          double error = 0.0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                for (unsigned int i=0; i<indices.size(); ++i)
                  for (unsigned int d=0; d<dim; ++d)
                    if (boundary_dofs.is_element(indices[i])
                        && fe.system_to_component_index(i).first
                        == this->introspection().component_indices.velocities[d])
                      error = std::max(error, std::abs(this->get_solution()[indices[i]] - expected[d]));
              }
          error = Utilities::MPI::max(error, this->get_mpi_communicator());
          AssertThrow(error < 1e-13, ExcMessage("Coupled solve changed prescribed boundary velocities."));

          // Exercise both production constitutive entry points. At zero the
          // old FE placeholder must be irrelevant; on a real step, changing
          // the previous profile must still change the history correction.
          const auto &model = Plugins::get_plugin_as_type<
            const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &fault = this->get_reconstructed_fault_manager().get_fault(0);
          const unsigned int segment = fault.n_cells()/2;
          typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs bulk;
          bulk.fault_index = 0;
          bulk.segment_index = segment;
          bulk.xi = 0.5;
          bulk.phase_field = 0.5;
          bulk.previous_phase_field = 0.0;
          bulk.temperature = 293.0;
          bulk.bulk_material_fractions = {0.5, 0.5};
          const auto zero_previous_bulk = model.evaluate_reconstructed_fault_bulk_point(bulk);
          bulk.previous_phase_field = bulk.phase_field;
          const auto matching_previous_bulk = model.evaluate_reconstructed_fault_bulk_point(bulk);

          typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs point;
          point.fault_index = bulk.fault_index;
          point.segment_index = segment;
          point.xi = bulk.xi;
          point.position = 0.5*(fault.vertex(segment)+fault.vertex(segment+1));
          point.phase_field = bulk.phase_field;
          point.previous_phase_field = 0.0;
          point.temperature = bulk.temperature;
          point.bulk_material_fractions = bulk.bulk_material_fractions;
          point.slip_rate = 1e-6;
          point.dynamic_pressure = 0.0;
          Tensor<1,dim> tangent = fault.vertex(segment+1)-fault.vertex(segment);
          tangent /= tangent.norm();
          Tensor<1,dim> normal;
          normal[0] = -tangent[1];
          normal[1] = tangent[0];
          point.slip_tensor = symmetrize(outer_product(tangent, normal));
          point.normal_tensor = symmetrize(outer_product(normal, normal));
          const auto zero_previous_surface = model.evaluate_reconstructed_fault_point(point);
          point.previous_phase_field = point.phase_field;
          const auto matching_previous_surface = model.evaluate_reconstructed_fault_point(point);
          if (this->get_timestep_number() == 0)
            {
              AssertThrow(zero_previous_bulk.history_correction == matching_previous_bulk.history_correction
                          && zero_previous_surface.residual_density == matching_previous_surface.residual_density,
                          ExcMessage("Initial mechanics treated the zero old FE phase field as physical history."));
            }
          else
            {
              AssertThrow(zero_previous_bulk.history_correction != matching_previous_bulk.history_correction
                          && zero_previous_surface.residual_density != matching_previous_surface.residual_density,
                          ExcMessage("Later mechanics ignored the actual previous phase profile."));
            }
          return {"Reconstructed-fault prescribed velocity:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyFaultPrescribedVelocity,
                                  "verify phase field fault stage i",
                                  "Verify physical base constraints and absence of double lifting.")
  }
}
