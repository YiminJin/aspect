/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/reconstructed_fault/manager.h>

#include <deal.II/fe/fe_values.h>

#include <map>

namespace aspect
{
  namespace
  {
    template <int dim>
    const MaterialModel::PhaseFieldFault<dim> &
    checked_phase_field_fault(const SimulatorAccess<dim> &simulator)
    {
      const auto *phase_field_fault =
        dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
          &simulator.get_material_model());
      AssertThrow(phase_field_fault != nullptr,
                  ExcMessage("The reconstructed-fault Stokes assembler requires the "
                             "'Phase field fault' material model."));
      return *phase_field_fault;
    }


    template <int dim>
    void
    validate_fault_vector(const ReconstructedFaultManager<dim> &fault_manager,
                          const std::vector<std::vector<double>> &values,
                          const std::string &name)
    {
      const auto &faults = fault_manager.get_faults();
      AssertThrow(values.size() == faults.size(),
                  ExcMessage("The " + name + " has the wrong number of faults."));
      for (unsigned int fault = 0; fault < faults.size(); ++fault)
        AssertThrow(values[fault].size() == faults[fault].n_vertices(),
                    ExcMessage("The " + name + " has the wrong number of vertices "
                               "for reconstructed fault "
                               + Utilities::int_to_string(fault) + "."));
    }


    template <int dim>
    std::vector<typename MaterialModel::PhaseFieldFault<dim>::
                ReconstructedFaultBulkPointResponse>
    evaluate_bulk_point_responses(
      const SimulatorAccess<dim> &simulator,
      const MaterialModel::PhaseFieldFault<dim> &phase_field_fault,
      const FEValues<dim> &fe_values,
      const LinearAlgebra::BlockVector &bulk_state,
      const std::vector<typename ReconstructedFaultManager<dim>::
                        StokesQPFaultAssociation> &associations)
    {
      AssertDimension(fe_values.n_quadrature_points, associations.size());
      const unsigned int n_q_points = fe_values.n_quadrature_points;
      const Introspection<dim> &introspection = simulator.introspection();

      std::vector<double> temperature(n_q_points);
      std::vector<double> phase_field(n_q_points);
      std::vector<double> previous_phase_field(n_q_points);
      fe_values[introspection.extractors.temperature].get_function_values(
        bulk_state, temperature);
      const FEValuesExtractors::Scalar phase_field_extractor(
        introspection.variable("phase_field").first_component_index);
      fe_values[phase_field_extractor].get_function_values(bulk_state, phase_field);
      fe_values[phase_field_extractor].get_function_values(
        simulator.get_old_solution(), previous_phase_field);

      const std::vector<unsigned int> &chemical_fields =
        introspection.chemical_composition_field_indices();
      std::vector<std::vector<double>> chemical_values(
        chemical_fields.size(), std::vector<double>(n_q_points));
      for (unsigned int c = 0; c < chemical_fields.size(); ++c)
        {
          const unsigned int component =
            introspection.component_indices.compositional_fields[chemical_fields[c]];
          fe_values[FEValuesExtractors::Scalar(component)].get_function_values(
            bulk_state, chemical_values[c]);
        }

      std::vector<typename MaterialModel::PhaseFieldFault<dim>::
                  ReconstructedFaultBulkPointResponse> responses(n_q_points);
      for (unsigned int q = 0; q < n_q_points; ++q)
        if (associations[q].active)
          {
            std::vector<double> compositions(chemical_fields.size());
            for (unsigned int c = 0; c < chemical_fields.size(); ++c)
              compositions[c] = chemical_values[c][q];

            typename MaterialModel::PhaseFieldFault<dim>::
              ReconstructedFaultBulkPointInputs inputs;
            inputs.fault_index = associations[q].fault_index;
            inputs.segment_index = associations[q].segment_index;
            inputs.xi = associations[q].xi;
            inputs.phase_field = phase_field[q];
            inputs.previous_phase_field = previous_phase_field[q];
            inputs.temperature = temperature[q];
            inputs.bulk_material_fractions =
              MaterialModel::MaterialUtilities::compute_composition_fractions(
                compositions);
            responses[q] =
              phase_field_fault.evaluate_reconstructed_fault_bulk_point(inputs);
          }
      return responses;
    }


    template <int dim>
    SymmetricTensor<2,dim>
    slip_tensor(const typename ReconstructedFaultManager<dim>::
                StokesQPFaultAssociation &association)
    {
      return symmetrize(outer_product(association.tangent,
                                     association.normal));
    }
  }


  namespace Assemblers
  {
    template <int dim>
    struct ReconstructedFaultStokes<dim>::BLinearization
    {
      std::map<CellId, std::vector<SymmetricTensor<2,dim>>> coefficients;
      unsigned int geometry_cache_rebuild_count = 0;
    };


    template <int dim>
    ReconstructedFaultStokes<dim>::ReconstructedFaultStokes(
      const Simulator<dim> &simulator)
      :
      SimulatorAccess<dim>(simulator),
      phase_field_fault(checked_phase_field_fault<dim>(*this))
    {}


    template <int dim>
    ReconstructedFaultStokes<dim>::~ReconstructedFaultStokes() = default;


    template <int dim>
    void
    ReconstructedFaultStokes<dim>::linearize_B(
      const LinearAlgebra::BlockVector &bulk_linearization_point)
    {
      AssertThrow(bulk_linearization_point.size()
                  == this->get_dof_handler().n_dofs(),
                  ExcMessage("The reconstructed-fault B linearization point has "
                             "the wrong size."));
      B_linearization.reset();
      phase_field_fault.validate_reconstructed_fault_constitutive_state();
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      fault_manager.prepare_stokes_qp_projection_cache();

      auto candidate = std::make_unique<BLinearization>();
      candidate->geometry_cache_rebuild_count =
        fault_manager.get_stokes_qp_cache_diagnostics().rebuild_count;
      const Quadrature<dim> &quadrature =
        this->introspection().quadratures.velocities;
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                              update_values | update_quadrature_points);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const auto &associations =
              fault_manager.get_stokes_qp_fault_associations(
                cell->id(), quadrature, fe_values.get_quadrature_points());
            const auto responses = evaluate_bulk_point_responses(
              *this, phase_field_fault, fe_values,
              bulk_linearization_point, associations);
            std::vector<SymmetricTensor<2,dim>> coefficients(quadrature.size());
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              if (associations[q].active)
                coefficients[q] = 2.0 * responses[q].kappa
                                  * responses[q].localization_factor
                                  * slip_tensor<dim>(associations[q]);
            candidate->coefficients.emplace(cell->id(), std::move(coefficients));
          }

      B_linearization = std::move(candidate);
      ++B_linearization_rebuild_count;
    }


    template <int dim>
    void
    ReconstructedFaultStokes<dim>::evaluate_slip_dependent_bulk_residual(
      const LinearAlgebra::BlockVector &bulk_state,
      const FaultVector &slip_rate,
      LinearAlgebra::BlockVector &result) const
    {
      AssertThrow(bulk_state.size() == this->get_dof_handler().n_dofs(),
                  ExcMessage("The reconstructed-fault bulk state has the wrong size."));
      AssertThrow(result.size() == this->get_dof_handler().n_dofs(),
                  ExcMessage("The reconstructed-fault bulk residual vector has the "
                             "wrong size."));
      result = 0;
      phase_field_fault.validate_reconstructed_fault_constitutive_state();
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      validate_fault_vector(fault_manager, slip_rate, "fault slip-rate vector");
      fault_manager.prepare_stokes_qp_projection_cache();

      const Quadrature<dim> &quadrature =
        this->introspection().quadratures.velocities;
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                              update_values | update_gradients
                              | update_quadrature_points | update_JxW_values);
      std::vector<types::global_dof_index> dof_indices(this->get_fe().dofs_per_cell);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const auto &associations =
              fault_manager.get_stokes_qp_fault_associations(
                cell->id(), quadrature, fe_values.get_quadrature_points());
            const auto responses = evaluate_bulk_point_responses(
              *this, phase_field_fault, fe_values, bulk_state, associations);
            Vector<double> local_residual(this->get_fe().dofs_per_cell);

            for (unsigned int q = 0; q < quadrature.size(); ++q)
              if (associations[q].active)
                {
                  const auto &association = associations[q];
                  const double V = association.shape_0
                                   * slip_rate[association.fault_index]
                                              [association.segment_index]
                                   + association.shape_1
                                   * slip_rate[association.fault_index]
                                              [association.segment_index+1];
                  const double crack_strain_rate =
                    responses[q].localization_factor * V
                    + responses[q].history_correction;
                  const SymmetricTensor<2,dim> stress =
                    2.0 * responses[q].kappa * crack_strain_rate
                    * slip_tensor<dim>(association);
                  for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
                    if (this->introspection().component_masks.velocities[
                          this->get_fe().system_to_component_index(i).first])
                      local_residual[i] -=
                        stress
                        * fe_values[this->introspection().extractors.velocities]
                            .symmetric_gradient(i,q)
                        * fe_values.JxW(q);
                }

            cell->get_dof_indices(dof_indices);
            this->get_current_constraints().distribute_local_to_global(
              local_residual, dof_indices, result);
          }
      result.compress(VectorOperation::add);
    }


    template <int dim>
    void
    ReconstructedFaultStokes<dim>::apply_B(
      const FaultVector &fault_direction,
      LinearAlgebra::BlockVector &result) const
    {
      AssertThrow(B_linearization != nullptr,
                  ExcMessage("B must be linearized before applying it."));
      AssertThrow(result.size() == this->get_dof_handler().n_dofs(),
                  ExcMessage("The reconstructed-fault B result vector has the wrong size."));
      result = 0;
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      validate_fault_vector(fault_manager, fault_direction,
                            "fault-direction vector");
      AssertThrow(B_linearization->geometry_cache_rebuild_count
                  == fault_manager.get_stokes_qp_cache_diagnostics().rebuild_count,
                  ExcMessage("The reconstructed-fault geometry cache changed after B "
                             "was linearized."));

      const Quadrature<dim> &quadrature =
        this->introspection().quadratures.velocities;
      FEValues<dim> fe_values(this->get_mapping(), this->get_fe(), quadrature,
                              update_gradients | update_quadrature_points
                              | update_JxW_values);
      std::vector<types::global_dof_index> dof_indices(this->get_fe().dofs_per_cell);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const auto &associations =
              fault_manager.get_stokes_qp_fault_associations(
                cell->id(), quadrature, fe_values.get_quadrature_points());
            const auto coefficient = B_linearization->coefficients.find(cell->id());
            Assert(coefficient != B_linearization->coefficients.end(),
                   ExcInternalError());
            AssertDimension(coefficient->second.size(), quadrature.size());
            Vector<double> local_result(this->get_fe().dofs_per_cell);

            for (unsigned int q = 0; q < quadrature.size(); ++q)
              if (associations[q].active)
                {
                  const auto &association = associations[q];
                  const double direction = association.shape_0
                                           * fault_direction[association.fault_index]
                                                            [association.segment_index]
                                           + association.shape_1
                                           * fault_direction[association.fault_index]
                                                            [association.segment_index+1];
                  for (unsigned int i = 0; i < this->get_fe().dofs_per_cell; ++i)
                    if (this->introspection().component_masks.velocities[
                          this->get_fe().system_to_component_index(i).first])
                      local_result[i] +=
                        direction * coefficient->second[q]
                        * fe_values[this->introspection().extractors.velocities]
                            .symmetric_gradient(i,q)
                        * fe_values.JxW(q);
                }

            cell->get_dof_indices(dof_indices);
            this->get_current_constraints().distribute_local_to_global(
              local_result, dof_indices, result);
          }
      result.compress(VectorOperation::add);
    }


    template <int dim>
    void
    ReconstructedFaultStokes<dim>::execute(
      internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
      internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      auto &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim> &>(
        scratch_base);
      auto &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim> &>(
        data_base);
      ReconstructedFaultManager<dim> &fault_manager =
        this->get_reconstructed_fault_manager();
      const Quadrature<dim> &quadrature =
        this->introspection().quadratures.velocities;
      const auto &associations = fault_manager.get_stokes_qp_fault_associations(
        scratch.cell->id(), quadrature,
        scratch.finite_element_values.get_quadrature_points());
      const auto responses = evaluate_bulk_point_responses(
        *this, phase_field_fault, scratch.finite_element_values,
        this->get_current_linearization_point(), associations);

      const FiniteElement<dim> &fe = this->get_fe();
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        if (associations[q].active)
          {
            const auto &association = associations[q];
            const double V = fault_manager.interpolate_slip_rate(
              association.fault_index, association.segment_index, association.xi);
            const double crack_strain_rate =
              responses[q].localization_factor * V
              + responses[q].history_correction;
            const SymmetricTensor<2,dim> stress =
              2.0 * responses[q].kappa * crack_strain_rate
              * slip_tensor<dim>(association);

            for (unsigned int i = 0, i_stokes = 0;
                 i_stokes < data.local_rhs.size(); ++i)
              if (this->introspection().is_stokes_component(
                    fe.system_to_component_index(i).first))
                {
                  if (this->introspection().component_masks.velocities[
                        fe.system_to_component_index(i).first])
                    data.local_rhs[i_stokes] +=
                      stress
                      * scratch.finite_element_values[
                          this->introspection().extractors.velocities]
                          .symmetric_gradient(i,q)
                      * scratch.finite_element_values.JxW(q);
                  ++i_stokes;
                }
          }
    }


    template <int dim>
    unsigned int
    ReconstructedFaultStokes<dim>::get_B_linearization_rebuild_count() const
    {
      return B_linearization_rebuild_count;
    }


#define INSTANTIATE(dim) template class ReconstructedFaultStokes<dim>;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
