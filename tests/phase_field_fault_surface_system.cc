/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "phase_field_fault_test_access.h"

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator_access.h>

#include <deal.II/numerics/vector_tools.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyPhaseFieldFaultSurfaceSystem : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());
          const auto *const_model =
            dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
              &this->get_material_model());
          AssertThrow(const_model != nullptr, ExcInternalError());
          auto &model =
            const_cast<MaterialModel::PhaseFieldFault<dim> &>(*const_model);
          MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
            ::initialize_cohesive_state_from_initial_fields(model);

          ReconstructedFaultManager<dim> &fault_manager =
            this->get_reconstructed_fault_manager();
          const bool stateful_friction =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::fault_friction(model).has_state_variable();
          if (stateful_friction)
            initialize_and_validate_state(model, fault_manager);

          using FaultVector = typename ReconstructedFaultSurfaceSystem<dim>::FaultVector;
          ReconstructedFaultSurfaceSystem<dim> surface_system(
            this->get_simulator());

          FaultVector V(fault_manager.get_faults().size());
          FaultVector direction(fault_manager.get_faults().size());
          for (unsigned int fault = 0; fault < V.size(); ++fault)
            {
              V[fault].assign(fault_manager.get_fault(fault).n_vertices(), 2.e-6);
              direction[fault].resize(V[fault].size());
              for (unsigned int vertex = 0; vertex < direction[fault].size(); ++vertex)
                direction[fault][vertex] =
                  2.e-6 * (0.25 + 0.5 * static_cast<double>(vertex+1)
                                  / static_cast<double>(direction[fault].size()+1));
            }

          AssertThrow(!fault_manager.slip_rates_are_initialized(),
                      ExcMessage("The Stage-F test fixture unexpectedly has committed slip rate."));
          const bool uses_adiabatic_pressure =
            verify_point_response_pressure_mode(model, fault_manager);
          const auto evaluated = surface_system.evaluate_surface_residual(
            this->get_solution(), V);
          assert_replicated(evaluated, this->get_mpi_communicator());
          const auto &linearized = surface_system.linearize_surface_system(
            this->get_solution(), V);
          assert_same_residual(evaluated, linearized);

          double previous_error = std::numeric_limits<double>::max();
          for (const double epsilon : {2.e-1, 1.e-1, 5.e-2, 2.5e-2})
            {
              FaultVector V_plus = V;
              FaultVector V_minus = V;
              for (unsigned int fault = 0; fault < V.size(); ++fault)
                for (unsigned int vertex = 0; vertex < V[fault].size(); ++vertex)
                  {
                    V_plus[fault][vertex] += epsilon*direction[fault][vertex];
                    V_minus[fault][vertex] -= epsilon*direction[fault][vertex];
                  }

              const auto residual_plus = surface_system.evaluate_surface_residual(
                this->get_solution(), V_plus);
              const auto residual_minus = surface_system.evaluate_surface_residual(
                this->get_solution(), V_minus);
              FaultVector minus_finite_difference(V.size());
              for (unsigned int fault = 0; fault < V.size(); ++fault)
                {
                  minus_finite_difference[fault].resize(V[fault].size());
                  for (unsigned int vertex = 0; vertex < V[fault].size(); ++vertex)
                    minus_finite_difference[fault][vertex] =
                      -(residual_plus.values[fault][vertex]
                        - residual_minus.values[fault][vertex])/(2.0*epsilon);
                }

              FaultVector recovered_direction;
              surface_system.solve_surface_jacobian(minus_finite_difference,
                                                    recovered_direction);
              assert_replicated(recovered_direction,
                                this->get_mpi_communicator());
              const double error = relative_maximum_error(recovered_direction,
                                                          direction);
              AssertThrow(error < 0.4*previous_error,
                          ExcMessage("The Stage-F K_V centered-difference error did not "
                                     "decrease at second order under step refinement: "
                                     "previous=" + Utilities::to_string(previous_error)
                                     + ", current=" + Utilities::to_string(error) + "."));
              previous_error = error;
            }
          AssertThrow(previous_error < 1.e-3,
                      ExcMessage("The Stage-F surface Jacobian does not match centered "
                                 "finite differences: relative error="
                                 + Utilities::to_string(previous_error) + "."));
          verify_surface_jacobian_signs(surface_system, V, stateful_friction);
          AssertThrow(!fault_manager.slip_rates_are_initialized(),
                      ExcMessage("Non-committing Stage-F evaluation changed committed "
                                 "slip-rate state."));

          verify_stage_G(model, fault_manager, surface_system, V, direction,
                         uses_adiabatic_pressure);

          return {"Reconstructed-fault surface and bulk coupling:", "verified"};
        }

      private:
        class BulkDirectionFunction : public Function<dim>
        {
          public:
            BulkDirectionFunction(const unsigned int n_components,
                                  const unsigned int velocity_component,
                                  const unsigned int pressure_component,
                                  const SymmetricTensor<2,dim> &strain_rate,
                                  const double pressure)
              :
              Function<dim>(n_components),
              velocity_component(velocity_component),
              pressure_component(pressure_component),
              strain_rate(strain_rate),
              pressure(pressure)
            {}

            double
            value(const Point<dim> &position,
                  const unsigned int component) const override
            {
              if (component >= velocity_component
                  && component < velocity_component+dim)
                {
                  const unsigned int d = component-velocity_component;
                  double result = 0.0;
                  for (unsigned int e = 0; e < dim; ++e)
                    result += strain_rate[d][e]*position[e];
                  return result;
                }
              if (component == pressure_component)
                return pressure;
              return 0.0;
            }

          private:
            const unsigned int velocity_component;
            const unsigned int pressure_component;
            const SymmetricTensor<2,dim> strain_rate;
            const double pressure;
        };


        static bool
        verify_point_response_pressure_mode(
          const MaterialModel::PhaseFieldFault<dim> &model,
          const ReconstructedFaultManager<dim> &fault_manager)
        {
          const ReconstructedFault<dim> &fault = fault_manager.get_fault(0);
          Assert(fault.n_cells() > 0, ExcInternalError());

          const unsigned int segment = fault.n_cells()/2;
          constexpr double xi = 0.5;
          Tensor<1,dim> tangent = fault.vertex(segment+1)-fault.vertex(segment);
          tangent /= tangent.norm();
          Tensor<1,dim> normal;
          normal[0] = -tangent[1];
          normal[1] = tangent[0];

          typename MaterialModel::PhaseFieldFault<dim>::
            ReconstructedFaultPointInputs inputs;
          inputs.fault_index = 0;
          inputs.segment_index = segment;
          inputs.xi = xi;
          inputs.position = (1.0-xi)*fault.vertex(segment)
                            + xi*fault.vertex(segment+1);
          inputs.slip_rate = 2.e-6;
          inputs.phase_field = 0.5;
          inputs.previous_phase_field = 0.4;
          inputs.temperature = 293.0;
          inputs.dynamic_pressure = 3.e6;
          inputs.bulk_material_fractions = {0.5, 0.5};
          inputs.strain_rate[0][0] = 1.e-15;
          inputs.strain_rate[1][1] = -0.5e-15;
          inputs.strain_rate[0][1] = 0.75e-15;
          inputs.slip_tensor = symmetrize(outer_product(tangent, normal));
          inputs.normal_tensor = symmetrize(outer_product(normal, normal));

          const auto baseline = model.evaluate_reconstructed_fault_point(inputs);

          constexpr double pressure_increment = 2.e5;
          auto pressure_inputs = inputs;
          pressure_inputs.dynamic_pressure += pressure_increment;
          const auto pressure_response =
            model.evaluate_reconstructed_fault_point(pressure_inputs);
          const double expected_pressure_change =
            baseline.uses_adiabatic_friction_pressure
            ? 0.0
            : -baseline.friction_coefficient*pressure_increment;
          assert_close(pressure_response.residual_density-baseline.residual_density,
                       expected_pressure_change,
                       "fault-pressure response");

          SymmetricTensor<2,dim> strain_rate_increment;
          strain_rate_increment[0][0] = 0.4e-5;
          strain_rate_increment[1][1] = -0.2e-5;
          strain_rate_increment[0][1] = 0.6e-5;
          auto strain_rate_inputs = inputs;
          strain_rate_inputs.strain_rate += strain_rate_increment;
          const auto strain_rate_response =
            model.evaluate_reconstructed_fault_point(strain_rate_inputs);
          const SymmetricTensor<2,dim> surface_derivative =
            baseline.uses_adiabatic_friction_pressure
            ? inputs.slip_tensor
            : inputs.slip_tensor
              + baseline.friction_coefficient*inputs.normal_tensor;
          const double expected_strain_rate_change =
            2.0*baseline.kappa*(surface_derivative*strain_rate_increment);
          assert_close(strain_rate_response.residual_density-baseline.residual_density,
                       expected_strain_rate_change,
                       "strain-rate response");
          return baseline.uses_adiabatic_friction_pressure;
        }


        LinearAlgebra::BlockVector
        make_bulk_direction(const SymmetricTensor<2,dim> &strain_rate,
                            const double pressure) const
        {
          LinearAlgebra::BlockVector owned(
            this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
          const BulkDirectionFunction function(
            this->get_fe().n_components(),
            this->introspection().variable("velocity").first_component_index,
            this->introspection().variable("pressure").first_component_index,
            strain_rate, pressure);
          VectorTools::interpolate(this->get_mapping(), this->get_dof_handler(),
                                   function, owned);
          this->get_current_constraints().set_zero(owned);
          owned.compress(VectorOperation::insert);

          LinearAlgebra::BlockVector ghosted(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          ghosted = owned;
          return ghosted;
        }


        LinearAlgebra::BlockVector
        perturb_bulk_state(const LinearAlgebra::BlockVector &state,
                           const double factor,
                           const LinearAlgebra::BlockVector &direction) const
        {
          LinearAlgebra::BlockVector owned(
            this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
          LinearAlgebra::BlockVector owned_direction(
            this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
          owned = state;
          owned_direction = direction;
          owned.add(factor, owned_direction);
          owned.compress(VectorOperation::insert);

          LinearAlgebra::BlockVector ghosted(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          ghosted = owned;
          return ghosted;
        }


        LinearAlgebra::BlockVector
        make_owned_system_vector() const
        {
          return LinearAlgebra::BlockVector(
            this->introspection().index_sets.system_partitioning,
            this->get_mpi_communicator());
        }


        static double
        fault_vector_maximum_norm(
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &values)
        {
          double norm = 0.0;
          for (const auto &fault : values)
            for (const double value : fault)
              norm = std::max(norm, std::abs(value));
          return norm;
        }


        static void
        assert_fault_vectors_close(
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &values,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &reference,
          const double relative_tolerance,
          const std::string &description)
        {
          AssertDimension(values.size(), reference.size());
          double error = 0.0;
          double scale = 0.0;
          for (unsigned int fault = 0; fault < values.size(); ++fault)
            {
              AssertDimension(values[fault].size(), reference[fault].size());
              for (unsigned int vertex = 0; vertex < values[fault].size(); ++vertex)
                {
                  error = std::max(error,
                                   std::abs(values[fault][vertex]
                                            - reference[fault][vertex]));
                  scale = std::max(scale, std::abs(reference[fault][vertex]));
                }
            }
          AssertThrow(error <= relative_tolerance*std::max(scale, 1.e-30),
                      ExcMessage("The Stage-G " + description
                                 + " check failed: error="
                                 + Utilities::to_string(error)
                                 + ", scale=" + Utilities::to_string(scale) + "."));
        }


        void
        verify_G_direction(
          ReconstructedFaultSurfaceSystem<dim> &surface_system,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &V,
          const LinearAlgebra::BlockVector &direction,
          const std::string &description) const
        {
          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector action(1);
          action[0].assign(1, 42.0);
          surface_system.apply_G(direction, action);

          constexpr double epsilon = 0.25;
          const LinearAlgebra::BlockVector plus = perturb_bulk_state(
            this->get_solution(), epsilon, direction);
          const LinearAlgebra::BlockVector minus = perturb_bulk_state(
            this->get_solution(), -epsilon, direction);
          const auto residual_plus = surface_system.evaluate_surface_residual(plus, V);
          const auto residual_minus = surface_system.evaluate_surface_residual(minus, V);
          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector
            finite_difference = action;
          for (unsigned int fault = 0; fault < finite_difference.size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < finite_difference[fault].size(); ++vertex)
              finite_difference[fault][vertex] =
                (residual_plus.values[fault][vertex]
                 - residual_minus.values[fault][vertex])/(2.0*epsilon);
          assert_fault_vectors_close(action, finite_difference, 2.e-7,
                                     description + " centered finite difference");
        }


        void
        verify_stage_G(
          MaterialModel::PhaseFieldFault<dim> &model,
          ReconstructedFaultManager<dim> &fault_manager,
          ReconstructedFaultSurfaceSystem<dim> &surface_system,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &V,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &fault_direction,
          const bool uses_adiabatic_pressure) const
        {
          MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
            ::scale_current_normalization_integrals(model, 1.125);
          surface_system.linearize_surface_system(this->get_solution(), V);

          SymmetricTensor<2,dim> velocity_strain_rate;
          velocity_strain_rate[0][0] = 1.5e-5;
          velocity_strain_rate[1][1] = -0.5e-5;
          velocity_strain_rate[0][1] = 0.75e-5;
          const LinearAlgebra::BlockVector velocity_direction =
            make_bulk_direction(velocity_strain_rate, 0.0);
          const LinearAlgebra::BlockVector pressure_direction =
            make_bulk_direction(SymmetricTensor<2,dim>(), 2.e8);
          const LinearAlgebra::BlockVector mixed_direction =
            make_bulk_direction(velocity_strain_rate, 2.e8);

          verify_G_direction(surface_system, V, velocity_direction,
                             "velocity-only G");
          verify_G_direction(surface_system, V, pressure_direction,
                             "pressure-only G");
          verify_G_direction(surface_system, V, mixed_direction,
                             "mixed G");

          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector pressure_action;
          surface_system.apply_G(pressure_direction, pressure_action);
          if (uses_adiabatic_pressure)
            AssertThrow(fault_vector_maximum_norm(pressure_action) == 0.0,
                        ExcMessage("Adiabatic-pressure Stage-G G contains a dynamic "
                                   "pressure contribution."));
          else
            AssertThrow(fault_vector_maximum_norm(pressure_action) > 0.0,
                        ExcMessage("Dynamic-pressure Stage-G G omitted its pressure "
                                   "contribution."));

          constexpr double pressure_scaling = 7.25;
          const LinearAlgebra::BlockVector unit_physical_pressure =
            make_bulk_direction(SymmetricTensor<2,dim>(), 1.0);
          const LinearAlgebra::BlockVector scaled_physical_pressure =
            make_bulk_direction(SymmetricTensor<2,dim>(), pressure_scaling);
          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector unit_action;
          typename ReconstructedFaultSurfaceSystem<dim>::FaultVector scaled_action;
          surface_system.apply_G(unit_physical_pressure, unit_action);
          surface_system.apply_G(scaled_physical_pressure, scaled_action);
          for (unsigned int fault = 0; fault < unit_action.size(); ++fault)
            for (unsigned int vertex = 0; vertex < unit_action[fault].size(); ++vertex)
              unit_action[fault][vertex] *= pressure_scaling;
          assert_fault_vectors_close(scaled_action, unit_action, 2.e-12,
                                     "physical-to-solver pressure scaling");

          Assemblers::ReconstructedFaultStokes<dim> bulk_assembler(
            this->get_simulator());
          const unsigned int geometry_rebuilds_before =
            fault_manager.get_stokes_qp_cache_diagnostics().rebuild_count;
          bulk_assembler.linearize_B(this->get_solution());
          const unsigned int geometry_rebuilds_after =
            fault_manager.get_stokes_qp_cache_diagnostics().rebuild_count;
          AssertThrow(geometry_rebuilds_after == geometry_rebuilds_before+1,
                      ExcMessage("The Stage-G geometry cache was not built exactly once."));
          AssertThrow(bulk_assembler.get_B_linearization_rebuild_count() == 1,
                      ExcMessage("The Stage-G B linearization cache was not built once."));

          auto bulk_fault_direction = fault_direction;
          auto bulk_V = V;
          for (unsigned int fault = 0; fault < bulk_V.size(); ++fault)
            for (unsigned int vertex = 0; vertex < bulk_V[fault].size(); ++vertex)
              {
                bulk_fault_direction[fault][vertex] *= 1.e5;
                bulk_V[fault][vertex] *= 1.e5;
              }
          LinearAlgebra::BlockVector B_direction = make_owned_system_vector();
          B_direction = this->get_solution();
          bulk_assembler.apply_B(bulk_fault_direction, B_direction);
          LinearAlgebra::BlockVector repeated_B_direction = make_owned_system_vector();
          bulk_assembler.apply_B(bulk_fault_direction, repeated_B_direction);
          LinearAlgebra::BlockVector repeated_error = repeated_B_direction;
          repeated_error.add(-1.0, B_direction);
          AssertThrow(repeated_error.l2_norm() == 0.0,
                      ExcMessage("Repeated Stage-G apply_B changed its frozen action."));
          AssertThrow(bulk_assembler.get_B_linearization_rebuild_count() == 1
                      && fault_manager.get_stokes_qp_cache_diagnostics().rebuild_count
                         == geometry_rebuilds_after,
                      ExcMessage("Stage-G apply_B rebuilt a frozen linearization cache."));

          typename Assemblers::ReconstructedFaultStokes<dim>::FaultVector zero_V = V;
          for (auto &fault : zero_V)
            std::fill(fault.begin(), fault.end(), 0.0);
          LinearAlgebra::BlockVector residual_zero = make_owned_system_vector();
          residual_zero = this->get_solution();
          bulk_assembler.evaluate_slip_dependent_bulk_residual(
            this->get_solution(), zero_V, residual_zero);
          AssertThrow(residual_zero.l2_norm() > 0.0,
                      ExcMessage("The Stage-G bulk residual omitted the nonzero frozen "
                                 "Maxwell/cohesive history contribution."));

          LinearAlgebra::BlockVector residual_V = make_owned_system_vector();
          bulk_assembler.evaluate_slip_dependent_bulk_residual(
            this->get_solution(), bulk_V, residual_V);
          LinearAlgebra::BlockVector B_V = make_owned_system_vector();
          bulk_assembler.apply_B(bulk_V, B_V);
          LinearAlgebra::BlockVector affine_error = residual_V;
          affine_error.add(-1.0, residual_zero);
          affine_error.add(1.0, B_V);
          AssertThrow(affine_error.l2_norm()
                      <= 2.e-12*std::max(B_V.l2_norm(), 1.e-30),
                      ExcMessage("The Stage-G absolute bulk residual does not satisfy "
                                 "R_bulk(V)=R_bulk(0)-B V."));

          constexpr double epsilon = 0.125;
          auto V_plus = bulk_V;
          auto V_minus = bulk_V;
          for (unsigned int fault = 0; fault < bulk_V.size(); ++fault)
            for (unsigned int vertex = 0; vertex < bulk_V[fault].size(); ++vertex)
              {
                V_plus[fault][vertex] += epsilon*bulk_fault_direction[fault][vertex];
                V_minus[fault][vertex] -= epsilon*bulk_fault_direction[fault][vertex];
              }
          LinearAlgebra::BlockVector residual_plus = make_owned_system_vector();
          LinearAlgebra::BlockVector residual_minus = make_owned_system_vector();
          bulk_assembler.evaluate_slip_dependent_bulk_residual(
            this->get_solution(), V_plus, residual_plus);
          bulk_assembler.evaluate_slip_dependent_bulk_residual(
            this->get_solution(), V_minus, residual_minus);
          LinearAlgebra::BlockVector B_finite_difference = residual_plus;
          B_finite_difference.add(-1.0, residual_minus);
          B_finite_difference *= 1.0/(2.0*epsilon);
          B_finite_difference.add(1.0, B_direction);
          AssertThrow(B_finite_difference.l2_norm()
                      <= 2.e-12*std::max(B_direction.l2_norm(), 1.e-30),
                      ExcMessage("The Stage-G apply_B action does not match the "
                                 "centered derivative of the bulk residual."));

          verify_additive_execute(fault_manager, bulk_assembler, V);
        }


        void
        verify_additive_execute(
          ReconstructedFaultManager<dim> &fault_manager,
          const Assemblers::ReconstructedFaultStokes<dim> &bulk_assembler,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &V) const
        {
          for (unsigned int fault = 0; fault < V.size(); ++fault)
            fault_manager.initialize_slip_rate(fault, V[fault]);

          LinearAlgebra::BlockVector saved_linearization_point(
            this->introspection().index_sets.system_partitioning,
            this->introspection().index_sets.system_relevant_partitioning,
            this->get_mpi_communicator());
          saved_linearization_point = this->get_current_linearization_point();
          auto &linearization_point =
            const_cast<LinearAlgebra::BlockVector &>(
              this->get_current_linearization_point());
          linearization_point = this->get_solution();

          const Quadrature<dim> &quadrature =
            this->introspection().quadratures.velocities;
          const unsigned int stokes_dofs_per_cell =
            dim * this->get_fe().base_element(
                    this->introspection().base_elements.velocities).dofs_per_cell
            + this->get_fe().base_element(
                this->introspection().base_elements.pressure).dofs_per_cell;
          internal::Assembly::Scratch::StokesSystem<dim> scratch(
            this->get_fe(), this->get_mapping(), quadrature,
            this->introspection().face_quadratures.velocities,
            update_values | update_gradients | update_quadrature_points
            | update_JxW_values,
            update_default,
            this->introspection().n_compositional_fields,
            stokes_dofs_per_cell,
            false, false, false, false, false, false);
          internal::Assembly::CopyData::StokesSystem<dim> data(
            stokes_dofs_per_cell, false);
          Vector<double> increment(stokes_dofs_per_cell);
          bool found_nonzero_cell = false;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                scratch.reinit(cell);
                const auto &associations =
                  fault_manager.get_stokes_qp_fault_associations(
                    cell->id(), quadrature,
                    scratch.finite_element_values.get_quadrature_points());
                if (!std::any_of(associations.begin(), associations.end(),
                                 [](const auto &association)
                                 {
                                   return association.active;
                                 }))
                  continue;

                data.local_rhs = 0.0;
                bulk_assembler.execute(scratch, data);
                if (data.local_rhs.l2_norm() > 0.0)
                  {
                    increment = data.local_rhs;
                    found_nonzero_cell = true;
                    break;
                  }
              }
          const unsigned int n_ranks_with_nonzero_cell = Utilities::MPI::sum(
            found_nonzero_cell ? 1U : 0U, this->get_mpi_communicator());
          AssertThrow(n_ranks_with_nonzero_cell > 0,
                      ExcMessage("The Stage-G additive assembler produced no local "
                                 "fault contribution."));
          if (!found_nonzero_cell)
            {
              linearization_point = saved_linearization_point;
              return;
            }

          data.local_rhs = 3.0;
          bulk_assembler.execute(scratch, data);
          for (unsigned int i = 0; i < data.local_rhs.size(); ++i)
            AssertThrow(std::abs(data.local_rhs[i]-(3.0+increment[i]))
                        <= 4.0*std::numeric_limits<double>::epsilon()
                           * std::max(3.0, std::abs(increment[i])),
                        ExcMessage("The Stage-G execute operation overwrote rather "
                                   "than accumulated into CopyData."));
          linearization_point = saved_linearization_point;
        }

        static void
        assert_close(const double value,
                     const double expected,
                     const std::string &description)
        {
          const double scale = std::max({1.0, std::abs(value), std::abs(expected)});
          AssertThrow(std::abs(value-expected) <= 2.e-6*scale,
                      ExcMessage("The independent Stage-F " + description
                                 + " identity failed: value="
                                 + Utilities::to_string(value)
                                 + ", expected="
                                 + Utilities::to_string(expected) + "."));
        }

        static void
        initialize_and_validate_state(
          MaterialModel::PhaseFieldFault<dim> &model,
          ReconstructedFaultManager<dim> &fault_manager)
        {
          std::string uninitialized_error;
          try
            {
              model.validate_reconstructed_fault_constitutive_state();
            }
          catch (const std::exception &exception)
            {
              uninitialized_error = exception.what();
            }
          AssertThrow(uninitialized_error.find("Theta") != std::string::npos,
                      ExcMessage("Missing rate-and-state Theta was not diagnosed clearly."));

          const unsigned int state_property = fault_manager.get_property_index(
            "phase field fault state");
          const unsigned int state_position =
            fault_manager.get_property_information()[state_property].position;
          for (unsigned int fault = 0;
               fault < fault_manager.get_faults().size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
              fault_manager.get_fault(fault).get_properties(vertex)[state_position] = 0.0;

          std::string invalid_error;
          try
            {
              model.validate_reconstructed_fault_constitutive_state();
            }
          catch (const std::exception &exception)
            {
              invalid_error = exception.what();
            }
          AssertThrow(invalid_error.find("positive") != std::string::npos
                      && invalid_error.find("Theta") != std::string::npos,
                      ExcMessage("Nonpositive rate-and-state Theta was not diagnosed clearly."));

          for (unsigned int fault = 0;
               fault < fault_manager.get_faults().size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
              fault_manager.get_fault(fault).get_properties(vertex)[state_position] = 2.e4;
          model.validate_reconstructed_fault_constitutive_state();
        }

        static void
        assert_replicated(
          const ReconstructedFaultSurfaceResidual &residual,
          const MPI_Comm communicator)
        {
          for (const auto &fault : residual.values)
            for (const double value : fault)
              AssertThrow(Utilities::MPI::min(value, communicator)
                          == Utilities::MPI::max(value, communicator),
                          ExcMessage("The Stage-F surface residual is not replicated "
                                     "identically across MPI ranks."));
        }

        static void
        assert_replicated(
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &values,
          const MPI_Comm communicator)
        {
          for (const auto &fault : values)
            for (const double value : fault)
              AssertThrow(Utilities::MPI::min(value, communicator)
                          == Utilities::MPI::max(value, communicator),
                          ExcMessage("The Stage-F K_V inverse result is not replicated "
                                     "identically across MPI ranks."));
        }

        static void
        assert_same_residual(
          const ReconstructedFaultSurfaceResidual &first,
          const ReconstructedFaultSurfaceResidual &second)
        {
          AssertDimension(first.values.size(), second.values.size());
          for (unsigned int fault = 0; fault < first.values.size(); ++fault)
            {
              AssertDimension(first.values[fault].size(), second.values[fault].size());
              for (unsigned int vertex = 0; vertex < first.values[fault].size(); ++vertex)
                AssertThrow(first.values[fault][vertex]
                            == second.values[fault][vertex],
                            ExcMessage("Residual-only and linearizing Stage-F assembly differ."));
            }
        }

        static double
        relative_maximum_error(const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &values,
                               const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &reference)
        {
          AssertDimension(values.size(), reference.size());
          double error = 0.0;
          double scale = 0.0;
          for (unsigned int fault = 0; fault < values.size(); ++fault)
            {
              AssertDimension(values[fault].size(), reference[fault].size());
              for (unsigned int vertex = 0; vertex < values[fault].size(); ++vertex)
                {
                  error = std::max(error,
                                   std::abs(values[fault][vertex]
                                            - reference[fault][vertex]));
                  scale = std::max(scale, std::abs(reference[fault][vertex]));
                }
            }
          return error/scale;
        }

        void
        verify_surface_jacobian_signs(
          ReconstructedFaultSurfaceSystem<dim> &surface_system,
          const typename ReconstructedFaultSurfaceSystem<dim>::FaultVector &V,
          const bool expect_positive_definite) const
        {
          double minimum_quadratic_form = std::numeric_limits<double>::max();
          double maximum_quadratic_form = -std::numeric_limits<double>::max();
          constexpr double epsilon = 5.e-2;
          for (unsigned int target_fault = 0; target_fault < V.size(); ++target_fault)
            for (unsigned int target_vertex = 0;
                 target_vertex < V[target_fault].size(); ++target_vertex)
              {
                typename ReconstructedFaultSurfaceSystem<dim>::FaultVector
                  V_plus = V;
                typename ReconstructedFaultSurfaceSystem<dim>::FaultVector
                  V_minus = V;
                const double direction = 0.5*V[target_fault][target_vertex];
                V_plus[target_fault][target_vertex] += epsilon*direction;
                V_minus[target_fault][target_vertex] -= epsilon*direction;
                const auto residual_plus = surface_system.evaluate_surface_residual(
                  this->get_solution(), V_plus);
                const auto residual_minus = surface_system.evaluate_surface_residual(
                  this->get_solution(), V_minus);
                const double quadratic_form =
                  -direction
                  * (residual_plus.values[target_fault][target_vertex]
                     - residual_minus.values[target_fault][target_vertex])
                  / (2.0*epsilon);
                minimum_quadratic_form = std::min(minimum_quadratic_form,
                                                   quadratic_form);
                maximum_quadratic_form = std::max(maximum_quadratic_form,
                                                   quadratic_form);
              }

          if (expect_positive_definite)
            AssertThrow(minimum_quadratic_form > 0.0,
                        ExcMessage("The positive-K_V Stage-F fixture has a nonpositive "
                                   "coordinate quadratic form."));
          else
            AssertThrow(minimum_quadratic_form < 0.0
                        && maximum_quadratic_form > 0.0,
                        ExcMessage("The rate-dependent Stage-F fixture does not exercise "
                                   "a sign-indefinite K_V: minimum coordinate form="
                                   + Utilities::to_string(minimum_quadratic_form)
                                   + ", maximum="
                                   + Utilities::to_string(maximum_quadratic_form) + "."));
        }
    };



    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhaseFieldFaultSurfaceSystem,
                                  "verify phase field fault surface system",
                                  "Verify the Stage-F/G non-committing surface and bulk "
                                  "residuals, K_V inverse, B and G actions, cache lifetimes, "
                                  "and configured fault-pressure mode.")
  }
}
