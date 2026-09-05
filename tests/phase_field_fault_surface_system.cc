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
#include <aspect/reconstructed_fault.h>
#include <aspect/simulator/reconstructed_fault_surface_system.h>
#include <aspect/simulator_access.h>

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

          return {"Reconstructed-fault surface system:", "verified"};
        }

      private:
        static void
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
                                  "Verify the Stage-F non-committing surface residual, "
                                  "K_V factorization, inverse, and centered finite-difference "
                                  "Jacobian in either configured fault-pressure mode.")
  }
}
