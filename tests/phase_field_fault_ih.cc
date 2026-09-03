/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyPhaseFieldFaultIh : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          const auto *const_model = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
            &this->get_material_model());
          AssertThrow(const_model != nullptr, ExcInternalError());
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(*const_model);
          MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
            ::initialize_cohesive_state_from_initial_fields(model);
          const auto &normalizations =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::compute_normalization_integrals(model);
          const double minimum_raw_phase_field =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::current_minimum_raw_normalization_phase_field(model);
          AssertThrow(std::isfinite(minimum_raw_phase_field)
                      && minimum_raw_phase_field >= -MaterialModel::internal::
                           PhaseFieldFaultTestAccess<dim>
                             ::normalization_phase_field_undershoot_tolerance(),
                      ExcMessage("Stage C lifecycle fixture violated the bounded-undershoot invariant."));

          unsigned int n_values = 0;
          double minimum = std::numeric_limits<double>::max();
          double maximum = 0.0;
          double sum = 0.0;
          for (const auto &fault_values : normalizations)
            for (const double value : fault_values)
              {
                AssertThrow(std::isfinite(value) && value > 0.0,
                            ExcMessage("Stage C produced an unusable nodal I_h."));
                ++n_values;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                sum += value;
              }
          AssertThrow(n_values > 0, ExcMessage("Stage C produced no nodal I_h values."));

          const auto &fault_manager = this->get_reconstructed_fault_manager();
          const unsigned int cohesive_property = fault_manager.get_property_index(
            "phase field fault cohesive traction");
          const unsigned int previous_I_h_property = fault_manager.get_property_index(
            "phase field fault previous I h");
          const unsigned int cohesive_position =
            fault_manager.get_property_information()[cohesive_property].position;
          const unsigned int previous_I_h_position =
            fault_manager.get_property_information()[previous_I_h_property].position;
          for (unsigned int fault = 0; fault < fault_manager.get_faults().size(); ++fault)
            for (unsigned int vertex = 0;
                 vertex < fault_manager.get_fault(fault).n_vertices(); ++vertex)
              {
                const ArrayView<const double> properties =
                  fault_manager.get_fault(fault).get_properties(vertex);
                AssertThrow(std::isfinite(properties[cohesive_position])
                            && properties[cohesive_position] >= 0.0,
                            ExcMessage("Stage D produced an invalid initial cohesive traction."));
                AssertThrow(properties[previous_I_h_position]
                            == normalizations[fault][vertex],
                            ExcMessage("Stage D did not commit the initial I_h consistently."));
              }
          const auto &cohesive_diagnostics =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::initial_cohesive_projection_diagnostics(model);
          AssertDimension(cohesive_diagnostics.size(), fault_manager.get_faults().size());
          for (const auto &diagnostic : cohesive_diagnostics)
            AssertThrow(std::isfinite(diagnostic.weighted_rms_residual)
                        && diagnostic.weighted_rms_residual >= 0.0
                        && std::isfinite(diagnostic.maximum_absolute_residual)
                        && diagnostic.maximum_absolute_residual >= 0.0
                        && std::isfinite(diagnostic.normalized_weighted_rms_residual)
                        && diagnostic.normalized_weighted_rms_residual >= 0.0
                        && std::isfinite(diagnostic.normalized_maximum_absolute_residual)
                        && diagnostic.normalized_maximum_absolute_residual >= 0.0,
                        ExcMessage("Stage D produced invalid cohesive-profile diagnostics."));

          AssertThrow(std::isfinite(minimum) && std::isfinite(maximum)
                      && std::isfinite(sum), ExcInternalError());
          return {"Distributed I_h:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhaseFieldFaultIh,
                                  "verify phase field fault I h",
                                  "Run a lifecycle smoke test of the private Stage C distributed "
                                  "I_h evaluator and Stage D cohesive initialization. Fault "
                                  "reconstruction accuracy is not tested.")
  }
}
