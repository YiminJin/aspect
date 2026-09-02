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
          const auto &normalizations =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
              ::compute_normalization_integrals(model);

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

          AssertThrow(std::isfinite(minimum) && std::isfinite(maximum)
                      && std::isfinite(sum), ExcInternalError());
          return {"Distributed I_h:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhaseFieldFaultIh,
                                  "verify phase field fault I h",
                                  "Run and verify the private Stage C distributed I_h evaluator.")
  }
}
