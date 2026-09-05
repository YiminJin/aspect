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
    class VerifySingularPhaseFieldFaultSurfaceSystem : public Interface<dim>,
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
          using SurfaceSystem = ReconstructedFaultSurfaceSystem<dim>;
          typename SurfaceSystem::FaultVector slip_rate(
            fault_manager.get_faults().size());
          for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
            slip_rate[fault].assign(
              fault_manager.get_fault(fault).n_vertices(), 2.e-6);

          SurfaceSystem surface_system(this->get_simulator());
          surface_system.linearize_surface_system(this->get_solution(), slip_rate);

          const auto &cached_associations =
            fault_manager.get_locally_owned_particle_fault_associations();
          auto &associations = const_cast<std::vector<typename
            ReconstructedFaultManager<dim>::ParticleFaultAssociation> &>(
              cached_associations);
          for (auto &association : associations)
            association.active = false;

          try
            {
              surface_system.linearize_surface_system(this->get_solution(), slip_rate);
            }
          catch (const std::exception &exception)
            {
              AssertThrow(std::string(exception.what()).find(
                            "Failed to factor reconstructed-fault K_V block")
                          != std::string::npos,
                          ExcMessage("A singular K_V did not produce the expected "
                                     "factorization diagnostic."));
              std::string invalidated_error;
              try
                {
                  typename SurfaceSystem::FaultVector result;
                  surface_system.solve_surface_jacobian(slip_rate, result);
                }
              catch (const std::exception &solve_exception)
                {
                  invalidated_error = solve_exception.what();
                }
              AssertThrow(invalidated_error.find("must be assembled")
                          != std::string::npos,
                          ExcMessage("A failed K_V factorization left a stale or partial "
                                     "surface linearization installed."));
              AssertThrow(false,
                          ExcMessage("Verified Stage-F singular K_V factorization "
                                     "diagnostic."));
            }

          AssertThrow(false,
                      ExcMessage("A deliberately singular Stage-F K_V unexpectedly "
                                 "factorized."));
          return {};
        }
    };



    ASPECT_REGISTER_POSTPROCESSOR(VerifySingularPhaseFieldFaultSurfaceSystem,
                                  "verify singular phase field fault surface system",
                                  "Inject an unsupported Stage-F surface algebra in a "
                                  "test fixture and verify the singular K_V factorization "
                                  "diagnostic.")
  }
}
