/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_simulator_assemblers_reconstructed_fault_stokes_h
#define _aspect_simulator_assemblers_reconstructed_fault_stokes_h

#include <aspect/simulator/assemblers/interface.h>
#include <aspect/simulator_access.h>

#include <array>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldFault;
  }

  namespace Assemblers
  {
    /** Bulk weak-form terms from frozen Maxwell history and fault slip. */
    template <int dim>
    class ReconstructedFaultStokes : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        using FaultVector = std::vector<std::vector<double>>;

        explicit ReconstructedFaultStokes(const Simulator<dim> &simulator);
        ~ReconstructedFaultStokes() override;

        /** Add -R from frozen bulk stress and active fault slip to the Stokes RHS. */
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch,
                     internal::Assembly::CopyData::CopyDataBase<dim> &data) const override;

        /** Freeze 2 kappa chi S at every associated Stokes quadrature point. */
        void linearize_B(const LinearAlgebra::BlockVector &bulk_linearization_point);

        /** Overwrite @p result with the complete fault-induced bulk residual. */
        void evaluate_slip_dependent_bulk_residual(
          const LinearAlgebra::BlockVector &bulk_state,
          const FaultVector &slip_rate,
          LinearAlgebra::BlockVector &result) const;

        /** Overwrite @p result with B times @p fault_direction. */
        void apply_B(const FaultVector &fault_direction,
                     LinearAlgebra::BlockVector &result) const;

        unsigned int get_B_linearization_rebuild_count() const;

      private:
        struct BLinearization;
        const MaterialModel::PhaseFieldFault<dim> &phase_field_fault;
        std::array<unsigned int, SymmetricTensor<2,dim>::n_independent_components>
          stress_composition_indices;
        std::unique_ptr<BLinearization> B_linearization;
        unsigned int B_linearization_rebuild_count = 0;
    };
  }
}

#endif
