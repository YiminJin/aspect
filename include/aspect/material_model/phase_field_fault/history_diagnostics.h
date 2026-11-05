/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_material_model_phase_field_fault_history_diagnostics_h
#define _aspect_material_model_phase_field_fault_history_diagnostics_h

#include <aspect/global.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/particles/particle_accessor.h>

#include <fstream>
#include <set>
#include <string>

namespace aspect
{
  namespace internal
  {
    /** Per-history-call streams only; never owns or resamples physical history. */
    template <int dim>
    class FaultHistoryDiagnostics
    {
      public:
        /** Open a rank-local optional stream for this call/step; no history reads. */
        void open_stress_cycle(const std::string &output_directory,
                               unsigned int step, MPI_Comm communicator, bool spin = false);
        /** Open source audit selected by the existing environment controls. */
        void open_source_history(const std::string &output_directory,
                                 unsigned int step, MPI_Comm communicator, bool spin = false);

        bool selects_stress_cycle(const Particles::ParticleAccessor<dim> &particle) const;
        bool source_history_is_open() const;

        /** Record already sampled spatial tensors (Pa) and gradient (1/s).
         * time is the endpoint (s), time_step its interval; eta_ve=kappa (Pa s).
         * Particle and optional rotated tensor are borrowed only for this call. */
        void record_stress_cycle(unsigned int step, double time, double time_step,
                                 unsigned int particle_index,
                                 const Particles::ParticleAccessor<dim> &particle,
                                 const Point<dim> &sample_point,
                                 double beta, double eta_ve,
                                 const Tensor<2, dim> &gradient,
                                 const SymmetricTensor<2, dim> &old_stress,
                                 const SymmetricTensor<2, dim> &effective_strain_rate,
                                 const SymmetricTensor<2, dim> &candidate_stress,
                                 double spin_angle = 0., const SymmetricTensor<2, dim> *rotated_history = nullptr);
        /** Record localization/source values without reconstructing or accepting
         * state. continued_V is m/s; continued_crack is a strain rate (1/s). */
        void record_source_history(const Particles::ParticleAccessor<dim> &particle,
                                   double phase_field, double continued_chi,
                                   double continued_V, double eta_ve, double beta,
                                   const Tensor<2, dim> &gradient,
                                   const SymmetricTensor<2, dim> &continued_crack,
                                   const SymmetricTensor<2, dim> &old_stress,
                                   const SymmetricTensor<2, dim> &candidate_stress,
                                   double spin_angle = 0., const SymmetricTensor<2, dim> *rotated_history = nullptr);

      private:
        bool spin_enabled = false;
        std::ofstream source_history_audit;
        std::set<std::string> trace_cells;
        std::ofstream stress_cycle_audit;
    };
  }
}

#endif
