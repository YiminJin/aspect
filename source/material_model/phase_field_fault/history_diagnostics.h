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
#include <deal.II/base/symmetric_tensor.h>
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
        void open_stress_cycle(const std::string &output_directory,
                               unsigned int step, MPI_Comm communicator);
        void open_source_history(const std::string &output_directory,
                                 unsigned int step, MPI_Comm communicator);

        bool selects_stress_cycle(const Particles::ParticleAccessor<dim> &particle) const;
        bool source_history_is_open() const;

        void record_stress_cycle(unsigned int step, double time, double time_step,
                                 unsigned int particle_index,
                                 const Particles::ParticleAccessor<dim> &particle,
                                 const Point<dim> &sample_point,
                                 double beta, double eta_ve,
                                 const Tensor<2,dim> &gradient,
                                 const SymmetricTensor<2,dim> &old_stress,
                                 const SymmetricTensor<2,dim> &effective_strain_rate,
                                 const SymmetricTensor<2,dim> &candidate_stress);
        void record_source_history(const Particles::ParticleAccessor<dim> &particle,
                                   double phase_field, double continued_chi,
                                   double continued_V, double eta_ve, double beta,
                                   const Tensor<2,dim> &gradient,
                                   const SymmetricTensor<2,dim> &continued_crack,
                                   const SymmetricTensor<2,dim> &old_stress,
                                   const SymmetricTensor<2,dim> &candidate_stress);

      private:
        std::ofstream source_history_audit;
        std::set<std::string> trace_cells;
        std::ofstream stress_cycle_audit;
    };
  }
}

#endif
