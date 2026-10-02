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

#include "history_diagnostics.h"

#include <iomanip>

namespace aspect
{
  namespace internal
  {
    template <int dim>
    void
    FaultHistoryDiagnostics<dim>::open_stress_cycle(
      const std::string &output_directory, const unsigned int step,
      const MPI_Comm communicator)
    {
      const auto rank=std::to_string(Utilities::MPI::this_mpi_process(communicator));
      std::ifstream cells(output_directory+"stress_trace_cells_rank"+rank+".txt");
      for (std::string id; cells>>id;) trace_cells.insert(id);
      stress_cycle_audit.open(output_directory+"stress_update_"
                             +std::to_string(step)+"_rank"+rank+".csv");
      stress_cycle_audit<<std::setprecision(17)
        <<"step,time_s,dt,particle_index,particle_id,cell,x,y,ref_x,ref_y,sample_x,sample_y,beta,kappa,grad_xx,grad_xy,grad_yx,grad_yy,old_xx,old_yy,old_xy,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,new_xx,new_yy,new_xy\n";
    }

    template <int dim>
    void
    FaultHistoryDiagnostics<dim>::open_source_history(
      const std::string &output_directory, const unsigned int step,
      const MPI_Comm communicator)
    {
      source_history_audit.open(output_directory+"continued_source_history_"
        +std::to_string(step)+"_rank"
        +std::to_string(Utilities::MPI::this_mpi_process(communicator))+".csv");
      source_history_audit<<std::setprecision(17)
        <<"id,x,y,phi,chi,V,kappa,beta,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,old_xx,old_yy,old_xy,new_xx,new_yy,new_xy\n";
    }

    template <int dim>
    bool
    FaultHistoryDiagnostics<dim>::selects_stress_cycle(
      const Particles::ParticleAccessor<dim> &particle) const
    {
      return stress_cycle_audit.is_open()
             && trace_cells.count(particle.get_surrounding_cell()->id().to_string());
    }

    template <int dim>
    bool
    FaultHistoryDiagnostics<dim>::source_history_is_open() const
    {
      return source_history_audit.is_open();
    }

    template <int dim>
    void
    FaultHistoryDiagnostics<dim>::record_stress_cycle(
      const unsigned int step, const double time, const double time_step,
      const unsigned int particle_index,
      const Particles::ParticleAccessor<dim> &particle,
      const Point<dim> &sample_point, const double beta, const double eta_ve,
      const Tensor<2,dim> &gradient,
      const SymmetricTensor<2,dim> &old_stress,
      const SymmetricTensor<2,dim> &effective_strain_rate,
      const SymmetricTensor<2,dim> &candidate_stress)
    {
      const auto x=particle.get_location(), r=particle.get_reference_location();
      stress_cycle_audit<<step<<','<<time<<','<<time_step<<','
        <<particle_index<<','<<particle.get_id()<<','<<particle.get_surrounding_cell()->id()<<','
        <<x[0]<<','<<x[1]<<','<<r[0]<<','<<r[1]<<','<<sample_point[0]<<','<<sample_point[1]
        <<','<<beta<<','<<eta_ve<<','
        <<gradient[0][0]<<','<<gradient[0][1]<<','<<gradient[1][0]<<','<<gradient[1][1];
      for (const auto &tensor:{old_stress,symmetrize(gradient),
                              symmetrize(gradient)-effective_strain_rate,candidate_stress})
        stress_cycle_audit<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
      stress_cycle_audit<<'\n';
    }

    template <int dim>
    void
    FaultHistoryDiagnostics<dim>::record_source_history(
      const Particles::ParticleAccessor<dim> &particle,
      const double phase_field, const double continued_chi, const double continued_V,
      const double eta_ve, const double beta, const Tensor<2,dim> &gradient,
      const SymmetricTensor<2,dim> &continued_crack,
      const SymmetricTensor<2,dim> &old_stress,
      const SymmetricTensor<2,dim> &candidate_stress)
    {
      const auto position=particle.get_location();
      source_history_audit<<particle.get_id()<<','<<position[0]<<','<<position[1]<<','
        <<phase_field<<','<<continued_chi<<','<<continued_V<<','
        <<eta_ve<<','<<beta;
      for (const auto &tensor : {symmetrize(gradient),
                                continued_crack,old_stress,candidate_stress})
        source_history_audit<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
      source_history_audit<<'\n';
    }

    template class FaultHistoryDiagnostics<2>;
    template class FaultHistoryDiagnostics<3>;
  }
}
