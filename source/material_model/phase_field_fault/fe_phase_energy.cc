/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

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

/* Restricted native P2 total-energy model. Existing particle history is unchanged. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/geometry_model/box.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/gravity_model/interface.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_phase_energy(const LinearAlgebra::BlockVector &trial,
                                                   bool tangent,bool positive_search) const
    {
      validate_fe_strain_restrictions();
      return evaluate_fe_phase_energy_at(trial,tangent,positive_search,
                                         fe_strain_state->displacement_history,fe_strain_state->Z);
    }

    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_endpoint_energy(const LinearAlgebra::BlockVector &trial,
                                                      const FEEndpointTrial &endpoint,
                                                      const std::vector<double> *phase_direction) const
    {
      return evaluate_fe_phase_energy_at(trial,false,false,endpoint.strain,endpoint.Z,phase_direction);
    }

    template<int dim> typename PhaseFieldFault<dim>::FEEndpointTrial
    PhaseFieldFault<dim>::make_fe_endpoint_trial(const LinearAlgebra::BlockVector &bulk,
                                                 const std::vector<std::vector<double>> &V) const
    {
      validate_fe_strain_restrictions();
      const auto &s=*fe_strain_state;
      const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      FEEndpointTrial endpoint;
      update_fe_strain_trial(bulk,c,endpoint.strain);
      endpoint.Z=s.Z;
      long double friction_potential=0.;
      AssertDimension(V.size(),1);
      AssertDimension(V[0].size(),s.Z.size());
      for (unsigned int i=0; i<s.Z.size(); ++i)
        endpoint.Z[i]=c.beta*s.Z[i]+c.eta_ve*V[0][i];
      const std::vector<double> fractions(elastic_shear_moduli.size(),1./elastic_shear_moduli.size());
      for (const auto &p:s.profiles)
        {
          const double v=ReconstructedFaultUtilities::interpolate_slip_rate(
                           V[0][p.segment],V[0][p.segment+1],p.xi);
          const double mu=fault_friction.friction_coefficient(fractions,v);
          AssertThrow(mu==fault_friction.friction_coefficient(fractions,minimum_fault_slip_rate())
                      && fault_friction.friction_coefficient_derivative_wrt_slip_rate(fractions,v)==0.,
                      ExcMessage("Endpoint variational solve requires fixed Coulomb friction."));
          const auto bg=reconstructed_fault_background_tractions(0,p.segment,p.xi);
          AssertThrow(bg.first==0.,ExcMessage("Endpoint verification prohibits an external shear-traction offset."));
          const Point<dim> x=(1-p.xi)*s.vertices[p.segment]+p.xi*s.vertices[p.segment+1];
          const double pressure=this->get_adiabatic_conditions().pressure(x)+bg.second;
          friction_potential+=static_cast<long double>(p.weight)*c.eta_ve/s.G*mu*pressure*v;
        }
      endpoint.friction_potential=friction_potential;
      return endpoint;
    }

    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_phase_energy_at(const LinearAlgebra::BlockVector &trial,
                                                      bool tangent,bool positive_search,
                                                      const LinearAlgebra::BlockVector &strain,
                                                      const std::vector<double> &Z_reference,
                                                      const std::vector<double> *phase_direction) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(), "Fault: FE phase energy");
      validate_fe_strain_restrictions();
      const auto &s=*fe_strain_state;
      FEPhaseEnergy result;
      long double bulk_energy=0.,cohesive_energy=0.;
      const unsigned int pb=this->introspection().variable("phase_field").block_index;
      const unsigned int n=trial.block(pb).size(),ns=Z_reference.size();
      AssertDimension(ns,s.Z.size());
      const auto normalization=evaluate_fe_normalization(trial,true,phase_direction);
      const auto &phase_values=normalization.phase_values;
      AssertThrow(!tangent || Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                  ExcMessage("Dense FE Hessians are serial diagnostics; MPI endpoint solves use the full matrix-free action."));
      result.gradient.assign(n,0.);
      result.roundoff_scale.assign(n,0.);
      result.normalization=normalization.values;
      result.normalization_action=normalization.action;
      if (tangent)
        result.hessian.reinit(n,n);
      if (!normalization.admissible)
        {
          result.admissible=false;
          return result;
        }
      // Assemble first in independent nodal (phi,I) coordinates, then apply
      // the normalization pullback once. This is the same full Hessian as
      // per-QP dense rank updates, without repeating nonlocal dense work.
      std::vector<double>forceI(ns,0.),scaleI(ns,0.);
      FullMatrix<double>cross,II;
      if (tangent)
        {
          cross.reinit(n,ns);
          II.reinit(ns,ns);
        }
      SymmetricTensor<2,dim>S;
      S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      const auto quad=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_gradients);
      std::vector<SymmetricTensor<2,dim>>R(quad.size());
      unsigned int point=0;
      for (const auto &entry:s.cells)
        {
          fe.reinit(entry.second);
          fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(strain,R);
          for (unsigned int q=0; q<quad.size(); ++q,++point)
            {
              const auto &p=s.bulk[point];
              AssertThrow(p.cell==entry.first,ExcInternalError());
              double phi=0.;
              for (unsigned int v=0; v<p.dofs.size(); ++v)
                phi+=p.shape[v]*phase_values[p.dofs[v]];
              const double w[2]= {1-p.xi,p.xi};
              const double I=w[0]*result.normalization[p.segment]+w[1]*result.normalization[p.segment+1];
              const double Z=w[0]*Z_reference[p.segment]+w[1]*Z_reference[p.segment+1];
              const double h=s.m*phi*(1+phi)/std::pow(1-phi,2);
              const double hp=s.m*(4/std::pow(1-phi,3)-3/std::pow(1-phi,2));
              const double hpp=s.m*(12/std::pow(1-phi,4)-6/std::pow(1-phi,3));
              // R[q] is a raw zero-zz plane-strain gradient. Deviate once.
              const auto tau=2*s.G*Utilities::Tensors::consistent_deviator(R[q])-2*Z*h/I*S;
              result.bulk+=p.weight*((tau*tau)+std::pow(trace(tau),2))/(4*s.G);
              bulk_energy+=static_cast<long double>(p.weight)*((tau*tau)+std::pow(trace(tau),2))/(4*s.G);
              const double c=-Z/s.G*(tau*S);
              const double positive_c=std::abs(Z/s.G)*(2*s.G*std::abs(R[q]*S)+std::abs(2*Z*h/I)*(S*S));
              const double a=2*Z*Z/s.G*(S*S);
              for (unsigned int v=0; v<p.dofs.size(); ++v)
                {
                  result.gradient[p.dofs[v]]+=p.weight*c*hp/I*p.shape[v];
                  result.roundoff_scale[p.dofs[v]]+=p.weight*positive_c*std::abs(hp/I*p.shape[v]);
                }
              for (unsigned int i=0; i<2; ++i)
                {
                  forceI[p.segment+i]-=p.weight*w[i]*c*h/(I*I);
                  scaleI[p.segment+i]+=p.weight*std::abs(w[i])*positive_c*std::abs(h/(I*I));
                }
              if (tangent)
                {
                  const double pp=a*hp*hp/(I*I)+(positive_search?0.:c*hpp/I);
                  const double pi=-a*hp*h/std::pow(I,3)-(positive_search?0.:c*hp/(I*I));
                  const double ii=a*h*h/std::pow(I,4)+(positive_search?0.:2*c*h/std::pow(I,3));
                  for (unsigned int v=0; v<p.dofs.size(); ++v)
                    {
                      for (unsigned int u=0; u<p.dofs.size(); ++u)
                        result.hessian(p.dofs[v],p.dofs[u])+=p.weight*pp*p.shape[v]*p.shape[u];
                      for (unsigned int i=0; i<2; ++i)
                        cross(p.dofs[v],p.segment+i)+=p.weight*pi*p.shape[v]*w[i];
                    }
                  for (unsigned int i=0; i<2; ++i)
                    for (unsigned int j=0; j<2; ++j)
                      II(p.segment+i,p.segment+j)+=p.weight*ii*w[i]*w[j];
                }
            }
        }
      aspect::internal::endpoint_sum(result.gradient,this->get_mpi_communicator());
      aspect::internal::endpoint_sum(result.roundoff_scale,this->get_mpi_communicator());
      aspect::internal::endpoint_sum(forceI,this->get_mpi_communicator());
      aspect::internal::endpoint_sum(scaleI,this->get_mpi_communicator());
      bulk_energy=aspect::internal::endpoint_energy_sum(bulk_energy,this->get_mpi_communicator());
      for (const auto &p:s.profiles)
        {
          const double w[2]= {1-p.xi,p.xi};
          const double I=w[0]*result.normalization[p.segment]+w[1]*result.normalization[p.segment+1];
          const double Z=w[0]*Z_reference[p.segment]+w[1]*Z_reference[p.segment+1];
          result.cohesive+=p.weight*Z*Z/(2*s.G*I);
          cohesive_energy+=static_cast<long double>(p.weight)*Z*Z/(2*s.G*I);
          const double c=-p.weight*Z*Z/(2*s.G*I*I);
          for (unsigned int i=0; i<2; ++i)
            {
              forceI[p.segment+i]+=c*w[i];
              scaleI[p.segment+i]+=std::abs(c*w[i]);
              if (tangent)
                for (unsigned int j=0; j<2; ++j)
                  II(p.segment+i,p.segment+j)+=p.weight*Z*Z/(s.G*I*I*I)*w[i]*w[j];
            }
        }
      add_fe_normalization_energy_pullback(normalization,forceI,scaleI,
                                           cross,II,tangent,positive_search,result);
      // The ordinary bulk subtotal is local; only its extended sum is global.
      // Retain the serial roundoff diagnostic without mislabelling the MPI one.
      this->get_pcout()<<(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1
                          ?"   FE energy summation: bulk old-new="
                          :"   FE energy summation: local bulk double minus global extended=")
                       <<static_cast<double>(result.bulk-bulk_energy)
                       <<", cohesive old-new="<<static_cast<double>(result.cohesive-cohesive_energy)<<std::endl;
      result.bulk=bulk_energy;
      result.cohesive=cohesive_energy;
      result.energy=result.bulk+result.cohesive;
      result.admissible=std::isfinite(result.energy);
      for (const auto g:result.gradient)
        result.admissible=result.admissible && std::isfinite(g);
      return result;
    }

    template<int dim> typename PhaseFieldFault<dim>::FEEndpointAction
    PhaseFieldFault<dim>::evaluate_fe_endpoint_action(
      const LinearAlgebra::BlockVector &trial, const FEEndpointTrial &endpoint,
      const std::vector<double> &direction,
      const LinearAlgebra::BlockVector *velocity_direction,
      const std::vector<std::vector<double>> *slip_direction, bool assemble_rhs) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(),"Fault: FE tangent pullback");
      validate_fe_strain_restrictions();
      const auto &s=*fe_strain_state;
      const auto &intro=this->introspection();
      const unsigned int pb=intro.variable("phase_field").block_index;
      const unsigned int n=trial.block(pb).size(),ns=s.Z.size();
      AssertDimension(direction.size(),n);
      AssertThrow((velocity_direction==nullptr)==(slip_direction==nullptr),ExcInternalError());
      const auto coefficients=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      const auto normalization=evaluate_fe_normalization(trial,false,&direction);
      AssertThrow(normalization.admissible,ExcMessage("Undefined tangent normalization."));
      const auto &phase_values=normalization.phase_values;
      const auto &I=normalization.values;
      const auto &dI=normalization.action;
      FEEndpointAction result;
      result.gradient.assign(n,0.);
      std::vector<double>force(ns,0.),dforce(ns,0.);
      SymmetricTensor<2,dim>S;
      S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      const auto quadrature=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quadrature,update_gradients);
      std::vector<SymmetricTensor<2,dim>>R(quadrature.size()),du(quadrature.size());
      Vector<double>cell_rhs(this->get_fe().n_dofs_per_cell());
      std::vector<types::global_dof_index>indices(cell_rhs.size());
      AffineConstraints<double>constraints(this->get_current_constraints());
      for (const auto &line:constraints.get_lines())
        constraints.set_inhomogeneity(line.index,0.);
      if (assemble_rhs)
        {
          result.bulk_rhs.reinit(intro.index_sets.system_partitioning,this->get_mpi_communicator());
          result.surface_rhs= {{}};
          result.surface_rhs[0].assign(ns,0.);
        }
      unsigned int point=0;
      for (const auto &entry:s.cells)
        {
          fe.reinit(entry.second);
          cell_rhs=0.;
          fe[intro.extractors.velocities].get_function_symmetric_gradients(endpoint.strain,R);
          if (velocity_direction)
            fe[intro.extractors.velocities].get_function_symmetric_gradients(*velocity_direction,du);
          for (unsigned int q=0; q<quadrature.size(); ++q,++point)
            {
              const auto &p=s.bulk[point];
              AssertThrow(p.cell==entry.first,ExcInternalError());
              double phi=0.,dp=0.;
              for (unsigned int v=0; v<p.dofs.size(); ++v)
                {
                  phi+=p.shape[v]*phase_values[p.dofs[v]];
                  dp+=p.shape[v]*direction[p.dofs[v]];
                }
              const double w[2]= {1-p.xi,p.xi};
              const double ii=w[0]*I[p.segment]+w[1]*I[p.segment+1];
              const double di=w[0]*dI[p.segment]+w[1]*dI[p.segment+1];
              const double Z=w[0]*endpoint.Z[p.segment]+w[1]*endpoint.Z[p.segment+1];
              const double dZ=slip_direction?coefficients.eta_ve*(w[0]*(*slip_direction)[0][p.segment]+w[1]*(*slip_direction)[0][p.segment+1]):0.;
              const double h=s.m*phi*(1+phi)/std::pow(1-phi,2);
              const double hp=s.m*(4/std::pow(1-phi,3)-3/std::pow(1-phi,2));
              const double hpp=s.m*(12/std::pow(1-phi,4)-6/std::pow(1-phi,3));
              const double chi=h/ii,dchi=hp*dp/ii-h*di/(ii*ii);
              const auto tau=2*s.G*Utilities::Tensors::consistent_deviator(R[q])-2*Z*chi*S;
              const auto dtau=(velocity_direction?2*coefficients.eta_ve*Utilities::Tensors::consistent_deviator(du[q]):SymmetricTensor<2,dim>())
                              -2*(dZ*chi+Z*dchi)*S;
              const double c=-Z/s.G*(tau*S),dc=-(dZ*(tau*S)+Z*(dtau*S))/s.G;
              for (unsigned int v=0; v<p.dofs.size(); ++v)
                result.gradient[p.dofs[v]]+=p.weight*p.shape[v]*(dc*hp/ii+c*(hpp*dp/ii-hp*di/(ii*ii)));
              for (unsigned int i=0; i<2; ++i)
                {
                  force[p.segment+i]-=p.weight*w[i]*c*h/(ii*ii);
                  dforce[p.segment+i]-=p.weight*w[i]*(dc*h/(ii*ii)+c*hp*dp/(ii*ii)-2*c*h*di/std::pow(ii,3));
                  if (assemble_rhs)
                    result.surface_rhs[0][p.segment+i]+=p.weight*w[i]*dchi*((tau*S)-2*Z*chi*(S*S));
                }
              if (assemble_rhs)
                for (unsigned int i=0; i<cell_rhs.size(); ++i)
                  cell_rhs[i]+=p.weight*2*Z*dchi*(S*fe[intro.extractors.velocities].symmetric_gradient(i,q));
            }
          if (assemble_rhs)
            {
              entry.second->get_dof_indices(indices);
              constraints.distribute_local_to_global(cell_rhs,indices,result.bulk_rhs);
            }
        }
      if (assemble_rhs)
        {
          result.bulk_rhs.compress(VectorOperation::add);
          aspect::internal::endpoint_sum(result.surface_rhs[0],this->get_mpi_communicator());
        }
      aspect::internal::endpoint_sum(result.gradient,this->get_mpi_communicator());
      aspect::internal::endpoint_sum(force,this->get_mpi_communicator());
      aspect::internal::endpoint_sum(dforce,this->get_mpi_communicator());
      for (const auto &p:s.profiles)
        {
          const double w[2]= {1-p.xi,p.xi};
          const double ii=w[0]*I[p.segment]+w[1]*I[p.segment+1];
          const double di=w[0]*dI[p.segment]+w[1]*dI[p.segment+1];
          const double Z=w[0]*endpoint.Z[p.segment]+w[1]*endpoint.Z[p.segment+1];
          const double dZ=slip_direction?coefficients.eta_ve*(w[0]*(*slip_direction)[0][p.segment]+w[1]*(*slip_direction)[0][p.segment+1]):0.;
          for (unsigned int i=0; i<2; ++i)
            {
              force[p.segment+i]-=p.weight*w[i]*Z*Z/(2*s.G*ii*ii);
              dforce[p.segment+i]-=p.weight*w[i]*(Z*dZ/(s.G*ii*ii)-Z*Z*di/(s.G*std::pow(ii,3)));
              if (assemble_rhs)
                result.surface_rhs[0][p.segment+i]+=p.weight*w[i]*Z*di/(ii*ii);
            }
        }
      add_fe_normalization_action_pullback(normalization,direction,force,dforce,result.gradient);
      return result;
    }

#define INSTANTIATE(dim) \
  template PhaseFieldFault<dim>::FEEndpointAction PhaseFieldFault<dim>::evaluate_fe_endpoint_action(const LinearAlgebra::BlockVector &,const FEEndpointTrial &,const std::vector<double> &,const LinearAlgebra::BlockVector *,const std::vector<std::vector<double>> *,bool) const; \
  template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_phase_energy(const LinearAlgebra::BlockVector &,bool,bool) const; \
  template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_phase_energy_at(const LinearAlgebra::BlockVector &,bool,bool,const LinearAlgebra::BlockVector &,const std::vector<double>&,const std::vector<double>*) const; \
  template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_endpoint_energy(const LinearAlgebra::BlockVector &,const FEEndpointTrial &,const std::vector<double>*) const; \
  template PhaseFieldFault<dim>::FEEndpointTrial PhaseFieldFault<dim>::make_fe_endpoint_trial(const LinearAlgebra::BlockVector &,const std::vector<std::vector<double>> &) const;
    INSTANTIATE(2)
    INSTANTIATE(3)
#undef INSTANTIATE
  }
}
