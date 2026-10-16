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
  namespace
  {
    // Second derivatives of the two endpoint values of a linear normal ray.
    // This small jet differentiates the analytic integral, not quadrature samples.
    struct RayJet : MaterialModel::internal::FENormalIntegral
    {
      RayJet(double value=0.)
      {
        v=value;
      }
      static RayJet variable(double value,unsigned int i)
      {
        RayJet x(value);
        x.d[i]=1.;
        return x;
      }
    };
    RayJet operator+(const RayJet &a,const RayJet &b)
    {
      RayJet r(a.v+b.v);
      for (unsigned int i=0; i<2; ++i)
        {
          r.d[i]=a.d[i]+b.d[i];
          for (unsigned int j=0; j<2; ++j)
            r.h[i][j]=a.h[i][j]+b.h[i][j];
        }
      return r;
    }
    RayJet operator*(const RayJet &a,const RayJet &b)
    {
      RayJet r(a.v*b.v);
      for (unsigned int i=0; i<2; ++i)
        {
          r.d[i]=a.d[i]*b.v+a.v*b.d[i];
          for (unsigned int j=0; j<2; ++j)
            // Group the symmetric pair before adding it to the Hessians.
            // Left-associated addition makes the two mixed entries follow
            // different cancellation paths for nearly constant normal rays.
            r.h[i][j]=a.h[i][j]*b.v+a.v*b.h[i][j]
                      +(a.d[i]*b.d[j]+b.d[i]*a.d[j]);
        }
      return r;
    }
    RayJet compose(const RayJet &x,double v,double first,double second)
    {
      RayJet r(v);
      for (unsigned int i=0; i<2; ++i)
        {
          r.d[i]=first*x.d[i];
          for (unsigned int j=0; j<2; ++j)
            r.h[i][j]=first*x.h[i][j]+second*(x.d[i]*x.d[j]);
        }
      return r;
    }
    RayJet inverse(const RayJet &x)
    {
      return compose(x,1./x.v,-1./(x.v*x.v),2./(x.v*x.v*x.v));
    }
    RayJet operator-(const RayJet &a,const RayJet &b)
    {
      return a+RayJet(-1.)*b;
    }
    RayJet operator/(const RayJet &a,const RayJet &b)
    {
      return a*inverse(b);
    }
    RayJet ray_integral(double left,double right,double length,double m)
    {
      const auto a=RayJet(1.)-RayJet::variable(left,0);
      const auto b=RayJet(1.)-RayJet::variable(right,1);
      const auto z=(b-a)/a;
      RayJet ratio;
      if (std::abs(z.v)<1.e-3)
        {
          RayJet power(1.);
          for (unsigned int k=0; k<13; ++k)
            {
              ratio=ratio+RayJet((k%2?-1.:1.)/(k+1))*power;
              power=power*z;
            }
        }
      else
        ratio=compose(z,std::log1p(z.v),1./(1.+z.v),-1./std::pow(1.+z.v,2))/z;
      return RayJet(m*length)*(RayJet(2.)/(a*b)-RayJet(3.)*ratio/a+RayJet(1.));
    }

    double ray_integral_value(double left,double right,double length,double m)
    {
      const double a=1.-left,b=1.-right,z=(b-a)*(1./a);
      double ratio=0.;
      if (std::abs(z)<1.e-3)
        {
          double power=1.;
          for (unsigned int k=0; k<13; ++k)
            {
              ratio+=((k%2?-1.:1.)/(k+1))*power;
              power*=z;
            }
        }
      else ratio=std::log1p(z)*(1./z);
      return (m*length)*(2.*(1./(a*b))-(3.*ratio)*(1./a)+1.);
    }
  }

  namespace MaterialModel
  {
    template <int dim>
    void PhaseFieldFault<dim>::prepare_fe_normalization_geometry(FEStrainState &state)
    {
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const auto extent=box.get_extents();
      const auto &intro=this->introspection();
      const unsigned int b=intro.variable("phase_field").block_index;
      const unsigned int ns=state.vertices.size()-1;
      const auto profiles=build_owned_normalization_profiles(true);
      prepare_cell_normalization_geometry(profiles);
      AssertThrow(normalization_cell_cache.supported,ExcMessage("FE strain requires affine normal-ray cell geometry."));
      const auto offset=state.displacement_history.get_block_indices().block_start(b);
      for (unsigned int k=0; k<profiles.size(); ++k)
        {
          typename FEStrainState::Profile p;
          p.segment=profiles[k].segment_index;
          p.xi=profiles[k].xi;
          p.weight=profiles[k].surface_weight;
          double covered=0.;
          for (const auto &cell:normalization_cell_cache.profiles[k].intervals)
            {
              typename FEStrainState::Ray ray;
              ray.length=cell.upper-cell.lower;
              ray.lower=cell.lower;
              covered+=ray.length;
              for (unsigned int v=0; v<ray.dofs.size(); ++v)
                {
                  ray.dofs[v]=cell.phase_dofs[v]-offset;
                  for (unsigned int end=0; end<2; ++end)
                    {
                      const auto unit=cell.reference_origin+(end?cell.upper:cell.lower)*cell.reference_direction;
                      double shape=1.;
                      for (unsigned int d=0; d<dim; ++d)
                        shape*=v&(1u<<d)?unit[d]:1.-unit[d];
                      ray.shape[end][v]=shape;
                    }
                }
              p.rays.push_back(ray);
            }
          state.profiles.push_back(p);
        }
      // Gather only immutable ray geometry from uniquely owned bulk cells.
      // Profile integration/adjoints are replicated; FE histories are not.
      std::vector<std::vector<typename FEStrainState::Ray>> local_rays;
      for (auto &profile:state.profiles)
        {
          local_rays.push_back(std::move(profile.rays));
          profile.rays.clear();
        }
      const auto rank_rays=Utilities::MPI::all_gather(this->get_mpi_communicator(),local_rays);
      for (const auto &rank:rank_rays)
        for (unsigned int k=0; k<rank.size(); ++k)
          state.profiles[k].rays.insert(state.profiles[k].rays.end(),rank[k].begin(),rank[k].end());
      for (auto &profile:state.profiles)
        {
          std::sort(profile.rays.begin(),profile.rays.end(),[](const auto &a,const auto &b)
          {
            return a.lower<b.lower;
          });
          double covered=0.;
          for (const auto &ray:profile.rays)
            covered+=ray.length;
          AssertThrow(std::abs(covered-extent[1])<1.e-12*std::max(1.,extent[1]),
                      ExcMessage("FE strain normal integration must cover the full Box exactly once."));
        }
      FullMatrix<double>mass(ns+1,ns+1),rhs(ns+1,profiles.size());
      for (unsigned int k=0; k<profiles.size(); ++k)
        {
          const auto &p=state.profiles[k];
          const double weights[2]= {1-p.xi,p.xi};
          for (unsigned int i=0; i<2; ++i)
            {
              rhs(p.segment+i,k)=p.weight*weights[i];
              for (unsigned int j=0; j<2; ++j)
                mass(p.segment+i,p.segment+j)+=p.weight*weights[i]*weights[j];
            }
        }
      mass.gauss_jordan();
      state.projection.reinit(ns+1,profiles.size());
      mass.mmult(state.projection,rhs);
    }

    template<int dim> typename PhaseFieldFault<dim>::FENormalization
    PhaseFieldFault<dim>::evaluate_fe_normalization(
      const LinearAlgebra::BlockVector &trial,bool assemble_dense_derivatives,
      const std::vector<double> *phase_direction) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(),"Fault: FE normalization");
      AssertThrow(fe_strain_state,ExcMessage("FE normalization geometry has not been initialized."));
      const auto &s=*fe_strain_state;
      const auto &faults=this->get_reconstructed_fault_manager().get_faults();
      AssertThrow(faults.size()==1 && faults[0].geometry_version()==s.geometry_version
                  && this->get_dof_handler().n_dofs()==s.n_dofs
                  && this->get_triangulation().n_global_active_cells()==s.n_cells_global
                  && this->get_triangulation().get_vertices()==s.mesh_vertices,
                  ExcMessage("FE normalization rejects changed mesh or fault geometry."));
      const unsigned int pb=this->introspection().variable("phase_field").block_index;
      const unsigned int n=trial.block(pb).size(),ns=s.vertices.size();
      FENormalization result;
      result.phase_values=aspect::internal::endpoint_values(trial.block(pb),this->get_mpi_communicator());
      result.values.assign(ns,0.);
      for (const double phi:result.phase_values)
        if (!std::isfinite(phi) || phi<0. || phi>=1.)
          {
            result.admissible=false;
            return result;
          }
      const bool derivative_data=assemble_dense_derivatives || phase_direction;
      FullMatrix<double>A,positive_A;
      if (assemble_dense_derivatives)
        {
          A.reinit(s.profiles.size(),n);
          positive_A.reinit(s.profiles.size(),n);
        }
      if (derivative_data)
        result.integrals.resize(s.profiles.size());
      if (phase_direction)
        {
          AssertDimension(phase_direction->size(),n);
          result.action.assign(ns,0.);
        }
      for (unsigned int k=0; k<s.profiles.size(); ++k)
        {
          double direct=0.,action=0.;
          for (const auto &ray:s.profiles[k].rays)
            {
              double endpoint[2]= {},direction[2]= {};
              for (unsigned int v=0; v<ray.dofs.size(); ++v)
                for (unsigned int e=0; e<2; ++e)
                  {
                    endpoint[e]+=ray.shape[e][v]*result.phase_values[ray.dofs[v]];
                    if (phase_direction)
                      direction[e]+=ray.shape[e][v]*(*phase_direction)[ray.dofs[v]];
                  }
              if (derivative_data)
                {
                  const auto f=ray_integral(endpoint[0],endpoint[1],ray.length,s.m);
                  result.integrals[k].push_back(f);
                  direct+=f.v;
                  if (phase_direction)
                    action+=f.d[0]*direction[0]+f.d[1]*direction[1];
                  if (assemble_dense_derivatives)
                    for (unsigned int v=0; v<ray.dofs.size(); ++v)
                      {
                        A(k,ray.dofs[v])+=f.d[0]*ray.shape[0][v]+f.d[1]*ray.shape[1][v];
                        positive_A(k,ray.dofs[v])+=std::abs(f.d[0]*ray.shape[0][v])+std::abs(f.d[1]*ray.shape[1][v]);
                      }
                }
              else direct+=ray_integral_value(endpoint[0],endpoint[1],ray.length,s.m);
            }
          for (unsigned int i=0; i<ns; ++i)
            {
              result.values[i]+=s.projection(i,k)*direct;
              if (phase_direction && !assemble_dense_derivatives)
                result.action[i]+=s.projection(i,k)*action;
            }
        }
      for (const double I:result.values)
        if (!(I>0.) || !std::isfinite(I))
          {
            result.admissible=false;
            return result;
          }
      if (assemble_dense_derivatives)
        {
          result.derivative.reinit(ns,n);
          s.projection.mmult(result.derivative,A);
          result.derivative_scale.reinit(ns,n);
          for (unsigned int i=0; i<ns; ++i)
            for (unsigned int k=0; k<s.profiles.size(); ++k)
              for (unsigned int j=0; j<n; ++j)
                result.derivative_scale(i,j)+=std::abs(s.projection(i,k))*positive_A(k,j);
          if (phase_direction)
            for (unsigned int i=0; i<ns; ++i)
              for (unsigned int j=0; j<n; ++j)
                result.action[i]+=result.derivative(i,j)*(*phase_direction)[j];
        }
      return result;
    }

    template<int dim> void PhaseFieldFault<dim>::add_fe_normalization_energy_pullback(
      const FENormalization &normalization,const std::vector<double> &forceI,
      const std::vector<double> &scaleI,const FullMatrix<double> &cross,
      const FullMatrix<double> &II,bool tangent,bool positive_search,FEPhaseEnergy &result) const
    {
      const auto &s=*fe_strain_state;
      const unsigned int n=result.gradient.size(),ns=normalization.values.size();
      const auto &Iderivative=normalization.derivative;
      const auto &positive_DI=normalization.derivative_scale;
      const auto &integrals=normalization.integrals;
      for (unsigned int j=0; j<n; ++j)
        for (unsigned int i=0; i<ns; ++i)
          {
            result.gradient[j]+=Iderivative(i,j)*forceI[i];
            result.roundoff_scale[j]+=positive_DI(i,j)*scaleI[i];
          }
      if (tangent)
        {
          FullMatrix<double>product(n,n),II_DI(ns,n),pullback(n,n);
          cross.mmult(product,Iderivative);
          II.mmult(II_DI,Iderivative);
          Iderivative.Tmmult(pullback,II_DI);
          for (unsigned int i=0; i<n; ++i)
            for (unsigned int j=0; j<n; ++j)
              result.hessian(i,j)+=product(i,j)+product(j,i)+pullback(i,j);
        }
      if (tangent && !positive_search)
        for (unsigned int k=0; k<s.profiles.size(); ++k)
          {
            double adjoint=0.;
            for (unsigned int i=0; i<ns; ++i)
              adjoint+=s.projection(i,k)*forceI[i];
            unsigned int j=0;
            for (const auto &ray:s.profiles[k].rays)
              {
                const auto &f=integrals[k][j++];
                for (unsigned int v=0; v<ray.dofs.size(); ++v)
                  for (unsigned int u=0; u<ray.dofs.size(); ++u)
                    for (unsigned int a=0; a<2; ++a)
                      for (unsigned int b=0; b<2; ++b)
                        result.hessian(ray.dofs[v],ray.dofs[u])+=adjoint*f.h[a][b]*ray.shape[a][v]*ray.shape[b][u];
              }
          }
    }

    template<int dim> void PhaseFieldFault<dim>::add_fe_normalization_action_pullback(
      const FENormalization &normalization,const std::vector<double> &direction,
      const std::vector<double> &force,const std::vector<double> &dforce,
      std::vector<double> &gradient) const
    {
      const auto &s=*fe_strain_state;
      const unsigned int ns=normalization.values.size();
      const auto &jets=normalization.integrals;
      // Matrix-free pullback DI^T dforce + (D DI[d])^T force. In particular,
      // the second term differentiates the analytic rays through P^T.
      for (unsigned int k=0; k<s.profiles.size(); ++k)
        {
          double adjoint=0.,dadjoint=0.;
          for (unsigned int i=0; i<ns; ++i)
            {
              adjoint+=s.projection(i,k)*force[i];
              dadjoint+=s.projection(i,k)*dforce[i];
            }
          unsigned int j=0;
          for (const auto &ray:s.profiles[k].rays)
            {
              const auto &jet=jets[k][j++];
              double dp[2]= {};
              for (unsigned int a=0; a<2; ++a)
                for (unsigned int v=0; v<ray.dofs.size(); ++v)
                  dp[a]+=ray.shape[a][v]*direction[ray.dofs[v]];
              for (unsigned int v=0; v<ray.dofs.size(); ++v)
                for (unsigned int a=0; a<2; ++a)
                  {
                    double second=0.;
                    for (unsigned int b=0; b<2; ++b)
                      second+=jet.h[a][b]*dp[b];
                    gradient[ray.dofs[v]]+=ray.shape[a][v]*(dadjoint*jet.d[a]+adjoint*second);
                  }
            }
        }
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::prepare_fe_normalization_geometry(FEStrainState &); \
  template PhaseFieldFault<dim>::FENormalization PhaseFieldFault<dim>::evaluate_fe_normalization(const LinearAlgebra::BlockVector &,bool,const std::vector<double> *) const; \
  template void PhaseFieldFault<dim>::add_fe_normalization_energy_pullback(const FENormalization &,const std::vector<double> &,const std::vector<double> &,const FullMatrix<double> &,const FullMatrix<double> &,bool,bool,FEPhaseEnergy &) const; \
  template void PhaseFieldFault<dim>::add_fe_normalization_action_pullback(const FENormalization &,const std::vector<double> &,const std::vector<double> &,const std::vector<double> &,std::vector<double> &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
