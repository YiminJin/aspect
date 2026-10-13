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
      const auto offset=state.accepted.get_block_indices().block_start(b);
      for(unsigned int k=0;k<profiles.size();++k)
        {
          typename FEStrainState::Profile p;
          p.segment=profiles[k].segment_index;p.xi=profiles[k].xi;p.weight=profiles[k].surface_weight;
          double covered=0.;
          for(const auto &cell:normalization_cell_cache.profiles[k].intervals)
            {
              typename FEStrainState::Ray ray;ray.length=cell.upper-cell.lower;ray.lower=cell.lower;covered+=ray.length;
              for(unsigned int v=0;v<ray.dofs.size();++v)
                {
                  ray.dofs[v]=cell.phase_dofs[v]-offset;
                  for(unsigned int end=0;end<2;++end)
                    {
                      const auto unit=cell.reference_origin+(end?cell.upper:cell.lower)*cell.reference_direction;
                      double shape=1.;for(unsigned int d=0;d<dim;++d)shape*=v&(1u<<d)?unit[d]:1.-unit[d];
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
      for(auto &profile:state.profiles){local_rays.push_back(std::move(profile.rays));profile.rays.clear();}
      const auto rank_rays=Utilities::MPI::all_gather(this->get_mpi_communicator(),local_rays);
      for(const auto &rank:rank_rays)for(unsigned int k=0;k<rank.size();++k)
        state.profiles[k].rays.insert(state.profiles[k].rays.end(),rank[k].begin(),rank[k].end());
      for(auto &profile:state.profiles)
        {
          std::sort(profile.rays.begin(),profile.rays.end(),[](const auto &a,const auto &b){return a.lower<b.lower;});
          double covered=0.;for(const auto &ray:profile.rays)covered+=ray.length;
          AssertThrow(std::abs(covered-extent[1])<1.e-12*std::max(1.,extent[1]),
                      ExcMessage("FE strain normal integration must cover the full Box exactly once."));
        }
      FullMatrix<double>mass(ns+1,ns+1),rhs(ns+1,profiles.size());
      for(unsigned int k=0;k<profiles.size();++k)
        {
          const auto&p=state.profiles[k];const double weights[2]={1-p.xi,p.xi};
          for(unsigned int i=0;i<2;++i)
            {rhs(p.segment+i,k)=p.weight*weights[i];
             for(unsigned int j=0;j<2;++j)mass(p.segment+i,p.segment+j)+=p.weight*weights[i]*weights[j];}
        }
      mass.gauss_jordan();state.projection.reinit(ns+1,profiles.size());mass.mmult(state.projection,rhs);
    }

#define INSTANTIATE(dim) \
    template void PhaseFieldFault<dim>::prepare_fe_normalization_geometry(FEStrainState &);
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
