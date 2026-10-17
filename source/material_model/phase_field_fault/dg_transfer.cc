/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <deal.II/distributed/solution_transfer.h>
#include <boost/serialization/map.hpp>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> struct PhaseFieldFault<dim>::DGTransfer
    {
      std::shared_ptr<FEStrainState> accepted;
      std::unique_ptr<parallel::distributed::SolutionTransfer<dim,LinearAlgebra::Vector>> tensor;
      std::map<types::particle_index,Point<dim>> particles;
    };

    template<int dim> void PhaseFieldFault<dim>::prepare_dg_history_transfer()
    {
      if (!dg_accepted_history || !fe_strain_state) return;
      validate_fe_strain_restrictions();
      AssertThrow(consistent_endpoint_phase && !dg_transfer,
                  ExcMessage("DG transfer requires an accepted fixed-reference endpoint."));
      const auto &s=*fe_strain_state;
      const auto &comm=this->get_mpi_communicator();
      const unsigned int block=this->introspection().variable("phase_field").block_index;
      AssertThrow(s.displacement_history.block(block)==this->get_solution().block(block),
                  ExcMessage("DG transfer rejects an unpublished physical phase trial."));
      unsigned long long refined=0,coarsened=0;
      for (const auto &cell:this->get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (cell->refine_flag_set()) ++refined;
            coarsened+=cell->coarsen_flag_set();
          }
      refined=Utilities::MPI::sum(refined,comm);
      coarsened=Utilities::MPI::sum(coarsened,comm);
      if (!refined && !coarsened) return;
      AssertThrow(!refined || !coarsened,
                  ExcMessage("Native DG endpoint transfer currently separates refinement from coarsening."));
      if (coarsened)
        {
          // Q1 equality at every child vertex proves equality throughout each
          // child. Reject lossy phase/obstacle coarsening; never repair by a cap
          // or an unselected obstacle projection. Gather physical vertex keys
          // because a coarsening family can straddle MPI ownership.
          const auto phase=aspect::internal::endpoint_values(s.displacement_history.block(block),comm);
          const auto offset=s.displacement_history.get_block_indices().block_start(block);
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          std::map<std::array<double,dim>,double> local,values;
          for (const auto &entry:s.cells)
            for (const auto v:entry.second->vertex_indices())
              {
                std::array<double,dim> key;
                for (unsigned int d=0; d<dim; ++d) key[d]=entry.second->vertex(v)[d];
                local[key]=phase[entry.second->vertex_dof_index(v,pc)-offset];
              }
          for (const auto &part:Utilities::MPI::all_gather(comm,local))
            values.insert(part.begin(),part.end());
          std::map<std::string,unsigned int> local_families,families;
          for (const auto &entry:s.cells)
            if (entry.second->coarsen_flag_set())
              {
                AssertThrow(entry.second->level()>0,ExcMessage("Cannot coarsen a coarse-grid cell."));
                ++local_families[entry.second->parent()->id().to_string()];
              }
          for (const auto &part:Utilities::MPI::all_gather(comm,local_families))
            for (const auto &family:part)families[family.first]+=family.second;
          for (const auto &family:families)
            AssertThrow(family.second==GeometryInfo<dim>::max_children_per_cell,
                        ExcMessage("DG transfer requires complete coarsening families."));
          double defect=0.;
          for (const auto &entry:s.cells)
            if (entry.second->coarsen_flag_set())
              {
                const auto parent=entry.second->parent();
                for (const auto v:entry.second->vertex_indices())
                  {
                    const auto x=entry.second->vertex(v);
                    double coarse=0.;
                    for (const auto w:parent->vertex_indices())
                      {
                        double shape=1.;
                        std::array<double,dim> key;
                        for (unsigned int d=0; d<dim; ++d)
                          {
                            key[d]=parent->vertex(w)[d];
                            const double xi=(x[d]-parent->vertex(0)[d])/(parent->vertex(1u<<d)[d]-parent->vertex(0)[d]);
                            shape*=((w>>d)&1u) ? xi : 1-xi;
                          }
                        coarse+=shape*values.at(key);
                      }
                    defect=std::max(defect,std::abs(coarse-phase[entry.second->vertex_dof_index(v,pc)-offset]));
                  }
              }
          defect=Utilities::MPI::max(defect,comm);
          AssertThrow(defect<=256*std::numeric_limits<double>::epsilon(),
                      ExcMessage("DG transfer rejects nonrepresentable physical phase/lower obstacle coarsening; defect="
                                 +Utilities::to_string(defect,17)));
        }
      auto transfer=std::make_shared<DGTransfer>();
      transfer->accepted=fe_strain_state;
      for (const auto &part:Utilities::MPI::all_gather(comm,s.particles))
        for (const auto &p:part)
          AssertThrow(transfer->particles.emplace(p).second,ExcMessage("Duplicate accepted particle owner."));
      transfer->tensor=std::make_unique<parallel::distributed::SolutionTransfer<dim,LinearAlgebra::Vector>>(s.dg_history->space->dofs);
      std::vector<const LinearAlgebra::Vector *> fields;
      for (const auto &component:s.dg_history->trial_stress) fields.push_back(&component);
      transfer->tensor->prepare_for_coarsening_and_refinement(fields);
      dg_transfer=std::move(transfer);
    }

    template<int dim> void PhaseFieldFault<dim>::finish_dg_history_transfer()
    {
      if (!dg_transfer) return;
      const auto transfer=dg_transfer;
      const auto old=transfer->accepted;
      const auto &comm=this->get_mpi_communicator();
      unsigned int fixed=1;
      for (const auto &p:this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
        if (!transfer->particles.count(p.get_id()) || transfer->particles.at(p.get_id())!=p.get_location()) fixed=0;
      AssertThrow(Utilities::MPI::min(fixed,comm)==1,ExcMessage("DG transfer moved or changed particles."));
      auto history=std::make_shared<DGHistory>();
      history->space=old->dg_history->space;
      auto &space=*history->space;
      space.dofs.distribute_dofs(space.element);
      space.relevant=DoFTools::extract_locally_relevant_dofs(space.dofs);
      space.cells.clear();
      for (const auto &cell:space.dofs.active_cell_iterators())
        if (!cell->is_artificial()) space.cells.emplace(cell->id(),cell);
      std::array<LinearAlgebra::Vector,SymmetricTensor<2,dim>::n_independent_components> owned;
      std::vector<LinearAlgebra::Vector *> fields;
      for (auto &component:owned)
        {
          component.reinit(space.dofs.locally_owned_dofs(),comm);
          fields.push_back(&component);
        }
      transfer->tensor->interpolate(fields);
      for (unsigned int c=0; c<owned.size(); ++c)
        {
          history->trial_stress[c].reinit(space.dofs.locally_owned_dofs(),space.relevant,comm);
          history->trial_stress[c]=owned[c];
        }
      // Rebuild quadrature, normalization rays, cell lookup and fixed-particle
      // ownership from native infrastructure. Only physical accepted fields and
      // counters survive; source-roundoff and point caches are scratch.
      fe_strain_state.reset();
      current_normalization_integrals= {old->I};
      prepare_fe_strain_history(false,true);
      auto &s=*fe_strain_state;
      AssertThrow(s.vertices==old->vertices && s.geometry_version==old->geometry_version
                  && s.n_particles_global==old->n_particles_global,
                  ExcMessage("DG transfer changed fixed surface geometry or particle population."));
      s.dg_history=std::move(history);
      s.Z=old->Z;
      s.I=old->I;
      s.initial_fracture=old->initial_fracture;
      s.committed_step=old->committed_step;
      s.accepted_publications=old->accepted_publications;
      invalidate_normalization_cache();
      dg_transfer.reset();
      validate_fe_strain_restrictions();
      this->get_reconstructed_fault_manager().get_locally_owned_particle_fault_associations();
      this->get_pcout()<<"   Transferred accepted DGQ2 history without physical advancement: "
                       <<old->n_cells_global<<" -> "<<s.n_cells_global<<" cells, publication "<<s.accepted_publications<<'.'<<std::endl;
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::prepare_dg_history_transfer(); \
  template void PhaseFieldFault<dim>::finish_dg_history_transfer();
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
