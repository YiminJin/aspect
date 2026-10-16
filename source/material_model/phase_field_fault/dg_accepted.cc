/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <set>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim> struct PhaseFieldFault<dim>::DGCheckpoint
    {
      unsigned int version=1,step=0,quadrature=0;
      std::uint64_t publications=0,geometry=0;
      double G=0.,eta=0.,m=0.,sense=0.;
      std::vector<double> Z,I,phase,coefficients;
      std::vector<std::string> cells;
      std::vector<Point<dim>> mesh,fault;
    };

    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::prepare_dg_accepted_step(
      const DGTransport &op,const LinearAlgebra::BlockVector &midpoint_bulk,const double time) const
    {
      validate_fe_strain_restrictions();
      AssertThrow(dg_accepted_history && op.space==fe_strain_state->dg_history->space
                  && op.relaxation_rate==0. && op.inflow.empty(),
                  ExcMessage("PF5 requires closed-boundary pure transport; relaxation/source are applied by native half steps."));
      const auto &state=*fe_strain_state;
      const double dt=this->get_timestep();
      AssertThrow(std::isfinite(time) && std::isfinite(dt) && dt>0.
                  && time+dt==this->get_time(),ExcMessage("PF5 candidate requires its prescribed interval clock."));
      const auto &rates=this->get_reconstructed_fault_manager().get_slip_rate(0);
      for (unsigned int j=0; j<rates.size(); ++j)
        AssertThrow(std::isfinite(rates[j]) && rates[j]>=0.
                    && rates[j]==rates[0]
                    && std::abs(state.Z[j]-state.Z[0])<=128*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(state.Z[0])),
                    ExcMessage("PF5 requires uniform surface V/Z; surface transport is not implemented."));
      const auto half=compute_maxwell_coefficients(state.eta,state.G,.5*dt);
      const auto first=make_dg_history_trial(*state.dg_history,midpoint_bulk,half);
      const DGField zero=[](const Point<dim> &,double)
      {
        return SymmetricTensor<2,dim>();
      };
      const auto transported=advance_dg_transport(op,*first,zero,zero,time,dt);
      const auto final=make_dg_history_trial(*transported,midpoint_bulk,half);
      // Reuse the native full-step surface law and immutable accepted I/Z.
      const auto cohesive=compute_surface_cohesive_candidates(dt, {rates});
      auto update=prepare_fe_strain_commit(midpoint_bulk,cohesive,final);
      update->dg_parent=state.dg_history;
      update->dg_cohesive=cohesive.at(0);
      update->dg_rates=rates;
      update->dg_step=this->get_timestep_number();
      update->dg_time=time;
      update->dg_dt=dt;
      return update;
    }

    template <int dim>
    template <class Archive>
    void PhaseFieldFault<dim>::serialize_dg_accepted_history(Archive &archive)
    {
      const bool stored=this->get_reconstructed_fault_manager().has_property("PF5 accepted DG history v1");
      AssertThrow(stored==dg_accepted_history,
                  ExcMessage("PF5 checkpoint mode mismatch; native/particle checkpoint conversion is forbidden."));
      if (!stored) return;
      if constexpr (Archive::is_saving::value)
        {
          validate_fe_strain_restrictions();
          const auto &s=*fe_strain_state;
          AssertThrow(s.committed_step==this->get_timestep_number(),
                      ExcMessage("PF5 checkpoints require an accepted history endpoint."));
          dg_checkpoint=std::make_shared<DGCheckpoint>();
          auto &p=*dg_checkpoint;
          p.step=s.committed_step;
          p.publications=s.accepted_publications;
          p.G=s.G;
          p.eta=s.eta;
          p.m=s.m;
          p.sense=s.shear_sense;
          p.Z=s.Z;
          p.I=s.I;
          p.mesh=s.mesh_vertices;
          p.fault=s.vertices;
          p.geometry=s.geometry_version;
          p.quadrature=fe_bulk_normal_quadrature_points;
          const unsigned int block=this->introspection().variable("phase_field").block_index;
          for (unsigned int i=0; i<s.displacement_history.block(block).size(); ++i)
            p.phase.push_back(s.displacement_history.block(block)[i]);
          std::vector<types::global_dof_index> indices(9);
          for (const auto &entry:s.dg_history->space->cells)
            {
              p.cells.push_back(entry.first.to_string());
              entry.second->get_dof_indices(indices);
              for (const auto &component:s.dg_history->trial_stress)
                for (const auto i:indices) p.coefficients.push_back(component[i]);
            }
        }
      else
        dg_checkpoint=std::make_shared<DGCheckpoint>();
      auto &p=*dg_checkpoint;
      archive &p.version &p.step &p.publications &p.geometry &p.G &p.eta &p.m &p.sense
      &p.quadrature &p.Z &p.I &p.phase &p.mesh &p.fault &p.cells &p.coefficients;
      AssertThrow(p.version==1 && p.G==elastic_shear_moduli[0] && p.eta==reference_viscosities[0]
                  && p.quadrature==fe_bulk_normal_quadrature_points,
                  ExcMessage("PF5 checkpoint version/material/quadrature mismatch."));
    }

    template <int dim>
    void PhaseFieldFault<dim>::restore_dg_accepted_history()
    {
      if (!dg_accepted_history) return;
      AssertThrow(dg_checkpoint && !fe_strain_state
                  && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                  ExcMessage("PF5 restore requires a same-mode serial native checkpoint."));
      const auto &p=*dg_checkpoint;
      AssertThrow(p.mesh==this->get_triangulation().get_vertices()
                  && p.cells.size()==this->get_triangulation().n_global_active_cells()
                  && std::set<std::string>(p.cells.begin(),p.cells.end()).size()==p.cells.size()
                  && p.step==this->get_timestep_number()
                  && p.coefficients.size()==27*p.cells.size(),
                  ExcMessage("PF5 restore rejects changed fixed mesh or malformed DG history."));
      current_normalization_integrals= {p.I};
      prepare_fe_strain_history(true);
      auto &s=*fe_strain_state;
      AssertThrow(s.vertices==p.fault && s.geometry_version==p.geometry
                  && s.shear_sense==p.sense && s.m==p.m
                  && p.Z.size()==s.Z.size() && p.I.size()==s.I.size(),
                  ExcMessage("PF5 restore rejects changed geometry/localization/orientation."));
      const unsigned int block=this->introspection().variable("phase_field").block_index;
      AssertThrow(p.phase.size()==s.displacement_history.block(block).size(),ExcMessage("PF5 phase layout mismatch."));
      for (unsigned int i=0; i<p.phase.size(); ++i)
        AssertThrow(p.phase[i]==s.displacement_history.block(block)[i],ExcMessage("PF5 accepted phase anchor mismatch."));
      std::vector<types::global_dof_index> indices(9);
      unsigned int offset=0;
      for (const auto &id:p.cells)
        {
          const auto entry=s.dg_history->space->cells.find(CellId(id));
          AssertThrow(entry!=s.dg_history->space->cells.end(),ExcMessage("PF5 DG CellId mismatch."));
          entry->second->get_dof_indices(indices);
          for (auto &component:s.dg_history->trial_stress)
            for (const auto i:indices)
              {
                const double value=p.coefficients[offset++];
                AssertThrow(std::isfinite(value),ExcMessage("Nonfinite checkpoint DG history."));
                component[i]=value;
              }
        }
      for (auto &component:s.dg_history->trial_stress) component.compress(VectorOperation::insert);
      for (unsigned int j=0; j<p.I.size(); ++j)
        AssertThrow(std::isfinite(p.I[j]) && p.I[j]>0. && std::isfinite(p.Z[j]) && p.Z[j]>=0.,
                    ExcMessage("Invalid PF5 checkpoint surface history."));
      s.Z=p.Z;
      s.I=p.I;
      s.committed_step=p.step;
      s.accepted_publications=p.publications;
      s.stress_samples.clear();
      // This transient constitutive cache is not history and is normally built
      // by mechanical-solve preparation. A prescribed restart bypasses that
      // solve, so rebuild it from the restored native temperature field.
      compute_fault_surface_temperatures();
      validate_fe_strain_restrictions();
      const auto &manager=this->get_reconstructed_fault_manager();
      const auto cp=manager.get_property_information()[fault_property_indices.cohesive_traction].position;
      const auto ip=manager.get_property_information()[fault_property_indices.previous_normalization_integral].position;
      for (unsigned int j=0; j<s.Z.size(); ++j)
        AssertThrow(manager.get_fault(0).get_properties(j)[ip]==s.I[j]
                    && std::abs(manager.get_fault(0).get_properties(j)[cp]*s.I[j]-s.Z[j])
                    <=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(s.Z[j])),
                    ExcMessage("PF5 checkpoint bulk/surface histories disagree."));
      this->get_pcout()<<"   Restored PF5 accepted DG history at step "<<p.step<<", publications "<<p.publications<<'.'<<std::endl;
    }

#define INSTANTIATE(dim) \
  template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::prepare_dg_accepted_step(const DGTransport &,const LinearAlgebra::BlockVector &,double) const; \
  template void PhaseFieldFault<dim>::serialize_dg_accepted_history<aspect::oarchive>(aspect::oarchive &); \
  template void PhaseFieldFault<dim>::serialize_dg_accepted_history<aspect::iarchive>(aspect::iarchive &); \
  template void PhaseFieldFault<dim>::restore_dg_accepted_history();
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
