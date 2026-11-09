/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <set>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim> struct PhaseFieldFault<dim>::DGCheckpoint
    {
      unsigned int version = 1, step = 0, quadrature = 0, clock_step = 0;
      double clock_time = 0., clock_dt = 0.;
      std::uint64_t publications = 0, geometry = 0, geometry_publications = 0;
      double G = 0., eta = 0., m = 0., sense = 0.;
      std::string material_law;
      std::vector<double> Z, I, phase, coefficients, residual_coefficients;
      std::vector<std::string> cells;
      std::vector<Point<dim>> mesh, fault;
    };

    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::prepare_dg_accepted_step(
      const DGTransport &op, const LinearAlgebra::BlockVector &midpoint_bulk, const double time) const
    {
      validate_fe_strain_restrictions();
      AssertThrow(dg_accepted_history && !mature_frictional_fault && !consistent_endpoint_phase && op.space == fe_strain_state->bulk_history.tensor->primary.space
                  && op.relaxation_rate == 0. && op.inflow.empty(),
                  ExcMessage("PF5 requires closed-boundary pure transport; relaxation/source are applied by native half steps."));
      const auto &state = *fe_strain_state;
      const double dt = this->get_timestep();
      AssertThrow(std::isfinite(time) && std::isfinite(dt) && dt > 0.
                  && time + dt == this->get_time(), ExcMessage("PF5 candidate requires its prescribed interval clock."));
      const auto &rates = this->get_reconstructed_fault_manager().get_slip_rate(0);
      for (unsigned int j = 0; j < rates.size(); ++j)
        AssertThrow(std::isfinite(rates[j]) && rates[j] >= 0.
                    && rates[j] == rates[0]
                    && std::abs(state.Z[j] - state.Z[0]) <= 128 * std::numeric_limits<double>::epsilon()*std::max(1., std::abs(state.Z[0])),
                    ExcMessage("PF5 requires uniform surface V/Z; surface transport is not implemented."));
      const auto half = compute_maxwell_coefficients(state.eta, state.G, .5 * dt);
      const auto first = make_dg_history_trial(*state.bulk_history.tensor, midpoint_bulk, half);
      const DGField zero = [](const Point<dim> &, double)
      {
        return SymmetricTensor<2, dim>();
      };
      const auto transported = advance_dg_transport(op, *first, zero, zero, time, dt);
      const auto final = make_dg_history_trial(*transported, midpoint_bulk, half);
      // Reuse the native full-step surface law and immutable accepted I/Z.
      const auto cohesive = compute_surface_cohesive_candidates(dt, {rates});
      auto update = prepare_fe_strain_commit(midpoint_bulk, cohesive, final);
      update->prescribed.parent = state.bulk_history.tensor;
      update->prescribed.cohesive = cohesive.at(0);
      update->prescribed.rates = rates;
      update->prescribed.step = this->get_timestep_number();
      update->prescribed.start_time = time;
      update->prescribed.dt = dt;
      return update;
    }

    template <int dim>
    template <class Archive>
    void
    PhaseFieldFault<dim>::serialize_dg_accepted_history(Archive &archive)
    {
      const bool stored = this->get_reconstructed_fault_manager().has_property("PF5 accepted DG history v1");
      AssertThrow(stored == dg_accepted_history,
                  ExcMessage("PF5 checkpoint mode mismatch; native/particle checkpoint conversion is forbidden."));
      if (!stored)
        return;
      if constexpr (Archive::is_saving::value)
        {
          validate_fe_strain_restrictions();
          const auto &s = *fe_strain_state;
          // Native outer checkpoints follow advance_time: their clock is one
          // step ahead of the accepted constitutive endpoint. PF5 manual
          // prescribed snapshots retain their existing at-endpoint contract.
          AssertThrow(uses_prescribed_dg_geometry()
                      ? (s.committed_step == 0 && this->get_timestep_number() == 0 && this->get_timestep() == 0.)
                      : (consistent_endpoint_phase || mature_frictional_fault)
                      ? s.committed_step + 1 == this->get_timestep_number()
                      : s.committed_step == this->get_timestep_number(),
                      ExcMessage("PF5 checkpoints require an accepted history endpoint."));
          if (consistent_endpoint_phase || mature_frictional_fault)
            {
              const unsigned int block = this->introspection().phase_field().block_index;
              AssertThrow(this->get_solution() == this->get_old_solution()
                          && s.bulk_history.native.block(block) == this->get_solution().block(block)
                          && s.accepted_publications == s.committed_step,
                          ExcMessage("PF6a checkpoints require the canonical accepted state after advance_time, not a trial."));
            }
          dg_checkpoint = std::make_shared<DGCheckpoint>();
          auto &p = *dg_checkpoint;
          p.version = dg_material_transport ? 7 : (jaumann_dg_stress ? (constitutive_spin==ConstitutiveSpin::bulk_slip_subtracted ? 8 : 6) : (uses_prescribed_dg_geometry() ? 5 : (mature_frictional_fault ? 4 : (consistent_endpoint_phase ? 3 : 1))));
          if (this->get_phase_field_handler().uses_dg_phase()) p.version=9;
          p.material_law = material_viscosity_identity;
          if (dg_material_transport && this->get_reconstructed_fault_manager().uses_full_normal_cross_section())
            p.material_law += "|association support:full normal cross-section";
          p.geometry_publications = s.geometry_publications;
          p.step = s.committed_step;
          p.clock_step = this->get_timestep_number();
          p.clock_time = this->get_time();
          p.clock_dt = this->get_timestep();
          p.publications = s.accepted_publications;
          p.G = s.G;
          p.eta = s.eta;
          p.m = s.m;
          p.sense = s.shear_sense;
          p.Z = s.Z;
          p.I = s.I;
          p.mesh = s.mesh_vertices;
          p.fault = s.vertices;
          p.geometry = s.geometry_version;
          p.quadrature = fe_bulk_normal_quadrature_points;
          const unsigned int block = this->introspection().phase_field().block_index;
          std::vector<types::global_dof_index> indices(9);
          if (p.version >= 3)
            {
              // Only uniquely owned CellIds contribute. Store native coefficients
              // and generalized interpolation anchors, not fictitious nodal
              // values for Bernstein. The outer phase-space metadata guards
              // basis identity before this archive is read. Anchors and cell
              // IDs avoid partition-dependent native DoF numbering.
              std::map<std::string, std::vector<double>> local, global;
              const auto pc = this->introspection().phase_field().first_component_index;
              for (const auto &entry : s.cells)
                {
                  auto &values = local[entry.first.to_string()];
                  const auto &element=this->get_fe().base_element(this->get_fe().component_to_base_index(pc).first);
                  std::vector<types::global_dof_index> native_indices(this->get_fe().n_dofs_per_cell());
                  entry.second->get_dof_indices(native_indices);
                  for (unsigned int v=0;v<element.n_dofs_per_cell();++v)
                    {
                      const auto point=p.version==9 ? this->get_mapping().transform_unit_to_real_cell(entry.second,element.get_generalized_support_points()[v]) : entry.second->vertex(v);
                      for (unsigned int d=0;d<dim;++d) values.push_back(point[d]);
                      values.push_back(s.bulk_history.native[native_indices[this->get_fe().component_to_system_index(pc,v)]]);
                    }
                  s.bulk_history.tensor->primary.space->cells.at(entry.first)->get_dof_indices(indices);
                  for (const auto &component : s.bulk_history.tensor->primary.coefficients)
                    for (const auto i : indices)
                      values.push_back(component[i]);
                  if (jaumann_dg_stress && !material_direct_stress)
                    {
                      AssertThrow(s.bulk_history.tensor->residual, ExcMessage("Objective checkpoint lacks residual history."));
                      for (const auto &component : s.bulk_history.tensor->residual->coefficients)
                        for (const auto i : indices)
                          values.push_back(component[i]);
                    }
                }
              for (const auto &part : Utilities::MPI::all_gather(this->get_mpi_communicator(), local))
                for (const auto &entry : part)
                  AssertThrow(global.emplace(entry).second, ExcMessage("Duplicate DG checkpoint CellId owner."));
              p.mesh.clear();
              for (const auto &entry : global)
                {
                  p.cells.push_back(entry.first);
                  unsigned int offset = 0;
                  for (unsigned int v = 0; v < (p.version==9 ? 9u : GeometryInfo<dim>::vertices_per_cell); ++v)
                    {
                      Point<dim> x;
                      for (unsigned int d = 0; d < dim; ++d)
                        x[d] = entry.second[offset++];
                      p.mesh.push_back(x);
                      p.phase.push_back(entry.second[offset++]);
                    }
                  AssertThrow(entry.second.size() == offset + (jaumann_dg_stress && !material_direct_stress ? 54 : 27), ExcMessage("Malformed complete DG checkpoint record."));
                  p.coefficients.insert(p.coefficients.end(), entry.second.begin() + offset, entry.second.begin() + offset + 27);
                  if (jaumann_dg_stress && !material_direct_stress)
                    p.residual_coefficients.insert(p.residual_coefficients.end(), entry.second.begin() + offset + 27, entry.second.end());
                }
            }
          else
            {
              for (unsigned int i = 0; i < s.bulk_history.native.block(block).size(); ++i)
                p.phase.push_back(s.bulk_history.native.block(block)[i]);
              for (const auto &entry : s.bulk_history.tensor->primary.space->cells)
                {
                  p.cells.push_back(entry.first.to_string());
                  entry.second->get_dof_indices(indices);
                  for (const auto &component : s.bulk_history.tensor->primary.coefficients)
                    for (const auto i : indices)
                      p.coefficients.push_back(component[i]);
                }
            }
        }
      else
        dg_checkpoint = std::make_shared<DGCheckpoint>();
      auto &p = *dg_checkpoint;
      archive &p.version &p.step &p.publications &p.geometry &p.G &p.eta &p.m &p.sense
      &p.quadrature &p.Z &p.I &p.phase &p.mesh &p.fault &p.cells &p.coefficients;
      if (p.version >= 2)
        archive &p.clock_step &p.clock_time &p.clock_dt;
      if (p.version == 5 || p.version == 7 || p.version == 9)
        archive &p.geometry_publications;
      if (p.version >= 6)
        archive &p.residual_coefficients;
      // Schema6 retains the old stationary total-spin meaning. Schema8 adds
      // an explicit constitutive identity to stationary mature DG only.
      if (p.version == 7 || p.version == 8 || p.version == 9)
        archive &p.material_law;
      const auto material_identity = material_viscosity_identity
                                     + ((dg_material_transport && this->get_reconstructed_fault_manager().uses_full_normal_cross_section())
                                        ? "|association support:full normal cross-section" : "");
      AssertThrow((this->get_phase_field_handler().uses_dg_phase()
                   ? (p.version==9 && p.material_law==material_identity)
                   : (dg_material_transport ? (p.version == 7 && p.material_law == material_identity) : (jaumann_dg_stress ? (constitutive_spin==ConstitutiveSpin::bulk_slip_subtracted ? (p.version == 8 && p.material_law == material_identity) : p.version == 6) : (uses_prescribed_dg_geometry() ? p.version == 5 : (mature_frictional_fault ? p.version == 4 : (consistent_endpoint_phase ? (p.version == 2 || p.version == 3) : p.version == 1)))))) && p.G == elastic_shear_moduli[0] && p.eta == reference_viscosities[0]
                  && p.quadrature == fe_bulk_normal_quadrature_points,
                  ExcMessage("PF5 checkpoint version/material/quadrature mismatch. Constitutive spin is part of the material identity; no automatic law conversion."));
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::restore_dg_accepted_history()
    {
      if (!dg_accepted_history)
        return;
      AssertThrow(dg_checkpoint && !fe_strain_state,
                  ExcMessage("DG restore requires a same-mode native checkpoint."));
      const auto &p = *dg_checkpoint;
      AssertThrow((p.version >= 3 || (Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 1
                                      && p.mesh == this->get_triangulation().get_vertices()))
                  && p.cells.size() == this->get_triangulation().n_global_active_cells()
                  && std::set<std::string>(p.cells.begin(), p.cells.end()).size() == p.cells.size()
                  && (uses_prescribed_dg_geometry()
                      ? (p.step == 0 && p.publications == 0 && this->get_timestep_number() == 0
                         && p.clock_step == 0 && p.clock_time == this->get_time() && p.clock_dt == 0. && this->get_timestep() == 0.)
                      : (consistent_endpoint_phase || mature_frictional_fault)
                      ? (p.step + 1 == this->get_timestep_number() && p.clock_step == this->get_timestep_number()
                         && p.clock_time == this->get_time() && p.clock_dt == this->get_timestep())
                      : p.step == this->get_timestep_number())
                  && p.coefficients.size() == 27 * p.cells.size()
                  && (material_direct_stress ? p.residual_coefficients.empty() : (!jaumann_dg_stress || p.residual_coefficients.size() == p.coefficients.size())),
                  ExcMessage("PF5 restore rejects changed fixed mesh or malformed DG history."));
      current_normalization_integrals = {p.I};
      prepare_fe_strain_history(true);
      auto &s = *fe_strain_state;
      AssertThrow(s.vertices == p.fault && s.geometry_version == p.geometry
                  && s.shear_sense == p.sense && s.m == p.m
                  && p.Z.size() == s.Z.size() && p.I.size() == s.I.size(),
                  ExcMessage("PF5 restore rejects changed geometry/localization/orientation."));
      const unsigned int block = this->introspection().phase_field().block_index;
      const auto pc = this->introspection().phase_field().first_component_index;
      if (p.version >= 3)
        AssertThrow(p.phase.size() == (p.version==9 ? 9u:4u) * p.cells.size() && p.mesh.size() == p.phase.size(),
                    ExcMessage("DG checkpoint physical phase/corner layout mismatch."));
      else
        {
          AssertThrow(p.phase.size() == s.bulk_history.native.block(block).size(), ExcMessage("PF5 phase layout mismatch."));
          for (unsigned int i = 0; i < p.phase.size(); ++i)
            AssertThrow(p.phase[i] == s.bulk_history.native.block(block)[i], ExcMessage("PF5 accepted phase anchor mismatch."));
        }
      std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components> owned;
      for (auto &component : owned)
        component.reinit(s.bulk_history.tensor->primary.space->dofs.locally_owned_dofs(), this->get_mpi_communicator());
      std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components> residual_owned;
      if (jaumann_dg_stress && !material_direct_stress)
        for (auto &component : residual_owned)
          component.reinit(s.bulk_history.tensor->primary.space->dofs.locally_owned_dofs(), this->get_mpi_communicator());
      std::vector<types::global_dof_index> indices(9);
      unsigned int offset = 0, cell_number = 0, matching_cells = 0;
      unsigned int matching_phase = 1;
      for (const auto &id : p.cells)
        {
          const auto entry = s.bulk_history.tensor->primary.space->cells.find(CellId(id));
          if (p.version < 3)
            AssertThrow(entry != s.bulk_history.tensor->primary.space->cells.end(), ExcMessage("PF5 DG CellId mismatch."));
          const auto native = s.cells.find(CellId(id));
          if (native != s.cells.end())
            ++matching_cells;
          if (p.version >= 3 && native != s.cells.end())
            {
              const auto &element=this->get_fe().base_element(this->get_fe().component_to_base_index(pc).first);
              std::vector<types::global_dof_index> native_indices(this->get_fe().n_dofs_per_cell());
              native->second->get_dof_indices(native_indices);
              for (unsigned int v=0;v<element.n_dofs_per_cell();++v)
                {
                  const unsigned int i=element.n_dofs_per_cell()*cell_number+v;
                  const auto point=p.version==9 ? this->get_mapping().transform_unit_to_real_cell(native->second,element.get_generalized_support_points()[v]) : native->second->vertex(v);
                  if (p.mesh[i]!=point || p.phase[i]!=s.bulk_history.native[native_indices[this->get_fe().component_to_system_index(pc,v)]])
                    matching_phase=0;
                }
            }
          const bool local = entry != s.bulk_history.tensor->primary.space->cells.end() && entry->second->is_locally_owned();
          if (local)
            entry->second->get_dof_indices(indices);
          for (auto &component : owned)
            for (unsigned int i = 0; i < 9; ++i)
              {
                const double value = p.coefficients[offset++];
                AssertThrow(std::isfinite(value), ExcMessage("Nonfinite checkpoint DG history."));
                if (local)
                  component[indices[i]] = value;
              }
          if (jaumann_dg_stress && !material_direct_stress)
            {
              unsigned int r = 27 * cell_number;
              for (auto &component : residual_owned)
                for (unsigned int i = 0; i < 9; ++i)
                  {
                    const double value = p.residual_coefficients[r++];
                    AssertThrow(std::isfinite(value), ExcMessage("Nonfinite complete DG residual checkpoint."));
                    if (local)
                      component[indices[i]] = value;
                  }
            }
          ++cell_number;
        }
      AssertThrow(Utilities::MPI::min(matching_phase, this->get_mpi_communicator()) == 1
                  && Utilities::MPI::sum(matching_cells, this->get_mpi_communicator()) == p.cells.size(),
                  ExcMessage("DG checkpoint physical mesh/phase anchor or CellId ownership mismatch."));
      for (unsigned int c = 0; c < owned.size(); ++c)
        {
          owned[c].compress(VectorOperation::insert);
          s.bulk_history.tensor->primary.coefficients[c] = owned[c];
        }
      if (jaumann_dg_stress && !material_direct_stress)
        {
          AssertThrow(s.bulk_history.tensor->residual, ExcMessage("Missing restored DG residual owner."));
          for (unsigned int c = 0; c < residual_owned.size(); ++c)
            {
              residual_owned[c].compress(VectorOperation::insert);
              s.bulk_history.tensor->residual->coefficients[c] = residual_owned[c];
            }
          s.objective.tensor.reset();
          s.objective.parent.reset();
        }
      for (unsigned int j = 0; j < p.I.size(); ++j)
        AssertThrow(std::isfinite(p.I[j]) && p.I[j] > 0. && std::isfinite(p.Z[j]) && p.Z[j] >= 0.,
                    ExcMessage("Invalid PF5 checkpoint surface history."));
      s.Z = p.Z;
      s.I = p.I;
      s.committed_step = p.step;
      s.accepted_publications = p.publications;
      s.geometry_publications = p.geometry_publications;
      s.observations.stress.clear();
      // This transient constitutive cache is not history and is normally built
      // by mechanical-solve preparation. A prescribed restart bypasses that
      // solve, so rebuild it from the restored native temperature field.
      compute_fault_surface_temperatures();
      validate_fe_strain_restrictions();
      const auto &manager = this->get_reconstructed_fault_manager();
      const auto cp = manager.get_property_information()[fault_property_indices.cohesive_traction].position;
      const auto ip = manager.get_property_information()[fault_property_indices.previous_normalization_integral].position;
      for (unsigned int j = 0; j < s.Z.size(); ++j)
        AssertThrow(manager.get_fault(0).get_properties(j)[ip] == s.I[j]
                    && (mature_frictional_fault
                        ? manager.get_fault(0).get_properties(j)[cp] == 0.
                        : std::abs(manager.get_fault(0).get_properties(j)[cp]*s.I[j] - s.Z[j])
                        <= 64 * std::numeric_limits<double>::epsilon()*std::max(1., std::abs(s.Z[j]))),
                    ExcMessage("PF5 checkpoint bulk/surface histories disagree."));
      this->get_pcout() << "   Restored PF5 accepted DG history at step " << p.step << ", publications " << p.publications << '.' << std::endl;
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
