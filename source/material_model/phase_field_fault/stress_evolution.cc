/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <cstdlib>
#include <iomanip>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> void PhaseFieldFault<dim>::transport_material_history(const MaterialHistoryInputs &input)
    {
      const auto &old = input.accepted;
      const double dt = input.dt;
      const auto &space = *old.dg_history->space;
      const auto comm = this->get_mpi_communicator();
      const auto project = [&](const typename DGHistory::MaterialField & field)
      {
        const auto &profile = old;
        return DGHistory::project(*this, *old.dg_history, profile.cells, profile.displacement_history,
                                  input.accepted_velocity, std::max(8u, fe_bulk_normal_quadrature_points), field);
      };
      // Evolve the whole R+B combination. Its subtraction uses the transported
      // profile/normalization below, so no geometry/product-rule term is dropped.
      std::shared_ptr<DGHistory> separated_memory;
      if (!material_direct_stress)
        {
          separated_memory = project([&](const auto &, const auto &, const auto & x, const double phi, const auto &)
          {
            return evaluate_mature_dg_memory(x, phi, old.Z, old.I);
          });
          for (unsigned c = 0; c < separated_memory->coefficients.size(); ++c)
            {
              LinearAlgebra::Vector owned(space.dofs.locally_owned_dofs(), comm);
              owned = separated_memory->coefficients[c];
              for (const auto i : owned.locally_owned_elements())
                owned[i] += old.dg_history->residual->coefficients[c][i];
              owned.compress(VectorOperation::insert);
              separated_memory->coefficients[c] = owned;
            }
        }
      auto primary = std::make_shared<DGHistory>(*old.dg_history);
      primary->residual.reset();
      // Optional diagnostic ledger, not constitutive state or checkpoint data.
      const bool energy_ledger = material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_ENERGY");
      const double energy_before = energy_ledger ? mature_tensor_energy(*primary, old) : 0.;

      const auto transport = make_dg_transport(input.accepted_velocity, std::max(8u, fe_bulk_normal_quadrature_points), false);
      const unsigned substeps = std::max(1u, static_cast<unsigned>(std::ceil(dt * transport->inverse_advection_time / .1)));
      AssertThrow(substeps <= 4096, ExcMessage("Material DG advection requires a smaller physical timestep (CFL substep budget)."));
      const DGField zero = [](const Point<dim> &, double)
      {
        return SymmetricTensor<2, dim>();
      };
      const auto point_less = [](const Point<dim> &a, const Point<dim> &b)
      {
        for (unsigned d = 0; d < dim; ++d)
          {
            if (a[d] < b[d])
              return true;
            if (b[d] < a[d])
              return false;
          }
        return false;
      };
      std::map<Point<dim>, SymmetricTensor<2, dim>, decltype(point_less)>
      primary_inflow(point_less), memory_inflow(point_less);
      if (material_trace_inflow)
        {
          std::map<types::global_dof_index, CellId> owner;
          for (const auto &cell : transport->cells)
            owner.emplace(cell.indices.front(), cell.id);
          for (const auto &load : transport->inflow)
            {
              const auto cell = space.cells.at(owner.at(load.indices.front()));
              const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, load.point);
              primary_inflow.emplace(load.point, primary->tensor_at(load.indices, unit));
              if (separated_memory)
                memory_inflow.emplace(load.point, separated_memory->tensor_at(load.indices, unit));
            }
        }
      const DGField primary_boundary = [&](const Point<dim> &x, double)
      {
        return material_trace_inflow ? primary_inflow.at(x) : SymmetricTensor<2, dim>();
      };
      const DGField memory_boundary = [&](const Point<dim> &x, double)
      {
        return material_trace_inflow ? memory_inflow.at(x) : SymmetricTensor<2, dim>();
      };
      {
        TimerOutput::Scope timer(this->get_computing_timer(), "Material tensor advection");
        for (unsigned k = 0; k < substeps; ++k)
          {
            const double start = input.time - dt + dt * k / substeps, sub_dt = dt / substeps;
            primary = advance_dg_transport(*transport, *primary, primary_boundary, zero, start, sub_dt);
            if (separated_memory)
              separated_memory = advance_dg_transport(*transport, *separated_memory, memory_boundary, zero, start, sub_dt);
          }
      }
      const double energy_advected = energy_ledger ? mature_tensor_energy(*primary, old) : 0.;
      auto &workspace = dg_material_trial->transport;
      if (material_direct_stress)
        workspace.physical_stress = primary;
      else workspace.separated_trial = primary;
      workspace.separated_total_memory = separated_memory;
      workspace.substeps = substeps;
      workspace.energy_ledger = energy_ledger;
      workspace.energy_before = energy_before;
      workspace.energy_advected = energy_advected;
    }
    template<int dim> double PhaseFieldFault<dim>::mature_tensor_energy(
      const DGHistory &history, const FEStrainState &old) const
    {
      const auto &space = *history.space;
      const auto comm = this->get_mpi_communicator();
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      double energy = 0.;
      FEValues<dim> fe(this->get_mapping(), space.element, rule, update_JxW_values);
      for (const auto &entry : old.cells)
        {
          fe.reinit(space.cells.at(entry.first));
          std::vector<types::global_dof_index> ids(9);
          space.cells.at(entry.first)->get_dof_indices(ids);
          for (unsigned q = 0; q < rule.size(); ++q)
            {
              const auto a = history.tensor_at(ids, rule.point(q));
              energy += fe.JxW(q) * (scalar_product(a, a) + trace(a) * trace(a)) / (4 * old.G);
            }
        }
      return Utilities::MPI::sum(energy, comm);
    }
    template<int dim> void PhaseFieldFault<dim>::rotate_material_history(const MaterialHistoryInputs &input)
    {
      const auto &old = input.accepted;
      const double dt = input.dt;
      auto next = dg_material_trial->candidate_history;
      const auto &workspace = dg_material_trial->transport;
      const auto &primary = material_direct_stress ? workspace.physical_stress : workspace.separated_trial;
      const auto &separated_memory = workspace.separated_total_memory;
      const auto project = [&](const typename DGHistory::MaterialField & field)
      {
        const auto &profile = *next;
        return DGHistory::project(*this, *old.dg_history, profile.cells, profile.displacement_history,
                                  input.accepted_velocity, std::max(8u, fe_bulk_normal_quadrature_points), field);
      };
      auto anchor = project([&](const auto & ids, const auto & u, const auto &, double, const auto & gradient)
      {
        if constexpr(dim == 2)
          return ReconstructedFaultUtilities::rotate_plane_strain_stress(primary->tensor_at(ids, u), gradient, dt);
        else return SymmetricTensor<2, dim>();
      });
      if (!material_direct_stress)
        anchor->residual = project([&](const auto & ids, const auto & u, const auto & x, double p, const auto & gradient)
        {
          if constexpr(dim == 2)
            return ReconstructedFaultUtilities::rotate_plane_strain_stress(separated_memory->tensor_at(ids, u), gradient, dt)
                   - evaluate_mature_dg_memory(x, p, next->Z, next->I);
          else return SymmetricTensor<2, dim>();
        });
      if (workspace.energy_ledger)
        this->get_pcout() << std::setprecision(17) << "   Direct transport energy: step=" << this->get_timestep_number()
                          << " dt=" << dt << " accepted=" << workspace.energy_before << " advected=" << workspace.energy_advected
                          << " rotated_projected=" << mature_tensor_energy(*anchor, old) << std::endl;
      next->objective_anchor = anchor;
      next->objective_parent = next->dg_history;
      next->objective_step = this->get_timestep_number();
      next->objective_dt = dt;
      old.dg_history->space->point_samples.clear();
      old.dg_history->space->point_geometry_generation = numbers::invalid_unsigned_int;
      this->get_pcout() << "   Material DG staged complete curve/profile/tensor/surface state; advection substeps=" << workspace.substeps << std::endl;
    }
#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::transport_material_history(const MaterialHistoryInputs &); \
  template double PhaseFieldFault<dim>::mature_tensor_energy(const DGHistory &,const FEStrainState &) const; \
  template void PhaseFieldFault<dim>::rotate_material_history(const MaterialHistoryInputs &);
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
