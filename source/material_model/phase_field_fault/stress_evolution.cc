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
    template <int dim>
    void
    PhaseFieldFault<dim>::transport_material_history(const MaterialHistoryInputs &input)
    {
      const auto &accepted = input.accepted;
      const double dt = input.dt;
      const auto &space = *accepted.bulk_history.tensor->primary.space;
      const auto comm = this->get_mpi_communicator();
      const auto project = [&](const MaterialTensorField & field)
      {
        const auto &profile = accepted;
        auto projected = std::make_shared<DGHistory>();
        projected->representation = accepted.bulk_history.tensor->representation;
        projected->primary = *project_material_tensor(accepted.bulk_history.tensor->primary, profile,
                                                      input.accepted_velocity, field);
        return projected;
      };
      // Evolve the whole R+B combination. Its subtraction uses the transported
      // profile/normalization below, so no geometry/product-rule term is dropped.
      std::shared_ptr<DGHistory> separated_memory;
      if (!material_direct_stress)
        {
          separated_memory = project([&](const auto &, const auto &, const auto & x, const double phi, const auto &)
          {
            return evaluate_mature_dg_memory(x, phi, accepted.Z, accepted.I);
          });
          for (unsigned c = 0; c < separated_memory->primary.coefficients.size(); ++c)
            {
              LinearAlgebra::Vector owned(space.dofs.locally_owned_dofs(), comm);
              owned = separated_memory->primary.coefficients[c];
              for (const auto i : owned.locally_owned_elements())
                owned[i] += accepted.bulk_history.tensor->residual->coefficients[c][i];
              owned.compress(VectorOperation::insert);
              separated_memory->primary.coefficients[c] = owned;
            }
        }
      auto transported = std::make_shared<DGHistory>(*accepted.bulk_history.tensor);
      transported->residual.reset();
      // Optional diagnostic ledger, not constitutive state or checkpoint data.
      const bool energy_ledger = material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_ENERGY");
      const double energy_before = energy_ledger ? mature_tensor_energy(*transported, accepted) : 0.;

      const auto transport = make_dg_transport(input.accepted_velocity, std::max(8u, fe_bulk_normal_quadrature_points), false);
      // Use the retained engineering substep target 0.1 (operator bound 0.125).
      // 4096 limits local work and requests a smaller physical step; it is not a
      // new constitutive cutoff or a claimed sharp DG stability theorem.
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
              primary_inflow.emplace(load.point, transported->primary.tensor_at(load.indices, unit));
              if (separated_memory)
                memory_inflow.emplace(load.point, separated_memory->primary.tensor_at(load.indices, unit));
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
            const double start = input.end_time - dt + dt * k / substeps, sub_dt = dt / substeps;
            transported = advance_dg_transport(*transport, *transported, primary_boundary, zero, start, sub_dt);
            if (separated_memory)
              separated_memory = advance_dg_transport(*transport, *separated_memory, memory_boundary, zero, start, sub_dt);
          }
      }
      const double energy_advected = energy_ledger ? mature_tensor_energy(*transported, accepted) : 0.;
      auto &workspace = dg_material_trial->transport;
      if (material_direct_stress)
        workspace.physical_stress = transported;
      else workspace.separated_trial = transported;
      workspace.separated_total_memory = separated_memory;
      workspace.substeps = substeps;
      workspace.energy_ledger = energy_ledger;
      workspace.energy_before = energy_before;
      workspace.energy_advected = energy_advected;
    }
    template <int dim>
    double
    PhaseFieldFault<dim>::mature_tensor_energy(
      const DGHistory &history, const FEStrainState &accepted) const
    {
      const auto &space = *history.primary.space;
      const auto comm = this->get_mpi_communicator();
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      double energy = 0.;
      FEValues<dim> fe(this->get_mapping(), space.element, rule, update_JxW_values);
      for (const auto &entry : accepted.cells)
        {
          fe.reinit(space.cells.at(entry.first));
          std::vector<types::global_dof_index> ids(9);
          space.cells.at(entry.first)->get_dof_indices(ids);
          for (unsigned q = 0; q < rule.size(); ++q)
            {
              const auto a = history.primary.tensor_at(ids, rule.point(q));
              // Reconstruct the plane-strain deviator component zz=-xx-yy;
              // its square supplies trace(a)^2 in the full 3-D contraction.
              energy += fe.JxW(q) * (scalar_product(a, a) + trace(a) * trace(a)) / (4 * accepted.G);
            }
        }
      return Utilities::MPI::sum(energy, comm);
    }
    template <int dim>
    void
    PhaseFieldFault<dim>::rotate_material_history(const MaterialHistoryInputs &input)
    {
      const auto &accepted = input.accepted;
      const double dt = input.dt;
      auto candidate = dg_material_trial->candidate_history;
      const auto &workspace = dg_material_trial->transport;
      const auto &transported = material_direct_stress ? workspace.physical_stress : workspace.separated_trial;
      const auto &separated_memory = workspace.separated_total_memory;
      const auto project = [&](const MaterialTensorField & field)
      {
        const auto &profile = *candidate;
        auto projected = std::make_shared<DGHistory>();
        projected->representation = accepted.bulk_history.tensor->representation;
        projected->primary = *project_material_tensor(accepted.bulk_history.tensor->primary, profile,
                                                      input.accepted_velocity, field);
        return projected;
      };
      // Accepted velocity supplies total material spin, lagged over this step.
      // Rotate then mass-project the advected field on the staged profile. The
      // resulting anchor must not be recomputed from a Newton velocity iterate.
      auto anchor = project([&](const auto & ids, const auto & u, const auto &, double, const auto & gradient)
      {
        if constexpr(dim == 2)
          return ReconstructedFaultUtilities::rotate_plane_strain_stress(transported->primary.tensor_at(ids, u), gradient, dt);
        else return SymmetricTensor<2, dim>();
      });
      if (!material_direct_stress)
        anchor->residual = project_material_tensor(accepted.bulk_history.tensor->primary, *candidate, input.accepted_velocity, [&](const auto & ids, const auto & u, const auto & x, double p, const auto & gradient)
        {
          if constexpr(dim == 2)
            return ReconstructedFaultUtilities::rotate_plane_strain_stress(separated_memory->primary.tensor_at(ids, u), gradient, dt)
                   - evaluate_mature_dg_memory(x, p, candidate->Z, candidate->I);
          else return SymmetricTensor<2, dim>();
        });
      if (workspace.energy_ledger)
        this->get_pcout() << std::setprecision(17) << "   Direct transport energy: step=" << this->get_timestep_number()
                          << " dt=" << dt << " accepted=" << workspace.energy_before << " advected=" << workspace.energy_advected
                          << " rotated_projected=" << mature_tensor_energy(*anchor, accepted) << std::endl;
      candidate->objective.tensor = anchor;
      candidate->objective.parent = candidate->bulk_history.tensor;
      candidate->objective.step = this->get_timestep_number();
      candidate->objective.dt = dt;
      accepted.observations.tensor_points->cells.clear();
      accepted.observations.tensor_points->geometry_generation = numbers::invalid_unsigned_int;
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
