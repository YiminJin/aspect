/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_history.h>
#include <aspect/material_model/phase_field_fault/stress_law.h>
#include <cstdlib>
#include <iomanip>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::make_material_reaction(const FEStrainState &s, const LinearAlgebra::BlockVector &bulk,
                                                 const std::vector<double> &rates) const
    {
      AssertThrow(dg_material_transport && s.objective_anchor,
                  ExcMessage("Projected material reaction requires its complete transported anchor."));
      TimerOutput::Scope timer(this->get_computing_timer(), "Material reaction projection");
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      auto update = std::make_shared<FEStrainUpdate>();
      update->displacement_history = s.displacement_history;
      update->I = s.I;
      update->Z.resize(s.Z.size());
      AssertDimension(rates.size(), s.Z.size());
      for (unsigned j = 0; j < s.Z.size(); ++j)
        {
          const auto c = compute_maxwell_coefficients(material_viscosity(s.vertices[j], s.eta), s.G, dt);
          update->Z[j] = c.beta * s.Z[j] + c.eta_ve * rates[j];
        }
      const MatureReactionField reaction_field = [&](const Point<dim> &x, const double phase,
                                                     const SymmetricTensor<2, dim> &strain, const SymmetricTensor<2, dim> &primary,
                                                     const SymmetricTensor<2, dim> &separated_residual)
      {
        const auto c = compute_maxwell_coefficients(material_viscosity(x, s.eta), s.G, dt);
        const auto B0 = material_direct_stress ? SymmetricTensor<2, dim>() : evaluate_mature_dg_memory(x, phase, s.Z, s.I);
        const auto B1 = material_direct_stress ? SymmetricTensor<2, dim>() : evaluate_mature_dg_memory(x, phase, update->Z, update->I);
        const auto twice_LV = evaluate_mature_dg_memory(x, phase, rates, s.I);
        if (material_direct_stress)
          return std::array<SymmetricTensor<2, dim>, 2>
          {
            MaterialModel::Rheology::mature_stress_endpoint(c.beta, c.eta_ve, primary, strain, twice_LV),
            SymmetricTensor<2, dim>()
          };
        return std::array<SymmetricTensor<2, dim>, 2>
        {
          c.beta *primary + 2 * c.eta_ve *Utilities::Tensors::consistent_deviator(strain),
          c.beta *(separated_residual + B0) + c.eta_ve *twice_LV - B1
        };
      };
      project_mature_reaction(s, bulk, *update, reaction_field);
      return update;
    }

    template<int dim> void PhaseFieldFault<dim>::project_mature_reaction(
      const FEStrainState &s, const LinearAlgebra::BlockVector &bulk,
      FEStrainUpdate &update, const MatureReactionField &field) const
    {
      const auto &anchor = *s.objective_anchor;
      const auto &space = *anchor.space;
      const auto comm = this->get_mpi_communicator();
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      update.dg_history = std::make_shared<DGHistory>();
      update.dg_history->space = anchor.space;
      update.dg_history->representation = anchor.representation;
      if (!material_direct_stress)
        {
          update.dg_history->residual = std::make_shared<DGHistory>();
          update.dg_history->residual->space = anchor.space;
        }
      constexpr unsigned components = SymmetricTensor<2, dim>::n_independent_components;
      const unsigned tensors = material_direct_stress ? 1 : 2;
      std::vector<LinearAlgebra::Vector> owned(tensors * components);
      for (auto &v : owned)
        v.reinit(space.dofs.locally_owned_dofs(), comm);
      const bool energy_ledger = material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_ENERGY");
      double local_energy = 0.;
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      FEValues<dim> native(this->get_mapping(), this->get_fe(), rule, update_gradients | update_values | update_quadrature_points);
      FEValues<dim> dg(this->get_mapping(), space.element, rule, update_values | update_JxW_values);
      std::vector<SymmetricTensor<2, dim>> strain(rule.size());
      std::vector<double> phase(rule.size());
      const unsigned n = space.element.n_dofs_per_cell();
      std::vector<types::global_dof_index> ids(n);
      for (const auto &entry : s.cells)
        {
          native.reinit(entry.second);
          dg.reinit(space.cells.at(entry.first));
          space.cells.at(entry.first)->get_dof_indices(ids);
          native[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk, strain);
          native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.displacement_history, phase);
          const auto T = evaluate_history_tensor(anchor, entry.first, rule);
          const auto R = anchor.residual ? evaluate_history_tensor(*anchor.residual, entry.first, rule)
                         : std::vector<SymmetricTensor<2, dim>>(rule.size());
          FullMatrix<double> mass(n, n);
          std::vector<Vector<double>> rhs(tensors * components);
          for (auto &v : rhs)
            v.reinit(n);
          for (unsigned q = 0; q < rule.size(); ++q)
            {
              const auto values = field(native.quadrature_point(q), phase[q], strain[q], T[q], R[q]);
              if (energy_ledger)
                local_energy += dg.JxW(q) * (scalar_product(values[0], values[0]) + trace(values[0]) * trace(values[0])) / (4 * s.G);
              for (unsigned i = 0; i < n; ++i)
                {
                  const double w = dg.JxW(q) * dg.shape_value(i, q);
                  for (unsigned j = 0; j < n; ++j)
                    mass(i, j) += w * dg.shape_value(j, q);
                  for (unsigned k = 0; k < tensors; ++k)
                    for (unsigned cpt = 0; cpt < components; ++cpt)
                      rhs[k * components + cpt][i] += w * values[k][SymmetricTensor<2, dim>::unrolled_to_component_indices(cpt)];
                }
            }
          mass.gauss_jordan();
          for (unsigned c = 0; c < rhs.size(); ++c)
            {
              Vector<double> coefficients(n);
              mass.vmult(coefficients, rhs[c]);
              for (unsigned i = 0; i < n; ++i)
                owned[c][ids[i]] = coefficients[i];
            }
        }
      for (unsigned c = 0; c < owned.size(); ++c)
        {
          owned[c].compress(VectorOperation::insert);
          auto &v = c < components ? update.dg_history->coefficients[c] : update.dg_history->residual->coefficients[c - components];
          v.reinit(space.dofs.locally_owned_dofs(), space.relevant, comm);
          v = owned[c];
        }
      if (energy_ledger)
        {
          double projected_energy = 0.;
          for (const auto &entry : s.cells)
            {
              dg.reinit(space.cells.at(entry.first));
              const auto values = evaluate_history_tensor(*update.dg_history, entry.first, rule);
              for (unsigned q = 0; q < rule.size(); ++q)
                projected_energy += dg.JxW(q) * (scalar_product(values[q], values[q]) + trace(values[q]) * trace(values[q])) / (4 * s.G);
            }
          local_energy = Utilities::MPI::sum(local_energy, comm);
          projected_energy = Utilities::MPI::sum(projected_energy, comm);
          this->get_pcout() << std::setprecision(17) << "   Direct source energy: step=" << this->get_timestep_number()
                            << " dt=" << dt << " publication=" << !material_reaction_solving
                            << " local=" << local_energy << " projected=" << projected_energy
                            << " loss=" << local_energy - projected_energy << std::endl;
        }
    }

    template<int dim> SymmetricTensor<2, dim> PhaseFieldFault<dim>::material_reaction_correction(
      const CellId &id, const Point<dim> &x) const
    {
      return evaluate_material_projection_correction(dg_material_reaction.get(), *fe_strain_state, id, x);
    }

    template<int dim> SymmetricTensor<2, dim> PhaseFieldFault<dim>::evaluate_material_projection_correction(
      const DGMaterialReaction *correction, const FEStrainState &s, const CellId &id, const Point<dim> &x) const
    {
      if (!correction)
        return {};
      const auto &r = *correction;
      const auto cell = s.evaluation_cells.at(id);
      const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, x);
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), Quadrature<dim>({unit}), update_gradients | update_values);
      fe.reinit(cell);
      std::vector<SymmetricTensor<2, dim>> strain(1);
      std::vector<double> phi(1);
      fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(r.bulk, strain);
      fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.displacement_history, phi);
      const double dt = this->get_timestep_number() > 0 ? this->get_timestep() : initial_time_step;
      const auto c = compute_maxwell_coefficients(material_viscosity(x, s.eta), s.G, dt);
      const auto physical = physical_history_stress(s, *r.candidate->dg_history, r.candidate->Z, s.I, id, x);
      const auto anchor = physical_history_stress(s, *s.objective_anchor, s.Z, s.I, id, x);
      const auto direct = MaterialModel::Rheology::mature_stress_endpoint(
                            c.beta, c.eta_ve, anchor, strain[0], evaluate_mature_dg_memory(x, phi[0], r.rates, s.I));
      return physical - direct;
    }

    template<int dim> double PhaseFieldFault<dim>::refresh_material_reaction(
      const LinearAlgebra::BlockVector &bulk, const std::vector<double> &rates)
    {
      auto next = std::make_shared<DGMaterialReaction>();
      next->bulk = bulk;
      next->rates = rates;
      next->candidate = make_material_reaction(*fe_strain_state, bulk, rates);
      auto previous = dg_material_reaction;
      double difference = 0., scale = 1.;
      const QGauss<dim> rule(std::max(8u, fe_bulk_normal_quadrature_points));
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), rule, update_quadrature_points);
      for (const auto &entry : fe_strain_state->cells)
        {
          fe.reinit(entry.second);
          for (const auto &x : fe.get_quadrature_points())
            {
              const auto before = evaluate_material_projection_correction(previous.get(), *fe_strain_state, entry.first, x);
              const auto after = evaluate_material_projection_correction(next.get(), *fe_strain_state, entry.first, x);
              difference = std::max(difference, (after - before).norm());
              scale = std::max({scale, before.norm(), after.norm(), evaluate_dg_history_stress(*next->candidate->dg_history, next->candidate->Z, fe_strain_state->I, entry.first, x).norm()});
            }
        }
      dg_material_reaction = next;
      fe_strain_state->stress_samples.clear();
      fe_strain_state->mechanical_samples.reset();
      difference = Utilities::MPI::max(difference, this->get_mpi_communicator());
      scale = Utilities::MPI::max(scale, this->get_mpi_communicator());
      return difference / scale;
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::project_mature_reaction(const FEStrainState &,const LinearAlgebra::BlockVector &,FEStrainUpdate &,const MatureReactionField &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_material_projection_correction(const DGMaterialReaction *,const FEStrainState &,const CellId &,const Point<dim> &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::make_material_reaction(const FEStrainState &,const LinearAlgebra::BlockVector &,const std::vector<double> &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::material_reaction_correction(const CellId &,const Point<dim> &) const; \
  template double PhaseFieldFault<dim>::refresh_material_reaction(const LinearAlgebra::BlockVector &,const std::vector<double> &);
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
