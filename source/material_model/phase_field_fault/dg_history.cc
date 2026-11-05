/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT, distributed under the GNU General Public License
  version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/utilities.h>
#include <aspect/reconstructed_fault/linear_performance.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <mutex>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      long double q2_roundoff_product(std::initializer_list<long double> factors)
      {
        long double result = 0.;
        for (const auto error : factors)
          result += error + result * error;
        return result;
      }
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &state) const
    {
      AssertThrow(dg_tensor_history && dim == 2 && this->get_parameters().stokes_velocity_degree == 2
                  && state.Z == std::vector<double>(state.Z.size(), 0.),
                  ExcMessage("Independent DG history initializes only an unloaded seed."));
      auto history = std::make_shared<DGHistory>();
      history->representation = material_direct_stress ? DGHistory::Representation::physical_stress
                                : DGHistory::Representation::separated_trial;
      history->primary.space = std::make_shared<typename DGHistory::Space>(this->get_triangulation());
      for (auto &component : history->primary.coefficients)
        {
          component.reinit(history->primary.space->dofs.locally_owned_dofs(), history->primary.space->relevant,
                           this->get_mpi_communicator());
          component = 0.;
        }
      if (jaumann_dg_stress && !material_direct_stress)
        {
          history->residual = std::make_shared<TensorField>();
          history->residual->space = history->primary.space;
          for (auto &component : history->residual->coefficients)
            {
              component.reinit(history->primary.space->dofs.locally_owned_dofs(), history->primary.space->relevant, this->get_mpi_communicator());
              component = 0.;
            }
        }
      state.bulk_history.tensor = std::move(history);
      this->get_pcout() << "   Accepted DGQ2 trial stress: independent native source; "
                        << (consistent_endpoint_phase ? "cohesive endpoint phase trials; fixed geometry; " : "fixed phase/geometry; ")
                        << (dg_accepted_history ? "accepted-history lifecycle/restart." : "no transport/restart.") << std::endl;
    }


    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::make_dg_history_trial(
      const LinearAlgebra::BlockVector &bulk, const MaxwellCoefficients &coefficients) const
    {
      AssertThrow(fe_strain_state && fe_strain_state->bulk_history.tensor,
                  ExcMessage("Missing authoritative DG trial-stress history."));
      return make_dg_history_trial(*fe_strain_state->bulk_history.tensor, bulk, coefficients);
    }


    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::make_dg_history_trial(
      const DGHistory &incoming, const LinearAlgebra::BlockVector &bulk,
      const MaxwellCoefficients &coefficients, const bool estimate_source_roundoff, const bool reference) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(), "DG native source/update");
      const auto &state = *fe_strain_state;
      AssertThrow(dg_tensor_history && incoming.primary.space == state.bulk_history.tensor->primary.space,
                  ExcMessage("Native DG source requires the accepted fixed space."));
      auto trial = std::make_shared<DGHistory>();
      trial->primary.space = incoming.primary.space;
      const auto &space = *trial->primary.space;
      std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components> owned;
      for (auto &component : owned)
        component.reinit(space.dofs.locally_owned_dofs(), this->get_mpi_communicator());

      // On the restricted affine Cartesian Q2 mesh, every component of the
      // native deviatoric symmetric gradient is in DGQ2. Three Gauss points
      // integrate both its scalar mass and source products exactly. This
      // projection stores the trial tensor, never the narrow chi*Z correction.
      const QGauss<dim> quadrature(3);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      FEValues<dim> source(this->get_mapping(), this->get_fe(), quadrature, update_gradients);
      FEValues<dim> dg(this->get_mapping(), space.element, quadrature, update_values | update_JxW_values);
      const unsigned int n = space.element.n_dofs_per_cell();
      std::vector<types::global_dof_index> indices(n);
      std::vector<types::global_dof_index> native_indices(this->get_fe().n_dofs_per_cell());
      std::vector<SymmetricTensor<2, dim>> gradient(quadrature.size());
      for (const auto &entry : state.cells)
        {
          source.reinit(entry.second);
          const auto dg_cell = space.cells.at(entry.first);
          dg.reinit(dg_cell);
          dg_cell->get_dof_indices(indices);
          entry.second->get_dof_indices(native_indices);
          source[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk, gradient);
          typename DGHistory::Space::SourceGeometry *geometry = nullptr;
          if (uses_stationary_mature_dg_optimizations() && !reference)
            {
              std::array<long double, 9+dim> key;
              for (unsigned int q = 0; q < 9; ++q)
                key[q] = dg.JxW(q);
              for (unsigned int d = 0; d < dim; ++d)
                key[9 + d] = static_cast<long double>(entry.second->vertex(1u << d)[d]) - entry.second->vertex(0)[d];
              geometry = &space.source_geometry[key];
            }
          const bool cached_mass = geometry && geometry->inverse.m() == n;
          FullMatrix<double> mass(n, n);
          if (cached_mass)
            mass = geometry->inverse;
          std::array<Vector<double>, SymmetricTensor<2, dim>::n_independent_components> rhs;
          for (auto &component : rhs)
            component.reinit(n);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const auto strain = Utilities::Tensors::consistent_deviator(gradient[q]);
              for (unsigned int i = 0; i < n; ++i)
                {
                  const double weight = dg.JxW(q) * dg.shape_value(i, q);
                  if (!cached_mass)
                    for (unsigned int j = 0; j < n; ++j)
                      mass(i, j) += weight * dg.shape_value(j, q);
                  for (unsigned int c = 0; c < rhs.size(); ++c)
                    rhs[c][i] += weight * strain[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)];
                }
            }
          if (!cached_mass)
            {
              aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::mass_inverse);
              mass.gauss_jordan();
              if (geometry)
                geometry->inverse = mass;
            }
          const auto gamma = [](unsigned int count)
          {
            const long double u = std::numeric_limits<double>::epsilon() / 2.;
            return count * u / (1 - count * u);
          };
          SymmetricTensor<2, dim> strain_scale;
          std::vector<long double> projection_factor;
          if (estimate_source_roundoff)
            {
              // Absolute operands of Q2 differentiation, rather than the small
              // resulting gradient. |L_i|<=1, |L_i'|<=4 on [0,1]. In particular,
              // a nearly constant velocity in a thin cell is ill conditioned.
              auto &roundoff = trial->source_roundoff[entry.first];
              std::array<long double, dim> velocity_sum {};
              for (unsigned int i = 0; i < native_indices.size(); ++i)
                {
                  const auto c = this->get_fe().system_to_component_index(i).first;
                  for (unsigned int a = 0; a < dim; ++a)
                    if (c == this->introspection().component_indices.velocities[a])
                      velocity_sum[a] += std::abs(bulk[native_indices[i]]);
                }
              long double volume = 1.;
              for (unsigned int b = 0; b < dim; ++b)
                {
                  const long double width = static_cast<long double>(entry.second->vertex(1u << b)[b]) - entry.second->vertex(0)[b];
                  volume *= width;
                  for (unsigned int a = 0; a < dim; ++a)
                    roundoff.gradient_scale[a][b] = std::nextafter(static_cast<double>(4 * velocity_sum[a] / width
                                                                                       * (1 + 2 * n * std::numeric_limits<long double>::epsilon())), std::numeric_limits<double>::infinity());
                }
              for (unsigned int a = 0; a < dim; ++a)
                for (unsigned int b = 0; b < dim; ++b)
                  if (a != b)
                    strain_scale[a][b] = .5 * (roundoff.gradient_scale[a][b] + roundoff.gradient_scale[b][a]);
                  else
                    for (unsigned int d = 0; d < dim; ++d)
                      strain_scale[a][a] += (a == d ? 2. / 3. : 1. / 3.) * roundoff.gradient_scale[d][d];

              if (geometry && !geometry->projection_factor.empty())
                projection_factor = geometry->projection_factor;
              else
                {
                  aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::roundoff_geometry);
                  // A posteriori inverse defect against the analytic tensor-product
                  // Q2 mass (not against its rounded assembly). This accounts for
                  // assembly/inversion without assuming a backward-stable inverse.
                  const auto gamma_long = [](unsigned int count)
                  {
                    const long double u = std::numeric_limits<long double>::epsilon() / 2.;
                    return count * u / (1 - count * u);
                  };
                  const int mass_1d[3][3] = {{4, 2, -1}, {2, 16, 2}, { -1, 2, 4}};
                  FullMatrix<long double> exact_mass(n, n), defect(n, n);
                  const auto &support = space.element.get_unit_support_points();
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        long double value = volume;
                        for (unsigned int d = 0; d < dim; ++d)
                          value *= mass_1d[std::lround(2 * support[i][d])][std::lround(2 * support[j][d])] / 30.L;
                        exact_mass(i, j) = value;
                      }
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        long double product = 0., absolute_product = 0.;
                        for (unsigned int k = 0; k < n; ++k)
                          {
                            product += mass(i, k) * exact_mass(k, j);
                            absolute_product += std::abs(mass(i, k) * exact_mass(k, j));
                          }
                        defect(i, j) = std::abs(product - (i == j ? 1.L : 0.L))
                                       + gamma_long(2 * n + 2 * dim + 4) * absolute_product;
                      }
                  std::vector<long double> rhs_weight(n, 0.);
                  projection_factor.assign(n, 0.);
                  // Monomial absolute coefficient sums are <=8 for L and <=12
                  // for L'; |L|<=1, |L'|<=4, |L''|<=8. Six operations bound the
                  // quadratic Horner/product evaluations (also the simultaneous
                  // derivative recurrence). Include rounded Gauss abscissae here.
                  const long double value_error = 8 * gamma(6) + 4 * gamma(4);
                  const long double derivative_relative_error = 3 * gamma(6) + 2 * gamma(4);
                  const long double gradient_relative_error = q2_roundoff_product({gamma(2 * n + 10), value_error, derivative_relative_error});
                  const long double shape_error = q2_roundoff_product({value_error, value_error});
                  const long double accumulation = gamma(2 * quadrature.size() + 10);
                  const long double rhs_relative_error = q2_roundoff_product({gradient_relative_error, accumulation});
                  long double weight_total = 0.;
                  for (unsigned int q = 0; q < quadrature.size(); ++q)
                    weight_total += std::abs(dg.JxW(q));
                  for (unsigned int j = 0; j < n; ++j)
                    for (unsigned int q = 0; q < quadrature.size(); ++q)
                      rhs_weight[j] += std::abs(dg.JxW(q) * dg.shape_value(j, q));
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      projection_factor[i] += defect(i, j) + std::abs(mass(i, j))
                                              * (rhs_relative_error * rhs_weight[j] + weight_total * shape_error * (1 + rhs_relative_error));
                  if (geometry)
                    geometry->projection_factor = projection_factor;
                }
            }
          Vector<double> projected(n);
          for (unsigned int c = 0; c < rhs.size(); ++c)
            {
              mass.vmult(projected, rhs[c]);
              const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
              for (unsigned int i = 0; i < n; ++i)
                {
                  if (estimate_source_roundoff)
                    {
                      long double multiply_scale = 0.;
                      for (unsigned int j = 0; j < n; ++j)
                        multiply_scale += std::abs(mass(i, j) * rhs[c][j]);
                      // Inverse defect, RHS operand errors, and inverse/RHS dot.
                      const long double projection_error = projection_factor[i] * strain_scale[ij] + gamma(2 * n) * multiply_scale;
                      trial->source_roundoff.at(entry.first).projection[ij] = std::max(trial->source_roundoff.at(entry.first).projection[ij], std::nextafter(static_cast<double>(projection_error
                                                                                       * (1 + 4 * n * std::numeric_limits<long double>::epsilon())), std::numeric_limits<double>::infinity()));
                    }
                  const double value = coefficients.beta * incoming.primary.coefficients[c][indices[i]]
                                       + 2 * coefficients.eta_ve * projected[i];
                  AssertThrow(std::isfinite(value), ExcMessage("Nonfinite DG trial-stress candidate."));
                  owned[c][indices[i]] = value;
                }
            }
        }
      for (unsigned int c = 0; c < owned.size(); ++c)
        {
          owned[c].compress(VectorOperation::insert);
          trial->primary.coefficients[c].reinit(space.dofs.locally_owned_dofs(), space.relevant,
                                                this->get_mpi_communicator());
          trial->primary.coefficients[c] = owned[c];
        }
      if (incoming.residual)
        {
          trial->residual = std::make_shared<TensorField>();
          trial->residual->space = incoming.primary.space;
          for (unsigned int c = 0; c < owned.size(); ++c)
            {
              LinearAlgebra::Vector scaled;
              scaled.reinit(incoming.primary.space->dofs.locally_owned_dofs(), this->get_mpi_communicator());
              for (const auto i : scaled.locally_owned_elements())
                scaled[i] = coefficients.beta * incoming.residual->coefficients[c][i];
              scaled.compress(VectorOperation::insert);
              auto &r = trial->residual->coefficients[c];
              r.reinit(incoming.residual->coefficients[c]);
              r = scaled;
            }
        }
      return trial;
    }


    template <int dim>
    double
    PhaseFieldFault<dim>::dg_source_publication_allowance(
      const DGHistory &trial, const CellId &id, const Point<dim> &point) const
    {
      const auto &source = trial.source_roundoff.at(id);
      const auto cell = trial.primary.space->cells.at(id);
      const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, point);
      const auto &element = trial.primary.space->element;
      long double lebesgue = 0.;
      for (unsigned int i = 0; i < element.n_dofs_per_cell(); ++i)
        lebesgue += std::abs(element.shape_value(i, unit));
      const long double u = std::numeric_limits<double>::epsilon() / 2.;
      const auto gamma = [&](unsigned int count)
      {
        return count * u / (1 - count * u);
      };
      const long double gradient_relative_error = q2_roundoff_product({gamma(2 * element.n_dofs_per_cell() + 10), 8 * gamma(6), 3 * gamma(6)});
      Tensor<2, dim> native_error;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          {
            long double coordinate_error = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const long double width = static_cast<long double>(cell->vertex(1u << d)[d]) - cell->vertex(0)[d];
                // Subtraction/division in both real-to-unit maps;
                // |L_i''|<=8, |L_i' L_j'|<=16.
                coordinate_error += (d == b ? 2 : 4) * gamma(4)
                                    * (std::abs(point[d]) + std::abs(cell->vertex(0)[d])) / width;
              }
            native_error[a][b] = (gradient_relative_error + 2 * coordinate_error) * source.gradient_scale[a][b];
          }
      SymmetricTensor<2, dim> strain_error;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          if (a != b)
            strain_error[a][b] = .5 * (native_error[a][b] + native_error[b][a]);
          else
            for (unsigned int d = 0; d < dim; ++d)
              strain_error[a][a] += (a == d ? 2. / 3. : 1. / 3.) * native_error[d][d];
      const auto source_error = 2 * compute_maxwell_coefficients(fe_strain_state->eta, fe_strain_state->G, this->get_timestep()).eta_ve * (strain_error + static_cast<double>(lebesgue) * source.projection);
      return std::sqrt(std::pow(source_error.norm(), 2) + std::pow(trace(source_error), 2));
    }



#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const LinearAlgebra::BlockVector &,const MaxwellCoefficients &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const DGHistory &,const LinearAlgebra::BlockVector &,const MaxwellCoefficients &,bool,bool) const; \
  template double PhaseFieldFault<dim>::dg_source_publication_allowance(const DGHistory &,const CellId &,const Point<dim> &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
