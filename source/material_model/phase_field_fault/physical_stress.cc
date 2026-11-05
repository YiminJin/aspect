/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT, distributed under the GNU General Public License
  version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_history.h>
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
    template<int dim> struct PhaseFieldFault<dim>::FEStrainState::MechanicalSamples
    {
      explicit MechanicalSamples(const Quadrature<dim> &q): points(q.get_points()), weights(q.get_weights()) {}
      std::vector<Point<dim>> points;
      std::vector<double> weights;
      std::map<CellId, std::vector<Point<dim>>> units;
      std::map<CellId, std::vector<double>> phase_values;
      std::map<CellId, std::vector<SymmetricTensor<2, dim>>> stress;
      std::weak_ptr<const DGHistory> history;
      unsigned int geometry = numbers::invalid_unsigned_int;
    };

    template<int dim>
    const std::vector<SymmetricTensor<2, dim>> &PhaseFieldFault<dim>::retained_dg_qp_stress(
      const CellId &id, const Quadrature<dim> &quadrature, const std::vector<Point<dim>> &positions) const
    {
      AssertThrow(uses_stationary_mature_dg_optimizations(), ExcMessage("Stationary DG QP cache used outside its scope."));
      // WorkStream may call from multiple workers. Publication is outside the
      // assembly, so returned cell arrays remain immutable during each assembly.
      static std::mutex mutex;
      std::lock_guard<std::mutex> guard(mutex);
      const auto &s = *fe_strain_state;
      auto &manager = this->get_reconstructed_fault_manager();
      const unsigned int generation = manager.get_stokes_qp_cache_diagnostics().rebuild_count;
      auto &pointer = s.mechanical_samples;
      if (!pointer || pointer->geometry != generation)
        {
          pointer = std::make_shared<typename FEStrainState::MechanicalSamples>(quadrature);
          pointer->geometry = generation;
        }
      auto &cache = *pointer;
      AssertThrow(cache.points == quadrature.get_points() && cache.weights == quadrature.get_weights(),
                  ExcMessage("Stationary DG mechanical cache requires its original quadrature."));
      if (cache.history.lock() != s.dg_history)
        {
          cache.stress.clear();
          cache.history = s.dg_history;
        }
      const auto found = cache.stress.find(id);
      if (found != cache.stress.end())
        return found->second;
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::history_cache_misses);
      const auto cell = s.evaluation_cells.at(id);
      // Associations are already prepared by the Stokes owner. No segment
      // search or coordinate-keyed stress map is needed at these known QPs.
      const auto &association = manager.get_stokes_qp_fault_associations(id, quadrature, positions);
      auto phase = cache.phase_values.find(id);
      if (phase == cache.phase_values.end())
        {
          auto &units = cache.units[id];
          for (const auto &x : positions)
            units.push_back(this->get_mapping().transform_real_to_unit_cell(cell, x));
          FEValues<dim> phase_fe(this->get_mapping(), this->get_fe(), Quadrature<dim>(units), update_values);
          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
          phase_fe.reinit(cell);
          std::vector<double> values(quadrature.size());
          phase_fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
          .get_function_values(s.displacement_history, values);
          phase = cache.phase_values.emplace(id, std::move(values)).first;
        }
      FEValues<dim> dg(this->get_mapping(), s.dg_history->space->element, Quadrature<dim>(cache.units.at(id)), update_values);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      const auto dg_cell = s.dg_history->space->cells.at(id);
      dg.reinit(dg_cell);
      std::vector<types::global_dof_index> indices(s.dg_history->space->element.n_dofs_per_cell());
      dg_cell->get_dof_indices(indices);
      std::vector<SymmetricTensor<2, dim>> stress(quadrature.size());
      for (unsigned int c = 0; c < s.dg_history->coefficients.size(); ++c)
        {
          const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
          for (unsigned int q = 0; q < stress.size(); ++q)
            for (unsigned int i = 0; i < indices.size(); ++i)
              stress[q][ij] += s.dg_history->coefficients[c][indices[i]] * dg.shape_value(i, q);
        }
      for (unsigned int q = 0; q < stress.size(); ++q)
        if (association[q].active)
          {
            const auto &a = association[q];
            const auto j = a.segment_index;
            const double xi = a.xi;
            const double z = (1 - xi) * s.Z.at(j) + xi * s.Z.at(j + 1), I = (1 - xi) * s.I.at(j) + xi * s.I.at(j + 1);
            const auto localization = evaluate_reconstructed_fault_localization(a.fault_index, j, xi, phase->second[q], phase->second[q], "mature DG QP stress");
            const auto S = manager.get_shear_sense(a.fault_index) * symmetrize(outer_product(a.tangent, a.normal));
            stress[q] -= (2 * z * localization.current_h / I) * S;
          }
      return cache.stress.emplace(id, std::move(stress)).first->second;
    }

    template <int dim>
    std::vector<SymmetricTensor<2, dim>> PhaseFieldFault<dim>::evaluate_history_tensor(
      const DGHistory &history, const CellId &id, const Quadrature<dim> &quadrature) const
    {
      FEValues<dim> fe(this->get_mapping(), history.space->element, quadrature, update_values);
      fe.reinit(history.space->cells.at(id));
      std::vector<SymmetricTensor<2, dim>> tensor(quadrature.size());
      std::vector<double> component(quadrature.size());
      for (unsigned int c = 0; c < history.coefficients.size(); ++c)
        {
          fe.get_function_values(history.coefficients[c], component);
          const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            tensor[q][ij] = component[q];
        }
      return tensor;
    }


    template<int dim> void PhaseFieldFault<dim>::prepare_dg_point_samples(
      const CellId &id, const std::vector<Point<dim>> &points) const
    {
      if (!uses_stationary_mature_dg_optimizations())
        return;
      const auto &s = *fe_strain_state;
      auto &space = *s.dg_history->space;
      const auto &manager = this->get_reconstructed_fault_manager();
      const auto generation = manager.get_stokes_qp_cache_diagnostics().rebuild_count;
      if (space.point_geometry_generation != generation)
        {
          space.point_samples.clear();
          space.point_geometry_generation = generation;
        }
      auto &data = space.point_samples[id];
      // Keep the few fixed, cell-owned layouts (publication particles and dense
      // diagnostic QPs) together. Adding a diagnostic layout must not evict the
      // publication layout or attach accepted stress values to geometry.
      std::vector<Point<dim>> missing;
      for (const auto &x : points)
        if (std::none_of(data.begin(), data.end(), [&](const auto & p)
        {
          return p.point == x;
        }))
      missing.push_back(x);
      if (missing.empty())
        return;
      const auto cell = s.evaluation_cells.at(id);
      std::vector<Point<dim>> units;
      for (const auto &x : missing)
        units.push_back(this->get_mapping().transform_real_to_unit_cell(cell, x));
      const Quadrature<dim> quadrature(units);
      FEValues<dim> dg(this->get_mapping(), space.element, quadrature, update_values);
      FEValues<dim> native(this->get_mapping(), this->get_fe(), quadrature, update_values);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
      dg.reinit(space.cells.at(id));
      native.reinit(cell);
      std::vector<double> phi(missing.size());
      native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
      .get_function_values(s.displacement_history, phi);
      const auto old_size = data.size();
      data.resize(old_size + missing.size());
      for (unsigned int q = 0; q < missing.size(); ++q)
        {
          auto &sample = data[old_size + q];
          sample.point = missing[q];
          sample.phase = phi[q];
          for (unsigned int i = 0; i < 9; ++i)
            sample.shape[i] = dg.shape_value(i, q);
          sample.source = manager.project_to_bulk_source(missing[q]);
        }
    }

    template <int dim>
    SymmetricTensor<2, dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(
      const DGHistory &history, const std::vector<double> &Z, const std::vector<double> &I,
      const CellId &id, const Point<dim> &point, double *evaluation_scale,
      const LinearAlgebra::BlockVector *phase_trial) const
    {
      return physical_history_stress(*fe_strain_state, history, Z, I, id, point, evaluation_scale, phase_trial);
    }

    template <int dim>
    SymmetricTensor<2, dim> PhaseFieldFault<dim>::physical_history_stress(
      const FEStrainState &state, const DGHistory &history,
      const std::vector<double> &Z, const std::vector<double> &I,
      const CellId &id, const Point<dim> &point, double *evaluation_scale,
      const LinearAlgebra::BlockVector *phase_trial) const
    {
      if (history.representation == DGHistory::Representation::physical_stress)
        {
          AssertThrow(material_direct_stress && !history.residual,
                      ExcMessage("Physical DG stress cannot own a separated residual."));
          const auto cell = history.space->cells.at(id);
          const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, point);
          const auto value = evaluate_history_tensor(history, id, Quadrature<dim>({unit}))[0];
          if (evaluation_scale)
            {
              *evaluation_scale = 0.;
              std::vector<types::global_dof_index> ids(history.space->element.n_dofs_per_cell());
              cell->get_dof_indices(ids);
              for (const auto &component : history.coefficients)
                for (unsigned i = 0; i < ids.size(); ++i)
                  *evaluation_scale += std::abs(component[ids[i]] * history.space->element.shape_value(i, unit));
            }
          return value;
        }
      const auto cell = history.space->cells.at(id);
      const typename DGHistory::Space::PointSample *sample = nullptr;
      if (uses_stationary_mature_dg_optimizations() && phase_trial == nullptr)
        {
          const auto found = history.space->point_samples.find(id);
          if (found != history.space->point_samples.end())
            for (const auto &p : found->second)
              if (p.point == point)
                {
                  sample = &p;
                  break;
                }
        }
      SymmetricTensor<2, dim> trial;
      double scale = 0.;
      std::vector<double> phi(1);
      std::vector<types::global_dof_index> indices(history.space->element.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      if (sample)
        {
          for (unsigned int c = 0; c < history.coefficients.size(); ++c)
            {
              const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  const double value = history.coefficients[c][indices[i]] * sample->shape[i];
                  trial[ij] += value;
                  scale += std::abs(value);
                }
            }
          phi[0] = sample->phase;
        }
      else
        {
          const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, point);
          FEValues<dim> dg(this->get_mapping(), history.space->element, Quadrature<dim>({unit}), update_values);
          dg.reinit(cell);
          for (unsigned int c = 0; c < history.coefficients.size(); ++c)
            {
              const auto ij = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  const double value = history.coefficients[c][indices[i]] * dg.shape_value(i, 0);
                  trial[ij] += value;
                  scale += std::abs(value);
                }
            }

          // Only the immutable phase block of this shared vector is sampled.
          // Neither its unused velocity block nor any mapped/particle stress is read.
          const auto native_cell = state.evaluation_cells.at(id);
          FEValues<dim> phase(this->get_mapping(), this->get_fe(), Quadrature<dim>({unit}), update_values);
          phase.reinit(native_cell);
          phase[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
          .get_function_values(phase_trial ? *phase_trial : state.displacement_history, phi);

          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
        }
      if (mature_frictional_fault)
        {
          // General oblique map, including the retained constant endpoint
          // continuation. Z is auxiliary Maxwell/slip memory, not I*C.
          const auto correction = evaluate_mature_dg_memory(point, phi[0], Z, I, sample ? &sample->source : nullptr);
          if (history.residual)
            {
              const auto unit = this->get_mapping().transform_real_to_unit_cell(cell, point);
              const auto r = evaluate_history_tensor(*history.residual, id, Quadrature<dim>({unit}))[0];
              trial -= r;
              // Include the independent residual operands in publication bounds.
              for (const auto &component : history.residual->coefficients)
                for (unsigned int i = 0; i < indices.size(); ++i)
                  scale += std::abs(component[indices[i]] * history.space->element.shape_value(i, unit));
            }
          if (evaluation_scale)
            *evaluation_scale = scale + correction.norm();
          return trial - correction;
        }
      if (uses_prescribed_dg_geometry())
        {
          const auto source = this->get_reconstructed_fault_manager().project_to_normal_profiles(point);
          if (!source.active)
            {
              if (evaluation_scale)
                *evaluation_scale = scale;
              return trial;
            }
          const auto j = source.segment_index;
          const double t = source.xi;
          const double z = (1 - t) * Z[j] + t * Z[j + 1], normalization = (1 - t) * I[j] + t * I[j + 1];
          AssertThrow(normalization > 0. && std::isfinite(normalization) && phi[0] >= 0. && phi[0] < 1.,
                      ExcMessage("Undefined prescribed finite DG localization."));
          const double correction = z * state.m * phi[0] * (1 + phi[0]) / std::pow(1 - phi[0], 2) / normalization;
          trial[0][1] -= state.shear_sense * correction;
          if (evaluation_scale)
            *evaluation_scale = scale + std::abs(correction);
          return trial;
        }
      const unsigned int segment = std::min<unsigned int>(Z.size() - 2, point[0] * (Z.size() - 1));
      const double xi = point[0] * (Z.size() - 1) - segment;
      const double z = (1 - xi) * Z[segment] + xi * Z[segment + 1];
      const double normalization = (1 - xi) * I[segment] + xi * I[segment + 1];
      AssertThrow(normalization > 0. && std::isfinite(normalization) && phi[0] >= 0. && phi[0] < 1.,
                  ExcMessage("Undefined fixed DG localization."));
      const double h = state.m * phi[0] * (1 + phi[0]) / std::pow(1 - phi[0], 2);
      SymmetricTensor<2, dim> S;
      S[0][1] = .5 * state.shear_sense;
      const double correction = 2 * z * h / normalization;
      if (evaluation_scale)
        *evaluation_scale = scale + std::abs(correction) * S.norm();
      return trial - correction * S;
    }



#define INSTANTIATE(dim) \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::physical_history_stress(const FEStrainState &,const DGHistory &,const std::vector<double> &,const std::vector<double> &,const CellId &,const Point<dim> &,double *,const LinearAlgebra::BlockVector *) const; \
  template const std::vector<SymmetricTensor<2,dim>> &PhaseFieldFault<dim>::retained_dg_qp_stress(const CellId &,const Quadrature<dim> &,const std::vector<Point<dim>> &) const; \
  template std::vector<SymmetricTensor<2,dim>> PhaseFieldFault<dim>::evaluate_history_tensor(const DGHistory &,const CellId &,const Quadrature<dim> &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(const DGHistory &,const std::vector<double> &,const std::vector<double> &,const CellId &,const Point<dim> &,double *,const LinearAlgebra::BlockVector *) const; \
  template void PhaseFieldFault<dim>::prepare_dg_point_samples(const CellId &,const std::vector<Point<dim>> &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
