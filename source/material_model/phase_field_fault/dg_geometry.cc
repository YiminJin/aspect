/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <aspect/simulator.h>
#include <aspect/phase_field.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/fe/fe_values.h>
#include <sstream>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      template <int dim>
      std::string surface_stamp(const ReconstructedFaultManager<dim> &manager)
      {
        std::ostringstream out;
        {
          aspect::oarchive archive(out);
          archive << manager;
        }
        for (unsigned int f = 0; f < manager.get_faults().size(); ++f)
          for (const double v : manager.get_slip_rate(f))
            out << std::hexfloat << v << ',';
        return out.str();
      }
    }
    template <int dim>
    void
    PhaseFieldFault<dim>::enable_prescribed_dg_geometry(
      const std::function<double(const Point<dim> &)> &phase)
    {
      AssertThrow(dim == 2 && phase && !fe_strain_state && homogeneous_mesh_data
                  && dg_accepted_history && dg_tensor_history && consistent_endpoint_phase
                  && !mature_frictional_fault && !fault_friction.has_state_variable()
                  && this->get_parameters().nonlinear_solver == Parameters<dim>::NonlinearSolver::no_Advection_no_Stokes
                  && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 1
                  && this->get_geometry_model().get_periodic_boundary_pairs().empty(),
                  ExcMessage("Prescribed cohesive DG geometry requires serial homogeneous frozen state, no solvers, no RSF/periodic geometry."));
      prescribed_geometry_phase = phase;
      this->get_reconstructed_fault_manager().register_property("prescribed cohesive DG geometry v1", 1);
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::initialize_prescribed_dg_geometry()
    {
      AssertThrow(uses_prescribed_dg_geometry() && !fe_strain_state
                  && !this->get_parameters().resume_computation && this->get_timestep_number() == 0,
                  ExcMessage("Prescribed geometry initialization is fresh-state only."));
      prepare_reconstructed_fault_phase_state();
      compute_fault_surface_temperatures();
    }

    template <int dim>
    std::shared_ptr<DGGeometryUpdate<dim>>
    PhaseFieldFault<dim>::prepare_dg_geometry_append(
      const std::vector<Point<dim>> &vertices, const std::vector<double> &widths,
      const std::vector<std::vector<double>> &properties, const std::vector<double> &rates)
    {
      AssertThrow(uses_prescribed_dg_geometry(), ExcMessage("Prescribed geometry setup is required."));
      validate_fe_strain_restrictions();
      auto update = std::make_shared<DGGeometryUpdate<dim>>();
      update->incoming = fe_strain_state;
      update->next = fe_strain_state;
      update->normalization = current_normalization_integrals;
      update->surface_temperatures = current_fault_surface_temperatures;
      update->identity = vertices.empty();
      auto &manager = this->get_reconstructed_fault_manager();
      update->parent_surface = surface_stamp(manager);
      update->manager = manager.clone_prescribed_geometry();
      if (update->identity)
        return update;
      const auto old = fe_strain_state;
      const auto ns = old->vertices.size();
      AssertThrow(old->Z.back() == 0. && manager.get_timestep_committed_slip_rate(0).back() == 0.,
                  ExcMessage("Prescribed zero birth requires zero old-tip Z/rate; general joins remain unqualified."));
      const double ds = old->vertices[1][0] - old->vertices[0][0];
      for (unsigned int j = 0; j < old->vertices.size(); ++j)
        AssertThrow(old->vertices[j][1] == .5 && std::abs(old->vertices[j][0] - j * ds) < 1e-13,
                    ExcMessage("Prescribed append requires the retained straight horizontal uniform seed."));
      for (unsigned int j = 0; j < vertices.size(); ++j)
        AssertThrow(vertices[j][1] == .5 && vertices[j][0] < 1.
                    && std::abs(vertices[j][0] - (ns + j)*ds) < 1e-13 && rates.at(j) == 0.,
                    ExcMessage("Prescribed append requires collinear uniform interior nodes and zero birth rate."));
      const auto cp = manager.get_property_information()[fault_property_indices.cohesive_traction].position;
      const auto ip = manager.get_property_information()[fault_property_indices.previous_normalization_integral].position;
      for (const auto &row : properties)
        AssertThrow(row.at(cp) == 0., ExcMessage("Prescribed new cohesive records must be unloaded."));
      update->manager->append_reconstructed_fault(0, vertices, widths, properties, rates);
      // Temporarily bind the fully private candidate to existing native geometry
      // builders. No callback/solver/publication runs here. Both exception and
      // success restore the live accepted owners and invalidate candidate caches.
      auto incoming_normalization = current_normalization_integrals;
      auto incoming_temperatures = current_fault_surface_temperatures;
      manager.swap_prescribed_geometry(*update->manager);
      bool bound = true;
      const auto restore = [&]() noexcept
      {
        if (!bound)
          return;
        manager.swap_prescribed_geometry(*update->manager);
        fe_strain_state = old;
        invalidate_normalization_cache();
        current_normalization_integrals.swap(incoming_normalization);
        current_fault_surface_temperatures.swap(incoming_temperatures);
        bound = false;
      };
      try
        {
          fe_strain_state.reset();
          invalidate_normalization_cache();
          compute_normalization_integrals();
          prepare_fe_strain_history(false, true);
          compute_fault_surface_temperatures();
          auto &next = *fe_strain_state;
          next.Z = old->Z;
          next.Z.resize(ns + vertices.size(), 0.);
          next.initial_fracture = old->initial_fracture;
          next.committed_step = old->committed_step;
          next.accepted_publications = old->accepted_publications;
          next.geometry_publications = old->geometry_publications + 1;
          for (unsigned int j = 0; j < next.Z.size(); ++j)
            {
              manager.get_fault(0).get_properties(j)[cp] = next.Z[j] / next.I[j];
              manager.get_fault(0).get_properties(j)[ip] = next.I[j];
            }
          const auto &space = *next.bulk_history.tensor->primary.space;
          const auto quad = mechanical_quadrature();
          FEValues<dim> dg(this->get_mapping(), space.element, quad, update_values | update_quadrature_points | update_JxW_values);
          FEValues<dim> fe(this->get_mapping(), this->get_fe(), quad, update_values);
          std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components> owned;
          for (auto &field : owned)
            field.reinit(space.dofs.locally_owned_dofs(), this->get_mpi_communicator());
          std::vector<types::global_dof_index> ids(9);
          std::vector<double> phi(quad.size());
          const auto correction = [&](const ReconstructedFaultManager<dim> &geometry,
                                      const std::vector<double> &Z, const std::vector<double> &I,
                                      const Point<dim> &x, const double phase)
          {
            const auto a = geometry.project_to_normal_profiles(x);
            if (!a.active)
              return 0.;
            const auto j = a.segment_index;
            const double t = a.xi;
            return ((1 - t) * Z[j] + t * Z[j + 1]) * old->m * phase * (1 + phase)
                   / (std::pow(1 - phase, 2) * ((1 - t) * I[j] + t * I[j + 1]));
          };
          for (const auto &entry : next.cells)
            {
              const auto cell = space.cells.at(entry.first);
              dg.reinit(cell);
              fe.reinit(entry.second);
              cell->get_dof_indices(ids);
              fe[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)]
              .get_function_values(next.bulk_history.native, phi);
              FullMatrix<double> mass(9, 9);
              Vector<double> rhs(9), coefficients(9);
              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  const double delta = old->shear_sense * (correction(manager, next.Z, next.I, dg.quadrature_point(q), phi[q])
                                                           - correction(*update->manager, old->Z, old->I, dg.quadrature_point(q), phi[q]));
                  for (unsigned int i = 0; i < 9; ++i)
                    {
                      for (unsigned int j = 0; j < 9; ++j)
                        mass(i, j) += dg.JxW(q) * dg.shape_value(i, q) * dg.shape_value(j, q);
                      rhs[i] += dg.JxW(q) * dg.shape_value(i, q) * delta;
                    }
                }
              mass.gauss_jordan();
              mass.vmult(coefficients, rhs);
              std::vector<types::global_dof_index> old_ids(9);
              old->bulk_history.tensor->primary.space->cells.at(entry.first)->get_dof_indices(old_ids);
              for (unsigned int c = 0; c < owned.size(); ++c)
                for (unsigned int i = 0; i < 9; ++i)
                  {
                    const auto component = SymmetricTensor<2, dim>::unrolled_to_component_indices(c);
                    owned[c][ids[i]] = old->bulk_history.tensor->primary.coefficients[c][old_ids[i]]
                                       + (component[0] == 0 && component[1] == 1 ? coefficients[i] : 0.);
                  }
            }
          for (unsigned int c = 0; c < owned.size(); ++c)
            {
              owned[c].compress(VectorOperation::insert);
              next.bulk_history.tensor->primary.coefficients[c] = owned[c];
            }
          update->next = fe_strain_state;
          update->normalization = current_normalization_integrals;
          update->surface_temperatures = current_fault_surface_temperatures;
          restore();
        }
      catch (...)
        {
          restore();
          throw;
        }
      return update;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::validate_dg_geometry_commit(
      const std::shared_ptr<DGGeometryUpdate<dim>> &u) const
    {
      AssertThrow(uses_prescribed_dg_geometry() && u && !u->published && u->incoming == fe_strain_state,
                  ExcMessage("Stale or duplicate native geometry publication."));
      validate_fe_strain_restrictions();
      AssertThrow(u->parent_surface == surface_stamp(this->get_reconstructed_fault_manager()),
                  ExcMessage("Surface histories changed after geometry preparation."));
      AssertThrow(u->next->bulk_history.native == u->incoming->bulk_history.native
                  && u->next->accepted_publications == u->incoming->accepted_publications
                  && u->next->committed_step == u->incoming->committed_step,
                  ExcMessage("Geometry transfer cannot change physical phase/clock/history publication."));
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::publish_dg_geometry_commit(
      const std::shared_ptr<DGGeometryUpdate<dim>> &u) noexcept
    {
      if (!u->identity)
        {
          this->get_reconstructed_fault_manager().swap_prescribed_geometry(*u->manager);
          fe_strain_state = u->next;
          invalidate_normalization_cache();
          current_normalization_integrals.swap(u->normalization);
          current_fault_surface_temperatures.swap(u->surface_temperatures);
        }
      u->published = true;
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::enable_prescribed_dg_geometry(const std::function<double(const Point<dim> &)> &); \
  template void PhaseFieldFault<dim>::initialize_prescribed_dg_geometry(); \
  template std::shared_ptr<DGGeometryUpdate<dim>> PhaseFieldFault<dim>::prepare_dg_geometry_append(const std::vector<Point<dim>> &,const std::vector<double> &,const std::vector<std::vector<double>> &,const std::vector<double> &); \
  template void PhaseFieldFault<dim>::validate_dg_geometry_commit(const std::shared_ptr<DGGeometryUpdate<dim>> &) const; \
  template void PhaseFieldFault<dim>::publish_dg_geometry_commit(const std::shared_ptr<DGGeometryUpdate<dim>> &) noexcept;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
