/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/surface_system.h>

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/reconstructed_fault/linear_performance.h>
#include <aspect/reconstructed_fault/sparse_coupling.h>
#include "surface_system_internal.h"
#include "surface_direct_internal.h"
#include "normal_filter_internal.h"

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <set>

namespace aspect
{
  namespace
  {
    template <int dim>
    class RestrictedSurfaceLinearSolve final
      : public ReconstructedFaultSurfaceLinearSolve<dim>
    {
      public:
        RestrictedSurfaceLinearSolve(
          const ReconstructedFaultSurfaceSystem<dim> &owner,
          const unsigned int generation,
          const ReconstructedFaultVector &diagonal,
          const ReconstructedFaultVector &off_diagonal,
          const ReconstructedFaultActiveSet &active_set)
          : owner(owner), generation(generation)
        {
          for (unsigned int fault=0; fault<diagonal.size(); ++fault)
            {
              internal::FaultLinearSection timing(internal::FaultLinearTiming::factor);
              factorizations.push_back(std::make_unique<internal::FaultSurfaceDirect>(
                diagonal[fault],off_diagonal[fault],active_set[fault],fault));
            }
        }

        void solve(const ReconstructedFaultVector &rhs,
                   ReconstructedFaultVector &solution) const override
        {
          AssertThrow(generation == owner.get_linearization_generation(),
                      ExcMessage("This restricted reconstructed-fault surface solve has been superseded."));
          AssertThrow(rhs.size() == factorizations.size(),
                      ExcMessage("The restricted K_V RHS has the wrong number of faults."));
          solution.resize(rhs.size());
          for (unsigned int fault=0; fault<rhs.size(); ++fault)
            {
              internal::FaultLinearSection timing(internal::FaultLinearTiming::inverse);
              factorizations[fault]->solve(rhs[fault],solution[fault]);
            }
        }

      private:
        const ReconstructedFaultSurfaceSystem<dim> &owner;
        const unsigned int generation;
        std::vector<std::unique_ptr<internal::FaultSurfaceDirect>> factorizations;
    };
  }


  template <int dim>
  typename ReconstructedFaultSurfaceSystem<dim>::SurfaceAssembly
  ReconstructedFaultSurfaceSystem<dim>::assemble_surface_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate,
    const bool assemble_jacobian) const
  {
    if (bulk_work_measure)
      return assemble_bulk_work_system(bulk_state, slip_rate, assemble_jacobian);
    return assemble_particle_system(bulk_state, slip_rate, assemble_jacobian);
  }


  template <int dim>
  struct ReconstructedFaultSurfaceSystem<dim>::SurfaceLinearization
  {
    struct CouplingPoint
    {
      Point<dim> position;
      unsigned int parent_index;
      unsigned int fault_index;
      unsigned int segment_index;
      double xi;
      double particle_domain_volume;
      double eta_ve;
      double friction_coefficient;
      SymmetricTensor<2,dim> slip_tensor;
      SymmetricTensor<2,dim> normal_tensor;
      bool uses_adiabatic_friction_pressure;
    };

    using FaultFactorization = internal::FaultSurfaceDirect;

    ReconstructedFaultSurfaceResidual residual;
    std::vector<std::vector<double>> diagonal;
    std::vector<std::vector<double>> off_diagonal;
    std::vector<std::vector<double>> mass_diagonal;
    std::vector<std::vector<double>> mass_off_diagonal;
    std::vector<std::unique_ptr<FaultFactorization>> factorizations;
    std::vector<CouplingPoint> coupling_points;
    std::unique_ptr<Utilities::MPI::RemotePointEvaluation<dim>> point_cache;
    std::unique_ptr<internal::FaultSparseCoupling> matrix;
    std::vector<std::shared_ptr<const internal::FaultNormalFilter>> normal_filters;
  };


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::ReconstructedFaultSurfaceSystem(
    const Simulator<dim> &simulator)
    :
    SimulatorAccess<dim>(simulator),
    phase_field_fault(
      Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
        this->get_material_model())),
    grid_cache(this->get_triangulation(), this->get_mapping())
  {
    performance_timer = std::make_unique<TimerOutput>(
      std::cout,
      std::getenv("ASPECT_FAULT_PERFORMANCE") && this->get_pcout().is_active()
      ? TimerOutput::summary : TimerOutput::never, TimerOutput::wall_times);
  }


  template <int dim>
  ReconstructedFaultSurfaceSystem<dim>::~ReconstructedFaultSurfaceSystem() = default;

  template <int dim>
  void ReconstructedFaultSurfaceSystem<dim>::set_normal_stress_filter(const std::string &mode, const double length)
  {
    AssertThrow(mode=="raw" || mode=="projected" || mode=="helmholtz",
                ExcMessage("Normal filter mode must be raw, projected or helmholtz."));
    AssertThrow(std::isfinite(length) && length>=0. && (mode=="helmholtz" || length==0.),
                ExcMessage("Only Helmholtz mode accepts a nonzero physical filter length."));
    if (mode==normal_filter_mode && length==normal_filter_length) return;
    normal_filter_mode=mode;normal_filter_length=length;
    normal_filter_cache.clear();surface_linearization.reset();++linearization_generation;
  }


  template <int dim>
  void ReconstructedFaultSurfaceSystem<dim>::set_normal_traction_diagnostic(
    std::function<bool(const Point<dim> &)> window,
    std::vector<std::pair<Point<dim>,Point<dim>>> lines)
  {
    normal_diagnostic_window = std::move(window);
    normal_diagnostic_lines = std::move(lines);
    normal_diagnostic.reset();
  }


  template <int dim>
  const typename ReconstructedFaultSurfaceSystem<dim>::NormalTractionDiagnostic &
  ReconstructedFaultSurfaceSystem<dim>::get_normal_traction_diagnostic() const
  {
    AssertThrow(normal_diagnostic, ExcMessage("No captured work-measure normal traction."));
    return *normal_diagnostic;
  }


  template <int dim>
  void ReconstructedFaultSurfaceSystem<dim>::enable_bulk_work_measure()
  {
    AssertThrow(dim==2 && phase_field_fault.is_mature_frictional_fault(),
                ExcMessage("Bulk work measure requires a frozen mature 2-D fault."));
    const auto &faults=this->get_reconstructed_fault_manager().get_faults();
    // Automatic completion has separately qualified each terminal neighborhood
    // and its common source association. The assembly below is already per fault
    // and uses each segment's frame; retain the legacy admission restriction.
    if (this->get_reconstructed_fault_manager().uses_automatic_boundary_completion())
      {
        if (!bulk_work_measure) surface_linearization.reset();
        bulk_work_measure=true;
        return;
      }
    AssertThrow(faults.size()==1,ExcMessage("Bulk work measure currently supports one straight fault."));
    const auto &fault=faults[0];
    auto tangent=fault.vertex(1)-fault.vertex(0);tangent/=tangent.norm();
    const double length=fault.vertex(fault.n_vertices()-1).distance(fault.vertex(0));
    for (unsigned int i=0;i<fault.n_vertices();++i)
      {
        const auto offset=fault.vertex(i)-fault.vertex(0);
        AssertThrow((offset-(offset*tangent)*tangent).norm()<1e-10*length,
                    ExcMessage("Bulk work measure requires a straight fault."));
      }
    if (!bulk_work_measure) surface_linearization.reset();
    bulk_work_measure=true;
  }


  template <int dim>
  ReconstructedFaultSurfaceResidual
  ReconstructedFaultSurfaceSystem<dim>::evaluate_surface_residual(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate) const
  {
    return assemble_surface_system(bulk_state, slip_rate, false).residual;
  }


  template <int dim>
  const ReconstructedFaultSurfaceResidual &
  ReconstructedFaultSurfaceSystem<dim>::linearize_surface_system(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate)
  {
    // Invalidate all previous semantic solves before building any new K_V data.
    TimerOutput::Scope timer(*performance_timer, "Fault: Linearization total");
    surface_linearization.reset();
    ++linearization_generation;
    normal_diagnostic.reset();
#ifndef DEAL_II_WITH_UMFPACK
    AssertThrow(false,
                ExcMessage("The coupled reconstructed-fault solver requires deal.II "
                           "with UMFPACK support."));
#else
    const SurfaceAssembly assembled =
      assemble_surface_system(bulk_state, slip_rate, true);
    if (assembled.normal_diagnostic && normal_diagnostic_observer)
      normal_diagnostic_observer(assembled.residual,*assembled.normal_diagnostic);
    auto candidate = std::make_unique<SurfaceLinearization>();
    candidate->residual = assembled.residual;
    candidate->diagonal = assembled.diagonal;
    candidate->off_diagonal = assembled.off_diagonal;
    candidate->mass_diagonal = assembled.mass_diagonal;
    candidate->mass_off_diagonal = assembled.mass_off_diagonal;
    candidate->normal_filters = assembled.normal_filters;
    candidate->coupling_points.reserve(assembled.coupling_points.size());
    for (const auto &point : assembled.coupling_points)
      candidate->coupling_points.push_back(
      {
        point.position,
        point.parent_index,
        point.fault_index,
        point.segment_index,
        point.xi,
        point.particle_domain_volume,
        point.eta_ve,
        point.friction_coefficient,
        point.slip_tensor,
        point.normal_tensor,
        point.uses_adiabatic_friction_pressure
      });
    candidate->factorizations.resize(assembled.diagonal.size());

    // Each reconstructed fault is an independent replicated Q1 block. Factor
    // the full nonsingular block once for this coupled linearization.
    for (unsigned int fault = 0; fault < assembled.diagonal.size(); ++fault)
      {
        internal::FaultLinearSection timing(internal::FaultLinearTiming::factor);
        candidate->factorizations[fault] =
          std::make_unique<typename SurfaceLinearization::FaultFactorization>(
            assembled.diagonal[fault], assembled.off_diagonal[fault],
            std::vector<bool>(assembled.diagonal[fault].size(),false), fault);
      }

    // G reuses the same particle points and constitutive coefficients as K_V;
    // cache their remote bulk-field lookup for the lifetime of this linearization.
    std::vector<Point<dim>> points;
    points.reserve(candidate->coupling_points.empty() ? 0 :
                   candidate->coupling_points.back().parent_index+1);
    for (const auto &point : candidate->coupling_points)
      if (point.parent_index == points.size())
        points.push_back(point.position);
    candidate->point_cache =
      std::make_unique<Utilities::MPI::RemotePointEvaluation<dim>>();
    TimerOutput::Scope lookup_timer(*performance_timer, "Fault: G lookup build");
    candidate->point_cache->reinit(grid_cache, points);
    lookup_timer.stop();
    if (std::getenv("ASPECT_FAULT_EXPLICIT_G") && candidate->normal_filters.empty())
      {
        internal::FaultLinearSection timing(internal::FaultLinearTiming::G_setup);
        using Coefficient=std::pair<SymmetricTensor<2,dim>,double>;
        using Row=std::pair<unsigned int,Coefficient>;
        using Rows=std::vector<Row>;
        std::vector<std::map<unsigned int,Coefficient>> parent_coefficients(points.size());
        std::vector<unsigned int> offsets(1,0);
        for (const auto &fault : assembled.diagonal)
          offsets.push_back(offsets.back()+fault.size());
        const auto &point_ptrs=candidate->point_cache->get_point_ptrs();

        // Integrate domain test functions first, holding each parent's bulk
        // sample P0 exactly as in G. Shared-face samples are averaged, not
        // assigned to an arbitrary incident cell: divide before RPE duplicates.
        for (const auto &point : candidate->coupling_points)
          {
            const auto multiplicity=point_ptrs[point.parent_index+1]-point_ptrs[point.parent_index];
            // VectorTools' avg evaluation returns zero for a missing request.
            // Preserve that existing action convention instead of dividing by zero.
            if (multiplicity==0) continue;
            const auto stress=2.*point.eta_ve*(point.slip_tensor
              +(point.uses_adiabatic_friction_pressure ? 0. : point.friction_coefficient)*point.normal_tensor);
            for (unsigned int end=0; end<2; ++end)
              {
                const double weight=point.particle_domain_volume*(end==0 ? 1.-point.xi : point.xi)/multiplicity;
                auto &c=parent_coefficients[point.parent_index][offsets[point.fault_index]+point.segment_index+end];
                c.first+=weight*stress;
                if (!point.uses_adiabatic_friction_pressure)
                  c.second-=weight*point.friction_coefficient;
              }
          }
        std::vector<Rows> input(points.size());
        for (unsigned int p=0; p<points.size(); ++p)
          for (const auto &row : parent_coefficients[p]) input[p].push_back(row);
        internal::FaultSparseCoupling::Entries entries;
        using Cache=Utilities::MPI::RemotePointEvaluation<dim>;
        candidate->point_cache->template process_and_evaluate<Rows>(input,
          [&](const ArrayView<const Rows> &values, const typename Cache::CellData &cells)
          {
            // RPE routes each coefficient to the original FE-cell owner.
            // Store physical, unconstrained columns; the solver continues to
            // distribute homogeneous constraints and pressure scaling before G.
            const auto &fe=this->get_fe();
            std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
            for (const auto c : cells.cell_indices())
              {
                const auto tria_cell=cells.get_active_cell_iterator(c);
                const typename DoFHandler<dim>::active_cell_iterator cell(
                  &this->get_triangulation(),tria_cell->level(),tria_cell->index(),&this->get_dof_handler());
                const auto unit_points=cells.get_unit_points(c);
                FEValues<dim> fe_values(this->get_mapping(),fe,
                  Quadrature<dim>(std::vector<Point<dim>>(unit_points.begin(),unit_points.end())),
                  update_values | update_gradients);
                fe_values.reinit(cell);
                cell->get_dof_indices(indices);
                const auto data=cells.get_data_view(c,values);
                for (unsigned int p=0; p<data.size(); ++p)
                  for (unsigned int i=0; i<indices.size(); ++i)
                    {
                      const auto component=fe.system_to_component_index(i).first;
                      const bool velocity=this->introspection().component_masks.velocities[component];
                      const bool pressure=component==this->introspection().component_indices.pressure;
                      if (!velocity && !pressure) continue;
                      for (const auto &row : data[p])
                        entries[{row.first,indices[i]}]+=velocity
                          ? row.second.first*fe_values[this->introspection().extractors.velocities].symmetric_gradient(i,p)
                          : row.second.second*fe_values[this->introspection().extractors.pressure].value(i,p);
                    }
              }
          });
        candidate->matrix=std::make_unique<internal::FaultSparseCoupling>();
        candidate->matrix->build(entries);
        if (std::getenv("ASPECT_FAULT_PERFORMANCE"))
          this->get_pcout() << "   Fault sparse G: rank0 entries=" << candidate->matrix->values.size()
                           << ", bytes=" << candidate->matrix->bytes() << std::endl;
      }
    normal_diagnostic = assembled.normal_diagnostic;
    surface_linearization = std::move(candidate);
#endif
    return surface_linearization->residual;
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::solve(
    const FaultVector &rhs,
    FaultVector &solution) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before applying its inverse."));
    AssertThrow(rhs.size() == surface_linearization->factorizations.size(),
                ExcMessage("The K_V right-hand side has the wrong number of faults."));
    solution.resize(rhs.size());
    for (unsigned int fault = 0; fault < rhs.size(); ++fault)
      {
        internal::FaultLinearSection timing(internal::FaultLinearTiming::inverse);
        // Replicated K_V/RHS require no communication. The same factor object
        // handles unrestricted and principal-free blocks, including safeguards.
        surface_linearization->factorizations[fault]->solve(rhs[fault],solution[fault]);
      }
  }


  template <int dim>
  std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
  ReconstructedFaultSurfaceSystem<dim>::create_restricted_linear_solve(
    const ReconstructedFaultActiveSet &active_set) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before restricting its inverse."));
    AssertThrow(active_set.size() == surface_linearization->diagonal.size(),
                ExcMessage("The reconstructed-fault active set has the wrong "
                           "number of faults."));
    for (unsigned int fault = 0; fault < active_set.size(); ++fault)
      AssertThrow(active_set[fault].size()
                  == surface_linearization->diagonal[fault].size(),
                  ExcMessage("The reconstructed-fault active set has the wrong "
                             "number of vertices for fault "
                             + Utilities::int_to_string(fault) + "."));

    return std::make_unique<RestrictedSurfaceLinearSolve<dim>>(
             *this,
             linearization_generation,
             surface_linearization->diagonal,
             surface_linearization->off_diagonal,
             active_set);
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_surface_jacobian(
    const FaultVector &direction,
    FaultVector &result) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("K_V must be assembled before applying it."));
    AssertThrow(direction.size() == surface_linearization->diagonal.size(),
                ExcMessage("The K_V direction has the wrong number of faults."));

    result.resize(direction.size());
    for (unsigned int fault = 0; fault < direction.size(); ++fault)
      {
        AssertThrow(direction[fault].size()
                    == surface_linearization->diagonal[fault].size(),
                    ExcMessage("The K_V direction has the wrong number of vertices "
                               "for reconstructed fault "
                               + Utilities::int_to_string(fault) + "."));
        result[fault].resize(direction[fault].size());
        for (unsigned int i = 0; i < direction[fault].size(); ++i)
          {
            double value = surface_linearization->diagonal[fault][i]
                           * direction[fault][i];
            if (i > 0)
              value += surface_linearization->off_diagonal[fault][i-1]
                       * direction[fault][i-1];
            if (i+1 < direction[fault].size())
              value += surface_linearization->off_diagonal[fault][i]
                       * direction[fault][i+1];
            result[fault][i] = value;
          }
      }
  }


  template <int dim>
  double
  ReconstructedFaultSurfaceSystem<dim>::surface_residual_rms(
    const ReconstructedFaultSurfaceResidual &residual,
    const ReconstructedFaultActiveSet &active_set) const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("The surface system must be linearized before "
                           "measuring its residual."));
    AssertDimension(residual.values.size(),
                    surface_linearization->mass_diagonal.size());
    AssertDimension(active_set.size(), residual.values.size());

    // For weak nodal residual r, sqrt(r^T M^-1 r) is the L2 norm of
    // its consistent-Q1 strong representation. Restrict both r and M to
    // the same free set used by the projected Newton solve.
    double squared_norm = 0.0;
    double measure = 0.0;
    for (unsigned int fault = 0; fault < residual.values.size(); ++fault)
      {
        const auto &mass_diagonal = surface_linearization->mass_diagonal[fault];
        const auto &mass_off_diagonal =
          surface_linearization->mass_off_diagonal[fault];
        AssertDimension(residual.values[fault].size(), mass_diagonal.size());
        AssertDimension(active_set[fault].size(), mass_diagonal.size());

        std::vector<double> restricted_diagonal = mass_diagonal;
        std::vector<double> restricted_off_diagonal = mass_off_diagonal;
        std::vector<double> restricted_rhs = residual.values[fault];
        for (unsigned int vertex = 0; vertex < restricted_rhs.size(); ++vertex)
          if (active_set[fault][vertex])
            {
              restricted_diagonal[vertex] = 1.0;
              restricted_rhs[vertex] = 0.0;
              if (vertex > 0)
                restricted_off_diagonal[vertex-1] = 0.0;
              if (vertex < restricted_off_diagonal.size())
                restricted_off_diagonal[vertex] = 0.0;
            }

        const std::vector<double> strong_residual =
          ReconstructedFaultUtilities::solve_tridiagonal_system(
            restricted_diagonal, restricted_off_diagonal, restricted_rhs);
        for (unsigned int vertex = 0; vertex < restricted_rhs.size(); ++vertex)
          if (!active_set[fault][vertex])
            {
              squared_norm += restricted_rhs[vertex]*strong_residual[vertex];
              measure += mass_diagonal[vertex];
            }
        for (unsigned int segment = 0;
             segment < mass_off_diagonal.size(); ++segment)
          if (!active_set[fault][segment] && !active_set[fault][segment+1])
            measure += 2.0*mass_off_diagonal[segment];
      }

    AssertThrow(std::isfinite(squared_norm) && std::isfinite(measure)
                && squared_norm >= 0.0 && measure >= 0.0,
                ExcMessage("The consistent reconstructed-fault surface residual "
                           "norm is not finite and nonnegative."));
    return measure > 0.0 ? std::sqrt(squared_norm/measure) : 0.0;
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_G(
    const LinearAlgebra::BlockVector &physical_bulk_direction,
    FaultVector &result) const
  {
    AssertThrow(surface_linearization,ExcMessage("G must be linearized before applying it."));
    AssertThrow(physical_bulk_direction.size()==this->get_dof_handler().n_dofs(),
                ExcMessage("The reconstructed-fault G direction has the wrong size."));
    if (!surface_linearization->matrix)
      { apply_G_reference(physical_bulk_direction,result); return; }
    {
      internal::FaultLinearSection timing(internal::FaultLinearTiming::G_sparse);
      unsigned int size=0;
      for (const auto &fault : surface_linearization->diagonal) size+=fault.size();
      std::vector<double> local(size),global(size);
      surface_linearization->matrix->add(physical_bulk_direction,local);
      Utilities::MPI::sum(local,this->get_mpi_communicator(),global);
      result=surface_linearization->diagonal;
      unsigned int i=0;
      for (auto &fault : result) for (auto &value : fault) value=global[i++];
    }
  }


  template <int dim>
  void
  ReconstructedFaultSurfaceSystem<dim>::apply_G_reference(
    const LinearAlgebra::BlockVector &physical_bulk_direction,
    FaultVector &result) const
  {
    TimerOutput::Scope timer(*performance_timer, "Fault: G total");
    internal::FaultLinearSection linear_timing(internal::FaultLinearTiming::G);
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("The surface system must be linearized before applying G."));
    AssertThrow(physical_bulk_direction.size()
                == this->get_dof_handler().n_dofs(),
                ExcMessage("The reconstructed-fault G direction has the wrong size."));
    const auto &faults = this->get_reconstructed_fault_manager().get_faults();
    result.resize(faults.size());
    unsigned int n_fault_dofs = 0;
    for (unsigned int fault = 0; fault < faults.size(); ++fault)
      {
        result[fault].assign(faults[fault].n_vertices(), 0.0);
        n_fault_dofs += faults[fault].n_vertices();
      }

    // The input is a homogeneous bulk perturbation with pressure already
    // converted from solver scaling to physical units by the condensed system.
    const unsigned int velocity_component =
      this->introspection().component_indices.velocities[0];
    const unsigned int pressure_component =
      this->introspection().component_indices.pressure;
    TimerOutput::Scope sample_timer(*performance_timer, "Fault: G FE sampling");
    const auto velocity_gradients = VectorTools::point_gradients<dim>(
                                      *surface_linearization->point_cache, this->get_dof_handler(),
                                      physical_bulk_direction, VectorTools::EvaluationFlags::avg,
                                      velocity_component);
    const std::vector<double> pressures = VectorTools::point_values<1>(
                                            *surface_linearization->point_cache, this->get_dof_handler(),
                                            physical_bulk_direction, VectorTools::EvaluationFlags::avg,
                                            pressure_component);
    AssertDimension(velocity_gradients.size(), pressures.size());
    sample_timer.stop();
    TimerOutput::Scope action_timer(*performance_timer, "Fault: G integrate/reduce");

    const bool filtering=!surface_linearization->normal_filters.empty();
    FaultVector normal_direction;
    if (filtering)
      {
        TimerOutput::Scope filter_timer(*performance_timer,"Fault: G normal filter");
        normal_direction=result;
        for (const auto &p:surface_linearization->coupling_points)
          {
            const double dsigma=pressures[p.parent_index]
              -2.*p.eta_ve*(symmetrize(velocity_gradients[p.parent_index])*p.normal_tensor);
            normal_direction[p.fault_index][p.segment_index]+=p.particle_domain_volume*(1.-p.xi)*dsigma;
            normal_direction[p.fault_index][p.segment_index+1]+=p.particle_domain_volume*p.xi*dsigma;
          }
        // The filter acts globally on each disconnected fault. Background is
        // fixed and has zero derivative; pressure is already in physical units.
        for (unsigned int f=0;f<normal_direction.size();++f)
          {
            const auto local=normal_direction[f];
            Utilities::MPI::sum(local,this->get_mpi_communicator(),normal_direction[f]);
            normal_direction[f]=surface_linearization->normal_filters[f]->solve(normal_direction[f]);
          }
      }

    // Apply the pointwise G action with its pressure-mode sign convention. The
    // adiabatic branch has neither the mu*N strain term nor the -mu*delta-p term.
    for (unsigned int p = 0;
         p < surface_linearization->coupling_points.size(); ++p)
      {
        const auto &point = surface_linearization->coupling_points[p];
        const SymmetricTensor<2,dim> strain_rate =
          symmetrize(velocity_gradients[point.parent_index]);
        const SymmetricTensor<2,dim> stress_direction =
          (point.uses_adiabatic_friction_pressure || filtering)
          ? 2.0 * point.eta_ve * point.slip_tensor
          : 2.0 * point.eta_ve
          * (point.slip_tensor
             + point.friction_coefficient * point.normal_tensor);
        double value = stress_direction * strain_rate;
        if (filtering)
          value-=point.friction_coefficient*((1.-point.xi)*normal_direction[point.fault_index][point.segment_index]
                                             +point.xi*normal_direction[point.fault_index][point.segment_index+1]);
        else if (!point.uses_adiabatic_friction_pressure)
          value -= point.friction_coefficient * pressures[point.parent_index];

        const double shape[2] = {1.0-point.xi, point.xi};
        result[point.fault_index][point.segment_index] +=
          point.particle_domain_volume * shape[0] * value;
        result[point.fault_index][point.segment_index+1] +=
          point.particle_domain_volume * shape[1] * value;
      }

    // Contributions originate on locally owned particles; sum them so the
    // returned fault vector is replicated identically on all ranks.
    std::vector<double> local_values(n_fault_dofs);
    unsigned int position = 0;
    for (const auto &fault_values : result)
      for (const double value : fault_values)
        local_values[position++] = value;
    std::vector<double> global_values(n_fault_dofs);
    Utilities::MPI::sum(local_values, this->get_mpi_communicator(), global_values);
    position = 0;
    for (auto &fault_values : result)
      for (double &value : fault_values)
        value = global_values[position++];
  }


  template <int dim>
  const ReconstructedFaultSurfaceResidual &
  ReconstructedFaultSurfaceSystem<dim>::get_linearization_residual() const
  {
    AssertThrow(surface_linearization != nullptr,
                ExcMessage("Surface weak diagnostics require a completed linearization."));
    return surface_linearization->residual;
  }


  template <int dim>
  unsigned int
  ReconstructedFaultSurfaceSystem<dim>::get_linearization_generation() const
  {
    return linearization_generation;
  }


#define INSTANTIATE(dim) template class ReconstructedFaultSurfaceSystem<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
