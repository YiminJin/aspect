/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>

#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/plugins.h>

namespace aspect
{
  namespace StokesSolver
  {
    namespace
    {
      template <int dim>
      void
      apply_stokes_matrix(const LinearAlgebra::BlockSparseMatrix &matrix,
                          const LinearAlgebra::BlockVector &source,
                          LinearAlgebra::BlockVector &result)
      {
        AssertDimension(source.n_blocks(), 2);
        AssertDimension(result.n_blocks(), 2);
        matrix.block(0,0).vmult(result.block(0), source.block(0));
        matrix.block(0,1).vmult_add(result.block(0), source.block(1));
        matrix.block(1,0).vmult(result.block(1), source.block(0));
        matrix.block(1,1).vmult_add(result.block(1), source.block(1));
      }


      AffineConstraints<double>
      make_homogeneous_constraints(const AffineConstraints<double> &constraints)
      {
        AffineConstraints<double> homogeneous(constraints);
        for (const auto &line : homogeneous.get_lines())
          homogeneous.set_inhomogeneity(line.index, 0.0);
        return homogeneous;
      }
    }


    template <int dim>
    ReconstructedFaultCondensedSystem<dim>::ReconstructedFaultCondensedSystem(
      const Simulator<dim> &simulator)
      :
      SimulatorAccess<dim>(simulator),
      surface_system(this->get_reconstructed_fault_surface_system()),
      stokes_coupling(this->get_reconstructed_fault_stokes_coupling())
    {
      AssertThrow(dim == 2,
                  ExcMessage("The coupled reconstructed-fault solver currently "
                             "supports only two dimensions."));
      AssertThrow(!this->get_parameters().include_melt_transport,
                  ExcMessage("The coupled reconstructed-fault solver does not "
                             "currently support melt transport."));
      AssertThrow(this->get_parameters().stokes_solver_type
                  == Parameters<dim>::StokesSolverType::block_amg,
                  ExcMessage("The coupled reconstructed-fault solver currently "
                             "requires the assembled block-AMG Stokes solver. "
                             "Direct and matrix-free/GMG solvers are unsupported."));
      AssertThrow(this->introspection().block_indices.velocities == 0
                  && this->introspection().block_indices.pressure == 1,
                  ExcMessage("The coupled reconstructed-fault solver requires "
                             "separate velocity and pressure blocks in positions 0 and 1."));

      // Open/traction boundaries or pressure-dependent equations may fix p.
      // Partial component masks are conservatively excluded from eligibility.
      const auto &material = Plugins::get_plugin_as_type<
        const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
      pressure_gauge_candidate = material.uses_adiabatic_friction_pressure()
                                 && !material.is_compressible()
                                 && !this->get_parameters().mesh_deformation_enabled;
      auto open = this->get_geometry_model().get_used_boundary_indicators();
      const auto &velocity = this->get_boundary_velocity_manager();
      for (const auto id : velocity.get_zero_boundary_velocity_indicators()) open.erase(id);
      for (const auto id : velocity.get_tangential_boundary_velocity_indicators()) open.erase(id);
      for (const auto id : velocity.get_prescribed_boundary_velocity_indicators())
        {
          const auto mask = velocity.get_component_mask(id);
          bool all_velocity_components = true;
          for (unsigned int c=0; c<dim; ++c)
            all_velocity_components &= mask[this->introspection().component_indices.velocities[c]];
          if (all_velocity_components) open.erase(id);
        }
      for (const auto &pair : this->get_geometry_model().get_periodic_boundary_pairs())
        {
          open.erase(pair.first.first);
          open.erase(pair.first.second);
        }
      pressure_gauge_candidate &= open.empty();
    }


    template <int dim>
    typename ReconstructedFaultCondensedSystem<dim>::Linearization
    ReconstructedFaultCondensedSystem<dim>::linearize(
      const LinearAlgebra::BlockSparseMatrix &bulk_matrix,
      const LinearAlgebra::BlockVector &physical_bulk_state,
      const FaultVector &slip_rate,
      const ReconstructedFaultSurfaceLinearSolve<dim> *surface_solve)
    {
      // Publish A, frozen B, G, and K_V as one generation. Rebuilding either
      // simulator-owned coupling component invalidates the returned view.
      ++active_generation;

      const ReconstructedFaultSurfaceResidual &residual =
        surface_system.linearize_surface_system(physical_bulk_state, slip_rate);
      stokes_coupling.linearize_B(physical_bulk_state);

      const auto system_constraints =
        std::make_shared<const AffineConstraints<double>>(
          make_homogeneous_constraints(this->get_current_constraints()));

      // Jacobian vectors are perturbations: retain hanging-node/periodic
      // relations but remove every inhomogeneous boundary value.
#if DEAL_II_VERSION_GTE(9,6,0)
      IndexSet stokes_dofs(this->get_dof_handler().n_dofs());
      stokes_dofs.add_range(
        0,
        this->introspection().index_sets.stokes_partitioning[0].size()
        + this->introspection().index_sets.stokes_partitioning[1].size());
      const auto stokes_constraints =
        std::make_shared<const AffineConstraints<double>>(
          make_homogeneous_constraints(
            this->get_current_constraints().get_view(stokes_dofs)));
#else
      const auto stokes_constraints = system_constraints;
#endif
      return Linearization(*this,
                           bulk_matrix,
                           surface_solve != nullptr ? *surface_solve : surface_system,
                           residual,
                           active_generation,
                           surface_system.get_linearization_generation(),
                           stokes_coupling.get_B_linearization_rebuild_count(),
                           system_constraints,
                           stokes_constraints);
    }


    template <int dim>
    ReconstructedFaultCondensedSystem<dim>::Linearization::Linearization(
      ReconstructedFaultCondensedSystem<dim> &owner,
      const LinearAlgebra::BlockSparseMatrix &bulk_matrix,
      const ReconstructedFaultSurfaceLinearSolve<dim> &surface_solve,
      const ReconstructedFaultSurfaceResidual &surface_residual,
      const unsigned int generation,
      const unsigned int surface_generation,
      const unsigned int B_generation,
      const std::shared_ptr<const AffineConstraints<double>> &system_constraints,
      const std::shared_ptr<const AffineConstraints<double>> &stokes_constraints)
      :
      owner(owner),
      bulk_matrix(bulk_matrix),
      surface_solve(surface_solve),
      residual(surface_residual),
      generation(generation),
      surface_generation(surface_generation),
      B_generation(B_generation),
      homogeneous_system_constraints(system_constraints),
      homogeneous_stokes_constraints(stokes_constraints)
    {}


    template <int dim>
    typename ReconstructedFaultCondensedSystem<dim>::Linearization
    ReconstructedFaultCondensedSystem<dim>::Linearization::with_surface_solve(
      const ReconstructedFaultSurfaceLinearSolve<dim> &new_surface_solve) const
    {
      assert_is_current();
      return owner.rebind_surface_solve(*this, new_surface_solve);
    }


    template <int dim>
    typename ReconstructedFaultCondensedSystem<dim>::Linearization
    ReconstructedFaultCondensedSystem<dim>::rebind_surface_solve(
      const Linearization &linearization,
      const ReconstructedFaultSurfaceLinearSolve<dim> &new_surface_solve)
    {
      linearization.assert_is_current();

      // An active-set update changes only the semantic K_V inverse. Preserve
      // A/B/G/residual data and supersede the unrestricted linearization view.
      ++active_generation;
      return Linearization(*this,
                           linearization.bulk_matrix,
                           new_surface_solve,
                           linearization.residual,
                           active_generation,
                           linearization.surface_generation,
                           linearization.B_generation,
                           linearization.homogeneous_system_constraints,
                           linearization.homogeneous_stokes_constraints);
    }


    template <int dim>
    void
    ReconstructedFaultCondensedSystem<dim>::Linearization::assert_is_current() const
    {
      AssertThrow(generation == owner.active_generation,
                  ExcMessage("This reconstructed-fault coupled linearization has "
                             "been superseded."));
      AssertThrow(surface_generation
                  == owner.surface_system.get_linearization_generation()
                  && B_generation
                  == owner.stokes_coupling.get_B_linearization_rebuild_count(),
                  ExcMessage("A reconstructed-fault coupling component was "
                             "relinearized independently. Rebuild the complete "
                             "coupled linearization."));
    }


    template <int dim>
    LinearAlgebra::BlockVector
    ReconstructedFaultCondensedSystem<dim>::Linearization::
    make_constrained_solver_direction(
      const LinearAlgebra::BlockVector &direction) const
    {
      AssertDimension(direction.n_blocks(), 2);

      // Krylov vectors use solver-scaled pressure and homogeneous constraints;
      // constrained algebraic entries must not contribute to A or B/G actions.
      LinearAlgebra::BlockVector constrained(
        owner.introspection().index_sets.stokes_partitioning,
        owner.get_mpi_communicator());
      constrained = direction;
      homogeneous_stokes_constraints->set_zero(constrained);
      constrained.compress(VectorOperation::insert);
      return constrained;
    }


    template <int dim>
    LinearAlgebra::BlockVector
    ReconstructedFaultCondensedSystem<dim>::Linearization::
    verified_pressure_nullspace(double &right_error, double &left_error) const
    {
      assert_is_current();
      LinearAlgebra::BlockVector q(owner.introspection().index_sets.stokes_partitioning,
                                   owner.get_mpi_communicator());
      right_error = left_error = 0.;
      if (!owner.pressure_gauge_candidate)
        return q;

      // A prescribed pressure DoF fixes the gauge, unlike hanging/periodic
      // relations. Never overwrite such a physical/algebraic constraint.
      bool pressure_is_fixed = false;
      for (const auto &line : homogeneous_stokes_constraints->get_lines())
        if (line.index >= q.block(0).size() && line.entries.empty())
          pressure_is_fixed = true;
      if (Utilities::MPI::max(static_cast<unsigned int>(pressure_is_fixed),
                             owner.get_mpi_communicator()) != 0)
        return q;

      if (!owner.get_parameters().use_locally_conservative_discretization)
        q.block(1) = 1.;
      else
        {
          // FE_DGP's first local basis function is the constant, as in
          // ASPECT's existing pressure-normalization implementation.
          const auto &fe = owner.get_fe();
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          const auto constant = fe.component_to_system_index(
            owner.introspection().component_indices.pressure, 0);
          for (const auto &cell : owner.get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                q[indices[constant]] = 1.;
              }
          q.compress(VectorOperation::insert);
        }
      homogeneous_stokes_constraints->set_zero(q);
      q /= q.l2_norm();

      LinearAlgebra::BlockVector right(q), left(q);
      vmult(right, q); // Includes G, the current free-set K_V solve, and B.
      right_error = right.l2_norm();

      // B has only velocity rows, so q^T B=0 identically. Thus these are
      // exactly the left-null tests for C=A-B K_V^{-1} G, not merely for A.
      bulk_matrix.block(1,0).Tvmult(left.block(0), q.block(1));
      bulk_matrix.block(1,1).Tvmult(left.block(1), q.block(1));
      homogeneous_stokes_constraints->set_zero(left);
      left_error = left.l2_norm();
      const double roundoff = 100.*std::numeric_limits<double>::epsilon();
      const double diagonal = bulk_matrix.block(1,1).frobenius_norm();
      if (right_error > roundoff*(bulk_matrix.block(0,1).frobenius_norm()+diagonal)
          || left_error > roundoff*(bulk_matrix.block(1,0).frobenius_norm()+diagonal))
        q = 0.;
      return q;
    }


    template <int dim>
    LinearAlgebra::BlockVector
    ReconstructedFaultCondensedSystem<dim>::Linearization::
    make_physical_bulk_direction(
      const LinearAlgebra::BlockVector &solver_direction) const
    {
      // G differentiates the physical constitutive law. Convert solver pressure
      // exactly once, then distribute only homogeneous perturbation constraints.
      LinearAlgebra::BlockVector owned(
        owner.introspection().index_sets.system_partitioning,
        owner.get_mpi_communicator());
      owned.block(0) = solver_direction.block(0);
      owned.block(1) = solver_direction.block(1);
      owned.block(1) *= owner.get_pressure_scaling();
      homogeneous_system_constraints->distribute(owned);
      owned.compress(VectorOperation::insert);

      // Remote point evaluation of G needs locally relevant ghost values, while
      // the scaling and constraint operations above act on the owned vector.
      LinearAlgebra::BlockVector ghosted(
        owner.introspection().index_sets.system_partitioning,
        owner.introspection().index_sets.system_relevant_partitioning,
        owner.get_mpi_communicator());
      ghosted = owned;
      return ghosted;
    }


    template <int dim>
    const ReconstructedFaultSurfaceResidual &
    ReconstructedFaultCondensedSystem<dim>::Linearization::surface_residual() const
    {
      assert_is_current();
      return residual;
    }


    template <int dim>
    void
    ReconstructedFaultCondensedSystem<dim>::Linearization::vmult(
      LinearAlgebra::BlockVector &result,
      const LinearAlgebra::BlockVector &direction) const
    {
      assert_is_current();

      // Apply the condensed Jacobian A-B*K_V^{-1}*G. The signs follow the
      // uncondensed fault row G*dx-K_V*dV=-R_Gamma.
      const LinearAlgebra::BlockVector constrained =
        make_constrained_solver_direction(direction);
      apply_stokes_matrix<dim>(bulk_matrix, constrained, result);

      FaultVector surface_direction;
      owner.surface_system.apply_G(make_physical_bulk_direction(constrained),
                                   surface_direction);
      FaultVector surface_solution;
      surface_solve.solve(surface_direction, surface_solution);
      LinearAlgebra::BlockVector bulk_correction(
        owner.introspection().index_sets.system_partitioning,
        owner.get_mpi_communicator());
      owner.stokes_coupling.apply_B(surface_solution, bulk_correction);
      result.block(0).add(-1.0, bulk_correction.block(0));
      result.block(1).add(-1.0, bulk_correction.block(1));
      homogeneous_stokes_constraints->set_zero(result);
      result.compress(VectorOperation::insert);
    }


    template <int dim>
    void
    ReconstructedFaultCondensedSystem<dim>::Linearization::build_condensed_rhs(
      const LinearAlgebra::BlockVector &bulk_newton_rhs,
      LinearAlgebra::BlockVector &result) const
    {
      assert_is_current();
      Assert(bulk_newton_rhs.n_blocks() >= 2, ExcInternalError());

      // ASPECT supplies -R_bulk. Condensation adds +B*K_V^{-1}*R_Gamma.
      result.block(0) = bulk_newton_rhs.block(0);
      result.block(1) = bulk_newton_rhs.block(1);

      FaultVector surface_solution;
      surface_solve.solve(surface_residual().values, surface_solution);
      LinearAlgebra::BlockVector bulk_correction(
        owner.introspection().index_sets.system_partitioning,
        owner.get_mpi_communicator());
      owner.stokes_coupling.apply_B(surface_solution, bulk_correction);
      result.block(0).add(1.0, bulk_correction.block(0));
      result.block(1).add(1.0, bulk_correction.block(1));
      homogeneous_stokes_constraints->set_zero(result);
      result.compress(VectorOperation::insert);
    }


    template <int dim>
    void
    ReconstructedFaultCondensedSystem<dim>::Linearization::
    recover_slip_rate_increment(
      const LinearAlgebra::BlockVector &bulk_increment,
      FaultVector &slip_rate_increment) const
    {
      assert_is_current();

      // Recover dV=K_V^{-1}(R_Gamma+G*dx); a restricted semantic solve
      // automatically returns exact zero on bound-active vertices.
      const LinearAlgebra::BlockVector constrained =
        make_constrained_solver_direction(bulk_increment);
      FaultVector rhs;
      owner.surface_system.apply_G(make_physical_bulk_direction(constrained), rhs);
      const FaultVector &surface_rhs = surface_residual().values;
      AssertDimension(rhs.size(), surface_rhs.size());
      for (unsigned int fault = 0; fault < rhs.size(); ++fault)
        {
          AssertDimension(rhs[fault].size(), surface_rhs[fault].size());
          for (unsigned int vertex = 0; vertex < rhs[fault].size(); ++vertex)
            rhs[fault][vertex] += surface_rhs[fault][vertex];
        }
      surface_solve.solve(rhs, slip_rate_increment);
    }


    template <int dim>
    void
    ReconstructedFaultCondensedSystem<dim>::Linearization::
    apply_uncondensed_jacobian(
      const LinearAlgebra::BlockVector &bulk_direction,
      const FaultVector &slip_rate_direction,
      LinearAlgebra::BlockVector &bulk_result,
      FaultVector &surface_result) const
    {
      assert_is_current();

      // Verification uses the original block signs [A,-B; G,-K_V] without
      // condensation, but with the same homogeneous/scaling conversions.
      const LinearAlgebra::BlockVector constrained =
        make_constrained_solver_direction(bulk_direction);
      apply_stokes_matrix<dim>(bulk_matrix, constrained, bulk_result);

      LinearAlgebra::BlockVector B_direction(
        owner.introspection().index_sets.system_partitioning,
        owner.get_mpi_communicator());
      owner.stokes_coupling.apply_B(slip_rate_direction, B_direction);
      bulk_result.block(0).add(-1.0, B_direction.block(0));
      bulk_result.block(1).add(-1.0, B_direction.block(1));
      homogeneous_stokes_constraints->set_zero(bulk_result);
      bulk_result.compress(VectorOperation::insert);

      owner.surface_system.apply_G(make_physical_bulk_direction(constrained),
                                   surface_result);
      FaultVector K_direction;
      owner.surface_system.apply_surface_jacobian(slip_rate_direction,
                                                  K_direction);
      AssertDimension(surface_result.size(), K_direction.size());
      for (unsigned int fault = 0; fault < surface_result.size(); ++fault)
        {
          AssertDimension(surface_result[fault].size(), K_direction[fault].size());
          for (unsigned int vertex = 0;
               vertex < surface_result[fault].size(); ++vertex)
            surface_result[fault][vertex] -= K_direction[fault][vertex];
        }
    }


#define INSTANTIATE(dim) template class ReconstructedFaultCondensedSystem<dim>;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
