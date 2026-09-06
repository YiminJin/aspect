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
    }


    template <int dim>
    typename ReconstructedFaultCondensedSystem<dim>::Linearization
    ReconstructedFaultCondensedSystem<dim>::linearize(
      const LinearAlgebra::BlockSparseMatrix &bulk_matrix,
      const LinearAlgebra::BlockVector &physical_bulk_state,
      const FaultVector &slip_rate,
      const ReconstructedFaultSurfaceLinearSolve<dim> *surface_solve)
    {
      ++active_generation;

      const ReconstructedFaultSurfaceResidual &residual =
        surface_system.linearize_surface_system(physical_bulk_state, slip_rate);
      stokes_coupling.linearize_B(physical_bulk_state);

      const auto system_constraints =
        std::make_shared<const AffineConstraints<double>>(
          make_homogeneous_constraints(this->get_current_constraints()));

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
      const ReconstructedFaultCondensedSystem<dim> &owner,
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
    make_physical_bulk_direction(
      const LinearAlgebra::BlockVector &solver_direction) const
    {
      LinearAlgebra::BlockVector owned(
        owner.introspection().index_sets.system_partitioning,
        owner.get_mpi_communicator());
      owned.block(0) = solver_direction.block(0);
      owned.block(1) = solver_direction.block(1);
      owned.block(1) *= owner.get_pressure_scaling();
      homogeneous_system_constraints->distribute(owned);
      owned.compress(VectorOperation::insert);

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
