/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/simulator.h>
#include <aspect/global.h>
#include <aspect/newton.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/solver/block_stokes_preconditioner.h>
#include <aspect/simulator/solver/stokes_matrix_free_local_smoothing.h>
#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <aspect/simulator/solver/reconstructed_fault_nonlinear.h>
#include <aspect/simulator/solver/reconstructed_fault_linear.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include "stokes_operators.h"
#include "../reconstructed_fault_residual_audit.h"
#include "../reconstructed_fault_interface_preconditioner.h"

#include <deal.II/base/signaling_nan.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_gmres.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace
  {
    using FaultVector = ReconstructedFaultVector;


    template <int dim>
    FaultVector
    current_slip_rate(const ReconstructedFaultManager<dim> &fault_manager)
    {
      FaultVector values(fault_manager.get_faults().size());
      for (unsigned int fault = 0; fault < values.size(); ++fault)
        values[fault] = fault_manager.get_slip_rate(fault);
      return values;
    }


  }



  template <int dim>
  void
  Simulator<dim>::solve_reconstructed_fault_stokes ()
  {
    unsigned int total_fault_krylov_iterations = 0;
    double minimum_fault_accepted_alpha = 1.0;
    AssertThrow(dim == 2, ExcNotImplemented());
    AssertThrow(newton_handler != nullptr,
                ExcMessage("The coupled reconstructed-fault solver requires "
                           "the Newton solver handler."));

    auto &phase_field_fault =
      Plugins::get_plugin_as_type<MaterialModel::PhaseFieldFault<dim>>(
        *material_model);

    // Complete all frozen constitutive histories before opening the nonlinear
    // V lifecycle. Only a fresh timestep-zero model may initialize missing state.
    phase_field_fault.prepare_reconstructed_fault_mechanical_solve();

    ReconstructedFaultManager<dim> &fault_manager =
      *reconstructed_fault_manager;
    ReconstructedFaultSurfaceSystem<dim> &surface_system =
      *reconstructed_fault_surface_system;
    StokesSolver::ReconstructedFaultCondensedSystem<dim> condensed_system(*this);

    // Keep the production solution immutable during Newton. working_x is the
    // last accepted bulk iterate; only convergence publishes it to solution.
    const LinearAlgebra::BlockVector production_solution(solution);
    const LinearAlgebra::BlockVector saved_linearization_point(
      current_linearization_point);
    LinearAlgebra::BlockVector working_x(current_linearization_point);
    working_x = solution;

    // A pressure shift is a surface-equation gauge only in prescribed-pressure
    // mode. Keep the adjustment private, just like the accepted bulk iterate;
    // failed trials/solves must not alter published normalization bookkeeping.
    const bool normalize_fault_pressure =
      phase_field_fault.uses_adiabatic_friction_pressure();
    double working_pressure_adjustment = last_pressure_normalization_adjustment;

    // Coupled residual evaluation temporarily changes ASPECT assembly controls.
    // Snapshot them once so both success and every exception restore the caller.
    const bool saved_assemble_fault_terms =
      assemble_reconstructed_fault_stokes_terms;
    const bool saved_assemble_newton_system = assemble_newton_stokes_system;
    const bool saved_assemble_newton_matrix = assemble_newton_stokes_matrix;
    const bool saved_rebuild_matrix = rebuild_stokes_matrix;
    const bool saved_rebuild_preconditioner = rebuild_stokes_preconditioner;
    const double saved_derivative_scaling =
      newton_handler->parameters.newton_derivative_scaling_factor;
    const AffineConstraints<double> saved_current_constraints(current_constraints);

    const unsigned int max_nonlinear_iterations =
      (pre_refinement_step < parameters.initial_adaptive_refinement)
      ? std::min(parameters.max_nonlinear_iterations,
                 parameters.max_nonlinear_iterations_in_prerefinement)
      : parameters.max_nonlinear_iterations;
    SolverControl nonlinear_solver_control(max_nonlinear_iterations,
                                           parameters.nonlinear_tolerance);

    struct CoupledResidual
    {
      double bulk_norm;
      ReconstructedFaultSurfaceResidual surface;
    };

    bool nonlinear_state_is_active = false;
    bool terminal_commit_complete = false;
    auto restore_simulator_state = [&]()
    {
      assemble_reconstructed_fault_stokes_terms = saved_assemble_fault_terms;
      assemble_newton_stokes_system = saved_assemble_newton_system;
      assemble_newton_stokes_matrix = saved_assemble_newton_matrix;
      rebuild_stokes_matrix = saved_rebuild_matrix;
      rebuild_stokes_preconditioner = saved_rebuild_preconditioner;
      newton_handler->parameters.newton_derivative_scaling_factor =
        saved_derivative_scaling;
      current_constraints.copy_from(saved_current_constraints);
    };

    try
      {
        // Open manager-owned current/trial V state and freeze the fault-to-QP
        // geometry used throughout this mechanical solve.
        fault_manager.begin_slip_rate_nonlinear_solve();
        nonlinear_state_is_active = true;
        fault_manager.prepare_stokes_qp_projection_cache();

        assemble_reconstructed_fault_stokes_terms = true;
        assemble_newton_stokes_system = true;
        // The PhaseFieldFault bulk Maxwell law is linear in the current
        // strain rate. Its nonlinear fault derivative is represented by the
        // explicit B/K_V/G blocks, not by ASPECT's viscosity-derivative output.
        newton_handler->parameters.newton_derivative_scaling_factor = 0.0;
        set_assemblers();
        // Lift the base iterate with the current physical solution constraints,
        // including changed boundary loading. Never publish this lift before
        // convergence: a failed solve must restore the pre-solve bulk solution.
        assemble_newton_stokes_system = false;
        compute_current_constraints();
        assemble_newton_stokes_system = true;
        LinearAlgebra::BlockVector lifted_solution(
          introspection.index_sets.system_partitioning, mpi_communicator);
        lifted_solution = working_x;
        current_constraints.distribute(lifted_solution);
        working_x = lifted_solution;
        if (normalize_fault_pressure)
          working_pressure_adjustment = normalize_pressure(working_x);

        // Every subsequent assembly is a Newton residual/direction problem.
        // The physical lift is already present in working_x, so eliminate with
        // homogeneous Stokes constraints to avoid subtracting it a second time.
        const types::global_dof_index n_stokes_dofs =
          working_x.block(introspection.block_indices.velocities).size()
          + working_x.block(introspection.block_indices.pressure).size();
        for (const auto &line : current_constraints.get_lines())
          if (line.index < n_stokes_dofs)
            current_constraints.set_inhomogeneity(line.index, 0.0);
        pressure_scaling = compute_pressure_scaling_factor();

        auto evaluate_coupled_residual =
          [&](const LinearAlgebra::BlockVector &bulk_state,
              const FaultVector &slip_rate) -> CoupledResidual
        {
          // Use the very same absolute V in bulk and surface evaluation. A
          // subtract/add reconstruction could lose a small bound-contact value.
          fault_manager.begin_slip_rate_trial();
          bool trial_is_active = true;
          try
            {
              fault_manager.set_slip_rate_trial_values(slip_rate);
              current_linearization_point = bulk_state;
              assemble_newton_stokes_matrix = false;
              rebuild_stokes_preconditioner = false;
              rebuild_stokes_matrix =
                !boundary_velocity_manager
                   .get_prescribed_boundary_velocity_indicators().empty();
              assemble_stokes_system();

              const double velocity_residual =
                system_rhs.block(introspection.block_indices.velocities).l2_norm();
              const double pressure_residual =
                system_rhs.block(introspection.block_indices.pressure).l2_norm();
              CoupledResidual result;
              result.bulk_norm = std::sqrt(
                velocity_residual*velocity_residual
                + pressure_residual*pressure_residual);
              result.surface = surface_system.evaluate_surface_residual(
                bulk_state, slip_rate);
              fault_manager.rollback_slip_rate_trial();
              trial_is_active = false;
              return result;
            }
          catch (...)
            {
              if (trial_is_active)
                fault_manager.rollback_slip_rate_trial();
              throw;
            }
        };

        // Freeze separate dimensional normalization scales for the entire solve.
        // Their floors reuse existing bulk and K_V action scales rather than a
        // reconstructed-fault tuning parameter.
        const FaultVector initial_slip_rate = current_slip_rate(fault_manager);
        const CoupledResidual initial_residual =
          evaluate_coupled_residual(working_x, initial_slip_rate);

        LinearAlgebra::BlockVector bulk_reference(working_x);
        bulk_reference.block(introspection.block_indices.velocities) = 0.0;
        const double aspect_bulk_reference =
          evaluate_coupled_residual(bulk_reference, initial_slip_rate).bulk_norm;

        const double scale_floor_factor = std::max(
          parameters.linear_stokes_solver_tolerance,
          std::sqrt(std::numeric_limits<double>::epsilon()));
        const double bulk_scale =
          internal::reconstructed_fault_residual_scale(
          initial_residual.bulk_norm,
          aspect_bulk_reference,
          scale_floor_factor);
        double surface_scale = numbers::signaling_nan<double>();
        double bulk_precision = 0.0;
        double bulk_convergence_scale = bulk_scale;
        double fault_preconditioner_setup_seconds = 0.;

        // Bound the continuity evaluation before cancellation, using the actual
        // FE gradients, physical base iterate and constrained left-null test.
        // The source/history/B terms have no pressure rows in this eligible path.
        auto pressure_assembly_scale = [&](const LinearAlgebra::BlockVector &q)
        {
          if (q.l2_norm()==0.)
            return 0.;
          LinearAlgebra::BlockVector owned(introspection.index_sets.system_partitioning,mpi_communicator);
          owned.block(1)=q.block(1);
          current_constraints.distribute(owned);
          LinearAlgebra::BlockVector test(introspection.index_sets.system_partitioning,
                                         introspection.index_sets.system_relevant_partitioning,mpi_communicator);
          test=owned;
          FEValues<dim> fe(*mapping,finite_element,introspection.quadratures.velocities,
                           update_values|update_gradients|update_JxW_values);
          Vector<double> u(finite_element.n_dofs_per_cell()),p(u.size());
          double local=0.;
          for (const auto &cell:dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);cell->get_dof_values(working_x,u);cell->get_dof_values(test,p);
                double origin[dim]={};
                for (unsigned int d=0;d<dim;++d)
                  for (unsigned int i=0;i<u.size();++i)
                    if (finite_element.system_to_component_index(i).first==introspection.component_indices.velocities[d])
                      {origin[d]=u[i];break;}
                for (unsigned int k=0;k<fe.n_quadrature_points;++k)
                  {
                    double pressure_test=0.,divergence_terms=0.;
                    for (unsigned int i=0;i<u.size();++i)
                      {
                        pressure_test+=std::abs(p[i]*fe[introspection.extractors.pressure].value(i,k));
                        for (unsigned int d=0;d<dim;++d)
                          if (finite_element.system_to_component_index(i).first==introspection.component_indices.velocities[d])
                            divergence_terms+=(std::abs(u[i])+std::abs(origin[d]))
                              *std::abs(fe[introspection.extractors.velocities].gradient(i,k)[d][d]);
                      }
                    local+=std::abs(pressure_scaling*fe.JxW(k))*pressure_test*divergence_terms;
                  }
              }
          return Utilities::MPI::sum(local,mpi_communicator);
        };

        auto solve_condensed_system =
          [&](const typename StokesSolver::ReconstructedFaultCondensedSystem<dim>
                      ::Linearization &linearization,
              const ReconstructedFaultActiveSet &active,
              const LinearAlgebra::BlockVector &rhs,
              LinearAlgebra::BlockVector &direction,
              const bool already_converged)
        {
          direction = 0.0;
          const double rhs_norm = rhs.l2_norm();
          if (rhs_norm == 0.0)
            return;

          double right_null_error,left_null_error;
          const auto q=linearization.verified_pressure_nullspace(right_null_error,left_null_error);
          LinearAlgebra::BlockVector compatible_rhs(rhs),residual(rhs);
          const double assembly_scale=pressure_assembly_scale(q);
          // A worst-case serial chain also bounds parallel cell/constraint
          // accumulation; the final dot product has at most two operations/row.
          const double assembly_operations=6.*finite_element.n_dofs_per_cell()
            +2.*introspection.quadratures.velocities.size()+4.*dim+16.
            +triangulation.n_global_active_cells();
          const double reduction_operations=2.*rhs.block(1).size();
          const double reduction_scale=q.block(1).linfty_norm()*rhs.block(1).l1_norm();
          const double nonlinear_target=parameters.nonlinear_tolerance*bulk_convergence_scale;
          const double compatibility_tolerance=internal::fault_pressure_compatibility_bound(
            assembly_scale,assembly_operations,reduction_scale,reduction_operations,nonlinear_target);
          if (std::getenv("ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC"))
            {
              std::ostringstream audit;
              audit<<std::setprecision(17)<<"      Fault compatibility audit: step="<<timestep_number
                <<", Newton="<<nonlinear_iteration<<", rhs null="<<q*rhs<<", rhs norm="<<rhs_norm
                <<", velocity="<<system_rhs.block(0).l2_norm()<<", continuity="<<system_rhs.block(1).l2_norm()
                <<", initial="<<initial_residual.bulk_norm<<", reference="<<aspect_bulk_reference
                <<", old roundoff="<<100.*std::numeric_limits<double>::epsilon()
                  *std::max({initial_residual.bulk_norm,aspect_bulk_reference,rhs_norm})
                <<", relative cap="<<parameters.nonlinear_tolerance*bulk_scale
                <<", bulk precision="<<bulk_precision<<", mixed target="<<nonlinear_target
                <<", assembly scale="<<assembly_scale<<", assembly operations="<<assembly_operations
                <<", reduction scale="<<reduction_scale<<", reduction operations="<<reduction_operations
                <<", compatibility bound="<<compatibility_tolerance<<", already converged="<<already_converged
                <<", pressure scaling="<<pressure_scaling<<", right null="<<right_null_error
                <<", left null="<<left_null_error<<", q norm="<<q.l2_norm()
                <<", right null scale="<<system_matrix.block(0,1).frobenius_norm()
                <<", left null scale="<<system_matrix.block(1,0).frobenius_norm();
              pcout<<audit.str()<<std::endl;
            }
          const double removed_rhs=internal::project_compatible_fault_rhs(q,compatibility_tolerance,compatible_rhs);
          // Still check compatibility, but do not solve an unused direction
          // after the unprojected bulk and all unprescribed surface rows pass.
          if (already_converged)
            return;

          TimerOutput::Scope linear_timer(computing_timer, "Fault: condensed linear solve");

          const double tolerance =
            parameters.linear_stokes_solver_tolerance*rhs_norm;
          const unsigned int budget = std::max(1U,
            parameters.n_cheap_stokes_solver_steps + parameters.n_expensive_stokes_solver_steps);
          PrimitiveVectorMemory<LinearAlgebra::BlockVector> memory;

          std::unique_ptr<internal::SchurComplementOperator> schur;
          if (parameters.use_bfbt)
            schur = std::make_unique<
              internal::WeightedBFBT<LinearAlgebra::PreconditionBase>>(
                system_preconditioner_matrix.block(1,1),
                *Mp_preconditioner,
                parameters.linear_solver_S_block_tolerance,
                inverse_lumped_mass_matrix.block(0),
                system_matrix);
          else
            schur = std::make_unique<
              internal::InverseWeightedMassMatrix<LinearAlgebra::PreconditionBase>>(
                system_preconditioner_matrix.block(1,1),
                *Mp_preconditioner,
                parameters.linear_solver_S_block_tolerance);

          const auto solve_with_velocity_preconditioner = [&](const auto &velocity_preconditioner)
          {
            internal::InverseVelocityBlock<
              std::decay_t<decltype(velocity_preconditioner)>,
              LinearAlgebra::Vector,
              LinearAlgebra::SparseMatrix> inverse_velocity(
                system_matrix.block(0,0),
                velocity_preconditioner,
                true,
                stokes_A_block_is_symmetric(),
                parameters.linear_solver_A_block_tolerance);
            const internal::BlockSchurPreconditioner<
              decltype(inverse_velocity),
              internal::SchurComplementOperator,
              LinearAlgebra::SparseMatrix,
              LinearAlgebra::BlockVector> preconditioner(
                inverse_velocity, *schur, system_matrix.block(0,1));

            // B and G are not assumed adjoints, so the condensed operator is
            // generally nonsymmetric and requires FGMRES rather than CG/MINRES.
            const internal::FaultPressureComplementOperator<
              typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization,
              LinearAlgebra::BlockVector> projected_operator{linearization, q};
            const internal::FaultPressureComplementOperator<
              decltype(preconditioner), LinearAlgebra::BlockVector>
              projected_preconditioner{preconditioner, q, true};
            const internal::FaultInterfacePreconditioner<dim,decltype(projected_preconditioner)>
              interface_preconditioner(projected_preconditioner,linearization,surface_system,active,rhs,pcout);
            const internal::FaultPressureComplementOperator<
              decltype(interface_preconditioner),LinearAlgebra::BlockVector>
              projected_interface{interface_preconditioner,q};

            unsigned int iterations = 0;
            while (iterations < budget)
              {
                SolverControl control(budget-iterations, tolerance);
                control.enable_history_data();
                SolverFGMRES<LinearAlgebra::BlockVector> solver(
                  control, memory,
                  typename SolverFGMRES<LinearAlgebra::BlockVector>::AdditionalData(
                    parameters.stokes_gmres_restart_length));
                bool solver_failed = false;
                try
                  {
                    internal::FaultLinearSection krylov_timer(internal::FaultLinearTiming::krylov_vectors);
                    solver.solve(projected_operator, direction, compatible_rhs, projected_interface);
                  }
                catch (const SolverControl::NoConvergence &)
                  {
                    solver_failed = true;
                  }
                iterations += std::max(1U, control.last_step());
                total_fault_krylov_iterations += std::max(1U, control.last_step());
                internal::project_fault_pressure(q, direction);

                // Arnoldi's residual estimate may disagree with the final vector.
                // Verify C*x-b afresh, retaining raw and null-component diagnostics.
                double raw_residual, residual_null_component;
                const double fresh = internal::fault_true_linear_residual(
                  linearization, q, direction, rhs, residual,
                  raw_residual, residual_null_component);
                std::ostringstream report;
                report << std::setprecision(17)
                       << "      Fault linear solve: iterations=" << iterations
                       << ", estimated=" << control.last_value() << ", fresh=" << fresh
                       << ", target=" << tolerance << ", raw=" << raw_residual
                       << ", rhs null=" << removed_rhs << ", residual null=" << residual_null_component
                       << ", compatibility bound=" << compatibility_tolerance
                       << ", pressure quotient=" << (q.l2_norm() > 0.)
                       << ", right null=" << right_null_error << ", left null=" << left_null_error;
                if (std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC"))
                  pcout << report.str() << std::endl;
                else
                  {
                    std::ostringstream progress;
                    progress << "      Fault linear solve: iterations=" << iterations
                             << std::scientific << std::setprecision(6)
                             << ", fresh=" << fresh << ", target=" << tolerance;
                    pcout << progress.str() << std::endl;
                  }
                AssertThrow(std::abs(residual_null_component) <= compatibility_tolerance,
                            ExcMessage("The full condensed residual has a significant pressure incompatibility."));
                if (fresh <= tolerance)
                  {
                    if (!signals.post_reconstructed_fault_linear_solver.empty())
                      signals.post_reconstructed_fault_linear_solver(
                        *this,
                        [&](auto &dst,const auto &src) { projected_operator.vmult(dst,src); },
                        [&](auto &dst,const auto &src) { projected_interface.vmult(dst,src); },
                        [&](auto &dst,const auto &src) { schur->vmult(dst,src); },
                        compatible_rhs,direction,tolerance,budget,fault_preconditioner_setup_seconds);
                    return;
                  }
                if (solver_failed || iterations >= budget)
                  throw SolverControl::NoConvergence(iterations, fresh);
                // Re-enter FGMRES from this vector with its freshly evaluated
                // residual, charging every restart to the same total budget.
              }
          };

          if (parameters.stokes_solver_type == Parameters<dim>::StokesSolverType::block_amg)
            solve_with_velocity_preconditioner(*Amg_preconditioner);
          else
            {
              AssertThrow(parameters.stokes_velocity_degree==2 && stokes_A_block_is_symmetric(),
                          ExcMessage("The fault velocity-GMG preconditioner requires symmetric Q2 bulk Stokes."));
              StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2> gmg(*this,parameters);
              gmg.initialize_simulator(*this);
              gmg.initialize();
              gmg.with_velocity_preconditioner([&](const auto &cycle)
              {
                // Only adapt vector storage. The approximate velocity inverse
                // still applies the original assembled fine-level A matrix.
                struct Adapter
                {
                  const typename StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2>::VelocityCycle &cycle;
                  mutable dealii::LinearAlgebra::distributed::Vector<double> input,output;
                  void vmult(LinearAlgebra::Vector &dst,const LinearAlgebra::Vector &src) const
                  {
                    internal::ChangeVectorTypes::copy(input,src);
                    cycle.vmult(output,input);
                    internal::ChangeVectorTypes::copy(dst,output);
                  }
                } adapter{cycle,
                  dealii::LinearAlgebra::distributed::Vector<double>(rhs.block(0).locally_owned_elements(),mpi_communicator),
                  dealii::LinearAlgebra::distributed::Vector<double>(rhs.block(0).locally_owned_elements(),mpi_communicator)};
                solve_with_velocity_preconditioner(adapter);
              });
            }
        };

        for (nonlinear_iteration = 0;
             nonlinear_iteration < max_nonlinear_iterations;
             ++nonlinear_iteration)
          {
            // Assemble mutually consistent A, R_bulk, frozen B, G, K_V, and
            // R_Gamma at the current accepted pair (working_x,current V).
            current_linearization_point = working_x;
            assemble_newton_stokes_matrix = true;
            rebuild_stokes_matrix = true;
            rebuild_stokes_preconditioner = true;
            assemble_stokes_system();
            const auto preconditioner_start=internal::FaultLinearTiming::Clock::now();
            build_stokes_preconditioner();
            fault_preconditioner_setup_seconds=std::chrono::duration<double>(
              internal::FaultLinearTiming::Clock::now()-preconditioner_start).count();

            const FaultVector slip_rate = current_slip_rate(fault_manager);
            internal::FaultLinearProfile linear_profile(pcout, timestep_number, nonlinear_iteration);
            using CondensedLinearization =
              typename StokesSolver::ReconstructedFaultCondensedSystem<dim>
                ::Linearization;
            auto linearization = std::make_unique<CondensedLinearization>(
              condensed_system.linearize(system_matrix, working_x, slip_rate));

            ReconstructedFaultActiveSet active_set =
              fault_manager.prescribed_slip_rate_mask();
            std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
              restricted_surface_solve;
            // Prescribed rates are already lifted into the base iterate. Their
            // perturbations vanish, so condensation uses K_FF^{-1} from the
            // first solve, not after an unrestricted direction has been taken.
            bool has_prescribed_vertices = false;
            for (unsigned int f = 0; f < active_set.size(); ++f)
              for (unsigned int v = 0; v < active_set[f].size(); ++v)
                if (active_set[f][v])
                  {
                    has_prescribed_vertices = true;
                    AssertThrow(slip_rate[f][v] >= phase_field_fault.minimum_fault_slip_rate(),
                                ExcMessage("Prescribed V is below the material's minimum slip rate."));
                  }
            if (has_prescribed_vertices)
              {
                restricted_surface_solve = surface_system.create_restricted_linear_solve(active_set);
                linearization = std::make_unique<CondensedLinearization>(
                  linearization->with_surface_solve(*restricted_surface_solve));
              }
            LinearAlgebra::BlockVector bulk_rhs(
              introspection.index_sets.stokes_partitioning, mpi_communicator);
            LinearAlgebra::BlockVector bulk_direction(
              introspection.index_sets.stokes_partitioning, mpi_communicator);
            FaultVector slip_rate_direction;
            if (nonlinear_iteration == 0)
              {
                // Fix the attainable bulk accuracy from A and the represented
                // initial state, not from stalled residuals. The same mixed
                // absolute/relative scale is used by convergence and merit.
                LinearAlgebra::BlockVector solver_state(
                  introspection.index_sets.stokes_partitioning, mpi_communicator);
                solver_state.block(0) = working_x.block(introspection.block_indices.velocities);
                solver_state.block(1) = working_x.block(introspection.block_indices.pressure);
                solver_state.block(1) /= pressure_scaling;
                bulk_precision = internal::reconstructed_fault_bulk_precision_scale(
                  system_matrix, solver_state, mpi_communicator);
                bulk_convergence_scale = bulk_scale + bulk_precision/parameters.nonlinear_tolerance;

                // Establish the surface reference before the first direction;
                // the stabilized-free-set RMS completes this scale below.
                // A characteristic K_V action supplies a physical traction scale.
                // Do not suppress it to roundoff: an initially balanced surface
                // still develops second-order residuals when bulk loading changes.
                FaultVector characteristic_slip_rate = slip_rate;
                for (auto &fault_values : characteristic_slip_rate)
                  for (double &value : fault_values)
                    value = std::max(phase_field_fault.minimum_fault_slip_rate(),
                                     std::abs(value));
                FaultVector characteristic_surface_action;
                surface_system.apply_surface_jacobian(
                  characteristic_slip_rate, characteristic_surface_action);
                ReconstructedFaultSurfaceResidual characteristic_residual;
                characteristic_residual.values =
                  std::move(characteristic_surface_action);
                const ReconstructedFaultActiveSet no_active_vertices =
                  fault_manager.prescribed_slip_rate_mask();
                const double surface_reference = std::max(
                  surface_system.surface_residual_rms(
                    linearization->surface_residual(), no_active_vertices),
                  surface_system.surface_residual_rms(
                    characteristic_residual, no_active_vertices));
                surface_scale = surface_reference;
              }


            const bool already_converged =
              std::hypot(system_rhs.block(0).l2_norm(),system_rhs.block(1).l2_norm())
                < parameters.nonlinear_tolerance*bulk_convergence_scale
              && internal::normalized_reconstructed_fault_residual(
                   surface_system.surface_residual_rms(linearization->surface_residual(),active_set),
                   surface_scale,"surface") < parameters.nonlinear_tolerance;

            // Projected Newton solve: start free, add only at-bound vertices
            // whose direction is outward, and rebuild only K_FF^{-1} until stable.
            while (true)
              {
                linearization->build_condensed_rhs(system_rhs, bulk_rhs);
                solve_condensed_system(*linearization, active_set, bulk_rhs, bulk_direction, already_converged);
                if (already_converged)
                  {
                    slip_rate_direction=slip_rate;
                    for (auto &values:slip_rate_direction)
                      std::fill(values.begin(),values.end(),0.);
                    break;
                  }
                linearization->recover_slip_rate_increment(
                  bulk_direction, slip_rate_direction);

                const unsigned int n_new_active_vertices =
                  internal::update_reconstructed_fault_active_set(
                    slip_rate,
                    slip_rate_direction,
                    phase_field_fault.minimum_fault_slip_rate(),
                    active_set);
                if (n_new_active_vertices == 0)
                  break;

                auto new_surface_solve =
                  surface_system.create_restricted_linear_solve(active_set);
                auto new_linearization =
                  std::make_unique<CondensedLinearization>(
                    linearization->with_surface_solve(*new_surface_solve));
                linearization = std::move(new_linearization);
                restricted_surface_solve = std::move(new_surface_solve);
              }

            linear_profile.report();

            // Active residual entries do not participate in convergence or the
            // merit function; the bulk and free-surface blocks remain separate.
            const double current_velocity_norm =
              system_rhs.block(introspection.block_indices.velocities).l2_norm();
            const double current_pressure_norm =
              system_rhs.block(introspection.block_indices.pressure).l2_norm();
            const double current_bulk_norm = std::sqrt(
              current_velocity_norm*current_velocity_norm
              + current_pressure_norm*current_pressure_norm);
            const double current_surface_norm =
              surface_system.surface_residual_rms(
                linearization->surface_residual(), active_set);

            // Preserve the original stabilized-free-set RMS floor. Removing
            // active rows also changes its mass normalization, so it can exceed
            // the preliminary all-unprescribed-row RMS used for the early check.
            if (nonlinear_iteration == 0)
              surface_scale = std::max(surface_scale,current_surface_norm);


            const double relative_bulk_residual =
              internal::normalized_reconstructed_fault_residual(
                current_bulk_norm, bulk_convergence_scale, "bulk");
            const double relative_surface_residual =
              internal::normalized_reconstructed_fault_residual(
                current_surface_norm, surface_scale, "surface");
            {
              std::ostringstream progress;
              progress << "      Relative nonlinear residuals (bulk, fault) after "
                       << "nonlinear iteration " << std::setw(2) << nonlinear_iteration << ": "
                       << std::scientific << std::setprecision(6)
                       << relative_bulk_residual << ", " << relative_surface_residual;
              pcout << progress.str() << std::endl;
            }
            if (std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC"))
            {
              std::ostringstream report;
              report << std::setprecision(17)
                     << "      Fault nonlinear residual: bulk=" << current_bulk_norm
                     << ", bulk scale=" << bulk_scale << ", surface=" << current_surface_norm
                     << ", surface scale=" << surface_scale
                     << ", velocity=" << current_velocity_norm << ", scaled continuity=" << current_pressure_norm
                     << ", bulk precision=" << bulk_precision
                     << ", bulk target=" << parameters.nonlinear_tolerance*bulk_convergence_scale
                     << ", velocity correction=" << bulk_direction.block(0).linfty_norm();
              pcout << report.str() << std::endl;
            }

            const double maximum_step_length =
              internal::reconstructed_fault_maximum_step_length(
                slip_rate, slip_rate_direction, active_set,
                phase_field_fault.minimum_fault_slip_rate());

            // Observational bound probe: hold the current bulk iterate fixed
            // and set every unprescribed rate to V_min. This is a weak F(V_min)
            // diagnostic, not an independent scalar root or an accepted trial.
            const bool bound_audit = std::getenv("ASPECT_FAULT_NONLINEAR_DIAGNOSTIC") != nullptr;
            if (bound_audit)
              {
                const auto prescribed = fault_manager.prescribed_slip_rate_mask();
                FaultVector lower_rates = slip_rate;
                for (unsigned int f = 0; f < lower_rates.size(); ++f)
                  for (unsigned int v = 0; v < lower_rates[f].size(); ++v)
                    if (!prescribed[f][v])
                      lower_rates[f][v] = phase_field_fault.minimum_fault_slip_rate();
                const auto lower_residual = surface_system.evaluate_surface_residual(working_x, lower_rates);
                unsigned int free = 0, lower_active = 0, prefers_lower = 0;
                double minimum = std::numeric_limits<double>::max(), minimum_free = minimum;
                std::ofstream out;
                if (pcout.is_active())
                  {
                    out.open(parameters.output_directory + "nonlinear_bounds_"
                             + Utilities::int_to_string(timestep_number) + ".csv",
                             nonlinear_iteration == 0 ? std::ios::out : std::ios::app);
                    if (nonlinear_iteration == 0)
                      out << "iteration,fault,vertex,V,dV,prescribed,lower_active,Fmin_weak_density,alpha_max,bulk,surface\n";
                    out << std::setprecision(17);
                  }
                for (unsigned int f = 0; f < slip_rate.size(); ++f)
                  for (unsigned int v = 0; v < slip_rate[f].size(); ++v)
                    {
                      double mass = lower_residual.mass_diagonal[f][v];
                      if (v > 0) mass += lower_residual.mass_off_diagonal[f][v-1];
                      if (v+1 < slip_rate[f].size()) mass += lower_residual.mass_off_diagonal[f][v];
                      const double density = lower_residual.values[f][v]/mass;
                      if (!prescribed[f][v])
                        {
                          minimum = std::min(minimum, slip_rate[f][v]);
                          lower_active += active_set[f][v];
                          free += !active_set[f][v];
                          // R = shear - resistance; a negative F_min requests
                          // still lower rates when the local tangent is negative.
                          prefers_lower += density < 0.0;
                          if (!active_set[f][v]) minimum_free = std::min(minimum_free, slip_rate[f][v]);
                        }
                      if (pcout.is_active())
                        out << nonlinear_iteration << ',' << f << ',' << v << ',' << slip_rate[f][v]
                            << ',' << slip_rate_direction[f][v] << ',' << prescribed[f][v]
                            << ',' << (active_set[f][v] && !prescribed[f][v]) << ',' << density
                            << ',' << maximum_step_length << ',' << current_bulk_norm << ','
                            << current_surface_norm << '\n';
                    }
                pcout << "      Fault bound audit: free=" << free << ", lower-active=" << lower_active
                      << ", min V=" << minimum << ", min free V=" << minimum_free
                      << ", negative Fmin=" << prefers_lower << ", alpha_max=" << maximum_step_length
                      << std::endl;
              }

            if (relative_bulk_residual < parameters.nonlinear_tolerance
                && relative_surface_residual < parameters.nonlinear_tolerance)
              {
                // Observe accepted physical slip with the still-frozen history.
                // This opt-in standalone evaluation writes separate QP moments;
                // its residual is discarded and never enters the solve.
                if (std::getenv("ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC"))
                  {
                    LinearAlgebra::BlockVector diagnostic_residual(system_rhs);
                    reconstructed_fault_stokes_coupling->evaluate_slip_dependent_bulk_residual(
                      working_x, slip_rate, diagnostic_residual);
                  }


                // Allocate and validate the complete accepted publication
                // state before the first constitutive or kinematic write.
                LinearAlgebra::BlockVector accepted_solution(solution);
                accepted_solution.block(introspection.block_indices.velocities) =
                  working_x.block(introspection.block_indices.velocities);
                accepted_solution.block(introspection.block_indices.pressure) =
                  working_x.block(introspection.block_indices.pressure);
                LinearAlgebra::BlockVector accepted_linearization(working_x);
                fault_manager.validate_slip_rate_nonlinear_commit();
                nonlinear_solver_control.check(
                  nonlinear_iteration,
                  std::max(relative_bulk_residual,
                           relative_surface_residual));
                restore_simulator_state();

                // The history operation performs all failure-capable work
                // before its writes. Everything that follows is a fixed-size,
                // non-allocating terminal mutation of one accepted state.
                phase_field_fault.commit_reconstructed_fault_mechanical_history(
                  accepted_solution);
                fault_manager.commit_slip_rate_nonlinear_solve();
                solution.swap(accepted_solution);
                current_linearization_point.swap(accepted_linearization);
                last_pressure_normalization_adjustment = working_pressure_adjustment;
                nonlinear_state_is_active = false;
                terminal_commit_complete = true;
                signals.post_reconstructed_fault_solver(
                  nonlinear_iteration, total_fault_krylov_iterations,
                  minimum_fault_accepted_alpha, active_set);
                signals.post_nonlinear_solver(nonlinear_solver_control);
                return;
              }

            const double current_merit = 0.5*(
              relative_bulk_residual*relative_bulk_residual
              + relative_surface_residual*relative_surface_residual);

            // Limit only free downward directions and allow exact arrival at
            // V_min; active outward directions were already removed by K_FF.
            const LinearAlgebra::BlockVector physical_bulk_direction =
              linearization->make_physical_bulk_direction(bulk_direction);

            // Trial arithmetic uses owned vectors; residual point evaluation
            // receives a ghosted physical-pressure vector after the update.
            LinearAlgebra::BlockVector owned_physical_bulk_direction(
              introspection.index_sets.system_partitioning,
              mpi_communicator);
            owned_physical_bulk_direction = physical_bulk_direction;

            LinearAlgebra::BlockVector accepted_trial_x(working_x);
            double accepted_trial_pressure_adjustment = working_pressure_adjustment;
            FaultVector accepted_trial_slip_rate;

            // Bounded K1-style affine audit: freeze A before any residual-only
            // assembly can overwrite it. B/G/K_V retain their original caches.
            // Only iteration 0 is instrumented; no solver budget is changed.
            const bool audit = std::getenv("ASPECT_K1_FLOOR_AUDIT") != nullptr
                               && nonlinear_iteration == 0;
            LinearAlgebra::BlockSparseMatrix audit_matrix;
            LinearAlgebra::BlockVector audit_rhs, audit_unknowns, audit_frozen;
            auto restore_audit_matrix = [&]()
            {
              for (unsigned int i = 0; i < 2; ++i)
                for (unsigned int j = 0; j < 2; ++j)
                  system_matrix.block(i,j).copy_from(audit_matrix.block(i,j));
            };
            auto audit_channel = [&](const LinearAlgebra::BlockVector &x,
                                     const FaultVector &v,
                                     const internal::FaultResidualAuditChannel channel)
            {
              internal::fault_residual_audit_channel = channel;
              try
                {
                  evaluate_coupled_residual(x, v);
                }
              catch (...)
                {
                  internal::fault_residual_audit_channel =
                    internal::FaultResidualAuditChannel::normal;
                  throw;
                }
              internal::fault_residual_audit_channel =
                internal::FaultResidualAuditChannel::normal;
              restore_audit_matrix();
              return LinearAlgebra::BlockVector(system_rhs);
            };
            if (audit)
              {
                audit_matrix.reinit(2,2);
                for (unsigned int i = 0; i < 2; ++i)
                  for (unsigned int j = 0; j < 2; ++j)
                    audit_matrix.block(i,j).copy_from(system_matrix.block(i,j));
                audit_matrix.collect_sizes();
                audit_rhs = system_rhs;
                audit_unknowns = audit_channel(working_x, slip_rate,
                                               internal::FaultResidualAuditChannel::unknowns);
                audit_frozen = audit_channel(working_x, slip_rate,
                                             internal::FaultResidualAuditChannel::frozen);
                system_rhs = audit_rhs;
                unsigned int active = 0, total = 0;
                for (const auto &fault : active_set)
                  for (const bool value : fault)
                    {
                      active += value;
                      ++total;
                    }
                pcout << "      Affine audit active=" << active
                      << " free=" << total-active << std::endl;
              }
            const auto line_search_result =
              internal::reconstructed_fault_armijo_line_search(
                maximum_step_length,
                newton_handler->parameters.max_newton_line_search_iterations,
                current_merit,
                [&](const double step_length)
                {
                  // Every candidate is reconstructed from the same accepted base,
                  // and the stabilized active set remains fixed across the search.
                  LinearAlgebra::BlockVector owned_trial_x(
                    introspection.index_sets.system_partitioning,
                    mpi_communicator);
                  owned_trial_x = working_x;
                  owned_trial_x.block(introspection.block_indices.velocities).add(
                    step_length,
                    owned_physical_bulk_direction.block(
                      introspection.block_indices.velocities));
                  owned_trial_x.block(introspection.block_indices.pressure).add(
                    step_length,
                    owned_physical_bulk_direction.block(
                      introspection.block_indices.pressure));
                  owned_trial_x.compress(VectorOperation::insert);
                  LinearAlgebra::BlockVector trial_x(
                    introspection.index_sets.system_partitioning,
                    introspection.index_sets.system_relevant_partitioning,
                    mpi_communicator);
                  trial_x = owned_trial_x;

                  // Normalize the physical candidate before residual/merit
                  // evaluation, not the homogeneous Newton direction. This
                  // removes arbitrary pressure offsets from cancellation in
                  // bulk assembly; true-pressure friction is left unchanged.
                  const double trial_pressure_adjustment = normalize_fault_pressure
                    ? normalize_pressure(trial_x) : working_pressure_adjustment;

                  FaultVector trial_slip_rate = slip_rate;
                  for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
                    for (unsigned int vertex = 0;
                         vertex < slip_rate[fault].size(); ++vertex)
                      {
                        trial_slip_rate[fault][vertex] = internal::reconstructed_fault_trial_value(
                          slip_rate[fault][vertex], slip_rate_direction[fault][vertex],
                          step_length, phase_field_fault.minimum_fault_slip_rate());
                      }

                  const CoupledResidual trial_residual =
                    evaluate_coupled_residual(trial_x, trial_slip_rate);
                  if (audit)
                    {
                      // Use the represented, pressure-normalized update, with
                      // homogeneous constraint rows removed and p in solver units.
                      const LinearAlgebra::BlockVector fresh_rhs(system_rhs);
                      LinearAlgebra::BlockVector actual(owned_trial_x);
                      actual = trial_x;
                      LinearAlgebra::BlockVector owned_base(owned_trial_x);
                      owned_base = working_x;
                      actual -= owned_base;
                      current_constraints.set_zero(actual);
                      LinearAlgebra::BlockVector dx(bulk_direction), action(bulk_direction);
                      dx.block(0) = actual.block(introspection.block_indices.velocities);
                      dx.block(1) = actual.block(introspection.block_indices.pressure);
                      dx.block(1) /= pressure_scaling;
                      internal::StokesBlock(audit_matrix).vmult(action, dx);
                      LinearAlgebra::BlockVector requested(bulk_direction);
                      internal::StokesBlock(audit_matrix).vmult(requested, bulk_direction);
                      requested *= step_length;
                      FaultVector dv = trial_slip_rate;
                      double v_representation_error = 0.0, dv_max = 0.0;
                      for (unsigned int f = 0; f < dv.size(); ++f)
                        for (unsigned int i = 0; i < dv[f].size(); ++i)
                          {
                            const double represented = slip_rate[f][i]
                              + (trial_slip_rate[f][i]-slip_rate[f][i]);
                            v_representation_error = std::max(v_representation_error,
                              std::abs(represented-trial_slip_rate[f][i]));
                            dv[f][i] = represented-slip_rate[f][i];
                            dv_max = std::max(dv_max, std::abs(dv[f][i]));
                          }
                      LinearAlgebra::BlockVector b_action(system_rhs);
                      reconstructed_fault_stokes_coupling->apply_B(dv, b_action);
                      const auto fresh_unknowns = audit_channel(trial_x, trial_slip_rate,
                        internal::FaultResidualAuditChannel::unknowns);
                      const auto fresh_frozen = audit_channel(trial_x, trial_slip_rate,
                        internal::FaultResidualAuditChannel::frozen);
                      // A block action may nearly cancel (notably continuity).
                      // Reuse the absolute-row-sum precision bound on this
                      // represented direction, not the small cancelled action.
                      const double action_precision =
                        internal::reconstructed_fault_bulk_precision_scale(
                          audit_matrix, dx, mpi_communicator);
                      for (unsigned int block = 0; block < 2; ++block)
                        {
                          // system_rhs=-R: affine prediction is rhs-A dx+B dV.
                          auto predicted = audit_rhs.block(block);
                          predicted -= action.block(block);
                          predicted += b_action.block(block);
                          auto error = fresh_rhs.block(block);
                          error -= predicted;
                          auto representation = action.block(block);
                          representation -= requested.block(block);
                          auto unknowns_error = fresh_unknowns.block(block);
                          unknowns_error -= audit_unknowns.block(block);
                          unknowns_error += action.block(block);
                          unknowns_error -= b_action.block(block);
                          auto frozen_error = fresh_frozen.block(block);
                          frozen_error -= audit_frozen.block(block);
                          const double accuracy = 1e-10*std::max({
                            audit_rhs.block(block).l2_norm(), action.block(block).l2_norm(),
                            b_action.block(block).l2_norm()})
                            + 32.0*action_precision
                            + 32.0*std::numeric_limits<double>::epsilon()
                              * audit_frozen.block(block).l2_norm();
                          AssertThrow(error.l2_norm() <= accuracy
                                      && frozen_error.l2_norm() == 0.0,
                                      ExcMessage("The represented coupled increment failed "
                                                 "the affine residual consistency regression."));
                          std::ostringstream report;
                          report << std::setprecision(17)
                            << "      Affine audit alpha=" << step_length << " block=" << block
                            << " base=" << audit_rhs.block(block).l2_norm()
                            << " predicted=" << predicted.l2_norm()
                            << " fresh=" << fresh_rhs.block(block).l2_norm()
                            << " affine_error=" << error.l2_norm()
                            << " represented_action_error=" << representation.l2_norm()
                            << " unknowns_base=" << audit_unknowns.block(block).l2_norm()
                            << " frozen_base=" << audit_frozen.block(block).l2_norm()
                            << " unknowns_affine_error=" << unknowns_error.l2_norm()
                            << " frozen_load_change=" << frozen_error.l2_norm()
                            << " test_accuracy=" << accuracy
                            << " B_dV=" << b_action.block(block).l2_norm()
                            << " dV_max=" << dv_max
                            << " V_representation_error=" << v_representation_error;
                          pcout << report.str() << std::endl;
                        }

                      // A separate admissible V probe is never an accepted
                      // candidate. It prevents an all-active solve from hiding
                      // accidental freezing of BV in either accumulator.
                      FaultVector probe_v = trial_slip_rate, probe_dv = trial_slip_rate;
                      for (unsigned int f = 0; f < probe_v.size(); ++f)
                        for (unsigned int i = 0; i < probe_v[f].size(); ++i)
                          {
                            probe_v[f][i] += 1e-12*(1.0+0.1*i);
                            probe_dv[f][i] = (slip_rate[f][i]
                              + (probe_v[f][i]-slip_rate[f][i]))
                              - (slip_rate[f][i]
                                 + (trial_slip_rate[f][i]-slip_rate[f][i]));
                          }
                      evaluate_coupled_residual(trial_x, probe_v);
                      const LinearAlgebra::BlockVector probe_rhs(system_rhs);
                      reconstructed_fault_stokes_coupling->apply_B(probe_dv, b_action);
                      const auto probe_frozen = audit_channel(trial_x, probe_v,
                        internal::FaultResidualAuditChannel::frozen);
                      AssertThrow(b_action.l2_norm() > 0.0,
                                  ExcMessage("The nonzero-V probe must have a nonzero weak action."));
                      for (unsigned int block = 0; block < 2; ++block)
                        {
                          auto error = probe_rhs.block(block);
                          error -= fresh_rhs.block(block);
                          error -= b_action.block(block);
                          auto frozen_change = probe_frozen.block(block);
                          frozen_change -= fresh_frozen.block(block);
                          const double accuracy = 1e-10*b_action.block(block).l2_norm()
                            + 32.0*std::numeric_limits<double>::epsilon()
                              * fresh_frozen.block(block).l2_norm();
                          AssertThrow(error.l2_norm() <= accuracy
                                      && frozen_change.l2_norm() == 0.0,
                                      ExcMessage("The nonzero-V probe changed the frozen load "
                                                 "or disagreed with B."));
                          std::ostringstream report;
                          report << std::setprecision(17)
                            << "      Affine nonzero-V probe: block=" << block
                            << " B_dV=" << b_action.block(block).l2_norm()
                            << " error=" << error.l2_norm()
                            << " frozen_change=" << frozen_change.l2_norm()
                            << " test_accuracy=" << accuracy;
                          pcout << report.str() << std::endl;
                        }
                      // Shadow assembly is observational: restore the actual
                      // production residual and the original linearization.
                      system_rhs = fresh_rhs;
                    }
                  const double trial_relative_bulk =
                    internal::normalized_reconstructed_fault_residual(
                      trial_residual.bulk_norm, bulk_convergence_scale, "bulk");
                  const double trial_relative_surface =
                    internal::normalized_reconstructed_fault_residual(
                      surface_system.surface_residual_rms(
                        trial_residual.surface, active_set),
                      surface_scale,
                      "surface");
                  const double trial_merit = 0.5*(
                    trial_relative_bulk*trial_relative_bulk
                    + trial_relative_surface*trial_relative_surface);
                  if (audit || bound_audit)
                    {
                      std::ostringstream report;
                      report << std::setprecision(17)
                        << (audit ? "      Affine audit merit: alpha=" : "      Fault trial merit: alpha=") << step_length
                        << " bulk_relative=" << trial_relative_bulk
                        << " surface_relative=" << trial_relative_surface
                        << " current=" << current_merit << " trial=" << trial_merit;
                      pcout << report.str() << std::endl;
                    }
                  accepted_trial_x = trial_x;
                  accepted_trial_pressure_adjustment = trial_pressure_adjustment;
                  accepted_trial_slip_rate = std::move(trial_slip_rate);
                  return trial_merit;
                },
                [&](const double)
                {
                  // Accepting replaces manager current V and working_x; the
                  // timestep-committed V still changes only at convergence.
                  fault_manager.begin_slip_rate_trial();
                  fault_manager.set_slip_rate_trial_values(accepted_trial_slip_rate);
                  fault_manager.accept_slip_rate_trial();
                  working_x = accepted_trial_x;
                  working_pressure_adjustment = accepted_trial_pressure_adjustment;
                });

            if (!line_search_result.accepted)
              {
                pcout << "   Coupled reconstructed-fault Newton line search "
                      << "exhausted all admissible candidates." << std::endl;
                throw ExcNonlinearSolverNoConvergence();
              }
            pcout << "      Reconstructed-fault line search accepted after "
                  << line_search_result.rejected_candidates
                  << " rejected candidates; alpha=" << line_search_result.step_length << "." << std::endl;
            minimum_fault_accepted_alpha = std::min(minimum_fault_accepted_alpha,
                                                     line_search_result.step_length);
          }

        nonlinear_solver_control.check(max_nonlinear_iterations,
                                       std::numeric_limits<double>::max());
        AssertThrow(false, ExcNonlinearSolverNoConvergence());
      }
    catch (...)
      {
        if (terminal_commit_complete)
          throw;
        // Any failure restores both externally visible bulk state and committed
        // manager V; constitutive histories were not yet made mutable.
        if (nonlinear_state_is_active)
          fault_manager.rollback_slip_rate_nonlinear_solve();
        solution = production_solution;
        current_linearization_point = saved_linearization_point;
        restore_simulator_state();
        nonlinear_solver_control.check(max_nonlinear_iterations,
                                       std::numeric_limits<double>::max());
        signals.post_nonlinear_solver(nonlinear_solver_control);
        throw;
      }
  }

}

// Explicit instantiation of the function implemented in this file.
namespace aspect
{
#define INSTANTIATE(dim) \
  template void Simulator<dim>::solve_reconstructed_fault_stokes ();

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
