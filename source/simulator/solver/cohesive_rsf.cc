/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/reconstructed_fault/cohesive_system.h>

namespace aspect
{
  template <int dim>
  void
  Simulator<dim>::solve_cohesive_rsf_propagation ()
  {
    if constexpr (dim == 2)
      {
        AssertThrow (parameters.cohesive_rsf_propagation && cohesive_rsf_system,
                     ExcMessage ("Native cohesive system was not selected."));
        TimerOutput::Scope timer (computing_timer, "Cohesive RSF: native mixed solve");
        // The experimental dispatch precedes the ordinary Newton setup, which
        // normally initializes this native algebraic scaling factor.
        pressure_scaling = compute_pressure_scaling_factor ();
        AssertThrow (std::isfinite (pressure_scaling) && pressure_scaling > 0.,
                     ExcMessage ("Invalid native cohesive pressure scaling."));
        const AffineConstraints<double> original_constraints (current_constraints);
        const auto original_linearization = current_linearization_point;
        const double original_normalization = last_pressure_normalization_adjustment;
        const bool original_newton = assemble_newton_stokes_system;
        // The ordinary native BCs define an FE particular increment. The
        // homogeneous Stokes inverse is reused for all remaining unknowns.
        LinearAlgebra::BlockVector particular_owned (introspection.index_sets.system_partitioning,
                                                     mpi_communicator);
        particular_owned = 0.;
        current_constraints.distribute (particular_owned);
        for (unsigned int b = 0; b < particular_owned.n_blocks (); ++b)
          if (b != introspection.block_indices.velocities)
            particular_owned.block (b) = 0.;
        LinearAlgebra::BlockVector particular_velocity (solution);
        particular_velocity = particular_owned;
        particular_owned.block (introspection.block_indices.velocities) *= time_step;
        LinearAlgebra::BlockVector particular (solution);
        particular = particular_owned;
        AffineConstraints<double> homogeneous (current_constraints);
        for (const auto &line : current_constraints.get_lines ())
          homogeneous.set_inhomogeneity (line.index, 0.);
        homogeneous.close ();
        current_constraints.copy_from (homogeneous);
        assemble_newton_stokes_system = false;
        auto restore = [&] ()
          {
            current_constraints.copy_from (original_constraints);
            current_linearization_point = original_linearization;
            last_pressure_normalization_adjustment = original_normalization;
            assemble_newton_stokes_system = original_newton;
          };
        try
          {
            if (!cohesive_rsf_system->bulk_matrix_ready ())
              {
                cohesive_rsf_system->assemble_bulk_matrix (system_matrix, pressure_scaling);
                rebuild_stokes_preconditioner = true;
                build_stokes_preconditioner ();
                rebuild_stokes_matrix = false;
              }
            auto inverse = [&] (const LinearAlgebra::BlockVector &rhs)
              {
                system_rhs = rhs;
                current_linearization_point = 0.;
                last_pressure_normalization_adjustment = 0.;
                auto field = std::make_shared<LinearAlgebra::BlockVector> ();
                field->reinit (solution);
                *field = 0.;
                if (rhs.l2_norm () != 0.)
                  solve_stokes (*field);
                return field;
              };
            if (timestep_number == 0)
              {
                AssertThrow (!cohesive_rsf_system->initialized (),
                             ExcMessage ("Native cohesive initialization published twice."));
                const auto initial = cohesive_rsf_system->initialize (particular_velocity, inverse);
                solution = *initial;
                old_solution = solution;
                old_old_solution = solution;
                restore ();
                current_linearization_point = solution;
                last_pressure_normalization_adjustment = 0.;
                return;
              }
            const auto incoming = solution;
            const auto trial = cohesive_rsf_system->solve (particular, incoming, inverse);
            solution = cohesive_rsf_system->solution (*trial);
            if (benchmark_endpoint_precommit)
              benchmark_endpoint_precommit ();
            pending_dg_endpoint_publication = [this, trial] (const bool accept)
              {
                if (accept)
                  cohesive_rsf_system->publish (trial);
              };
            restore ();
            current_linearization_point = solution;
            // This owner publishes a volume-mean-zero physical perturbation;
            // a reference inverse's pressure-guess adjustment is not history.
            last_pressure_normalization_adjustment = 0.;
          }
        catch (...)
          {
            restore ();
            throw;
          }
      }
    else
      AssertThrow (false, ExcMessage ("Native cohesive RSF pilot is two dimensional."));
  }

  template void Simulator<2>::solve_cohesive_rsf_propagation ();
  template void Simulator<3>::solve_cohesive_rsf_propagation ();
}
