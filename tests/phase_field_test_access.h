#ifndef _aspect_tests_phase_field_test_access_h
#define _aspect_tests_phase_field_test_access_h

#include <aspect/phase_field.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace internal
  {
    template <int dim>
    class PhaseFieldTestAccess
    {
      public:
        static void endpoint_trial(Simulator<dim> &simulator,
                                   const LinearAlgebra::BlockVector &guess,
                                   ReconstructedFaultVector &rates)
        { simulator.compute_current_constraints();simulator.solve_reconstructed_fault_endpoint_phase(false,&guess,&rates); }
        static auto endpoint_defects(Simulator<dim>&simulator,const LinearAlgebra::BlockVector &bulk,
                                    const ReconstructedFaultVector &rates)
        { return simulator.endpoint_mechanical_defects(bulk,rates); }
        static std::function<void()> diagnostic_endpoint_clock(Simulator<dim> &simulator,double time,double dt)
        {
          const auto old_time=simulator.time,old_dt=simulator.time_step;
          const auto old_step=simulator.timestep_number;
          const auto old_iteration=simulator.nonlinear_iteration;
          const auto old_pressure=simulator.last_pressure_normalization_adjustment;
          const auto old_solution=simulator.old_solution;
          const auto old_statistics=simulator.statistics;
          simulator.time=time;simulator.time_step=dt;simulator.timestep_number=1;simulator.nonlinear_iteration=0;
          simulator.endpoint_velocity_inverse={};simulator.endpoint_bulk_direct.reset();simulator.rebuild_stokes_matrix=true;simulator.rebuild_stokes_preconditioner=true;
          return [&simulator,old_time,old_dt,old_step,old_iteration,old_pressure,old_solution,old_statistics]()
            {
              simulator.time=old_time;simulator.time_step=old_dt;simulator.timestep_number=old_step;
              simulator.nonlinear_iteration=old_iteration;simulator.last_pressure_normalization_adjustment=old_pressure;
              simulator.old_solution=old_solution;simulator.endpoint_velocity_inverse={};simulator.endpoint_bulk_direct.reset();
              simulator.statistics=old_statistics;
              simulator.rebuild_stokes_matrix=true;simulator.rebuild_stokes_preconditioner=true;
            };
        }
        static void diagnostic_old_phase(Simulator<dim> &simulator,const LinearAlgebra::BlockVector &reference)
        { simulator.old_solution.block(simulator.introspection.variable("phase_field").block_index)=reference.block(simulator.introspection.variable("phase_field").block_index); }
        static void complete_step(Simulator<dim> &simulator)
        { simulator.compute_current_constraints();simulator.solve_timestep(); }
        static void endpoint_failure_hook(Simulator<dim> &simulator,std::function<void()> hook)
        { simulator.benchmark_endpoint_precommit=std::move(hook); }
        static auto endpoint_bookkeeping(const Simulator<dim> &simulator)
        {
          std::ostringstream statistics;simulator.statistics.write_text(statistics);
          return std::make_tuple(simulator.time,simulator.time_step,simulator.old_time_step,
                                 simulator.timestep_number,simulator.nonlinear_iteration,
                                 simulator.last_pressure_normalization_adjustment,statistics.str());
        }
        static auto endpoint_tangent(Simulator<dim> &simulator,
                                     const LinearAlgebra::BlockVector &bulk,
                                     const ReconstructedFaultVector &slip,
                                     const LinearAlgebra::BlockVector &strain,
                                     const std::vector<double> &Z)
        { return simulator.make_reconstructed_fault_endpoint_tangent(bulk,slip,strain,Z); }
        static double set_stokes_tolerance(Simulator<dim> &simulator,double value)
        { const double old=simulator.parameters.linear_stokes_solver_tolerance;
          simulator.parameters.linear_stokes_solver_tolerance=value;return old; }
        static void mechanical_trial(Simulator<dim> &simulator,ReconstructedFaultVector &rates)
        {simulator.compute_current_constraints();simulator.solve_reconstructed_fault_stokes(false,&rates);}
        static unsigned int endpoint_linear_solve(Simulator<dim> &simulator,
          const typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization &linearization,
          const ReconstructedFaultActiveSet &active,const LinearAlgebra::BlockVector &rhs,
          const LinearAlgebra::BlockVector &bulk,LinearAlgebra::BlockVector &direction,bool direct)
        {
          const bool previous=simulator.phase_field_handler->solver_parameters.endpoint_direct_mechanics;
          simulator.phase_field_handler->solver_parameters.endpoint_direct_mechanics=direct;
          const AffineConstraints<double> saved(simulator.current_constraints);
          for(const auto &line:simulator.current_constraints.get_lines())simulator.current_constraints.set_inhomogeneity(line.index,0.);
          const double norm=rhs.l2_norm();
          typename Simulator<dim>::ReconstructedFaultLinearSolveScales scales{norm,norm,norm,0.,norm};
          unsigned int iterations=0;
          try{simulator.solve_reconstructed_fault_condensed_system(linearization,active,rhs,bulk,scales,false,0.,direction,iterations);}
          catch(...){simulator.current_constraints.copy_from(saved);simulator.phase_field_handler->solver_parameters.endpoint_direct_mechanics=previous;throw;}
          simulator.current_constraints.copy_from(saved);
          simulator.phase_field_handler->solver_parameters.endpoint_direct_mechanics=previous;
          return iterations;
        }
        // Native material functions for the independent integration reference.
        // Caller supplies actual particle fractions; no constitutive formula is copied.
        struct Coefficients
        {
          double diffusion=0, reaction=0, tangent=0, energy=0;
          bool valid=true;
        };

        static Coefficients coefficients(const PhaseFieldHandler<dim> &handler,
                                         const std::vector<double> &fractions,
                                         const double phi, const double H)
        {
          AssertThrow(fractions.size()==handler.degradation_functions.size(), ExcInternalError());
          Coefficients c;
          const double ell=handler.geometric_function->get_length_scale();
          for (unsigned int j=0; j<fractions.size(); ++j)
            if (fractions[j]>0)
              {
                const auto &g=*handler.degradation_functions[j];
                c.valid=c.valid && g.is_in_domain(phi);
                if (!c.valid) return c;
                const double E=handler.critical_energy_densities[j];
                c.diffusion+=fractions[j]*2*E*ell*ell;
                c.reaction+=fractions[j]*(H*g.first_derivative(phi)+E*handler.geometric_function->first_derivative(phi));
                c.tangent+=fractions[j]*(H*g.second_derivative(phi)+E*handler.geometric_function->second_derivative(phi));
                c.energy+=fractions[j]*(H*g.value(phi)+E*handler.geometric_function->value(phi));
              }
          return c;
        }

        static const PhaseField::SolverParameters &
        parameters(const PhaseFieldHandler<dim> &handler)
        {
          return handler.solver_parameters;
        }

        static unsigned int solve(const PhaseFieldHandler<dim> &handler,
                                  const LinearAlgebra::BlockSparseMatrix &matrix,
                                  const LinearAlgebra::BlockVector &rhs,
                                  LinearAlgebra::BlockVector &update)
        {
          return handler.solve_phase_field_system(matrix,rhs,update);
        }

        static bool assemble(const PhaseFieldHandler<dim> &handler,
                             LinearAlgebra::BlockSparseMatrix &matrix,
                             LinearAlgebra::BlockVector &rhs,
                             const LinearAlgebra::BlockVector &state,
                             const bool assemble_jacobian = true,
                             double *roundoff_allowance = nullptr,
                             double *energy = nullptr,
                             bool omit_fe_elastic_energy = false)
        {
          return handler.assemble_phase_field_system(matrix, rhs, state, assemble_jacobian,roundoff_allowance,energy,omit_fe_elastic_energy);
        }

        static void constrain(const PhaseFieldHandler<dim> &handler,
                              LinearAlgebra::BlockSparseMatrix &matrix,
                              LinearAlgebra::BlockVector &rhs,
                              const LinearAlgebra::BlockVector &state,
                              const LinearAlgebra::BlockVector *lower_bound = nullptr)
        { handler.constrain_phase_newton_system(matrix,rhs,state,lower_bound); }

        static void bounded(const PhaseFieldHandler<dim> &handler,
                            LinearAlgebra::BlockSparseMatrix &matrix,
                            LinearAlgebra::BlockVector &rhs,
                            LinearAlgebra::BlockVector &state,
                            const LinearAlgebra::BlockVector *lower_bound = nullptr)
        { handler.solve_bound_constrained_phase(matrix,rhs,state,lower_bound); }

        static const std::vector<types::global_dof_index> &
        vertex_dofs(const PhaseFieldHandler<dim> &handler)
        {
          return handler.vertex_to_dof_indices;
        }

        // Diagnostic decomposition of the single-material production residual.
        // The complete assembly remains the independent consistency check.
        static std::pair<double,double>
        single_material_coefficients(const PhaseFieldHandler<dim> &handler,
                                     const double phi, const double H)
        {
          AssertThrow(handler.degradation_functions.size() == 1,
                      ExcMessage("This phase diagnostic requires one material."));
          const double energy = handler.critical_energy_densities[0];
          const double ell = handler.geometric_function->get_length_scale();
          return {2*energy*ell*ell,
                  H*handler.degradation_functions[0]->first_derivative(phi)
                  +energy*handler.geometric_function->first_derivative(phi)};
        }
    };
  }
}
#endif
