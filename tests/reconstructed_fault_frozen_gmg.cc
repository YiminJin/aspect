/* Bounded frozen-system AMG/GMG comparison. No physical state is committed. */
#include <aspect/simulator_signals.h>
#include <aspect/simulator/solver/stokes_matrix_free_local_smoothing.h>
#include <aspect/simulator/solver/block_stokes_preconditioner.h>
#include <aspect/reconstructed_fault/linear_performance.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <deal.II/lac/solver_gmres.h>
#include <fstream>
#include <sys/resource.h>

namespace aspect
{
  namespace FrozenGMGTest
  {
    template <typename Vector>
    struct Action
    {
      std::function<void(Vector &,const Vector &)> action;
      void vmult(Vector &dst,const Vector &src) const { action(dst,src); }
    };

    template <int dim>
    void compare(const SimulatorAccess<dim> &sim,
                 const typename SimulatorSignals<dim>::FaultBulkAction &operator_action,
                 const typename SimulatorSignals<dim>::FaultBulkAction &amg_action,
                 const typename SimulatorSignals<dim>::FaultPressureAction &pressure_action,
                 const LinearAlgebra::BlockVector &rhs,
                 const LinearAlgebra::BlockVector &accepted_direction,
                 double tolerance,unsigned int budget,double amg_setup)
    {
      const unsigned int step=std::getenv("ASPECT_FROZEN_GMG_STEP") ? std::atoi(std::getenv("ASPECT_FROZEN_GMG_STEP")) : 2;
      const unsigned int iteration=std::getenv("ASPECT_FROZEN_GMG_NEWTON") ? std::atoi(std::getenv("ASPECT_FROZEN_GMG_NEWTON")) : 4;
      if (sim.get_timestep_number()!=step || sim.get_nonlinear_iteration()!=iteration) return;
      AssertThrow(dim==2 && sim.get_parameters().stokes_velocity_degree==2,
                  ExcMessage("This bounded GMG fixture requires 2-D Q2 velocity."));
      AssertThrow(!std::getenv("ASPECT_FAULT_INTERFACE_MODES")
                  && !Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
                    sim.get_material_model()).uses_adiabatic_friction_pressure(),
                  ExcMessage("This probe uses absolute-pressure BP3 without an interface correction."));

      using Vector=LinearAlgebra::BlockVector;
      using Clock=internal::FaultLinearTiming::Clock;
      const auto comm=sim.get_mpi_communicator();
      const auto maximum=[&](double x) { return Utilities::MPI::max(x,comm); };
      const auto rss=[&]() { struct rusage usage;getrusage(RUSAGE_SELF,&usage);return maximum(usage.ru_maxrss); };
      Action<Vector> op{operator_action};
      Vector initial_state(sim.introspection().index_sets.system_partitioning,comm);
      initial_state=sim.get_solution();
      Vector initial_linearization(initial_state);
      initial_linearization=sim.get_current_linearization_point();
      Vector reference(rhs); reference=0.;
      std::ofstream output;
      if (Utilities::MPI::this_mpi_process(comm)==0)
        {
          output.open(sim.get_output_directory()+"frozen_gmg.csv");
          output<<std::setprecision(17)
            <<"backend,step,newton,rhs_norm,tolerance,setup_s,solve_s,iterations,fresh,estimated,direction_difference_relative,preconditioner_calls,preconditioner_s,operator_calls,operator_s,B_s,G_s,inverse_s,peak_rank_RSS_KiB\n";
        }
      const auto solve=[&](const char *name,const auto &prec,double setup,Vector &solution)
      {
        MPI_Barrier(comm);
        auto &timing=internal::FaultLinearTiming::get();
        timing.charge();const auto before=timing.seconds;
        const auto start=Clock::now();
        double preconditioner_seconds=0.,operator_seconds=0.;
        unsigned int preconditioner_calls=0,operator_calls=0;
        Action<Vector> measured_prec{[&](auto &dst,const auto &src)
          { const auto t=Clock::now();dst=0.;prec.vmult(dst,src);
            preconditioner_seconds+=std::chrono::duration<double>(Clock::now()-t).count();++preconditioner_calls; }};
        Action<Vector> measured_op{[&](auto &dst,const auto &src)
          { const auto t=Clock::now();op.vmult(dst,src);
            operator_seconds+=std::chrono::duration<double>(Clock::now()-t).count();++operator_calls; }};
        solution=0.;unsigned int iterations=0;double fresh=0.,estimated=0.;
        Vector residual(rhs);
        do
          {
            SolverControl control(budget-iterations,tolerance);
            SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(
              sim.get_parameters().stokes_gmres_restart_length));
            bool failed=false;
            try { solver.solve(measured_op,solution,rhs,measured_prec); }
            catch (const SolverControl::NoConvergence &) { failed=true; }
            iterations+=std::max(1u,control.last_step());estimated=control.last_value();
            op.vmult(residual,solution);residual-=rhs;fresh=residual.l2_norm();
            AssertThrow(fresh<=tolerance || (!failed && iterations<budget),
                        ExcMessage(std::string(name)+" failed its fresh residual test."));
          }
        while (fresh>tolerance);
        const double elapsed=maximum(std::chrono::duration<double>(Clock::now()-start).count());
        timing.charge();const auto after=timing.seconds;
        Vector difference(solution);difference-=accepted_direction;
        const double relative=difference.l2_norm()/accepted_direction.l2_norm();
        const double peak_rss=rss();
        const double B=maximum(after[internal::FaultLinearTiming::B_sparse]-before[internal::FaultLinearTiming::B_sparse]);
        const double G=maximum(after[internal::FaultLinearTiming::G_sparse]-before[internal::FaultLinearTiming::G_sparse]);
        const double inverse=maximum(after[internal::FaultLinearTiming::inverse]-before[internal::FaultLinearTiming::inverse]);
        const double prec_time=maximum(preconditioner_seconds),op_time=maximum(operator_seconds);
        const double setup_time=maximum(setup);
        const double rhs_norm=rhs.l2_norm();
        if (Utilities::MPI::this_mpi_process(comm)==0)
          { output<<name<<','<<step<<','<<iteration<<','<<rhs_norm<<','<<tolerance<<','
                  <<setup_time<<','<<elapsed<<','<<iterations<<','<<fresh<<','<<estimated<<','<<relative<<','
                  <<preconditioner_calls<<','<<prec_time<<','<<operator_calls<<','<<op_time<<','<<B<<','<<G<<','
                  <<inverse<<','<<peak_rss<<'\n';output.flush(); }
        sim.get_pcout()<<"Frozen "<<name<<": setup="<<setup_time<<", solve="<<elapsed
          <<", iterations="<<iterations<<", fresh="<<fresh<<", target="<<tolerance<<std::endl;
      };
      Action<Vector> amg{amg_action};
      solve("AMG",amg,amg_setup,reference);

      const auto start=Clock::now();
      StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2> gmg(
        const_cast<Simulator<dim>&>(sim.get_simulator()),sim.get_parameters());
      gmg.initialize_simulator(sim.get_simulator());gmg.initialize();
      gmg.with_velocity_preconditioner([&](const auto &cycle)
      {
        dealii::LinearAlgebra::distributed::Vector<double> in(rhs.block(0).locally_owned_elements(),comm),out(in);
        Action<LinearAlgebra::Vector> velocity_cycle{[&](auto &dst,const auto &src)
          { internal::ChangeVectorTypes::copy(in,src);cycle.vmult(out,in);internal::ChangeVectorTypes::copy(dst,out); }};
        internal::InverseVelocityBlock<decltype(velocity_cycle),LinearAlgebra::Vector,LinearAlgebra::SparseMatrix>
          velocity(sim.get_system_matrix().block(0,0),velocity_cycle,true,true,
                   sim.get_parameters().linear_solver_A_block_tolerance);
        Action<LinearAlgebra::Vector> pressure{pressure_action};
        internal::BlockSchurPreconditioner<decltype(velocity),decltype(pressure),LinearAlgebra::SparseMatrix,Vector>
          preconditioner(velocity,pressure,sim.get_system_matrix().block(0,1));
        Vector direction(rhs);
        const double setup=std::chrono::duration<double>(Clock::now()-start).count();
        solve("GMG",preconditioner,setup,direction);
      });

      // All work was on private vectors and borrowed operators. The caller's
      // existing exception path rolls back the still-uncommitted Newton step.
      Vector current(initial_state);
      current=sim.get_solution();initial_state-=current;
      current=sim.get_current_linearization_point();initial_linearization-=current;
      AssertThrow(initial_state.l2_norm()==0. && initial_linearization.l2_norm()==0.,
                  ExcMessage("Frozen preconditioner probe mutated the bulk state."));
      sim.get_pcout()<<"FROZEN AMG/GMG COMPARISON PASSED; intentional stop before trial/history commit."<<std::endl;
      AssertThrow(false,ExcMessage("Intentional frozen GMG comparison stop."));
    }

    template <int dim> void connect(SimulatorSignals<dim> &signals)
    { signals.post_reconstructed_fault_linear_solver.connect(&compare<dim>); }
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(FrozenGMGTest::connect<2>,FrozenGMGTest::connect<3>)
}
