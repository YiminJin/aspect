/* Bounded frozen-system AMG/GMG comparison. No physical state is committed. */
#include <aspect/simulator_signals.h>
#include <aspect/simulator/solver/stokes_matrix_free_local_smoothing.h>
#include <aspect/simulator/solver/block_stokes_preconditioner.h>
#include <aspect/reconstructed_fault/linear_performance.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <deal.II/lac/solver_gmres.h>
#include <boost/archive/binary_oarchive.hpp>
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
      // The fixture selects 'default solver': core builds the hierarchy before
      // resolving reconstructed-fault mechanics to AMG. Never label a borrowed
      // production GMG action as AMG just to obtain that hierarchy.
      AssertThrow(sim.get_parameters().stokes_solver_type ==
                    Parameters<dim>::StokesSolverType::block_amg,
                  ExcMessage("The frozen AMG probe requires the production AMG action."));
      AssertThrow(sim.get_triangulation().is_multilevel_hierarchy_constructed(),
                  ExcMessage("The frozen comparison requires a multigrid hierarchy; "
                             "select default solver in this reconstructed-fault fixture."));
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
      Vector initial_operator_action(rhs),initial_rhs_action(rhs);
      op.vmult(initial_operator_action,accepted_direction);
      op.vmult(initial_rhs_action,rhs);
      const auto vector_snapshot=[&](const Vector &source)
      {
        std::ostringstream bytes;
        boost::archive::binary_oarchive archive(bytes);
        archive << source.n_blocks();
        for (unsigned int b=0; b<source.n_blocks(); ++b)
          {
            const auto &block=source.block(b);
            archive << block.size() << block.locally_owned_elements().n_elements();
            for (const auto index : block.locally_owned_elements())
              archive << index << block[index];
          }
        return bytes.str();
      };
      const auto state_snapshot=[&]()
      {
        std::ostringstream bytes;
        boost::archive::binary_oarchive archive(bytes);
        const auto &manager=sim.get_reconstructed_fault_manager();
        // Serialization captures geometry, all registered surface histories,
        // and committed V; active V and prescribed masks are transient.
        archive << manager << manager.prescribed_slip_rate_mask();
        for (unsigned int f=0; f<manager.get_faults().size(); ++f)
          archive << manager.get_slip_rate(f);
        for (const auto *source : {&sim.get_solution(), &sim.get_current_linearization_point(),
                                  &sim.get_old_solution(), &sim.get_old_old_solution()})
          archive << vector_snapshot(*source);
        for (const auto &particle : sim.get_phase_field_handler()
             .get_associated_particle_manager().get_particle_handler())
          {
            archive << particle.get_id() << particle.get_location();
            const auto values=particle.get_properties();
            archive << std::vector<double>(values.begin(),values.end());
          }
        archive << vector_snapshot(rhs) << vector_snapshot(accepted_direction);
        return bytes.str();
      };
      const std::string frozen_state=state_snapshot();
      const auto write_snapshot=[&](const std::string &name,const std::string &bytes)
      {
        std::ofstream file(sim.get_output_directory()+name+"."+
                           std::to_string(Utilities::MPI::this_mpi_process(comm))+".bin",
                           std::ios::binary);
        file.write(bytes.data(),bytes.size());
        AssertThrow(file.good(),ExcMessage("Cannot write frozen comparison evidence."));
      };
      write_snapshot("frozen_state",frozen_state);
      write_snapshot("frozen_operator_direction",vector_snapshot(initial_operator_action));
      write_snapshot("frozen_operator_rhs",vector_snapshot(initial_rhs_action));
      sim.get_pcout()<<"Frozen backend verification: production=block AMG; hierarchy=present; "
                      "AMG=borrowed production action; GMG=explicit local-smoothing velocity cycle."
                     <<std::endl;
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
        write_snapshot(std::string("frozen_direction_")+name,vector_snapshot(solution));
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
        unsigned int cycle_calls=0;
        Action<LinearAlgebra::Vector> velocity_cycle{[&](auto &dst,const auto &src)
          { internal::ChangeVectorTypes::copy(in,src);cycle.vmult(out,in);++cycle_calls;
            internal::ChangeVectorTypes::copy(dst,out); }};
        internal::InverseVelocityBlock<decltype(velocity_cycle),LinearAlgebra::Vector,LinearAlgebra::SparseMatrix>
          velocity(sim.get_system_matrix().block(0,0),velocity_cycle,true,true,
                   sim.get_parameters().linear_solver_A_block_tolerance);
        Action<LinearAlgebra::Vector> pressure{pressure_action};
        internal::BlockSchurPreconditioner<decltype(velocity),decltype(pressure),LinearAlgebra::SparseMatrix,Vector>
          preconditioner(velocity,pressure,sim.get_system_matrix().block(0,1));
        Vector direction(rhs);
        const double setup=std::chrono::duration<double>(Clock::now()-start).count();
        solve("GMG",preconditioner,setup,direction);
        AssertThrow(Utilities::MPI::min(cycle_calls,comm)>0,
                    ExcMessage("The frozen GMG solve did not apply its velocity cycle."));
        sim.get_pcout()<<"Frozen GMG velocity cycle applied on every rank."<<std::endl;
      });

      // All work was on private vectors and borrowed operators. The caller's
      // existing exception path rolls back the still-uncommitted Newton step.
      Vector current(initial_state);
      current=sim.get_solution();initial_state-=current;
      current=sim.get_current_linearization_point();initial_linearization-=current;
      AssertThrow(initial_state.l2_norm()==0. && initial_linearization.l2_norm()==0.,
                  ExcMessage("Frozen preconditioner probe mutated the bulk state."));
      Vector check(rhs);
      op.vmult(check,accepted_direction);check-=initial_operator_action;
      const double direction_action_change=check.l2_norm();
      op.vmult(check,rhs);check-=initial_rhs_action;
      AssertThrow(direction_action_change==0. && check.l2_norm()==0.,
                  ExcMessage("Frozen comparison changed the borrowed operator action."));
      const unsigned int unchanged=state_snapshot()==frozen_state;
      AssertThrow(Utilities::MPI::min(unchanged,comm)==1,
                  ExcMessage("Frozen comparison changed bulk, surface, particle, V or RHS state."));
      sim.get_pcout()<<"Frozen state and RHS preserved exactly on every rank; operator actions unchanged."
                     <<std::endl;
      sim.get_pcout()<<"FROZEN AMG/GMG COMPARISON PASSED; intentional stop before trial/history commit."<<std::endl;
      AssertThrow(false,ExcMessage("Intentional frozen GMG comparison stop."));
    }

    template <int dim> void connect(SimulatorSignals<dim> &signals)
    { signals.post_reconstructed_fault_linear_solver.connect(&compare<dim>); }
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(FrozenGMGTest::connect<2>,FrozenGMGTest::connect<3>)
}
