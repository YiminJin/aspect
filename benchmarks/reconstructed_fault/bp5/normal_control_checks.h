// Initialization-only derivative check for the explicit normal-feedback control.
namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    void check_normal_control(const SimulatorAccess<dim> &sim)
    {
      AssertThrow(sim.get_timestep_number()==0 && sim.get_time()==0.,
                  ExcMessage("BP5 normal control cannot advance physical history."));
      const auto &intro=sim.introspection();
      const auto comm=sim.get_mpi_communicator();
      auto &surface=sim.get_reconstructed_fault_surface_system();
      const ReconstructedFaultVector V{sim.get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0)};
      const auto &base=sim.get_current_linearization_point();
      const auto residual=surface.evaluate_surface_residual(base,V);
      AffineConstraints<double> homogeneous;
      homogeneous.copy_from(sim.get_current_constraints());
      for (const auto &line:sim.get_current_constraints().get_lines())
        homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.close();
      std::ofstream out;
      if (sim.get_pcout().is_active())
        {
          out.open(sim.get_output_directory()+"normal_control_G.csv");
          out<<std::setprecision(17)<<"direction,G_max_Pa,FD_error_Pa,reference_action_error_Pa\n";
        }
      for (unsigned int mode=0;mode<2;++mode)
        {
          LinearAlgebra::BlockVector delta(intro.index_sets.system_partitioning,comm);
          if (mode==0) delta.block(intro.block_indices.pressure)=100.;
          else
            {
              delta.block(intro.block_indices.velocities)=base.block(intro.block_indices.velocities);
              delta.block(intro.block_indices.velocities)*=0.1;
            }
          homogeneous.distribute(delta);
          LinearAlgebra::BlockVector direction(intro.index_sets.system_partitioning,
                                                intro.index_sets.system_relevant_partitioning,comm);
          direction=delta;
          ReconstructedFaultVector G, reference;
          surface.apply_G(direction,G);
          surface.apply_G_reference(direction,reference);
          LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
          LinearAlgebra::BlockVector trial(intro.index_sets.system_partitioning,
                                            intro.index_sets.system_relevant_partitioning,comm);
          owned=base; owned.add(1.,delta); trial=owned;
          const auto plus=surface.evaluate_surface_residual(trial,V);
          owned=base; owned.add(-1.,delta); trial=owned;
          const auto minus=surface.evaluate_surface_residual(trial,V);
          double error=0., action=0., scale=0.;
          for (unsigned int i=0;i<V[0].size();++i)
            {
              double mass=residual.mass_diagonal[0][i];
              if (i) mass+=residual.mass_off_diagonal[0][i-1];
              if (i+1<V[0].size()) mass+=residual.mass_off_diagonal[0][i];
              error=std::max(error,std::abs((plus.values[0][i]-minus.values[0][i])/2.-G[0][i])/mass);
              action=std::max(action,std::abs(G[0][i]-reference[0][i])/mass);
              scale=std::max(scale,std::abs(G[0][i])/mass);
            }
          AssertThrow(error<1e-5 && action<1e-5 && (mode!=0 || scale==0.),
                      ExcMessage("Prescribed-normal bulk-work G disagrees with its residual/reference action."));
          if (out) out<<(mode==0?"pressure":"velocity")<<','<<scale<<','<<error<<','<<action<<'\n';
        }
    }
  }
}
