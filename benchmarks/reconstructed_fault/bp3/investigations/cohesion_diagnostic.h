// Archived K5 probe; not included by the maintained BP3 plugin.
namespace aspect
{
  namespace BP3Benchmark
  {
    bool cohesion_derivative_ready=false;

    template <int dim>
    void check_frozen_cohesion(const SimulatorAccess<dim> &sim,AffineConstraints<double> &pending)
    {
      if (!cohesion_derivative_ready) return;
      cohesion_derivative_ready=false;
      const auto &intro=sim.introspection();
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,sim.get_mpi_communicator());
      owned=sim.get_solution();
      AffineConstraints<double> physical(pending);physical.close();physical.distribute(owned);
      LinearAlgebra::BlockVector bulk(sim.get_current_linearization_point());bulk=owned;
      ReconstructedFaultVector V(1,sim.get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0));
      auto &surface=sim.get_reconstructed_fault_surface_system();
      const auto base=surface.linearize_surface_system(bulk,V);
      // The fully frictional replay also makes the continued bottom source an
      // unknown. Check that endpoint's same work-equation tangent before step 1.
      const std::vector<unsigned int> nodes=std::getenv("ASPECT_BP3_FULLY_FRICTIONAL_REPLAY")
        ? std::vector<unsigned int>{796,0} : std::vector<unsigned int>{796};
      for (const unsigned int node:nodes)
        {
          ReconstructedFaultVector direction(1,std::vector<double>(V[0].size(),0.)),K;
          direction[0][node]=1.;surface.apply_surface_jacobian(direction,K);
          for (const double h:{1e-15,1e-16})
            {
              auto trial=V;trial[0][node]+=h;
              const auto r=surface.evaluate_surface_residual(bulk,trial);
              AssertThrow(r.cohesive_traction==base.cohesive_traction,
                          ExcMessage("Frozen resistance changed with trial V."));
              double error=0.,scale=0.;
              for (unsigned int i=(node==0 ? 0 : node-1);i<=node+1;++i)
                {error=std::max(error,std::abs((r.values[0][i]-base.values[0][i])/h+K[0][i]));scale=std::max(scale,std::abs(K[0][i]));}
              sim.get_pcout()<<"Frozen cohesion K check: node="<<node<<", h="<<h<<", relative error="<<error/scale<<std::endl;
              AssertThrow(error/scale<1e-4,ExcMessage("Frozen resistance K derivative mismatch."));
            }
        }
      LinearAlgebra::BlockVector delta(owned);delta=0.;delta.block(intro.block_indices.pressure)=1.;
      AffineConstraints<double> homogeneous(pending);
      for (const auto &line:pending.get_lines())homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.close();homogeneous.distribute(delta);
      LinearAlgebra::BlockVector ghost(bulk);ghost=delta;
      ReconstructedFaultVector G;surface.apply_G(ghost,G);
      auto trial=bulk;trial.add(100.,ghost);
      const auto r=surface.evaluate_surface_residual(trial,V);
      double error=0.,scale=0.;
      const bool full=std::getenv("ASPECT_BP3_FULLY_FRICTIONAL_REPLAY");
      for (unsigned int i=(full ? 0 : 795);i<(full ? V[0].size() : 798);++i)
        {error=std::max(error,std::abs((r.values[0][i]-base.values[0][i])/100.-G[0][i]));scale=std::max(scale,std::abs(G[0][i]));}
      sim.get_pcout()<<"Frozen cohesion G pressure check: relative error="<<error/scale<<std::endl;
      AssertThrow(error/scale<1e-7,ExcMessage("Frozen resistance G derivative mismatch."));
    }

    template <int dim>
    void snapshot_initial_cohesion(const SimulatorAccess<dim> &sim,const std::vector<double> &V)
    {
      const char *path=std::getenv("ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC");
      if (!path || sim.get_timestep_number()!=0) return;
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto C=manager.get_property_information()[manager.get_property_index("phase field fault cohesive traction")].position;
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      bool okay=true;
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(path);out<<std::setprecision(17)<<fault.n_vertices()<<'\n';
          for (unsigned int i=0;i<V.size();++i)
            out<<fault.vertex(i)[0]<<' '<<fault.vertex(i)[1]<<' '<<fault.get_properties(i)[C]
               <<' '<<V[i]<<' '<<fault.get_properties(i)[I]<<'\n';
          out.close();okay=static_cast<bool>(out);
        }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(okay),sim.get_mpi_communicator()),
                  ExcMessage("Cannot snapshot initial cohesive resistance."));
    }
  }
}
