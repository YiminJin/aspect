namespace aspect
{
  namespace BP3Benchmark
  {
    bool fully_frictional=false;
    bool work_derivative_ready=false;

    template <int dim>
    void check_work_derivatives(const SimulatorAccess<dim> &sim,AffineConstraints<double> &pending)
    {
      if (!work_derivative_ready) return;
      work_derivative_ready=false;
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
      const std::vector<unsigned int> nodes=fully_frictional
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
                          ExcMessage("Mature resistance changed with trial V."));
              double error=0.,scale=0.;
              for (unsigned int i=(node==0 ? 0 : node-1);i<=node+1;++i)
                {error=std::max(error,std::abs((r.values[0][i]-base.values[0][i])/h+K[0][i]));scale=std::max(scale,std::abs(K[0][i]));}
              sim.get_pcout()<<"Mature work K check: node="<<node<<", h="<<h<<", relative error="<<error/scale<<std::endl;
              AssertThrow(error/scale<1e-4,ExcMessage("Mature resistance K derivative mismatch."));
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
      const bool full=fully_frictional;
      for (unsigned int i=(full ? 0 : 795);i<(full ? V[0].size() : 798);++i)
        {error=std::max(error,std::abs((r.values[0][i]-base.values[0][i])/100.-G[0][i]));scale=std::max(scale,std::abs(G[0][i]));}
      sim.get_pcout()<<"Mature work G pressure check: relative error="<<error/scale<<std::endl;
      AssertThrow(error/scale<1e-7,ExcMessage("Mature resistance G derivative mismatch."));
    }

    // Retired probes require the archived experimental tree. Refuse rather
    // than silently running different physics when an old launcher is used.
    void reject_retired_investigation_selectors()
    {
      for (const char *name : {
          "ASPECT_BP3_FULLY_FRICTIONAL_REPLAY", "ASPECT_BP3_SPLIT_TRACE_REPLAY",
          "ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC", "ASPECT_FAULT_WITHIN_STEP_STATE",
          "ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC", "ASPECT_BP3_COUPLED_STATE_REPLAY",
          "ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC",
          "ASPECT_FAULT_THETA_UPDATE_DIAGNOSTIC", "ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC",
          "ASPECT_FAULT_THETA_QP_EXPORT", "ASPECT_BP3_THETA_EXACT_DIAGNOSTIC",
          "ASPECT_BP3_NOTCH_BOUNDARY_PROBE", "ASPECT_BP3_JUNCTION_DIAGNOSTIC",
          "ASPECT_FAULT_HISTORY_FE", "ASPECT_BP3_FROZEN_SURFACE",
          "ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC", "ASPECT_BP3_SHARED_CLOCK",
          "ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC", "ASPECT_BP3_WORK_MEASURE"})
        AssertThrow(!std::getenv(name),
                    ExcMessage(std::string("Retired BP3 investigation selector: ")+name+
                      ". Use the archived investigation snapshot, not the research configuration."));
    }
  }
}
