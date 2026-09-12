/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include "../../uniform_shear.cc"

namespace aspect
{
  namespace K3PhaseFloor
  {
    template <int dim>
    void run(const SimulatorAccess<dim> &sim)
    {
      const auto step=sim.get_timestep_number();
      if (step==0 || step>2) return;
      AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(sim.get_mpi_communicator())==1,
                  ExcMessage("The frozen phase-floor audit is one-rank 2-D."));
      const auto fingerprint=K3::fingerprint(sim);
      const auto &handler=sim.get_phase_field_handler();
      const auto &pm=handler.get_associated_particle_manager();
      const auto &constraints=sim.get_current_constraints();
      const auto &dofs=internal::PhaseFieldTestAccess<dim>::vertex_dofs(handler);
      const auto block=sim.introspection().variable("phase_field").block_index;
      const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
      const std::string prefix=sim.get_output_directory()+"audit"+std::to_string(step);
      LinearAlgebra::BlockSparseMatrix matrix,eval_matrix;
      for (auto *m : {&matrix,&eval_matrix})
        {
          m->reinit(sim.get_system_matrix().n_block_rows(),sim.get_system_matrix().n_block_cols());
          for (unsigned int r=0;r<m->n_block_rows();++r)
            for (unsigned int c=0;c<m->n_block_cols();++c)
              m->block(r,c).copy_from(sim.get_system_matrix().block(r,c));
          m->collect_sizes();
        }
      LinearAlgebra::BlockVector state(sim.get_solution()),trial(state);
      LinearAlgebra::BlockVector rhs(sim.introspection().index_sets.system_partitioning,sim.get_mpi_communicator());
      LinearAlgebra::BlockVector update(rhs),owned(rhs),trial_rhs(rhs),action(rhs),represented(rhs);
      auto assemble=[&](const auto &value,auto &result,const bool jacobian)
      {
        AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,jacobian?matrix:eval_matrix,result,value,jacobian),
                    ExcMessage("Diagnostic phase candidate inadmissible."));
      };
      // Export actual CPDI coefficients once; offline extended-precision
      // evaluation uses these fixed domains, constraints and parent histories.
      std::ofstream weights(prefix+"_weights.csv");
      weights << std::setprecision(17) << "id,dof,w,gx,gy,volume,H\n";
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto &p : pm.get_particle_handler().particles_in_cell(cell))
            {
              const auto domain=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
              for (unsigned int i=0;i<domain.n_relevant_vertices();++i)
                weights << p.get_id() << ',' << dofs[domain.relevant_vertex_index(i)] << ','
                        << domain.weighting_function_value(i) << ',' << domain.weighting_function_gradient(i)[0]
                        << ',' << domain.weighting_function_gradient(i)[1] << ',' << domain.volume()
                        << ',' << p.get_properties()[H] << '\n';
            }
      weights.close();
      std::ofstream cfile(prefix+"_constraints.csv");
      cfile << std::setprecision(17) << "slave,master,weight\n";
      for (const auto dof : dofs)
        if (dof!=numbers::invalid_dof_index && constraints.is_constrained(dof))
          for (const auto &entry : *constraints.get_constraint_entries(dof))
            cfile << dof << ',' << entry.first << ',' << entry.second << '\n';
      cfile.close();
      assemble(state,rhs,false);
      const double initial=rhs.block(block).l2_norm();
      std::ofstream history(prefix+"_iterations.csv");
      history << std::setprecision(17) << "iteration,initial,base,linear_iterations,fresh_linear,update,alpha,trial,relative\n";
      // Mirror the unchanged production update/line-search on private data.
      // Ten iterations suffice to capture the recorded fixed plateau; the
      // live solver still has its original 50-iteration budget.
      for (unsigned int iteration=0;iteration<10;++iteration)
        {
          assemble(state,rhs,true);
          update=0;
          const auto linear=internal::PhaseFieldTestAccess<dim>::solve(handler,matrix,rhs,update);
          owned=update;
          constraints.set_zero(owned);
          matrix.block(block,block).vmult(action.block(block),owned.block(block));
          action.block(block)-=rhs.block(block);
          const double fresh_linear=action.block(block).l2_norm();
          // Keep the original Jacobian intact while fresh residual assembly
          // uses a separate matrix: the production residual clears its matrix.
          {
            {
              std::vector<double> lengths{0.,1.,.5,.125};
              if (update.block(block).linfty_norm()<1e-10)
                lengths.insert(lengths.end(),{1e4,-1e4,1e6,-1e6});
              for (const double alpha : lengths)
                {
                  owned.block(block)=state.block(block);
                  owned.block(block).sadd(1.,alpha,update.block(block));
                  trial.block(block)=owned.block(block);
                  represented.block(block)=owned.block(block);
                  represented.block(block)-=state.block(block);
                  constraints.set_zero(represented);
                  matrix.block(block,block).vmult(action.block(block),represented.block(block));
                  assemble(trial,trial_rhs,false);
                  const std::string name=prefix+"_i"+std::to_string(iteration)+"_a"+std::to_string(alpha)+".csv";
                  std::ofstream nodes(name);
                  nodes << std::setprecision(17) << "dof,x,y,constrained,phi,rhs,update,represented,Jrepresented,trial_phi,trial_rhs\n";
                  const auto &vertices=sim.get_triangulation().get_vertices();
                  for (unsigned int v=0;v<dofs.size();++v)
                    if (dofs[v]!=numbers::invalid_dof_index)
                      {
                        const auto d=dofs[v];
                        nodes << d << ',' << vertices[v][0] << ',' << vertices[v][1] << ',' << constraints.is_constrained(d)
                              << ',' << state[d] << ',' << rhs[d] << ',' << update[d] << ',' << represented[d]
                              << ',' << action[d] << ',' << trial[d] << ',' << trial_rhs[d] << '\n';
                      }
                }
            }
          }
          double alpha=1.,residual=0;
          for (unsigned int search=0;search<=3;++search)
            {
              owned.block(block)=state.block(block);
              owned.block(block).sadd(1.,alpha,update.block(block));
              trial.block(block)=owned.block(block);
              assemble(trial,trial_rhs,false);
              residual=trial_rhs.block(block).l2_norm();
              if (residual<(1-1e-4*alpha)*rhs.block(block).l2_norm() || search==3) break;
              alpha*=.5;
            }
          history << iteration << ',' << initial << ',' << rhs.block(block).l2_norm() << ',' << linear
                  << ',' << fresh_linear << ',' << update.block(block).l2_norm() << ',' << alpha
                  << ',' << residual << ',' << residual/initial << std::endl;
          state.block(block)=trial.block(block);
          if (residual/initial<=1e-8) break;
        }
      AssertThrow(fingerprint==K3::fingerprint(sim),ExcMessage("Phase-floor audit mutated live state."));
      sim.get_pcout() << "K3_PHASE_FLOOR_AUDIT step=" << step << " restored=1" << std::endl;
      if (step==2) AssertThrow(false,ExcMessage("K3_PHASE_FLOOR_AUDIT_COMPLETE"));
    }
  }
  template <int dim> void connect_k3_phase_floor(SimulatorSignals<dim> &signals)
  { signals.start_timestep.connect(&K3PhaseFloor::run<dim>); }
  namespace k3_phase_floor_registration
  { ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_k3_phase_floor<2>,connect_k3_phase_floor<3>) }
}
