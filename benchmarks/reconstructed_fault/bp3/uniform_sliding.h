// Observational mature-fault test. No constitutive state is re-published here.
#include <aspect/material_model/utilities.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    inline bool uniform_sliding_test()
    { return std::getenv("ASPECT_BP3_UNIFORM_SLIDING") != nullptr; }

    template <int dim>
    void export_uniform_sliding(const SimulatorAccess<dim> &sim)
    {
      if (!uniform_sliding_test()) return;
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
        sim.get_material_model());
      AssertThrow(dim==2 && model.is_mature_frictional_fault(),
                  ExcMessage("Uniform sliding diagnostic requires the 2-D mature model."));
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto &V=manager.get_timestep_committed_slip_rate(0);
      for (double v:V) AssertThrow(v==BP3::Vp,ExcMessage("Uniform sliding rate was not prescribed exactly."));
      const auto &intro=sim.introspection();
      const auto &working=sim.get_current_linearization_point();
      // This vector retains the mechanically constrained OLD FE history even
      // after particle histories commit. Do not use the new particle stress.
      const auto &quadrature=intro.quadratures.velocities;
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quadrature,
        update_values|update_gradients|update_quadrature_points|update_JxW_values);
      const unsigned int nq=quadrature.size();
      std::vector<double> phi(nq),temperature(nq),pressure(nq);
      std::vector<Tensor<1,dim>> velocity(nq);
      std::vector<SymmetricTensor<2,dim>> strain(nq);
      std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
      std::array<unsigned int,3> fields;
      fields.fill(numbers::invalid_unsigned_int);
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") fields[m.second.second]=m.first;
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      const auto rank=Utilities::MPI::this_mpi_process(sim.get_mpi_communicator());
      const bool all_source_qps=std::getenv("ASPECT_BP3_ALL_SOURCE_QPS");
      std::ofstream out(sim.get_output_directory()+"uniform_bulk_"+std::to_string(sim.get_timestep_number())
                        +"_rank"+std::to_string(rank)+".csv");
      out.exceptions(std::ios::failbit|std::ios::badbit);
      out<<std::setprecision(17)<<"cell,qp,x,y,xd,r,weight,segment,xi,phi,Ih,chi,kappa,ux,uy,p,eps_xx,eps_yy,eps_xy,crack_xx,crack_yy,crack_xy,elastic_norm,old_xx,old_yy,old_xy,tau_xx,tau_yy,tau_xy,tauN,sigma_n,source_active\n";
      manager.prepare_stokes_qp_projection_cache();
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            const auto &assoc=manager.get_stokes_qp_fault_associations(cell->id(),quadrature,fe.get_quadrature_points());
            fe[intro.extractors.velocities].get_function_values(working,velocity);
            fe[intro.extractors.velocities].get_function_symmetric_gradients(working,strain);
            fe[intro.extractors.pressure].get_function_values(working,pressure);
            fe[intro.extractors.temperature].get_function_values(working,temperature);
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
            for (unsigned int c=0;c<composition.size();++c)
              fe[intro.extractors.compositional_fields[c]].get_function_values(working,composition[c]);
            for (unsigned int q=0;q<nq;++q)
              {
                const auto p=fe.quadrature_point(q);
                const double xd=BP3::down_dip(p[0],p[1]);
                const bool audit_point=all_source_qps && phi[q]>0.
                  && (p[1]<2000. || p[1]>BP3::box_size-2000. || (xd>59000. && xd<61000.));
                if ((assoc[q].active && (!all_source_qps || phi[q]>0.)) || audit_point)
                {
                  auto a=assoc[q];
                  // Inactive points still need the actual bulk Maxwell
                  // coefficients and a reference frame for tensor diagnostics.
                  // Their source remains exactly zero, as in assembly.
                  if (!a.active)
                    {
                      a.fault_index=0;
                      unsigned int j=0;
                      while (j+1<fault.n_cells() && fault.vertex(j+1)[1]<p[1]) ++j;
                      a.segment_index=j;
                      a.tangent=fault.vertex(j+1)-fault.vertex(j);
                      const double length=a.tangent.norm();a.tangent/=length;
                      a.normal[0]=-a.tangent[1];a.normal[1]=a.tangent[0];
                      a.xi=std::clamp((p-fault.vertex(j))*a.tangent/length,0.,1.);
                    }
                  std::vector<double> chemical,all(composition.size());
                  for (unsigned int c=0;c<all.size();++c) all[c]=composition[c][q];
                  for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(all[c]);
                  typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs input;
                  input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                  input.phase_field=phi[q];input.previous_phase_field=phi[q];
                  input.temperature=temperature[q];
                  input.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
                  const auto response=model.evaluate_reconstructed_fault_bulk_point(input);
                  AssertThrow(response.history_correction==0.,ExcMessage("Mature fixed-profile history correction is not zero."));
                  const auto S=symmetrize(outer_product(a.tangent,a.normal));
                  const auto N=symmetrize(outer_product(a.normal,a.normal));
                  const double chi=a.active ? response.localization_factor : 0.;
                  const auto crack=chi*BP3::Vp*S;
                  SymmetricTensor<2,dim> old;
                  old[0][0]=all[fields[0]];old[1][1]=all[fields[1]];old[0][1]=all[fields[2]];
                  const auto stress=2*response.kappa*(strain[q]-crack)
                    +model.evaluate_frozen_maxwell_stress(temperature[q],all,old);
                  const double Ih=(1-a.xi)*fault.get_properties(a.segment_index)[I]
                                  +a.xi*fault.get_properties(a.segment_index+1)[I];
                  out<<cell->id().to_string()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<BP3::down_dip(p[0],p[1])
                     <<','<<(BP3::trace_x-p[0])*BP3::sine-(BP3::box_size-p[1])*BP3::cosine
                     <<','<<fe.JxW(q)<<','<<a.segment_index<<','<<a.xi<<','<<phi[q]<<','<<Ih<<','
                     <<chi<<','<<response.kappa<<','<<velocity[q][0]<<','<<velocity[q][1]<<','<<pressure[q];
                  for (const auto &tensor:{strain[q],crack})
                    out<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
                  out<<','<<(strain[q]-crack).norm();
                  for (const auto &tensor:{old,stress})
                    out<<','<<tensor[0][0]<<','<<tensor[1][1]<<','<<tensor[0][1];
                  out<<','<<stress*N<<','<<BP3::sigma0+pressure[q]-stress*N<<','<<a.active<<'\n';
                }
              }
          }
      // Independent one-dimensional reference uses the prescribed distance
      // profile, not the endpoint-dependent projected FE I_h. Export nodes for
      // offline high-order integration of h and construction of u_plate(r).
      if (sim.get_timestep_number()==0 && sim.get_pcout().is_active())
        {
          const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(BP3::core_phi);
          std::ofstream profile(sim.get_output_directory()+"uniform_reference_profile.csv");
          profile<<std::setprecision(17)<<"r,phi,h\n";
          for (const double r:profiles[0]->get_coordinate_values())
            {
              const double phi=profiles[0]->value(r);
              const double g=sim.get_phase_field_handler().energetic_degradation({1.,0.},phi);
              profile<<r<<','<<phi<<','<<1/g-1<<'\n';
            }
        }
      if (std::getenv("ASPECT_BP3_TOP_SOURCE_EXPERIMENT"))
        {
          // Record the geometry mismatch explicitly: these real parents enter
          // Maxwell source subtraction but still supply no surface R/K/G weight.
          std::ofstream parents(sim.get_output_directory()+"top_source_parents_"
            +std::to_string(sim.get_timestep_number())+"_rank"+std::to_string(rank)+".csv");
          parents<<std::setprecision(17)<<"id,x,y,volume,surface_active,surface_weight,source_segment,source_xi\n";
          for (const auto &a:manager.get_locally_owned_particle_fault_associations())
            if (a.position[1]>BP3::box_size-2000.)
              {
                const auto source=manager.project_to_bulk_source(a.position);
                if (!source.active) continue;
                double weight=0.;for (const auto &q:a.quadrature) weight+=q.weight;
                parents<<a.particle_id<<','<<a.position[0]<<','<<a.position[1]<<','
                  <<a.particle_domain_volume<<','<<a.active<<','<<weight<<','
                  <<source.segment_index<<','<<source.xi<<'\n';
              }
          if (sim.get_timestep_number()==0)
            {
              // Exercise the production B and absolute bulk residual with an
              // independent endpoint perturbation, without altering manager V.
              Assemblers::ReconstructedFaultStokes<dim> coupling(sim.get_simulator());
              coupling.linearize_B(working);
              typename Assemblers::ReconstructedFaultStokes<dim>::FaultVector
                base(1,V),direction(1,std::vector<double>(V.size(),0.)),plus=base,minus=base;
              direction[0].back()=BP3::Vp;
              constexpr double epsilon=.125;
              plus[0].back()+=epsilon*BP3::Vp;minus[0].back()-=epsilon*BP3::Vp;
              const auto &partition=sim.introspection().index_sets.system_partitioning;
              LinearAlgebra::BlockVector action(partition,sim.get_mpi_communicator()),
                reference(partition,sim.get_mpi_communicator()),
                rplus(partition,sim.get_mpi_communicator()),rminus(partition,sim.get_mpi_communicator());
              coupling.apply_B(direction,action);
              coupling.apply_B_reference(direction,reference);
              reference.add(-1.,action);
              coupling.evaluate_slip_dependent_bulk_residual(working,plus,rplus);
              coupling.evaluate_slip_dependent_bulk_residual(working,minus,rminus);
              rplus.add(-1.,rminus);rplus*=1./(2*epsilon);rplus.add(1.,action);
              const double scale=action.l2_norm(),fd=rplus.l2_norm()/scale,
                           sparse_error=reference.l2_norm()/scale;
              AssertThrow(scale>0. && fd<2e-12 && sparse_error<2e-12,
                          ExcMessage("Top endpoint source derivative does not match production B."));
              AssertThrow(manager.get_timestep_committed_slip_rate(0)==base[0],
                          ExcMessage("Noncommitting endpoint probe changed V."));
              if (rank==0)
                {
                  std::ofstream probe(sim.get_output_directory()+"top_endpoint_B_check.csv");
                  probe<<std::setprecision(17)<<"B_norm,finite_difference_relative,sparse_reference_relative\n"
                    <<scale<<','<<fd<<','<<sparse_error<<'\n';
                }
            }
        }
    }
  }
}
