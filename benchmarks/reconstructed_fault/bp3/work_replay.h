// Committing replay observations. Never refresh FE histories or update stress here.
#include <deal.II/numerics/vector_tools_evaluate.h>
namespace aspect
{
  namespace BP3Benchmark
  {
    bool work_measure_replay=false;

    // Fixed-profile audit baseline, captured at initialization or immediately
    // after loading the explicitly qualified step-9 comparison checkpoint.
    std::map<types::particle_index,double> work_initial_H;
    std::vector<Point<2>> work_initial_geometry;
    std::vector<double> work_initial_I;

    template <int dim>
    void capture_work_invariants(const SimulatorAccess<dim> &sim)
    {
      const auto comm=sim.get_mpi_communicator();
      const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
      const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
      std::vector<std::pair<types::particle_index,double>> local;
      for (const auto &p:pm.get_particle_handler()) local.emplace_back(p.get_id(),p.get_properties()[H]);
      work_initial_H.clear();work_initial_geometry.clear();work_initial_I.clear();
      for (const auto &part:Utilities::MPI::all_gather(comm,local))
        for (const auto &entry:part) work_initial_H.emplace(entry);
      AssertThrow(!work_initial_H.empty(),ExcMessage("Work history audit must capture after particle deserialization."));
      sim.get_pcout()<<"Work history invariant capture: "<<work_initial_H.size()<<" real particle IDs before evolution."<<std::endl;
      const auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      for (unsigned int j=0;j<fault.n_vertices();++j)
        {
          Point<2> p;p[0]=fault.vertex(j)[0];p[1]=fault.vertex(j)[1];
          work_initial_geometry.push_back(p);work_initial_I.push_back(fault.get_properties(j)[I]);
        }
    }

    template <int dim>
    std::vector<double> export_work_replay(
      const SimulatorAccess<dim> &sim,const ReconstructedFaultSurfaceResidual &weak)
    {
      const auto &intro=sim.introspection();const auto comm=sim.get_mpi_communicator();
      auto &manager=sim.get_reconstructed_fault_manager();const auto &fault=manager.get_fault(0);
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      const auto &working=sim.get_current_linearization_point();
      const auto &V=manager.get_timestep_committed_slip_rate(0);
      const unsigned int step=sim.get_timestep_number(),n=V.size();
      const unsigned int rank=Utilities::MPI::this_mpi_process(comm);
      const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
      const auto &particles=pm.get_particle_handler();
      const auto &info=pm.get_property_manager().get_data_info();
      const auto H=info.get_position_by_field_name("crack_driving_force");
      const auto stress_position=info.get_position_by_field_name("maxwell stress");
      std::array<unsigned int,3> fields;
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") fields[m.second.second]=m.first;
      const auto I=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
      Tensor<1,dim> t=fault.vertex(1)-fault.vertex(0);t/=t.norm();
      Tensor<1,dim> normal;normal[0]=-t[1];normal[1]=t[0];
      const auto S=symmetrize(outer_product(t,normal)),N=symmetrize(outer_product(normal,normal));

      // Stable-ID inert history, geometry and completed profile remain fixed
      // even when particles move to another MPI owner. Capture only at t=0.
      if (step==0) capture_work_invariants(sim);
      const auto &initial_H=work_initial_H;
      const auto &initial_geometry=work_initial_geometry;
      const auto &initial_I=work_initial_I;
      bool history_ok=true;
      for (const auto &p:particles)
        {
          const auto old=initial_H.find(p.get_id());
          history_ok=history_ok && old!=initial_H.end() && old->second==p.get_properties()[H];
        }
      for (unsigned int j=0;j<n;++j)
        history_ok=history_ok && initial_geometry[j][0]==fault.vertex(j)[0]
          && initial_geometry[j][1]==fault.vertex(j)[1] && initial_I[j]==fault.get_properties(j)[I];
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(history_ok),comm),
                  ExcMessage("Work replay changed inert H, geometry or completed I_h."));

      struct Sample
      {
        SymmetricTensor<2,dim> tau;
        double chi,kappa,rate,bg,q,sigma,Ih;
      };
      const auto evaluate=[&](const unsigned int segment,const double xi,const bool active,
                             const double phi,const double temp,const double pressure,
                             const SymmetricTensor<2,dim> &eps,const std::vector<double> &all)
      {
        std::vector<double> chemical;
        for (const auto c:intro.chemical_composition_field_indices()) chemical.push_back(all[c]);
        typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs in;
        in.fault_index=0;in.segment_index=segment;in.xi=xi;in.phase_field=phi;in.previous_phase_field=phi;
        in.temperature=temp;in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
        const auto r=model.evaluate_reconstructed_fault_bulk_point(in);
        AssertThrow(r.history_correction==0.,ExcMessage("Mature frozen-profile history correction changed."));
        Sample result;
        result.chi=active ? r.localization_factor : 0.;result.kappa=r.kappa;
        result.rate=(1-xi)*V[segment]+xi*V[segment+1];
        SymmetricTensor<2,dim> old;
        old[0][0]=all[fields[0]];old[1][1]=all[fields[1]];old[0][1]=all[fields[2]];
        result.tau=2*r.kappa*(eps-result.chi*result.rate*S)+model.evaluate_frozen_maxwell_stress(temp,all,old);
        const auto bg=model.reconstructed_fault_background_tractions(0,segment,xi);
        result.bg=bg.first;result.q=bg.first+result.tau*S;result.sigma=bg.second+pressure-result.tau*N;
        result.Ih=(1-xi)*fault.get_properties(segment)[I]+xi*fault.get_properties(segment+1)[I];
        return result;
      };

      // Native work diagnostics use the accepted velocity/pressure and still
      // frozen working FE history, not the newly committed particle stress.
      const auto &quad=intro.quadratures.velocities;
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,
        update_values|update_gradients|update_quadrature_points|update_JxW_values);
      const unsigned int nq=quad.size();
      std::vector<double> phi(nq),temp(nq),pressure(nq);
      std::vector<SymmetricTensor<2,dim>> eps(nq);
      std::vector<std::vector<double>> composition(intro.n_compositional_fields,std::vector<double>(nq));
      std::vector<double> native(6*n),common(6*n);
      std::ofstream raw(sim.get_output_directory()+"work_qp_"+std::to_string(step)+"_rank"+std::to_string(rank)+".csv");
      raw.exceptions(std::ios::failbit|std::ios::badbit);
      raw<<std::setprecision(17)<<"cell,qp,x,y,xd,r,JxW,source_active,segment,xi,phi,Ih,chi,V,p,tau_xx,tau_yy,tau_xy,tauN,sigma_n,q,eps_xx,eps_yy,eps_xy,elastic_norm\n";
      const auto accumulate=[](std::vector<double> &v,unsigned int j,double xi,double weight,double p,const Sample &s,double tauN)
      {
        const double values[]={1.,p,tauN,s.q,s.sigma,s.bg};
        for (unsigned int end=0;end<2;++end)
          for (unsigned int c=0;c<6;++c) v[6*(j+end)+c]+=weight*(end ? xi : 1-xi)*values[c];
      };
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);const auto &a=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
            fe[intro.extractors.temperature].get_function_values(working,temp);
            fe[intro.extractors.pressure].get_function_values(working,pressure);
            fe[intro.extractors.velocities].get_function_symmetric_gradients(working,eps);
            for (unsigned int c=0;c<composition.size();++c) fe[intro.extractors.compositional_fields[c]].get_function_values(working,composition[c]);
            for (unsigned int q=0;q<nq;++q)
              {
                const auto p=fe.quadrature_point(q);const double xd=BP3::down_dip(p[0],p[1]);
                const bool window=(xd>=13000. && xd<=20000.) || (xd>=37000. && xd<=43000.)
                  || p[1]<2000. || p[1]>98000. || (xd>59000. && xd<61000.);
                if (!a[q].active && !(phi[q]>0. && window)) continue;
                std::vector<double> all(composition.size());
                for (unsigned int c=0;c<all.size();++c) all[c]=composition[c][q];
                const auto j=a[q].active ? a[q].segment_index : 0u;
                const auto xi=a[q].active ? a[q].xi : 0.;
                const auto s=evaluate(j,xi,a[q].active,phi[q],temp[q],pressure[q],eps[q],all);
                if (a[q].active && s.chi>0.) accumulate(native,j,xi,fe.JxW(q)*s.chi,pressure[q],s,s.tau*N);
                if (window && phi[q]>0.)
                  {
                    raw<<cell->id().to_string()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<xd<<','
                       <<(BP3::trace_x-p[0])*BP3::sine-(BP3::box_size-p[1])*BP3::cosine<<','<<fe.JxW(q)<<','
                       <<a[q].active<<','<<j<<','<<xi<<','<<phi[q]<<','<<s.Ih<<','<<s.chi<<','<<s.rate<<','<<pressure[q]<<','
                       <<s.tau[0][0]<<','<<s.tau[1][1]<<','<<s.tau[0][1]<<','<<s.tau*N<<','<<s.sigma<<','<<s.q<<','
                       <<eps[q][0][0]<<','<<eps[q][1][1]<<','<<eps[q][0][1]<<','<<(eps[q]-s.chi*s.rate*S).norm()<<'\n';
                  }
              }
          }

      // Common observer: the old parent-P0 FE samples and domain test weights.
      // This is output only and never supplies the new mechanical equation.
      const auto &associations=manager.get_locally_owned_particle_fault_associations();
      std::vector<Point<dim>> points;
      for (const auto &p:particles) points.push_back(p.get_location());
      Utilities::MPI::RemotePointEvaluation<dim> cache;
      cache.reinit(sim.get_phase_field_handler().get_grid_cache(),points);
      const auto gradients=VectorTools::point_gradients<dim>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.velocities[0]);
      const auto ph=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.variable("phase_field").first_component_index);
      const auto T=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.temperature);
      const auto p=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.pressure);
      std::vector<std::vector<double>> comps(intro.n_compositional_fields);
      for (unsigned int c=0;c<comps.size();++c)
        comps[c]=VectorTools::point_values<1>(cache,sim.get_dof_handler(),working,VectorTools::EvaluationFlags::avg,intro.component_indices.compositional_fields[c]);
      unsigned int k=0;double first_error=0.,first_scale=0.,first_old=0.;
      for (const auto &particle:particles)
        {
          Assert(particle.get_id()==associations[k].particle_id,ExcInternalError());
          std::vector<double> all(comps.size());for (unsigned int c=0;c<all.size();++c) all[c]=comps[c][k];
          for (const auto &q:associations[k].quadrature)
            {
              const auto s=evaluate(q.segment_index,q.xi,true,ph[k],T[k],p[k],symmetrize(gradients[k]),all);
              accumulate(common,q.segment_index,q.xi,q.weight,p[k],s,s.tau*N);
            }
          if (step==1)
            {
              const auto a=manager.project_to_bulk_source(particle.get_location());
              const auto s=evaluate(a.active ? a.segment_index : 0u,a.active ? a.xi : 0.,a.active,
                                    ph[k],T[k],p[k],symmetrize(gradients[k]),all);
              // Timestep zero retained zero particle stress. This independent
              // first finite-step formula must match the newly committed array.
              const auto expected=2*s.kappa*(symmetrize(gradients[k])-s.chi*s.rate*S);
              for (unsigned int c=0;c<3;++c)
                {
                  first_old=std::max(first_old,std::abs(all[fields[c]]));
                  const double actual=particle.get_properties()[stress_position+c];
                  first_error=std::max(first_error,std::abs(actual-expected[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]));
                  first_scale=std::max(first_scale,std::abs(actual));
                }
            }
          ++k;
        }
      if (step==1)
        {
          first_error=Utilities::MPI::max(first_error,comm);first_scale=Utilities::MPI::max(first_scale,comm);
          first_old=Utilities::MPI::max(first_old,comm);
          AssertThrow(first_old==0. && first_scale>0. && first_error<=1e-6+1e-11*first_scale,
                      ExcMessage("Work replay first Maxwell publication disagrees with the retained-zero-history update."));
          if (sim.get_pcout().is_active())
            {
              std::ofstream gate(sim.get_output_directory()+"first_update_maxwell.csv");
              gate<<std::setprecision(17)<<"error_Pa,scale_Pa,old_FE_max_Pa,stable_H_ids\n"
                  <<first_error<<','<<first_scale<<','<<first_old<<','<<initial_H.size()<<'\n';
            }
        }

      for (auto *v:{&native,&common}) {const auto local=*v;Utilities::MPI::sum(local,comm,*v);}
      std::vector<double> background(n);
      double consistency=0.;
      for (unsigned int j=0;j<n;++j)
        {
          background[j]=native[6*j+5];
          consistency=std::max(consistency,std::abs(native[6*j+3]-weak.shear_traction[0][j])/native[6*j]);
          consistency=std::max(consistency,std::abs(native[6*j+4]-weak.normal_traction[0][j])/native[6*j]);
        }
      AssertThrow(consistency<1e-5,ExcMessage("Accepted work observer does not reproduce frozen mechanical traction."));
      if (sim.get_pcout().is_active())
        {
          for (unsigned int mode=0;mode<2;++mode)
            {
              const auto &v=mode==0 ? native : common;
              std::ofstream out(sim.get_output_directory()+(mode==0 ? "work_weak_" : "common_fe_weak_")+std::to_string(step)+".csv");
              out<<std::setprecision(17)<<"node,xd,weight,p,tauN,q,sigma,bg\n";
              for (unsigned int j=0;j<n;++j)
                {out<<j<<','<<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]);for(unsigned int c=0;c<6;++c)out<<','<<v[6*j+c];out<<'\n';}
            }
          sim.get_pcout()<<"Work replay accepted observation: step="<<step<<", frozen weak stress discrepancy="<<consistency
                         <<" Pa; inert H and geometry unchanged."<<std::endl;
        }
      return ReconstructedFaultUtilities::solve_tridiagonal_system(weak.mass_diagonal[0],weak.mass_off_diagonal[0],background);
    }
  }
}
