/* Nonpublishing checks of the actual staged native material attempt.
 * Independent scalar source/frame and component rotation references. */
      void bulk_spin_checks()
      {
        auto &m=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        if(m.constitutive_spin!=Model::ConstitutiveSpin::bulk_slip_subtracted)return;
        auto &trial=*m.dg_material_trial;
        const auto &old=*trial.accepted_history;
        const auto &candidate=*trial.candidate_history;
        const auto &spin=*trial.accepted_spin;
        const auto comm=this->get_mpi_communicator();
        AssertThrow(trial.bound && spin.parent.lock()==old.bulk_history.tensor
          && spin.geometry_version==trial.manager->get_fault(0).geometry_version()
          && spin.committed_rates==trial.manager->get_timestep_committed_slip_rate(0),
          ExcMessage("Bulk spin lost its accepted parent/geometry/rate identity."));
        const QGauss<dim> rule(spin.quadrature_order);
        FEValues<dim> values(this->get_mapping(),this->get_fe(),rule,
          update_values|update_gradients|update_quadrature_points);
        std::vector<double> phi(rule.size()),next_phi(rule.size());
        std::vector<Tensor<2,dim>> gradients(rule.size());
        std::map<CellId,std::vector<double>> reference;
        double spin_error=0.,mixed_state_gap=0.,source_error=0.;
        unsigned active=0,inactive=0;
        const auto source=[&](const auto &manager,const auto &state,const Point<dim> &x,double p,const auto &rates)
          {
            Tensor<2,dim> J;
            const auto a=manager.project_to_bulk_source(x);
            if(a.active)
              {
                const auto &f=manager.get_fault(a.fault_index);
                auto t=f.vertex(a.segment_index+1)-f.vertex(a.segment_index);t/=t.norm();
                Tensor<1,dim> n;n[0]=-t[1];n[1]=t[0];
                const auto j=a.segment_index;
                const double h=50.*p*(1+p)/((1-p)*(1-p)); // fixed common fixture, G=10, Gc=1, ell=.15
                const double v=(1-a.xi)*rates[j]+a.xi*rates[j+1];
                const double I=(1-a.xi)*state.I[j]+a.xi*state.I[j+1];
                for(unsigned k=0;k<dim;++k)for(unsigned l=0;l<dim;++l)
                  J[k][l]=h*v/I*manager.get_shear_sense(a.fault_index)*t[k]*n[l];
              }
            return J;
          };
        auto &live=this->get_reconstructed_fault_manager();
        for(const auto &entry:old.cells)
          {
            values.reinit(entry.second);
            values[this->introspection().extractors.velocities].get_function_gradients(*spin.velocity,gradients);
            const auto phase=FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index);
            values[phase].get_function_values(old.bulk_history.native,phi);
            values[phase].get_function_values(candidate.bulk_history.native,next_phi);
            auto &omega=reference[entry.first];omega.resize(rule.size());
            for(unsigned q=0;q<rule.size();++q)
              {
                const auto x=values.quadrature_point(q);
                const auto J=source(*trial.manager,old,x,phi[q],spin.committed_rates);
                omega[q]=.5*(gradients[q][1][0]-gradients[q][0][1]-J[1][0]+J[0][1]);
                spin_error=std::max(spin_error,std::abs(omega[q]-spin.angular_rates.at(entry.first)[q]));
                const auto next=source(live,candidate,x,next_phi[q],spin.committed_rates);
                mixed_state_gap=std::max(mixed_state_gap,.5*std::abs(J[1][0]-J[0][1]-next[1][0]+next[0][1]));
                const auto sample=m.sample_mature_localization(*trial.manager,x,phi[q],old.I);
                source_error=std::max(source_error,(2*symmetrize(J)-sample.twice_symmetric(spin.committed_rates)).norm());
                if(sample.active)++active;else ++inactive;
              }
          }
        const auto rotate=[](const SymmetricTensor<2,dim> &T,double angle)
          {
            SymmetricTensor<2,dim> out;
            const double c=std::cos(angle),s=std::sin(angle);
            out[0][0]=c*c*T[0][0]+s*s*T[1][1]-2*c*s*T[0][1];
            out[1][1]=s*s*T[0][0]+c*c*T[1][1]+2*c*s*T[0][1];
            out[0][1]=c*s*(T[0][0]-T[1][1])+(c*c-s*s)*T[0][1];
            return out;
          };
        const auto &transported=m.material_direct_stress ? trial.transport.physical_stress : trial.transport.separated_trial;
        const auto expected=m.project_material_tensor(old.bulk_history.tensor->primary,candidate,*spin.velocity,
          [&](const auto &cell,unsigned q,const auto &ids,const auto &u,const auto &,double,const auto &)
            {return rotate(transported->primary.tensor_at(ids,u),trial.dt*reference.at(cell)[q]);});
        const auto difference=[&](const auto &a,const auto &b)
          {double error=0.;for(unsigned c=0;c<3;++c)for(auto i:a.coefficients[c].locally_owned_elements())
             error=std::max(error,std::abs(a.coefficients[c][i]-b.coefficients[c][i]));return error;};
        double rotation_error=difference(*expected,candidate.objective.tensor->primary);
        if(!m.material_direct_stress)
          {
            const auto expected_R=m.project_material_tensor(old.bulk_history.tensor->primary,candidate,*spin.velocity,
              [&](const auto &cell,unsigned q,const auto &ids,const auto &u,const auto &x,double p,const auto &)
                {return rotate(trial.transport.separated_total_memory->primary.tensor_at(ids,u),trial.dt*reference.at(cell)[q])
                     -m.localize_surface_values(x,p,candidate.Z,candidate.I);});
            rotation_error=std::max(rotation_error,difference(*expected_R,*candidate.objective.tensor->residual));
          }
        // Private nonlinear rates must not change an already frozen anchor.
        const auto frozen=candidate.objective.tensor;
        live.begin_slip_rate_nonlinear_solve();live.begin_slip_rate_trial();
        auto changed=spin.committed_rates;for(auto &v:changed)v=3*v+.01;
        live.set_slip_rate_trial_values({changed});
        m.prepare_objective_dg_anchor();
        AssertThrow(candidate.objective.tensor==frozen && trial.accepted_spin->angular_rates==spin.angular_rates,
                    ExcMessage("Newton rates changed the frozen spin/anchor."));
        live.rollback_slip_rate_trial();live.rollback_slip_rate_nonlinear_solve();
        spin_error=Utilities::MPI::max(spin_error,comm);
        source_error=Utilities::MPI::max(source_error,comm);
        rotation_error=Utilities::MPI::max(rotation_error,comm);
        mixed_state_gap=Utilities::MPI::max(mixed_state_gap,comm);
        active=Utilities::MPI::sum(active,comm);inactive=Utilities::MPI::sum(inactive,comm);
        this->get_pcout()<<std::setprecision(17)<<"Bulk spin audit: spin="<<spin_error<<" source="<<source_error<<" rotation="<<rotation_error<<" mixed="<<mixed_state_gap<<std::endl;
        AssertThrow(spin_error<2e-12 && source_error<2e-12 && rotation_error<2e-11,
                    ExcMessage("Native accepted bulk-spin/reference mismatch."));
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {
            const auto file=this->get_output_directory()+"bulk-spin-native.csv";
            const bool exists=std::ifstream(file).good();std::ofstream out(file,std::ios::app);
            if(!exists)out<<"step,dt,spin_Linf,source_Linf,rotation_coeff_Linf,candidate_spin_gap,active,inactive\n";
            out<<std::setprecision(17)<<spin.step<<','<<trial.dt<<','<<spin_error<<','<<source_error<<','<<rotation_error<<','<<mixed_state_gap<<','<<active<<','<<inactive<<'\n';
          }
      }
