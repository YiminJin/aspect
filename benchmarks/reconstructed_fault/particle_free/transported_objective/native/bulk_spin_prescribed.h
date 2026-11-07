/* Scratch prescribed-slip experiment: the ordinary native operations, without
 * publishing a synthetic state or changing the following coupled initial data. */
      void bulk_spin_prescribed()
      {
        using Stress=SymmetricTensor<2,dim>;
        using History=typename Model::DGHistory;
        auto &m=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        auto &sim=const_cast<Simulator<dim> &>(this->get_simulator());
        auto &manager=this->get_reconstructed_fault_manager();
        const auto original=m.fe_strain_state;
        const auto original_manager=manager.clone_prescribed_geometry();
        const auto before=snapshot();
        const auto old_spin=m.constitutive_spin;
        const bool old_direct=m.material_direct_stress,old_moving=m.dg_material_transport;
        const auto original_velocity=this->get_old_solution();
        const auto normalization=m.current_normalization_integrals,temperatures=m.current_fault_surface_temperatures;
        const auto comm=this->get_mpi_communicator();
        Stress initial;initial[0][0]=7;initial[1][1]=-2;initial[0][1]=1.3;
        const auto rotated=[&](double angle)
          {Stress r;double c=std::cos(2*angle),s=std::sin(2*angle);
           r[0][0]=2.5+4.5*c-1.3*s;r[1][1]=5-r[0][0];r[0][1]=4.5*s+1.3*c;return r;};
        struct Flow:Function<dim>
        {double q,w;Flow(unsigned n,double q,double w):Function<dim>(n),q(q),w(w){}
         double value(const Point<dim> &x,unsigned c) const override
         {return c==0 ? (q-w)*x[1] : (c==1 ? w*x[0] : 0.);}};
        const auto velocity=[&](double q,double w)
          {LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
           VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),Flow(this->get_fe().n_components(),q,w),owned);
           LinearAlgebra::BlockVector v(original_velocity);v=owned;return v;};
        const auto set_rates=[&](double v)
          {manager.begin_slip_rate_nonlinear_solve();manager.begin_slip_rate_trial();
           manager.set_slip_rate_trial_values({std::vector<double>(original->I.size(),v)});
           manager.accept_slip_rate_trial();manager.commit_slip_rate_nonlinear_solve();};
        const auto restore=[&]()
          {m.finish_dg_material_step(false);m.fe_strain_state=original;
           m.constitutive_spin=old_spin;m.material_direct_stress=old_direct;m.dg_material_transport=old_moving;
           sim.old_solution=original_velocity;m.current_normalization_integrals=normalization;m.current_fault_surface_temperatures=temperatures;};
        std::ofstream out;if(Utilities::MPI::this_mpi_process(comm)==0)
          {out.open(this->get_output_directory()+"bulk-spin-prescribed.csv");
           out<<"path,spin,dt,rigid_spin,bulk_spin_max,rotation_Linf,reaction_interior_Linf,boundary_profile_remainder_Linf\n"<<std::setprecision(17);}
        // Full normal sections span [-2,2]; uniform phi makes h/I=1/4.
        for(const auto I:original->I)AssertThrow(std::abs(I-75.)<1e-9,ExcMessage("Prescribed fixture needs phi=.2 on horizontal full normal sections."));
        for(const bool total:{false,true})for(const double rigid:{0.,.3})for(const double factor:{1.,.5})
          {
            m.fe_strain_state=std::make_shared<typename Model::FEStrainState>(*original);
            auto &s=*m.fe_strain_state;s.Z.assign(s.Z.size(),0.);
            set_rates(.8);const auto v=velocity(.2,rigid);
            const auto op=m.make_dg_transport(v,8,false);
            auto parent=m.project_dg_transport_field(*op,[&](const auto &,double){return initial;},0.,8);
            parent->representation=old_direct ? History::Representation::physical_stress : History::Representation::separated_trial;
            if(!old_direct)parent->residual=std::make_shared<typename Model::TensorField>(m.project_dg_transport_field(*op,[](const auto &,double){return Stress();},0.,8)->primary);
            s.bulk_history.tensor=parent;s.objective.tensor.reset();s.objective.parent.reset();
            m.constitutive_spin=total ? Model::ConstitutiveSpin::total_material : Model::ConstitutiveSpin::bulk_slip_subtracted;
            const double dt=factor*this->get_timestep();
            auto bulk=original_velocity,linear=original_velocity;
            const auto input=m.prepare_material_history(v,this->get_time(),dt);
            m.transport_material_history(input);m.stage_material_geometry(input,bulk,linear);m.rotate_material_history(input);
            if(!total && factor==1.)bulk_spin_checks();
            double omega=std::numeric_limits<double>::quiet_NaN();if(m.dg_material_trial->accepted_spin)omega=m.dg_material_trial->accepted_spin->maximum[2];
            const auto expected=rotated(dt*(rigid-(total ? .1 : 0.)));
            double error=0.,reaction_error=0.,boundary_remainder=0.;
            for(const auto &cell:s.cells)for(const auto &T:m.evaluate_history_tensor(*m.fe_strain_state->objective.tensor,cell.first,QGauss<dim>(3)))
              error=std::max(error,(T-expected).norm());
            // The endpoint law uses the simulator interval. Test it at that
            // interval, with stationary pure-slip geometry and constant eta.
            if(rigid==0. && factor==1.)
              {
                const auto endpoint=m.make_material_reaction(*m.fe_strain_state,v,std::vector<double>(s.I.size(),.8),false);
                const double beta=std::exp(-s.G*this->get_timestep()/s.eta);
                FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(3),update_quadrature_points|update_values);
                std::vector<double> phase(9);
                for(const auto &cell:s.cells)
                  {
                    const auto T=m.evaluate_history_tensor(*endpoint->bulk_history.tensor,cell.first,QGauss<dim>(3));
                    const auto R=endpoint->bulk_history.tensor->residual ? m.evaluate_history_tensor(*endpoint->bulk_history.tensor->residual,cell.first,QGauss<dim>(3)) : std::vector<Stress>(9);
                    fe.reinit(cell.second);fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(m.fe_strain_state->bulk_history.native,phase);
                    for(unsigned q=0;q<9;++q)
                      {const auto B=old_direct ? Stress() : m.localize_surface_values(fe.quadrature_point(q),phase[q],endpoint->Z,endpoint->I);
                       const double defect=(T[q]-R[q]-B-beta*expected).norm();
                       if(std::abs(fe.quadrature_point(q)[0])<1.)reaction_error=std::max(reaction_error,defect);
                       else boundary_remainder=std::max(boundary_remainder,defect);}

                  }
              }
            error=Utilities::MPI::max(error,comm);reaction_error=Utilities::MPI::max(reaction_error,comm);boundary_remainder=Utilities::MPI::max(boundary_remainder,comm);
            this->get_pcout()<<"Prescribed audit: total="<<total<<" rigid="<<rigid<<" factor="<<factor<<" omega="<<omega<<" rotation="<<error<<" reaction="<<reaction_error<<std::endl;
            AssertThrow(error<2e-10 && reaction_error<2e-10,ExcMessage("Prescribed native slip did not preserve the analytic anisotropic stress."));
            if(out)out<<"moving,"<<total<<','<<dt<<','<<rigid<<','<<omega<<','<<error<<','<<(rigid==0. && factor==1. ? reaction_error : std::numeric_limits<double>::quiet_NaN())<<','<<(rigid==0. && factor==1. ? boundary_remainder : std::numeric_limits<double>::quiet_NaN())<<'\n';
            restore();
          }
        // Actual stationary anchor, including zero-total/nonzero-fault spin.
        m.dg_material_transport=false;m.material_direct_stress=false;
        for(const auto pair:{std::make_pair(.2,.8),std::make_pair(0.,.8),std::make_pair(0.,0.)})
          {
            m.fe_strain_state=std::make_shared<typename Model::FEStrainState>(*original);
            auto &s=*m.fe_strain_state;const double shear=pair.first,rate=pair.second;set_rates(rate);const auto v=velocity(shear,0.);sim.old_solution=v;
            const auto op=m.make_dg_transport(v,8,false);
            auto parent=m.project_dg_transport_field(*op,[&](const auto &,double){return initial;},0.,8);
            parent->representation=History::Representation::separated_trial;
            parent->residual=std::make_shared<typename Model::TensorField>(m.project_dg_transport_field(*op,[](const auto &,double){return Stress();},0.,8)->primary);
            s.bulk_history.tensor=parent;s.Z.assign(s.Z.size(),.13);s.objective.tensor.reset();s.objective.parent.reset();
            m.constitutive_spin=Model::ConstitutiveSpin::bulk_slip_subtracted;m.prepare_objective_dg_anchor();
            const double angle=this->get_timestep()*(rate/8-.5*shear);const auto expected=rotated(angle);double error=0.;
            Stress residual;residual[0][0]=-.0325*std::sin(2*angle);residual[1][1]=-residual[0][0];residual[0][1]=.0325*(std::cos(2*angle)-1);
            for(const auto &cell:s.cells)for(const auto &T:m.evaluate_history_tensor(*s.objective.tensor,cell.first,QGauss<dim>(3)))error=std::max(error,(T-expected).norm());
            for(const auto &cell:s.cells)for(const auto &R:m.evaluate_history_tensor(*s.objective.tensor->residual,cell.first,QGauss<dim>(3)))error=std::max(error,(R-residual).norm());
            if(rate==0.)for(unsigned c=0;c<3;++c)for(auto i:parent->primary.coefficients[c].locally_owned_elements())AssertThrow(parent->primary.coefficients[c][i]==s.objective.tensor->primary.coefficients[c][i],ExcMessage("Exact-zero branch changed native coefficients."));
            error=Utilities::MPI::max(error,comm);AssertThrow(error<2e-10,ExcMessage("Stationary selected-spin shortcut used total spin."));
            if(out)out<<"stationary,0,"<<this->get_timestep()<<",0,"<<rate/8-.5*shear<<','<<error<<",0,0\n";
          }
        restore();manager.swap_prescribed_geometry(*original_manager);
        // Exercise schema8 and retained schema6 serialization without publishing
        // synthetic geometry/history or pretending to run a stationary trajectory.
        const auto saved_checkpoint=m.dg_checkpoint;
        const auto saved_identity=m.material_viscosity_identity;
        const bool saved_bounded=m.material_bounded_profile,saved_trace=m.material_trace_inflow;
        m.dg_material_transport=false;m.material_direct_stress=false;
        m.material_bounded_profile=false;m.material_trace_inflow=false;
        m.fe_strain_state=std::make_shared<typename Model::FEStrainState>(*original);
        auto parent=std::make_shared<History>(*original->bulk_history.tensor);
        parent->representation=History::Representation::separated_trial;
        parent->residual=std::make_shared<typename Model::TensorField>(parent->primary);
        m.fe_strain_state->bulk_history.tensor=parent;
        const auto identity=saved_identity.substr(0,saved_identity.find("|profile transport:"));
        std::ofstream archive_results;if(Utilities::MPI::this_mpi_process(comm)==0)
          {archive_results.open(this->get_output_directory()+"bulk-spin-stationary-archive.csv");archive_results<<"schema,bytes,same_law_loaded,cross_law_refused\n";}
        for(const bool bulk_spin:{false,true})
          {
            m.constitutive_spin=bulk_spin ? Model::ConstitutiveSpin::bulk_slip_subtracted : Model::ConstitutiveSpin::total_material;
            m.material_viscosity_identity=identity+(bulk_spin ? "|constitutive spin:bulk slip subtracted v1" : "");
            std::ostringstream buffer;{aspect::oarchive archive(buffer);m.serialize_dg_accepted_history(archive);}
            {std::istringstream input(buffer.str());aspect::iarchive archive(input);m.serialize_dg_accepted_history(archive);}
            m.constitutive_spin=bulk_spin ? Model::ConstitutiveSpin::total_material : Model::ConstitutiveSpin::bulk_slip_subtracted;
            bool refused=false;
            try {std::istringstream input(buffer.str());aspect::iarchive archive(input);m.serialize_dg_accepted_history(archive);}
            catch(const std::exception &exception){refused=std::string(exception.what()).find("PF5 checkpoint version/material/quadrature mismatch")!=std::string::npos;}
            AssertThrow(refused,ExcMessage("Stationary archive silently changed its constitutive spin."));
            if(archive_results)archive_results<<(bulk_spin ? 8 : 6)<<','<<buffer.str().size()<<",1,1\n";
          }
        m.dg_checkpoint=saved_checkpoint;m.material_viscosity_identity=saved_identity;
        m.material_bounded_profile=saved_bounded;m.material_trace_inflow=saved_trace;restore();
        AssertThrow(snapshot()==before,ExcMessage("Prescribed scratch check modified accepted physical state."));
      }
