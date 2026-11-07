/* Nonpublishing native physical-history checks, included in MomentCycle. */
      void true_stress_checks() const
      {
        using Tensor=SymmetricTensor<2,dim>;
        using History=typename Model::DGHistory;
        auto &m=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        auto &s=*m.fe_strain_state;
        const auto accepted=snapshot();
        const auto saved_anchor=s.objective.tensor;
        const auto saved_Z=s.Z;
        const auto comm=this->get_mpi_communicator();
        struct Flow : Function<dim>
        {
          Flow(unsigned n,double amplitude):Function<dim>(n),a(amplitude){}
          double a;
          double value(const Point<dim> &x,unsigned c) const override
          {return c==0 ? .4+a*(.03*x[0]-.07*x[1]) : (c==1 ? a*(.09*x[0]-.03*x[1]) : 0.);}
        };
        const auto velocity=[&](double a)
          {
            LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
            VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),Flow(this->get_fe().n_components(),a),owned);
            LinearAlgebra::BlockVector result(this->get_solution());result=owned;return result;
          };
        const auto bulk=velocity(1.);
        // The projection's explicit profile and gradient must win over the
        // model's currently bound accepted fields. Analytic constant output
        // avoids using the production projection as its own reference.
        auto projection_state=s;
        LinearAlgebra::BlockVector explicit_fields(this->introspection().index_sets.system_partitioning,comm);
        explicit_fields=velocity(2.);
        explicit_fields.block(this->introspection().variable("phase_field").block_index)=.25;
        projection_state.bulk_history.native=explicit_fields;
        const auto explicit_velocity=velocity(2.);
        const auto projected=m.project_material_tensor(s.bulk_history.tensor->primary,projection_state,explicit_velocity,
          [](const auto &,unsigned int,const auto &,const auto &,const auto &,double phase,const auto &gradient)
          {Tensor value;value[0][0]=phase+gradient[0][0];value[1][1]=gradient[1][1];
           value[0][1]=.5*(gradient[0][1]+gradient[1][0]);return value;});
        Tensor reference;reference[0][0]=.31;reference[1][1]=-.06;reference[0][1]=.02;
        double projection_error=0.;
        for(const auto &cell:s.cells)
          for(const auto &value:m.evaluate_history_tensor(*projected,cell.first,QGauss<dim>(3)))
            projection_error=std::max(projection_error,(value-reference).norm());
        projection_error=Utilities::MPI::max(projection_error,comm);
        AssertThrow(projection_error<3e-12,ExcMessage("Material projection ignored its explicit profile/gradient inputs."));
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {std::ofstream out(this->get_output_directory()+"mature-projection-inputs.csv");
           out<<std::setprecision(17)<<"tensor_Linf,tolerance\n"<<projection_error<<",3e-12\n";}

        const auto op=m.make_dg_transport(bulk,8,false);
        const auto zero=[](const Point<dim> &,double){return Tensor();};
        auto anchor=m.project_dg_transport_field(*op,zero,0.,8);
        anchor->representation=History::Representation::physical_stress;
        s.objective.tensor=anchor;s.Z.assign(s.Z.size(),0.);
        std::vector<double> rates(s.Z.size());
        for(unsigned j=0;j<rates.size();++j)rates[j]=.002*(1+.2*std::cos(s.vertices[j][0]));
        const double epsilon=1e-5,dt=m.initial_time_step;
        auto candidate=m.make_material_reaction(*m.fe_strain_state,bulk,rates, false);
        for(auto &z:s.Z)z=1.;
        const auto changed_diagnostic=m.make_material_reaction(*m.fe_strain_state,bulk,rates, false);
        double diagnostic_error=0.;
        for(unsigned c=0;c<candidate->bulk_history.tensor->primary.coefficients.size();++c)
          for(const auto i:candidate->bulk_history.tensor->primary.coefficients[c].locally_owned_elements())
            diagnostic_error=std::max(diagnostic_error,std::abs(candidate->bulk_history.tensor->primary.coefficients[c][i]-changed_diagnostic->bulk_history.tensor->primary.coefficients[c][i]));
        AssertThrow(Utilities::MPI::max(diagnostic_error,comm)==0.,ExcMessage("Diagnostic Z changed direct reaction/mechanics."));
        s.Z.assign(s.Z.size(),0.);
        const auto perturbed_bulk=velocity(1.+epsilon);
        auto cv=m.make_material_reaction(*m.fe_strain_state,perturbed_bulk,rates, false);
        auto perturbed_rates=rates;for(auto &v:perturbed_rates)v*=1.+epsilon;
        auto cs=m.make_material_reaction(*m.fe_strain_state,bulk,perturbed_rates, false);
        // Same frozen geometry, coefficients, zero physical parent, v and V.
        // Only the history representation changes in this scratch calculation.
        anchor->representation=History::Representation::separated_trial;
        anchor->residual=std::make_shared<typename Model::TensorField>(m.project_dg_transport_field(*op,zero,0.,8)->primary);
        m.material_direct_stress=false;
        auto separated=m.make_material_reaction(*m.fe_strain_state,bulk,rates, false);
        m.material_direct_stress=true;
        anchor->representation=History::Representation::physical_stress;anchor->residual.reset();
        const auto dot=[](const Tensor &a,const Tensor &b){return scalar_product(a,b)+trace(a)*trace(b);};
        std::ofstream out;
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {out.open(this->get_output_directory()+"true-stress-source.csv");out<<std::setprecision(17)<<"q,candidate_L2,local_projection_L2,within_DG_L2,analytic_remainder_L2,decomposition_Linf,velocity_fd_L2,slip_fd_L2,velocity_full_minus_inner_L2,slip_full_minus_inner_L2,projection_energy_loss\n";}
        for(unsigned order:{8u,12u})
          {
            const QGauss<dim> rule(order);
            FEValues<dim> native(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points|update_JxW_values);
            std::vector<double> phi(rule.size());std::array<double,10> sum{};
            for(const auto &entry:s.cells)
              {
                native.reinit(entry.second);
                native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.bulk_history.native,phi);
                const auto direct=m.evaluate_history_tensor(*candidate->bulk_history.tensor,entry.first,rule);
                const auto dv=m.evaluate_history_tensor(*cv->bulk_history.tensor,entry.first,rule);
                const auto ds=m.evaluate_history_tensor(*cs->bulk_history.tensor,entry.first,rule);
                const auto sepT=m.evaluate_history_tensor(*separated->bulk_history.tensor,entry.first,rule);
                const auto sepR=m.evaluate_history_tensor(*separated->bulk_history.tensor->residual,entry.first,rule);
                // Independent higher-order projection of analytic B and local
                // source, using the actual cell mass and Gauss8 below.
                const QGauss<dim> qr(8);
                FEValues<dim> massfe(this->get_mapping(),op->space->element,qr,update_values|update_JxW_values|update_quadrature_points);
                FEValues<dim> phasefe(this->get_mapping(),this->get_fe(),qr,update_values);
                massfe.reinit(op->space->cells.at(entry.first));phasefe.reinit(entry.second);
                std::vector<double> p(qr.size());phasefe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.bulk_history.native,p);
                FullMatrix<double> M(9,9);std::array<Vector<double>,12> rhs;
                for(auto &r:rhs)r.reinit(9);
                for(unsigned q=0;q<qr.size();++q)
                  {
                    const auto x=massfe.quadrature_point(q);const double eta=m.material_viscosity(x,s.eta),k=-eta*std::expm1(-s.G*dt/eta);
                    Tensor D;D[0][0]=.03;D[1][1]=-.03;D[0][1]=.01;
                    const auto slip=m.localize_surface_values(x,p[q],rates,s.I);
                    const auto B=m.localize_surface_values(x,p[q],separated->Z,s.I);
                    const std::array<Tensor,4> fields={2*k*D-k*slip,B,2*k*D,-k*slip};
                    for(unsigned i=0;i<9;++i)
                      {const double w=massfe.JxW(q)*massfe.shape_value(i,q);
                       for(unsigned j=0;j<9;++j)M(i,j)+=w*massfe.shape_value(j,q);
                       for(unsigned f=0;f<4;++f)for(unsigned c=0;c<3;++c)rhs[3*f+c][i]+=w*fields[f][Tensor::unrolled_to_component_indices(c)];}
                  }
                M.gauss_jordan();std::array<Vector<double>,12> coefficients;
                for(unsigned c=0;c<12;++c){coefficients[c].reinit(9);M.vmult(coefficients[c],rhs[c]);}
                for(unsigned q=0;q<rule.size();++q)
                  {
                    const auto x=native.quadrature_point(q);const double w=native.JxW(q),eta=m.material_viscosity(x,s.eta),k=-eta*std::expm1(-s.G*dt/eta);
                    std::array<Tensor,4> projected;
                    for(unsigned f=0;f<4;++f)for(unsigned c=0;c<3;++c)for(unsigned i=0;i<9;++i)
                      projected[f][Tensor::unrolled_to_component_indices(c)]+=coefficients[3*f+c][i]*op->space->element.shape_value(i,rule.point(q));
                    Tensor D;D[0][0]=.03;D[1][1]=-.03;D[0][1]=.01;
                    const auto slip=m.localize_surface_values(x,phi[q],rates,s.I),B=m.localize_surface_values(x,phi[q],separated->Z,s.I);
                    const auto local=2*k*D-k*slip;
                    const auto sep=sepT[q]-sepR[q]-B,Psep=sepT[q]-sepR[q]-projected[1];
                    const auto within=direct[q]-Psep,remainder=Psep-sep;
                    sum[0]+=w*dot(direct[q],direct[q]);sum[1]+=w*dot(direct[q]-local,direct[q]-local);
                    sum[2]+=w*dot(within,within);sum[3]+=w*dot(remainder,remainder);
                    sum[4]=std::max(sum[4],(direct[q]-sep-within-remainder).norm());
                    const auto ev=(dv[q]-direct[q])/epsilon-projected[2],es=(ds[q]-direct[q])/epsilon-projected[3];
                    sum[5]+=w*dot(ev,ev);sum[6]+=w*dot(es,es);
                    const auto iv=projected[2]-2*k*D,is=projected[3]+k*slip;
                    sum[7]+=w*dot(iv,iv);sum[8]+=w*dot(is,is);
                    sum[9]+=w*(dot(local,local)-dot(direct[q],direct[q]))/(4*s.G);
                    AssertThrow((direct[q]-projected[0]).norm()<1e-11,ExcMessage("Direct source differs from independent projection."));
                  }
              }
            for(unsigned i=0;i<sum.size();++i)sum[i]=i==4 ? Utilities::MPI::max(sum[i],comm) : Utilities::MPI::sum(sum[i],comm);
            AssertThrow(std::sqrt(sum[5])<1e-8 && std::sqrt(sum[6])<1e-8 && sum[4]<1e-12,
                        ExcMessage("Full projected velocity/slip derivative or decomposition failed."));
            if(order==8)AssertThrow(sum[9]>=-1e-12,ExcMessage("Native orthogonal projection gained energy."));
            if(out.is_open()){out<<order;for(unsigned i=0;i<sum.size();++i)out<<','<<(i==4 || i==9 ? sum[i] : std::sqrt(sum[i]));out<<'\n';}
          }
        if(!m.material_variable_viscosity)
          {
            double maximum=0.;
            for(bool forced:{false,true})
              {
                struct RotationFlow : Function<dim>
                {
                  RotationFlow(unsigned n):Function<dim>(n){}
                  double value(const Point<dim> &x,unsigned c) const override
                  {return c==0 ? .4-.08*x[1] : (c==1 ? .08*x[0] : 0.);}
                };
                auto flow=bulk;
                if(!forced)
                  {
                    LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
                    VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),RotationFlow(this->get_fe().n_components()),owned);flow=owned;
                  }
                const auto advection=m.make_dg_transport(flow,8,false);
                Tensor expected;expected[0][0]=1.;expected[1][1]=-.3;expected[0][1]=.4;
                auto h=m.project_dg_transport_field(*advection,[&](const Point<dim> &,double){return expected;},0.,8);
                h->representation=History::Representation::physical_stress;
                for(unsigned step=0;step<4;++step)
                  {
                    const auto inflow=[&](const Point<dim> &,double){return expected;};
                    const unsigned substeps=std::max(1u,unsigned(std::ceil(dt*advection->inverse_advection_time/.1)));
                    for(unsigned k=0;k<substeps;++k)h=m.advance_dg_transport(*advection,*h,inflow,zero,step*dt+k*dt/substeps,dt/substeps);
                    const QGauss<dim> qr(3);
                    for(const auto &cell:advection->cells)
                      for(const auto &v:m.evaluate_history_tensor(*h,cell.id,qr))maximum=std::max(maximum,(v-expected).norm());
                    dealii::Tensor<2,dim> gradient;gradient[0][1]=forced ? -.07 : -.08;gradient[1][0]=forced ? .09 : .08;
                    if(forced){gradient[0][0]=.03;gradient[1][1]=-.03;}
                    SymmetricTensor<2,dim> rotated;
                    if constexpr(dim==2)rotated=ReconstructedFaultUtilities::rotate_plane_strain_stress(expected,gradient,dt);
                    s.objective.tensor=m.project_dg_transport_field(*advection,[&](const Point<dim> &,double){return rotated;},0.,8);
                    s.objective.tensor->representation=History::Representation::physical_stress;
                    const auto update=m.make_material_reaction(*m.fe_strain_state,flow,std::vector<double>(rates.size(),0.), false);
                    const double c=std::cos(.08*dt),sn=std::sin(.08*dt),beta=std::exp(-s.G*dt/s.eta),k=-s.eta*std::expm1(-s.G*dt/s.eta);
                    const auto old=expected;
                    expected[0][0]=beta*(c*c*old[0][0]+sn*sn*old[1][1]-2*c*sn*old[0][1]);
                    expected[1][1]=beta*(sn*sn*old[0][0]+c*c*old[1][1]+2*c*sn*old[0][1]);
                    expected[0][1]=beta*(c*sn*(old[0][0]-old[1][1])+(c*c-sn*sn)*old[0][1]);
                    if(forced){expected[0][0]+=2*k*.03;expected[1][1]-=2*k*.03;expected[0][1]+=2*k*.01;}
                    h=update->bulk_history.tensor;
                    for(const auto &cell:advection->cells)
                      for(const auto &v:m.evaluate_history_tensor(*h,cell.id,qr))maximum=std::max(maximum,std::sqrt(dot(v-expected,v-expected)));
                  }
              }
            maximum=Utilities::MPI::max(maximum,comm);
            AssertThrow(maximum<1e-11,ExcMessage("Native uniform spin/relaxation or forced recurrence failed."));
            if(Utilities::MPI::this_mpi_process(comm)==0)
              {std::ofstream file(this->get_output_directory()+"true-stress-uniform.csv");file<<std::setprecision(17)<<"full3_max_error\n"<<maximum<<'\n';}
          }
        if(m.material_variable_viscosity)true_stress_moving_checks();
        s.objective.tensor=saved_anchor;s.Z=saved_Z;
        AssertThrow(snapshot()==accepted,ExcMessage("Direct scratch checks changed accepted state."));
      }
