/* Included inside the alternative diagnostic MomentCycle class. */
      void boundary_profile_checks() const
      {
        using Tensor = SymmetricTensor<2,dim>;
        using History = typename Model::DGHistory;
        const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
        const auto accepted=snapshot();
        const auto comm=this->get_mpi_communicator();
        const auto dot3=[](const Tensor &a,const Tensor &b)
          {return scalar_product(a,b)+trace(a)*trace(b);};
        const double G=10., finish=.4;
        struct Flow : Function<dim>
        {
          Flow(unsigned n,double speed):Function<dim>(n),speed(speed) {}
          double speed;
          double value(const Point<dim> &,unsigned component) const override
          {return component==0 ? speed : 0.;}
        };
        std::ofstream out;
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {
            out.open(this->get_output_directory()+"boundary-profile-transport.csv");
            out<<std::setprecision(17)
               <<"kind,direction,cells,dt,steps,relative_L2_consistent,relative_L2_extended,interior_relative_L2,initial_energy,final_energy,exact_final_energy,boundary_flux_integral,upwind_dissipation_integral,energy_trapezoid_remainder,max_semidiscrete_identity_defect\n";
          }
        for(const double direction:{-1.,1.})
          {
            const double speed=.4*direction;
            LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
            VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),
                                    Flow(this->get_fe().n_components(),speed),owned);
            LinearAlgebra::BlockVector velocity(this->get_solution());velocity=owned;
            const auto op=model.make_dg_transport(velocity,8,false);
            const auto &space=*op->space;
            const auto value=[&](const History &h,const auto &cell,const Point<dim> &x)
              {
                const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,x);
                std::vector<types::global_dof_index> ids(9);cell->get_dof_indices(ids);
                Tensor a;
                for(unsigned c=0;c<3;++c)
                  for(unsigned i=0;i<9;++i)
                    a[Tensor::unrolled_to_component_indices(c)]+=h.trial_stress[c][ids[i]]*space.element.shape_value(i,unit);
                return a;
              };
            for(unsigned kind=0;kind<3;++kind)
              {
                // 0: representable polynomial with exact inflow;
                // 1: identical initial field, deliberately zero inflow;
                // 2: compact C3 bump, exactly zero boundary data.
                const auto exact=[=](const Point<dim> &x,double t)
                  {
                    const double u=x[0]-speed*t, v=x[1];
                    const double f=kind==2 ? std::pow(std::max(0.,1-u*u),4)*std::pow(std::max(0.,1-v*v),4)
                                          : 1.+.15*u+.1*v+.08*u*v;
                    Tensor a;a[0][0]=f;a[1][1]=-.3*f;a[0][1]=.45*f;return a;
                  };
                const auto zero=[](const Point<dim> &,double){return Tensor();};
                const typename Model::DGField inflow=kind==0 ? typename Model::DGField(exact) : typename Model::DGField(zero);
                const auto energy=[&](const History &h,double time)
                  {
                    // Independently integrate volume energy and face flux/jumps.
                    // The algebraic RHS uses the native operator's actual rows.
                    History rhs;rhs.space=op->space;
                    std::array<LinearAlgebra::Vector,3> rows;
                    for(unsigned c=0;c<3;++c)
                      {
                        rows[c].reinit(space.dofs.locally_owned_dofs(),comm);
                        for(const auto &block:op->blocks)
                          for(unsigned i=0;i<block.rows.size();++i)
                            for(unsigned j=0;j<block.columns.size();++j)
                              rows[c][block.rows[i]]+=block.values(i,j)*h.trial_stress[c][block.columns[j]];
                        for(const auto &load:op->inflow)
                          {
                            const auto a=inflow(load.point,time);
                            for(unsigned i=0;i<load.indices.size();++i)
                              rows[c][load.indices[i]]+=load.weights[i]*a[Tensor::unrolled_to_component_indices(c)];
                          }
                        rows[c].compress(VectorOperation::insert);
                        rhs.trial_stress[c].reinit(space.dofs.locally_owned_dofs(),space.relevant,comm);
                        rhs.trial_stress[c]=rows[c];
                      }
                    const QGauss<dim> rule(8);const QGauss<1> face(8);
                    FEValues<dim> fe(this->get_mapping(),space.element,rule,update_quadrature_points|update_JxW_values);
                    std::array<double,4> result{}; // E, dE/dt, physical flux, upwind loss
                    for(const auto &entry:op->cells)
                      {
                        const auto cell=space.cells.at(entry.id);fe.reinit(cell);
                        const auto a=model.evaluate_dg_trial_tensor(h,entry.id,rule);
                        const auto da=model.evaluate_dg_trial_tensor(rhs,entry.id,rule);
                        for(unsigned q=0;q<rule.size();++q)
                          {
                            result[0]+=fe.JxW(q)*dot3(a[q],a[q])/(4*G);
                            result[1]+=fe.JxW(q)*dot3(a[q],da[q])/(2*G);
                          }
                        for(unsigned side=0;side<2;++side)
                          {
                            const double vn=(side ? speed : -speed);
                            const double y0=cell->vertex(0)[1],height=cell->vertex(2)[1]-y0;
                            for(unsigned q=0;q<face.size();++q)
                              {
                                Point<dim> x;x[0]=cell->vertex(side)[0];x[1]=y0+height*face.point(q)[0];
                                const auto a0=value(h,cell,x);
                                const double weight=height*face.weight(q);
                                if(cell->at_boundary(side))
                                  {
                                    const auto g=inflow(x,time);
                                    const auto flux_value=vn>=0 ? a0 : g;
                                    result[2]+=weight*vn*dot3(flux_value,flux_value)/(4*G);
                                    if(vn<0)result[3]+=weight*(-vn)*dot3(a0-g,a0-g)/(4*G);
                                  }
                                else
                                  {
                                    const auto jump=a0-value(h,cell->neighbor(side),x);
                                    // Each interior face has two owned-cell contributions globally.
                                    result[3]+=.5*weight*std::abs(vn)*dot3(jump,jump)/(4*G);
                                  }
                              }
                          }
                      }
                    for(auto &v:result)v=Utilities::MPI::sum(v,comm);
                    return result;
                  };
                const bool middle=this->get_triangulation().n_global_active_cells()==256;
                const std::vector<double> intervals=middle ? std::vector<double>{.025,.0125,.00625} : std::vector<double>{.00625};
                for(const double dt:intervals)
                  {
                    auto h=model.project_dg_transport_field(*op,exact,0.,8);
                    const unsigned steps=std::lround(finish/dt);
                    auto previous=energy(*h,0.);const double initial=previous[0];
                    double flux=0.,loss=0.,identity=std::abs(previous[1]+previous[2]+previous[3]);
                    for(unsigned k=0;k<steps;++k)
                      {
                        h=model.advance_dg_transport(*op,*h,inflow,zero,k*dt,dt);
                        const auto current=energy(*h,(k+1)*dt);
                        flux+=.5*dt*(previous[2]+current[2]);loss+=.5*dt*(previous[3]+current[3]);
                        identity=std::max(identity,std::abs(current[1]+current[2]+current[3]));
                        previous=current;
                      }
                    const QGauss<dim> rule(10);
                    FEValues<dim> fe(this->get_mapping(),space.element,rule,update_quadrature_points|update_JxW_values);
                    std::array<double,7> errors{};
                    for(const auto &entry:op->cells)
                      {
                        fe.reinit(space.cells.at(entry.id));
                        const auto actual=model.evaluate_dg_trial_tensor(*h,entry.id,rule);
                        for(unsigned q=0;q<rule.size();++q)
                          {
                            const auto x=fe.quadrature_point(q);const auto extended=exact(x,finish);
                            auto expected=extended;
                            if(kind==1 && std::abs(x[0]-speed*finish)>2.)expected=Tensor();
                            const auto e=actual[q]-expected, ee=actual[q]-extended;
                            const double w=fe.JxW(q);
                            errors[0]+=w*dot3(e,e);errors[1]+=w*dot3(expected,expected);
                            errors[2]+=w*dot3(ee,ee);errors[3]+=w*dot3(extended,extended);
                            if(std::abs(x[0])<1.25 && std::abs(x[1])<1.25)
                              {errors[4]+=w*dot3(ee,ee);errors[5]+=w*dot3(extended,extended);}
                            errors[6]+=w*dot3(expected,expected)/(4*G);
                          }
                      }
                    for(auto &v:errors)v=Utilities::MPI::sum(v,comm);
                    AssertThrow(identity<1e-10,ExcMessage("DG material energy volume/face identity failed."));
                    if(kind==0)AssertThrow(std::sqrt(errors[0]/errors[1])<1e-10,
                                           ExcMessage("Representable translating tensor with exact inflow failed."));
                    if(out.is_open())out<<kind<<','<<direction<<','<<this->get_triangulation().n_global_active_cells()<<','<<dt<<','<<steps<<','
                      <<std::sqrt(errors[0]/errors[1])<<','<<std::sqrt(errors[2]/errors[3])<<','<<std::sqrt(errors[4]/errors[5])<<','
                      <<initial<<','<<previous[0]<<','<<errors[6]<<','<<flux<<','<<loss<<','<<previous[0]-initial+flux+loss<<','<<identity<<'\n';
                  }
              }
          }
        AssertThrow(snapshot()==accepted,ExcMessage("Scratch boundary/profile experiment changed accepted state."));
      }
