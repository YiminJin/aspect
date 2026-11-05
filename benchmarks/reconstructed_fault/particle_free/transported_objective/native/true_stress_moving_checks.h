/* Prescribed shear/translation, layered relaxation and exact split inflow.
 * Scratch tensors exercise native DG transport and projected material reaction.
 * The characteristic reference uses independent component rotations and nested
 * Gauss integration of the scalar relaxation along straight characteristics. */
      void true_stress_moving_checks() const
      {
        using Tensor=SymmetricTensor<2,dim>;using History=typename Model::DGHistory;
        auto &m=const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        auto &s=*m.fe_strain_state;const auto comm=this->get_mpi_communicator();
        const double saved_dt=m.initial_time_step;const auto saved_anchor=s.objective.tensor;
        struct Flow : Function<dim>
        {
          Flow(unsigned n):Function<dim>(n){}
          double value(const Point<dim> &x,unsigned c) const override{return c==0 ? .4+.01*x[1] : 0.;}
        };
        LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
        VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),Flow(this->get_fe().n_components()),owned);
        LinearAlgebra::BlockVector bulk(this->get_solution());bulk=owned;
        const auto op=m.make_dg_transport(bulk,8,false);const auto &space=*op->space;
        const unsigned n=unsigned(std::sqrt(this->get_triangulation().n_global_active_cells()));const double h=4./n;
        std::map<std::pair<unsigned,unsigned>,CellId> cells;
        for(const auto &entry:op->cells)
          {const auto center=space.cells.at(entry.id)->center();cells.emplace(std::make_pair(unsigned((center[0]+2)/h),unsigned((center[1]+2)/h)),entry.id);}
        const auto sample=[&](const History &history,const Point<dim> &x)
          {
            const auto id=cells.at({unsigned((x[0]+2)/h),unsigned((x[1]+2)/h)});const auto cell=space.cells.at(id);
            const auto u=this->get_mapping().transform_real_to_unit_cell(cell,x);
            std::vector<types::global_dof_index> ids(9);cell->get_dof_indices(ids);Tensor value;
            for(unsigned c=0;c<3;++c)for(unsigned i=0;i<9;++i)value[Tensor::unrolled_to_component_indices(c)]+=history.primary.coefficients[c][ids[i]]*space.element.shape_value(i,u);
            return value;
          };
        const auto rotate=[](const Tensor &a,double angle)
          {const double c=std::cos(angle),sn=std::sin(angle);Tensor r;
           r[0][0]=c*c*a[0][0]+sn*sn*a[1][1]-2*c*sn*a[0][1];
           r[1][1]=sn*sn*a[0][0]+c*c*a[1][1]+2*c*sn*a[0][1];
           r[0][1]=c*sn*(a[0][0]-a[1][1])+(c*c-sn*sn)*a[0][1];return r;};
        const QGauss<1> time_rule(16);
        const auto exact=[&](const Point<dim> &x,double time)
          {
            const double speed=.4+.01*x[1];const auto rate=[&](double t){return 1./(1.+.5*std::tanh(4*(x[0]-speed*(time-t))));};
            const auto damping=[&](double start)
              {double integral=0.;for(unsigned q=0;q<time_rule.size();++q)integral+=time_rule.weight(q)*rate(start+(time-start)*time_rule.point(q)[0]);return std::exp(-(time-start)*integral);};
            Tensor initial;const double u=x[0]-speed*time,f=1.+.15*u+.1*x[1]+.08*u*x[1];
            initial[0][0]=f;initial[1][1]=-.3*f;initial[0][1]=.45*f;
            Tensor value=damping(0.)*rotate(initial,-.005*time),source;source[0][1]=.1;
            for(unsigned q=0;q<time_rule.size();++q)
              {const double t=time*time_rule.point(q)[0];value+=time*time_rule.weight(q)*damping(t)*rotate(source,-.005*(time-t));}
            return value;
          };
        for(double x:{-1.,0.,1.}){Point<dim> p;p[0]=x;AssertThrow(std::abs(m.material_viscosity(p,10.)-10*(1+.5*std::tanh(4*x)))<1e-13,ExcMessage("Layered reference fixture changed."));}
        const typename Model::DGField zero=[](const Point<dim> &,double){return Tensor();};
        std::ofstream out;if(Utilities::MPI::this_mpi_process(comm)==0)
          {out.open(this->get_output_directory()+"true-stress-moving.csv");out<<std::setprecision(17)<<"cells,dt,end,full3_relative_L2,full3_Linf\n";}
        for(double dt:{.05,.025})
          {
            m.initial_time_step=dt;
            auto history=m.project_dg_transport_field(*op,exact,0.,8);history->representation=History::Representation::physical_stress;
            const unsigned steps=unsigned(std::round(.2/dt));
            for(unsigned step=0;step<steps;++step)
              {
                const double time=step*dt;
                const typename Model::DGField inflow=[&](Point<dim> x,double t){x[0]-=(t-time)*(.4+.01*x[1]);return exact(x,time);};
                const unsigned count=std::max(1u,unsigned(std::ceil(dt*op->inverse_advection_time/.1)));
                for(unsigned k=0;k<count;++k)history=m.advance_dg_transport(*op,*history,inflow,zero,time+k*dt/count,dt/count);
                dealii::Tensor<2,dim> gradient;gradient[0][1]=.01;
                s.objective.tensor=m.project_dg_transport_field(*op,[&](const Point<dim> &x,double)
                  {if constexpr(dim==2)return ReconstructedFaultUtilities::rotate_plane_strain_stress(sample(*history,x),gradient,dt);else return Tensor();},0.,8);
                s.objective.tensor->representation=History::Representation::physical_stress;
                history=m.make_material_reaction(*m.fe_strain_state,bulk,std::vector<double>(s.Z.size(),0.))->bulk_history.tensor;
              }
            double error=0.,scale=0.,maximum=0.;const QGauss<dim> rule(12);
            FEValues<dim> fe(this->get_mapping(),space.element,rule,update_quadrature_points|update_JxW_values);
            const auto norm=[](const Tensor &a){return scalar_product(a,a)+trace(a)*trace(a);};
            for(const auto &cell:op->cells)
              {fe.reinit(space.cells.at(cell.id));const auto values=m.evaluate_history_tensor(*history,cell.id,rule);
               for(unsigned q=0;q<rule.size();++q){const auto expected=exact(fe.quadrature_point(q),.2);error+=fe.JxW(q)*norm(values[q]-expected);scale+=fe.JxW(q)*norm(expected);maximum=std::max(maximum,std::sqrt(norm(values[q]-expected)));}}
            error=Utilities::MPI::sum(error,comm);scale=Utilities::MPI::sum(scale,comm);maximum=Utilities::MPI::max(maximum,comm);
            if(out.is_open())out<<n*n<<','<<dt<<",.2,"<<std::sqrt(error/scale)<<','<<maximum<<'\n';
          }
        m.initial_time_step=saved_dt;s.objective.tensor=saved_anchor;
      }
