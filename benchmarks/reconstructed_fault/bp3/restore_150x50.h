// Explicit restored-chart benchmark; no selector changes existing BP3/BP5.
#include <deal.II/base/quadrature_lib.h>
#include <numeric>

namespace aspect
{
  namespace BP3Restore
  {
    inline std::vector<double> radius, phi, integral;
    inline double degradation_scale=0.;
    inline std::string filter_mode="helmholtz";
    inline double filter_length=20.;
    inline std::vector<double> incoming_theta;

    inline double h(const double p)
    {return degradation_scale*p*(1+p)/((1-p)*(1-p));}

    // Integral of the SAME tabulated stationary profile used for completion.
    // Integrate the final table interval, rather than differentiating a coarse
    // piecewise-linear cumulative table in the bottom boundary layer.
    inline double cumulative(const double r)
    {
      AssertThrow(!radius.empty(),ExcMessage("Missing restored BP3 loading profile."));
      const double a=std::abs(r);
      if (a>=radius.back()) return r<0 ? 0. : 1.;
      const unsigned int i=std::upper_bound(radius.begin(),radius.end(),a)-radius.begin()-1;
      static const QGauss<1> quadrature(8);
      double value=integral[i];
      for (unsigned int q=0;q<quadrature.size();++q)
        {
          const double x=radius[i]+(a-radius[i])*quadrature.point(q)[0];
          const double p=phi[i]+(phi[i+1]-phi[i])*(x-radius[i])/(radius[i+1]-radius[i]);
          value+=(a-radius[i])*quadrature.weight(q)*h(p);
        }
      return .5+(r<0 ? -.5 : .5)*value/integral.back();
    }

    template <int dim>
    Tensor<1,dim> loading(const SimulatorAccess<dim> &,const Point<dim> &p)
    {
      const double rate=-BP3::Vp*(cumulative(BP3::signed_normal(p[0],p[1]))-.5);
      Tensor<1,dim> u;u[0]=rate*BP3::cosine;u[1]=-rate*BP3::sine;
      return u;
    }

    template <int dim>
    void before_mechanics(const SimulatorAccess<dim> &sim,bool temperature,unsigned int,const SolverControl &)
    {
      if (!temperature) return;
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      incoming_theta.resize(fault.n_vertices());
      for(unsigned int i=0;i<fault.n_vertices();++i) incoming_theta[i]=fault.get_properties(i)[state];
      if(sim.get_timestep_number()!=0) return;
      const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(BP3::core_phi);
      double error=0.;
      for(unsigned int i=0;i<radius.size();++i)
        error=std::max(error,std::abs(profiles[0]->value(radius[i])-phi[i]));
      AssertThrow(error<2e-11,ExcMessage("Restored completion/loading profile differs from the production profile."));
      const auto &handler=sim.get_phase_field_handler();
      for(const double p:{.1,.3,.6})
        AssertThrow(std::abs((1./handler.energetic_degradation({1.,0.},p)-1.)/h(p)-1.)<1e-12,
                    ExcMessage("Restored profile degradation differs from the production law."));
      sim.get_pcout()<<"Restored BP3 stationary profile: max error="<<error
                     <<", full normal integral="<<2*integral.back()<<std::endl;
    }
  }

  namespace Postprocess
  {
    template <int dim>
    class BP3RestoredMonitor : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");prm.enter_subsection("BP3 restored monitor");
        prm.declare_entry("Stationary profile file","",Patterns::Anything());
        prm.declare_entry("Friction normal input","helmholtz",Patterns::Selection("raw|helmholtz"));
        prm.declare_entry("Normal filter length","20",Patterns::Double(0));
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Postprocess");prm.enter_subsection("BP3 restored monitor");
        profile_path=prm.get("Stationary profile file");
        BP3Restore::filter_mode=prm.get("Friction normal input");
        BP3Restore::filter_length=prm.get_double("Normal filter length");
        prm.leave_subsection();prm.leave_subsection();
      }
      void initialize() override
      {
        using namespace BP3Restore;
        std::istringstream in(Utilities::read_and_distribute_file_content(
          Utilities::expand_ASPECT_SOURCE_DIR(profile_path),this->get_mpi_communicator()));
        unsigned int n=0;
        AssertThrow(in>>n>>degradation_scale && n>2 && degradation_scale>0.,ExcMessage("Invalid restored loading table."));
        radius.resize(n);phi.resize(n);integral.resize(n);
        for(unsigned int i=0;i<n;++i)
          AssertThrow(in>>radius[i]>>phi[i]>>integral[i]
                      && (i==0 || (radius[i]>radius[i-1] && integral[i]>=integral[i-1])),
                      ExcMessage("Invalid restored profile row."));
        AssertThrow(radius[0]==0. && integral[0]==0. && integral.back()>0.,ExcMessage("Invalid profile normalization."));
        this->get_reconstructed_fault_surface_system().set_normal_stress_filter(filter_mode,filter_length);
        this->get_reconstructed_fault_surface_system().set_normal_traction_diagnostic(
          [](const Point<dim> &p)
          {
            const double s=BP3::down_dip(p[0],p[1]);
            return s<200. || s>BP3::box_size/BP3::sine-200.
                   || std::abs(s-15000.)<100. || std::abs(s-18000.)<100. || std::abs(s-40000.)<100.;
          });
        this->get_signals().post_advection_solver.connect(&before_mechanics<dim>);
      }
      void save(std::map<std::string,std::string> &status) const override
      {status["BP3 restored model"]=identity();}
      void load(const std::map<std::string,std::string> &status) override
      {
        const auto entry=status.find("BP3 restored model");
        AssertThrow(entry!=status.end() && entry->second==identity(),
                    ExcMessage("Cannot change restored BP3 geometry, profile or normal filter across restart."));
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        const auto &w=this->get_reconstructed_fault_surface_system().get_linearization_residual();
        const auto &raw=w.raw_normal_traction.empty() ? w.normal_traction[0] : w.raw_normal_traction[0];
        const auto solve=[&](const auto &rhs){return ReconstructedFaultUtilities::solve_tridiagonal_system(
          w.mass_diagonal[0],w.mass_off_diagonal[0],rhs);};
        const auto raw_q1=solve(raw),filtered=solve(w.normal_traction[0]);
        const auto &V=manager.get_timestep_committed_slip_rate(0);
        const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
        const auto step=this->get_timestep_number();
        const auto comm=this->get_mpi_communicator();
        // These samples were captured in mechanics before history publication;
        // do not recompute current stress from already updated particles.
        const auto &diagnostic=this->get_reconstructed_fault_surface_system().get_normal_traction_diagnostic();
        AssertThrow(diagnostic.step==step,ExcMessage("Restored monitor has stale mechanical samples."));
        const auto rank=Utilities::MPI::this_mpi_process(comm);
        std::ofstream samples(this->get_output_directory()+"restored_raw_"+std::to_string(step)
                              +"_rank"+std::to_string(rank)+".csv");
        samples<<std::setprecision(17)<<"cell,qp,segment,xi,x,y,s,r,cell_h,JxW,work_weight,V,p,deviatoric,sigma_raw,sigma_friction,mu,phi,chi,Ih,working_old_xx,working_old_yy,working_old_xy,particle_interpolated_xx,particle_interpolated_yy,particle_interpolated_xy,constitutive_dt\n";
        for(const auto &q:diagnostic.samples)
          {
            const double rate=manager.interpolate_slip_rate(q.fault,q.segment,q.xi);
            samples<<q.cell<<','<<q.qp<<','<<q.segment<<','<<q.xi<<','<<q.position[0]<<','<<q.position[1]
                   <<','<<BP3::down_dip(q.surface_position[0],q.surface_position[1])
                   <<','<<BP3::signed_normal(q.position[0],q.position[1])<<','<<q.cell_size<<','<<q.JxW<<','<<q.weight
                   <<','<<rate<<','<<q.pressure<<','<<q.deviatoric<<','<<q.total<<','<<q.friction_normal<<','<<q.friction_coefficient
                   <<','<<q.phase<<','<<q.chi<<','<<q.I_h
                   <<','<<q.incoming_stress[0][0]<<','<<q.incoming_stress[1][1]<<','<<q.incoming_stress[0][1]
                   <<','<<q.particle_interpolated_stress[0][0]<<','<<q.particle_interpolated_stress[1][1]<<','<<q.particle_interpolated_stress[0][1]
                   <<','<<q.stress_time_step<<'\n';
          }
        std::ofstream particles(this->get_output_directory()+"restored_incoming_particles_"+std::to_string(step)
                                +"_rank"+std::to_string(rank)+".csv");
        particles<<std::setprecision(17)<<"cell,id,rank,x,y,old_xx,old_yy,old_xy\n";
        for(const auto &p:diagnostic.particles)
          particles<<p.cell<<','<<p.id<<','<<p.owner_rank<<','<<p.position[0]<<','<<p.position[1]
                   <<','<<p.stress[0][0]<<','<<p.stress[1][1]<<','<<p.stress[0][1]<<'\n';
        // Flux uses physical velocities on ALL boundaries, including the free
        // top; the three prescribed sides need not balance on their own.
        FEFaceValues<dim> face(this->get_mapping(),this->get_fe(),QGauss<dim-1>(3),
          update_values|update_normal_vectors|update_JxW_values);
        std::vector<Tensor<1,dim>> velocity(face.n_quadrature_points);
        double local_flux[4]={0,0,0,0};
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          if(cell->is_locally_owned())
            for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell;++f)
              if(cell->face(f)->at_boundary())
                {
                  face.reinit(cell,f);
                  face[this->introspection().extractors.velocities].get_function_values(this->get_solution(),velocity);
                  const auto id=cell->face(f)->boundary_id();
                  AssertThrow(id<4,ExcMessage("Restored BP3 requires Cartesian Box boundaries."));
                  for(unsigned int q=0;q<velocity.size();++q)
                    local_flux[id]+=velocity[q]*face.normal_vector(q)*face.JxW(q);
                }
        double flux[4]; for(unsigned int b=0;b<4;++b) flux[b]=Utilities::MPI::sum(local_flux[b],comm);
        const double minimum=w.raw_normal_traction.empty()?w.minimum_normal_traction:w.minimum_raw_normal_traction;
        AssertThrow(w.minimum_normal_traction>0.,ExcMessage("Restored BP3 friction normal input lost compression."));
        if(this->get_pcout().is_active())
          {
            std::ofstream out(this->get_output_directory()+"restored_fault_"+std::to_string(step)+".csv");
            out<<std::setprecision(17)<<"node,s_m,time_s,V,Theta_in,Theta_committed,Omega,raw_sigma_row,raw_sigma_Q1,friction_sigma_Q1,V_chord,raw_chord,filtered_chord,work_mass\n";
            double max_chord=0.,mean_error=0.;
            const auto chord=[&](const auto &z,unsigned int i)
              {
                if(i==0 || i+1==z.size()) return 0.;
                const double left=(fault.vertex(i)-fault.vertex(i-1)).norm(),right=(fault.vertex(i+1)-fault.vertex(i)).norm();
                return z[i]-(right*z[i-1]+left*z[i+1])/(left+right);
              };
            for(unsigned int i=0;i<V.size();++i)
              {
                const double mass=w.mass_diagonal[0][i]+(i?w.mass_off_diagonal[0][i-1]:0.)
                                  +(i+1<V.size()?w.mass_off_diagonal[0][i]:0.);
                mean_error+=w.normal_traction[0][i]-raw[i];
                max_chord=std::max(max_chord,std::abs(chord(V,i)));
                const double theta=fault.get_properties(i)[state];
                out<<i<<','<<BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1])<<','<<this->get_time()<<','<<V[i]
                   <<','<<BP3Restore::incoming_theta.at(i)<<','<<theta<<','<<V[i]*theta/.008
                   <<','<<raw[i]/mass<<','<<raw_q1[i]<<','<<filtered[i]<<','<<chord(V,i)
                   <<','<<chord(raw_q1,i)<<','<<chord(filtered,i)<<','<<mass<<'\n';
              }
            std::ofstream summary(this->get_output_directory()+"restored_growth.csv",std::ios::app);
            if(step==0) summary<<"step,time,dt,min_raw_sigma,min_friction_sigma,max_V_chord,filter_weak_mean_error,flux_left,flux_right,flux_bottom,flux_top\n";
            summary<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<minimum<<','<<w.minimum_normal_traction
                   <<','<<max_chord<<','<<mean_error;
            for(double f:flux) summary<<','<<f;
            summary<<'\n';
          }
        return {"Restored BP3 monitor","raw/filtered traction and endpoint growth recorded"};
      }
    private:
      std::string profile_path;
      std::string identity() const
      {
        std::ostringstream out;
        out<<std::setprecision(17)<<"BP3 150x50 thrust a=-1 ell20 Q2 LLS-unlimited v1 "
           <<BP3Restore::filter_mode<<' '<<BP3Restore::filter_length<<' '<<BP3Restore::degradation_scale;
        for(unsigned int i=0;i<BP3Restore::radius.size();++i)
          out<<' '<<BP3Restore::radius[i]<<' '<<BP3Restore::phi[i]<<' '<<BP3Restore::integral[i];
        return out.str();
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(BP3RestoredMonitor,"BP3 restored monitor",
      "Accepted raw and filtered traction, unsmoothed slip-rate variation, state timing and boundary flux.")
  }
}
