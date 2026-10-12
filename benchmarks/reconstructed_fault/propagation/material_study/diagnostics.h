  // Read-only instrumentation for this serial, fixed 32x32 P2 fixture.
  std::ofstream study_particles;
  std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>> study_profiles;
  std::vector<double> study_slip;
  std::map<types::particle_index,double> study_phi0;
  double study_cpdi_gap=0, study_initial_H_error=0, study_cpdi_min=1, study_cpdi_max=0;
  double study_core() const {
    return this->get_reconstructed_fault_manager().get_prescribed_faults().at(0).core_phase_field_values.front();
  }
  double study_energy() const {
    return A::coefficients(this->get_phase_field_handler(), {.5,.5}, 0., 0.).reaction;
  }
  double study_m() const {
    return (1./this->get_phase_field_handler().energetic_degradation({.5,.5}, .5)-1.)/3.;
  }
  void study_phase(const std::string &kind, const std::vector<double> &Ih) {
    const auto &state=this->get_solution();
    const auto &h=this->get_phase_field_handler();
    const auto &dofs=A::vertex_dofs(h);
    const std::string suffix=kind+"-"+std::to_string(this->get_timestep_number())+".csv";
    std::ofstream out(this->get_output_directory()+"study-phase-"+suffix);
    out<<std::setprecision(17)<<"x,y,phi,g\n";
    std::vector<std::vector<double>> grid(33,std::vector<double>(33,numbers::signaling_nan<double>()));
    double maximum=-1, xmax=0,ymax=0;
    for (unsigned int v=0;v<dofs.size();++v)
      if(dofs[v]!=numbers::invalid_dof_index && state.locally_owned_elements().is_element(dofs[v])) {
        const auto &p=this->get_triangulation().get_vertices()[v];
        const double phi=state[dofs[v]];
        out<<p[0]<<','<<p[1]<<','<<phi<<','<<h.energetic_degradation({.5,.5},phi)<<'\n';
        grid.at(std::lround(32*p[0])).at(std::lround(32*p[1]))=phi;
        if(phi>maximum){maximum=phi;xmax=p[0];ymax=p[1];}
      }
    out.close();
    // Exact integration of a transverse Q1 profile for p=1, h=m(2/u^2-3/u+1).
    // Native g is independently checked by analyze.py; no live cache is changed.
    std::ofstream norm(this->get_output_directory()+"study-normalization-"+suffix);
    norm<<std::setprecision(17)<<"x,I_direct,I_native,source_integral,phi_fault\n";
    double minI=1e300, max_relative=0, left=0,right=0;
    for(unsigned int k=0;k<=320;++k){
      const double x=k/320., z=32*x;
      const unsigned int j=std::min(31u,static_cast<unsigned int>(z));
      const long double t=z-j;
      long double direct=0;
      for(unsigned int n=0;n<32;++n){
        const long double u0=1-((1-t)*grid[j][n]+t*grid[j+1][n]);
        const long double u1=1-((1-t)*grid[j][n+1]+t*grid[j+1][n+1]);
        if(u0<=0 || u1<=0) {direct=std::numeric_limits<long double>::infinity();break;}
        const long double d=(u1-u0)/u0;
        const long double logmean=(d==0 ? 1.L : std::log1p(d)/d)/u0;
        direct+=study_m()/32.L*(2/(u0*u1)-3*logmean+1);
      }
      const double sf=x*(Ih.size()-1);
      const unsigned int si=std::min<unsigned int>(Ih.size()-2,static_cast<unsigned int>(sf));
      const double interpolated=(1-(sf-si))*Ih[si]+(sf-si)*Ih[si+1];
      norm<<x<<','<<direct<<','<<interpolated<<','<<direct/interpolated<<','
          <<(1-t)*grid[j][16]+t*grid[j+1][16]<<'\n';
      minI=std::min(minI,interpolated);
      max_relative=std::max(max_relative,std::abs(static_cast<double>(direct/interpolated)-1));
      if(k==0)left=direct/interpolated;
      if(k==320)right=direct/interpolated;
    }
    norm.close();
    std::ofstream summary(this->get_output_directory()+"study-state-"+suffix);
    summary<<std::setprecision(17)<<"time,phi_max,x_max,y_max,I_min,max_source_error,left_source_integral,right_source_integral\n"
      <<this->get_time()<<','<<maximum<<','<<xmax<<','<<ymax<<','<<minI<<','<<max_relative<<','<<left<<','<<right<<'\n';
    summary.close();
    AssertThrow(std::isfinite(max_relative) && maximum<1 && minI>0,
                ExcMessage("P2 material-study normalization inadmissible; entry retained."));
  }
  void study_begin(const unsigned int step){
    const auto &h=this->get_phase_field_handler();
    if(step==0){
      study_profiles=h.get_phase_field_profiles(study_core());
      const auto Gc=material().get_critical_energy_release_rates();
      const auto Hc=material().get_critical_crack_driving_forces();
      std::ofstream params(this->get_output_directory()+"study-runtime.csv");
      params<<std::setprecision(17)<<"constituent,Gc,Hc,energy,m_native,m_formula,core,ell,p\n";
      for(unsigned int i=0;i<Gc.size();++i){
        std::vector<double> fraction(Gc.size(),0);fraction[i]=1;
        const double E=A::coefficients(h,fraction,0,0).reaction;
        const double m=(1/h.energetic_degradation(fraction,.5)-1)/3;
        const double expected=Gc[i]/((8./3.)*.15625*Hc[i]);
        AssertThrow(std::abs(m/expected-1)<1e-13,ExcMessage("Native m calibration mismatch."));
        AssertThrow(std::abs(h.energetic_degradation(fraction,.25)-.5625/(.5625+m*.25*1.25))<1e-14,
                    ExcMessage("Expected p=1 degradation mismatch."));
        params<<i<<','<<Gc[i]<<','<<Hc[i]<<','<<E<<','<<m<<','<<expected<<','<<study_core()<<",0.15625,1\n";
        this->get_pcout()<<std::setprecision(17)<<"P2 runtime constituent "<<i<<": Gc="<<Gc[i]<<" Hc="<<Hc[i]<<" E="<<E<<" m="<<m<<" core="<<study_core()<<std::endl;
      }
      study_slip.assign(property("phase field fault cohesive traction").size(),0.);
    }
    study_cpdi_gap=0;study_initial_H_error=0;study_cpdi_min=1;study_cpdi_max=0;
    study_particles.open(this->get_output_directory()+"study-particles-"+std::to_string(step)+".csv");
    study_particles<<std::setprecision(17)<<"id,x,y,volume,phi_point,phi_cpdi,phi_initial_cpdi,H_initial,H_incoming,H_committed,candidate,driving_initial,driving_incoming,driving_committed\n";
  }
  template<class ParticleIterator>
  void study_particle(const ParticleIterator &p,const double phi,const double old_phi,
                      const double Hin,const double Hnew,const double candidate,const double volume){
    (void)old_phi;
    const auto &h=this->get_phase_field_handler();
    const auto domain=pm().get_particle_domain_handler().get_particle_domain(p.get_local_index());
    const auto &dofs=A::vertex_dofs(h);
    double cpdi=0;
    for(unsigned int j=0;j<domain.n_relevant_vertices();++j)
      cpdi+=domain.weighting_function_value(j)*this->get_solution()[dofs[domain.relevant_vertex_index(j)]];
    study_cpdi_gap=std::max(study_cpdi_gap,std::abs(cpdi-phi));
    study_cpdi_min=std::min(study_cpdi_min,cpdi);study_cpdi_max=std::max(study_cpdi_max,cpdi);
    AssertThrow(cpdi>=-3e-14 && cpdi<=1+3e-14,ExcMessage("P2 CPDI phase bounds failed."));
    if(this->get_timestep_number()==0){
      study_phi0[p.get_id()]=cpdi;
      const double initial=study_profiles[0]->value(std::abs(p.get_location()[1]-.5));
      const double expected=initial>1e-6 ? h.stationary_crack_driving_force({.5,.5},initial,study_core()) : material().get_critical_crack_driving_forces()[0];
      study_initial_H_error=std::max(study_initial_H_error,std::abs(expected-Hin)/std::max(1.,expected));
      AssertThrow(study_initial_H_error<1e-13,ExcMessage("Fresh native H initialization mismatch."));
    }
    const double x=p.get_location()[0];
    if(x<1./32 || x>31./32 || std::abs(x-.5)<1./64){
      const double E=study_energy();
      auto drive=[&](double phase,double H){return (E-A::coefficients(h,{.5,.5},phase,H).reaction)/E;};
      study_particles<<p.get_id()<<','<<x<<','<<p.get_location()[1]<<','<<volume<<','<<phi<<','<<cpdi<<','<<study_phi0.at(p.get_id())<<','
         <<initial_H.at(p.get_id())<<','<<Hin<<','<<Hnew<<','<<candidate<<','<<drive(study_phi0.at(p.get_id()),initial_H.at(p.get_id()))<<','<<drive(cpdi,Hin)<<','<<drive(cpdi,Hnew)<<'\n';
    }
  }
  void study_end(){
    study_particles.close();
    const unsigned int step=this->get_timestep_number();
    const auto &manager=this->get_reconstructed_fault_manager();
    const auto &fault=manager.get_fault(0);
    const auto &vel=manager.get_timestep_committed_slip_rate(0);
    const auto q=property("phase field fault cohesive traction");
    std::vector<Point<dim>> points;
    for(unsigned int i=0;i<fault.n_vertices();++i)points.push_back(fault.vertex(i));
    Utilities::MPI::RemotePointEvaluation<dim> cache;
    cache.reinit(this->get_phase_field_handler().get_grid_cache(),points);
    const auto phi=VectorTools::point_values<1>(cache,this->get_dof_handler(),this->get_solution(),
                      VectorTools::EvaluationFlags::avg,this->introspection().variable("phase_field").first_component_index);
    std::ofstream out(this->get_output_directory()+"study-fault-"+std::to_string(step)+".csv");
    out<<std::setprecision(17)<<"x,phi,q,V,slip\n";
    for(unsigned int i=0;i<points.size();++i){
      if(step>0)study_slip[i]+=this->get_timestep()*vel[i];
      out<<points[i][0]<<','<<phi[i]<<','<<q[i]<<','<<vel[i]<<','<<study_slip[i]<<'\n';
    }
    std::ofstream summary(this->get_output_directory()+"study-particle-summary-"+std::to_string(step)+".csv");
    summary<<std::setprecision(17)<<"cpdi_point_max_difference,cpdi_min,cpdi_max,native_initial_H_relative_error\n"
           <<study_cpdi_gap<<','<<study_cpdi_min<<','<<study_cpdi_max<<','<<study_initial_H_error<<'\n';
  }
