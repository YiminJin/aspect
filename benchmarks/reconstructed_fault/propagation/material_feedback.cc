// Read-only material comparison built on the audited sequential P2 observer.
#include "coupled_feedback.cc"
#include <array>

namespace aspect {
namespace Postprocess {
template <int dim> class MaterialFeedback : public CoupledFeedback<dim> {
  using Base = CoupledFeedback<dim>;
  using A = internal::PhaseFieldTestAccess<dim>;
  using F = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
protected:
  double energy_scale = 0, degradation_m = 0;
  std::vector<double> slip;
  std::array<double, 33*33> nodal;

  void calibration() {
    if (energy_scale != 0) return;
    auto &h = this->get_phase_field_handler();
    const auto Gc = this->material().get_critical_energy_release_rates();
    const auto Hc = this->material().get_critical_crack_driving_forces();
    std::ofstream out(this->get_output_directory()+"material-calibration.csv");
    out << std::setprecision(17) << "constituent,Gc,Hc,ell,c0,E,m,m_expected,core\n";
    for (unsigned int j=0; j<Gc.size(); ++j) {
      std::vector<double> fractions(Gc.size(),0.); fractions[j]=1.;
      // For AT1, reaction(phi=0,H=0)=E; g'(0)=-m in the native Lorentz law.
      const double E = A::coefficients(h,fractions,0,0).reaction;
      const double m = E-A::coefficients(h,fractions,0,1).reaction;
      const double ell=h.get_length_scale(), c0=Gc[j]/(E*ell);
      const double expected=Gc[j]/((8./3.)*ell*Hc[j]);
      AssertThrow(std::abs(c0-8./3.)<1e-14 && std::abs(m/expected-1)<1e-13
                  && Hc[j]==.5, ExcMessage("Native material calibration mismatch."));
      if (j==0) { energy_scale=E; degradation_m=m; }
      else AssertThrow(E==energy_scale && m==degradation_m,
                       ExcMessage("Study requires identical constituent phase parameters."));
      out << j << ',' << Gc[j] << ',' << Hc[j] << ',' << ell << ',' << c0 << ','
          << E << ',' << m << ',' << expected << ',' << this->nominal_core() << '\n';
      this->get_pcout() << "P2 material calibration: constituent=" << j
        << " Gc=" << std::setprecision(17) << Gc[j] << " Hc=" << Hc[j]
        << " E=" << E << " m=" << m << " core=" << this->nominal_core() << std::endl;
    }
  }
  double cpdi(const typename Particles::ParticleHandler<dim>::particle_iterator &p) const {
    const auto domain=this->pm().get_particle_domain_handler().get_particle_domain(p->get_local_index());
    const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
    double value=0;
    for (unsigned int j=0;j<domain.n_relevant_vertices();++j)
      value+=domain.weighting_function_value(j)*this->get_solution()[dofs[domain.relevant_vertex_index(j)]];
    return value;
  }
  double point_phase(const double x, const double y) const {
    const unsigned int i=std::min(31u,static_cast<unsigned int>(32*x));
    const unsigned int j=std::min(31u,static_cast<unsigned int>(32*y));
    const double s=32*x-i,t=32*y-j;
    return (1-s)*((1-t)*nodal[33*i+j]+t*nodal[33*i+j+1])
          +s*((1-t)*nodal[33*(i+1)+j]+t*nodal[33*(i+1)+j+1]);
  }
  double direct_integral(const double x) const {
    // Exact transverse integration of this fixed Cartesian Q1 field for p=1.
    // h=m*(2/u^2-3/u+1), u=1-phi. No quadrature can hide a saturated vertex.
    long double result=0;
    for (unsigned int j=0;j<32;++j) {
      const long double a=1-static_cast<long double>(point_phase(x,j/32.));
      const long double b=1-static_cast<long double>(point_phase(x,(j+1)/32.));
      AssertThrow(a>0 && b>0,ExcMessage("P2 material normalization is singular at phi=1."));
      const long double z=(b-a)/a;
      const long double mean_inverse=(z==0 ? 1 : std::log1p(z)/z)/a;
      result+=degradation_m/32.*(2/(a*b)-3*mean_inverse+1);
    }
    return static_cast<double>(result);
  }
  void phase_snapshot(const std::string &stage) {
    calibration();
    AssertThrow(this->get_triangulation().n_global_active_cells()==1024,
                ExcMessage("Material comparison uses the fixed 32x32 Cartesian mesh."));
    const auto step=this->get_timestep_number();
    const auto &h=this->get_phase_field_handler();
    const auto &dofs=A::vertex_dofs(h);
    std::ofstream out(this->get_output_directory()+"material-phase-"+stage+"-"+std::to_string(step)+".csv");
    out << std::setprecision(17) << "x,y,phi,old,g\n";
    double minimum=1,maximum=0,healing=0,change=0; Point<dim> location;
    unsigned int count=0;
    for (unsigned int v=0;v<dofs.size();++v)
      if (dofs[v]!=numbers::invalid_dof_index && this->get_solution().locally_owned_elements().is_element(dofs[v])) {
        const auto &p=this->get_triangulation().get_vertices()[v];
        const double phi=this->get_solution()[dofs[v]],old=this->get_old_solution()[dofs[v]];
        AssertThrow(std::abs(p[0]*32-std::round(p[0]*32))<1e-12 &&
                    std::abs(p[1]*32-std::round(p[1]*32))<1e-12,ExcMessage("Non-Cartesian study mesh."));
        nodal[33*std::lround(32*p[0])+std::lround(32*p[1])]=phi;++count;
        minimum=std::min(minimum,phi);
        if (phi>maximum) {maximum=phi;location=p;}
        if (step>0) {healing=std::max(healing,old-phi);change=std::max(change,std::abs(phi-old));}
        out << p[0] << ',' << p[1] << ',' << phi << ',' << old << ','
            << h.energetic_degradation({.5,.5},phi) << '\n';
      }
    out.close();
    AssertThrow(count==1089,ExcMessage("Incomplete serial Q1 snapshot."));
    double cpmin=1,cpmax=0;
    for (auto p=this->pm().get_particle_handler().begin();p!=this->pm().get_particle_handler().end();++p) {
      const double value=cpdi(p);cpmin=std::min(cpmin,value);cpmax=std::max(cpmax,value);
    }
    std::ofstream metrics(this->get_output_directory()+"material-phase-metrics-"+stage+"-"+std::to_string(step)+".csv");
    metrics << std::setprecision(17) << "time,min,max,max_x,max_y,increment,healing,cpdi_min,cpdi_max\n"
      << this->get_time() << ',' << minimum << ',' << maximum << ',' << location[0] << ',' << location[1]
      << ',' << change << ',' << healing << ',' << cpmin << ',' << cpmax << '\n';
    metrics.close();
    AssertThrow(minimum>=0 && maximum<=1 && cpmin>=-3e-14 && cpmax<=1+3e-14,
                ExcMessage("P2 material study phase bounds failed; entry field retained."));
    AssertThrow(healing<=4.922e-12,
                ExcMessage("P2 material study resolved phase healing; entry field retained."));
    AssertThrow(maximum<1,ExcMessage("P2 material study saturated field: normalization undefined."));
    auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(this->material());
    const auto I=F::compute_normalization_integrals(model).at(0);
    AssertThrow(*std::min_element(I.begin(),I.end())>0,
                ExcMessage("P2 material study nonpositive consistent normalization; entry field retained."));
    const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
    std::ofstream norm(this->get_output_directory()+"material-normalization-"+stage+"-"+std::to_string(step)+".csv");
    norm << std::setprecision(17) << "x,direct,projected,normalized_source_integral\n";
    for (unsigned int k=0;k<=200;++k) {
      const double x=k/200.;
      const unsigned int segment=std::min(fault.n_cells()-1,static_cast<unsigned int>(x*fault.n_cells()));
      const double xi=(x-fault.vertex(segment)[0])/(fault.vertex(segment+1)[0]-fault.vertex(segment)[0]);
      const double projected=(1-xi)*I[segment]+xi*I[segment+1], direct=direct_integral(x);
      norm << x << ',' << direct << ',' << projected << ',' << direct/projected << '\n';
    }
  }
public:
  void initialize() override {
    // Capture/reject bad phase consumption before mechanics can publish H.
    this->get_signals().post_particle_management.connect([this](Particle::Manager<dim>&) {
      if (this->get_timestep_number()>0 && !this->get_reconstructed_fault_manager().get_faults().empty())
        phase_snapshot("entry");
    });
    Base::initialize();
  }
  std::pair<std::string,std::string> execute(TableHandler &table) override {
    const auto result=Base::execute(table);
    phase_snapshot("accepted");
    const auto step=this->get_timestep_number();
    const double dt=this->get_timestep();
    auto &manager=this->get_reconstructed_fault_manager();
    const auto &fault=manager.get_fault(0);
    const auto &V=manager.get_timestep_committed_slip_rate(0);
    const auto q=this->property("phase field fault cohesive traction");
    if (step==0) slip.assign(q.size(),0.);
    std::ofstream local(this->get_output_directory()+"material-local.csv",step==0?std::ios::out:std::ios::app);
    if (step==0) local << "step,time,dt,x,phi,q,V,slip\n";
    for (unsigned int i=0;i<q.size();++i) {
      slip[i]+=dt*V[i];
      local << std::setprecision(17) << step << ',' << this->get_time() << ',' << dt << ',' << fault.vertex(i)[0]
            << ',' << point_phase(fault.vertex(i)[0],.5) << ',' << q[i] << ',' << V[i] << ',' << slip[i] << '\n';
    }
    // Small fixed-location particle probes and a global CPDI-vs-pointwise metric.
    std::array<double,3> nearest{{1e100,1e100,1e100}};
    std::array<types::particle_index,3> selected;
    const std::array<double,3> targets{{0.,.5,1.}};
    for (const auto &p:this->pm().get_particle_handler())
      for (unsigned int i=0;i<3;++i) {
        const double d=std::pow(p.get_location()[0]-targets[i],2)+std::pow(p.get_location()[1]-.5,2);
        if (d<nearest[i]) {nearest[i]=d;selected[i]=p.get_id();}
      }
    std::ofstream particles(this->get_output_directory()+"material-particles.csv",step==0?std::ios::out:std::ios::app);
    if (step==0) particles << "step,time,target,x,y,H0,H_in,H_new,phi_cpdi,phi_point,drive_in_cpdi,drive_new_cpdi,drive_in_point\n";
    double difference=0,drive_max=0;
    auto &h=this->get_phase_field_handler();
    const auto derivative=[&](double phi) {return A::coefficients(h,{.5,.5},phi,1).reaction-A::coefficients(h,{.5,.5},phi,0).reaction;};
    for (auto p=this->pm().get_particle_handler().begin();p!=this->pm().get_particle_handler().end();++p) {
      const double cp=cpdi(p),point=point_phase(p->get_location()[0],p->get_location()[1]);
      const double Hin=this->incoming.at(p->get_id()),H=p->get_properties()[this->hi()];
      difference=std::max(difference,std::abs(cp-point));
      drive_max=std::max(drive_max,-Hin*derivative(cp)/energy_scale);
      for (unsigned int i=0;i<3;++i)
        if (p->get_id()==selected[i])
          particles << std::setprecision(17) << step << ',' << this->get_time() << ',' << targets[i]
            << ',' << p->get_location()[0] << ',' << p->get_location()[1] << ',' << this->initial_H.at(p->get_id())
            << ',' << Hin << ',' << H << ',' << cp << ',' << point << ',' << -Hin*derivative(cp)/energy_scale
            << ',' << -H*derivative(cp)/energy_scale << ',' << -Hin*derivative(point)/energy_scale << '\n';
    }
    std::ofstream diagnostic(this->get_output_directory()+"material-sampling.csv",step==0?std::ios::out:std::ios::app);
    if (step==0) diagnostic << "step,time,max_cpdi_point_difference,max_incoming_normalized_driving\n";
    diagnostic << std::setprecision(17) << step << ',' << this->get_time() << ',' << difference << ',' << drive_max << '\n';
    return result;
  }
};
ASPECT_REGISTER_POSTPROCESSOR(MaterialFeedback,"P2 material feedback",
                             "Fresh fixed-geometry material comparison with native mechanical history.")
}
}
