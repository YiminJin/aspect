/* Benchmark-only parameter/initial-history checks; no state is changed. */
#include "../benchmarks/reconstructed_fault/bp3/bp3_model.h"
#include <aspect/particle/manager.h>

namespace aspect
{
  template <int dim>
  void check_bp3_length_scale(const SimulatorAccess<dim> &sim, bool temperature,
                             unsigned int, const SolverControl &)
  {
    if (!std::getenv("ASPECT_BP3_LENGTH_STUDY") || !temperature || sim.get_timestep_number()!=0) return;
    const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
    const auto &friction=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::fault_friction(model);
    const double dc=model.characteristic_fault_slip_distance();
    double aging_error=0., friction_error=0.;
    unsigned int count=0;
    // Independent configured instances exercise the production update for both
    // old and new parameters; they never replace the live material object.
    for (double trial_dc:{.008,.024})
      {
        auto law=friction;
        ParameterHandler prm;
        MaterialModel::Rheology::FaultFriction<dim>::declare_parameters(prm);
        prm.set("Characteristic slip distance",std::to_string(trial_dc));
        prm.set("Minimum slip rate","1e-20");
        prm.set("Reference friction coefficients","0.6");
        prm.set("Direct effect parameters","0.010, 0.025");
        prm.set("Evolution effect parameters","0.015");
        prm.set("Use regularized formulation","true");
        law.parse_parameters(prm);
        AssertThrow(law.get_characteristic_slip_distance()==trial_dc,ExcMessage("Dc parameter not honored."));
        for (double xd:{0.,15000.,16500.,18000.,40000.,100000.})
          {
            const double old=BP3::theta0(xd,trial_dc), f=BP3::fraction(xd);
            const double target=(BP3::tau0-BP3::damping*BP3::Vinit)/BP3::sigma0;
            friction_error=std::max(friction_error,std::abs(law.friction_coefficient({1-f,f},BP3::Vinit,old)/target-1));
            for(double v:{1e-20,3.0611065791756763e-17,1e-9,1e-6})
              for(double dt:{0.,1.,4e6,376359254.47685081})
                {
                  const double expected=BP3::aging_state_reference(v,old,dt,trial_dc);
                  aging_error=std::max(aging_error,std::abs(law.update_state({1-f,f},v,old,dt)/expected-1));
                  ++count;
                }
          }
      }
    auto &manager=sim.get_reconstructed_fault_manager();
    const auto &fault=manager.get_fault(0);
    const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
    double nodal_error=0., particle_error=0.;
    for(unsigned int j=0;j<fault.n_vertices();++j)
      {
        const auto p=fault.vertex(j);
        const double expected=BP3::configured_initial_state(BP3::down_dip(p[0],p[1]),friction);
        nodal_error=std::max(nodal_error,std::abs(fault.get_properties(j)[state]/expected-1));
        AssertThrow(std::getenv("ASPECT_BP5_SHORT_TEST") || std::abs(expected/BP3::theta0(BP3::down_dip(p[0],p[1]),.008)-dc/.008)<1e-12,
                    ExcMessage("Initial Theta/Dc changed."));
      }
    auto &particles=sim.get_phase_field_handler().get_associated_particle_manager();
    const auto &info=particles.get_property_manager().get_data_info();
    unsigned int checked_fields=0;
    for(const std::string name:{"phase field fault state","initial theta_initial"})
      if(info.fieldname_exists(name))
        {
          ++checked_fields;const unsigned int pos=info.get_position_by_field_name(name);
          for(const auto &p:particles.get_particle_handler())
            {
              const double expected=BP3::configured_initial_state((BP3::box_size-p.get_location()[1])/BP3::sine,friction);
              particle_error=std::max(particle_error,std::abs(p.get_properties()[pos]/expected-1));
            }
        }
    particle_error=Utilities::MPI::max(particle_error,sim.get_mpi_communicator());
    if (std::getenv("ASPECT_BP5_SHORT_TEST"))
      {
        AssertThrow(dc==.1,ExcMessage("BP5 diagnostic requires configured Dc=.1."));
        std::ofstream out;
        if(sim.get_pcout().is_active())
          {out.open(sim.get_output_directory()+"friction_configuration.csv");out<<std::setprecision(17)<<"xd,a,b,Dc,Theta0,mu,muV,nominal_q_Pa,inverse_error\n";}
        for (double xd:{0.,15000.,16500.,17166.666666666668,18000.,40000.,100000.})
          {
            const double f=BP3::fraction(xd), a=.004+.036*f;
            const double theta=BP3::configured_initial_state(xd,friction);
            const double z=BP3::Vinit/(2e-6)*std::exp((.6+.03*std::log(theta*1e-6/.1))/a);
            const double expected=a*std::asinh(z);
            const double mu=friction.friction_coefficient({1-f,f},BP3::Vinit,theta);
            const double derivative=friction.friction_coefficient_derivative_wrt_slip_rate({1-f,f},BP3::Vinit,theta);
            const double error=std::abs(mu/expected-1);
            AssertThrow(error<1e-12 && std::abs(derivative/(a/BP3::Vinit*z/std::hypot(1.,z))-1)<1e-12,
                        ExcMessage("BP5 live material parameters differ from requested coefficients."));
            for(double v:{1e-20,1e-9,1e-6})
              for(double dt:{0.,1.,4e6})
                AssertThrow(std::abs(friction.update_state({1-f,f},v,theta,dt)/BP3::aging_state_reference(v,theta,dt,dc)-1)<1e-12,
                            ExcMessage("BP5 configured aging law mismatch."));
            if(out)out<<xd<<','<<a<<",0.03,"<<dc<<','<<theta<<','<<mu<<','<<derivative<<','
                      <<BP3::sigma0*mu+BP3::damping*BP3::Vinit<<','<<error<<'\n';
          }
      }
    AssertThrow(checked_fields>0 && nodal_error<1e-12 && particle_error<1e-12 && aging_error<1e-12 && friction_error<1e-12,
                ExcMessage("BP3 configured-Dc initialization/update consistency failed."));
    if(sim.get_pcout().is_active())
      {
        std::ofstream out(sim.get_output_directory()+"Dc_consistency.csv");
        out<<std::setprecision(17)<<"Dc,production_aging_cases,aging_error,initial_friction_error,nodal_Theta_error,particle_Theta_error,particle_fields\n"
           <<dc<<','<<count<<','<<aging_error<<','<<friction_error<<','<<nodal_error<<','<<particle_error<<','<<checked_fields<<'\n';
      }
  }
}
