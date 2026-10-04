// Exercise the maintained audit with native transport, outflow and ID reuse.
#include "../../bp3/plugin/runtime.h"
#include "../../bp3/plugin/particle_initialization.h"
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <boost/serialization/map.hpp>
#include <fstream>
#include <iomanip>
namespace aspect
{
  namespace { bool attach_birth_audit=true; }
  template <int dim>
  void connect_birth_test(SimulatorSignals<dim> &signals)
  {
    signals.post_simulator_initialization.connect([](const SimulatorAccess<dim> &sim)
    {
      // The standalone transport fixture omitted BP3's post-manager startup
      // initializer. Apply the same shared initializer before attaching its audit.
      sim.get_signals().post_set_initial_state.connect([](const SimulatorAccess<dim> &sim)
      {
        auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
        const auto &initializer=pm.get_property_manager().template get_matching_active_plugin<Particle::Property::BP3FrozenCrackDrivingForce<dim>>();
        const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
        for(auto &p:pm.get_particle_handler())p.get_properties()[H]=initializer.initial_H(p.get_location());
      });
      if(attach_birth_audit)BP3Benchmark::connect_particle_history_audit(sim);
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_birth_test<2>,connect_birth_test<3>)
  namespace Postprocess
  {
    template <int dim>
    class BirthAudit : public Interface<dim>, public SimulatorAccess<dim>
    {
      bool repeat_native_step=false;
      public:
        static void declare_parameters(ParameterHandler &prm)
        { prm.enter_subsection("Postprocess");prm.declare_entry("Attach birth identity audit","true",Patterns::Bool());prm.declare_entry("Repeat native birth step","false",Patterns::Bool());prm.leave_subsection(); }
        void parse_parameters(ParameterHandler &prm) override
        { prm.enter_subsection("Postprocess");attach_birth_audit=prm.get_bool("Attach birth identity audit");repeat_native_step=prm.get_bool("Repeat native birth step");prm.leave_subsection(); }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          if(!attach_birth_audit)return {"Birth identity audit","disabled control"};
          auto &pm=this->get_particle_manager(0);
          const auto &info=pm.get_property_manager().get_data_info();
          const auto H=info.get_position_by_field_name("crack_driving_force");
          const auto stress=info.get_position_by_field_name("maxwell stress");
          const auto step=this->get_timestep_number();
          if(step==3 && repeat_native_step)
            {
              const auto snapshot=[&]()
              {
                std::ostringstream state;state<<std::setprecision(17)<<pm.get_random_number_state();
                std::map<types::particle_index,std::string> rows;
                for(const auto &p:pm.get_particle_handler())
                  {
                    std::ostringstream row;row<<std::setprecision(17)<<p.get_location()<<' '
                      <<p.get_reference_location()<<' '<<p.get_surrounding_cell()->id().to_string();
                    for(const double x:p.get_properties())row<<' '<<x;
                    row<<' '<<BP3Benchmark::work_initial_H.at(p.get_id())<<' '
                       <<BP3Benchmark::attempt_births.count(p.get_id());
                    rows[p.get_id()]=row.str();
                  }
                for(const auto &row:rows)state<<row.first<<' '<<row.second<<'\n';
                return state.str();
              };
              const auto before=snapshot();
              // Native backup was taken before this move. Exercise the actual
              // restore/advance path over the outflow and reused-ID birth.
              pm.restore_particles();
              pm.advance_timestep();
              AssertThrow(Utilities::MPI::min(static_cast<unsigned>(before==snapshot()),this->get_mpi_communicator()),
                          ExcMessage("Native reuse-step retry changed particles, RNG, membership or audit identity."));
              this->get_pcout()<<"REUSED_ID_NATIVE_RETRY_PASS"<<std::endl;
            }
          const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream out(this->get_output_directory()+"birth_identity_"+std::to_string(step)
                            +"_rank"+std::to_string(rank)+".csv");
          out<<std::setprecision(17)<<"id,x,y,born,H,baseline,tau_xx,tau_yy,tau_xy\n";
          for(const auto &p:pm.get_particle_handler())
            out<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','
               <<(step && BP3Benchmark::attempt_births.count(p.get_id()))<<','
               <<p.get_properties()[H]<<','<<BP3Benchmark::work_initial_H.at(p.get_id())<<','
               <<p.get_properties()[stress]<<','<<p.get_properties()[stress+1]<<','
               <<p.get_properties()[stress+2]<<'\n';
          AssertThrow(out,ExcMessage("Cannot write birth-identity evidence."));
          return {"Birth identity audit","passed"};
        }
        void save(std::map<std::string,std::string> &status) const override
        {
          std::ostringstream stream;
          { aspect::oarchive archive(stream); archive<<BP3Benchmark::checkpoint_particle_H; }
          status["birth test audit"]=stream.str();
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          std::istringstream stream(status.at("birth test audit"));
          aspect::iarchive archive(stream); archive>>BP3Benchmark::work_initial_H;
          BP3Benchmark::restore_particle_audit(0);
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(BirthAudit,"birth identity audit",
      "Exercise the production audit without changing transported history.")
  }
}
