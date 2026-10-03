#include "../../bp3/plugin/particle_initialization.h"
#include <aspect/particle/manager.h>
#include <aspect/postprocess/interface.h>
namespace BP3 { double weakening_length=15000.; }
namespace aspect { namespace Postprocess {
  template <int dim>
  class EvolvingBirthGuard : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
    void initialize() override
    {
      this->get_signals().post_simulator_initialization.connect([this](const auto &){
        this->get_signals().post_set_initial_state.connect([this](const auto &){
          auto &pm=this->get_particle_manager(0);
          const auto &property=pm.get_property_manager().template get_matching_active_plugin<Particle::Property::BP3FrozenCrackDrivingForce<dim>>();
          AssertThrow(!property.uses_stationary_initialization() && property.late_initialization_mode()==Particle::Property::interpolate,
                      ExcMessage("An evolving model must retain generic H history interpolation."));
          const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          // Deliberate history input for this isolated transfer probe, not a
          // physical evolution run. It must not be replaced by Hc or H(phi).
          for(auto &p:pm.get_particle_handler())p.get_properties()[H]=123.;
          pm.get_particle_handler().exchange_ghost_particles();
          bool checked=false,valid=true;
          for(const auto &cell:this->get_triangulation().active_cell_iterators())if(cell->is_locally_owned())
            {
              const auto values=pm.get_property_manager().initialize_late_particle(cell->center(),pm.get_particle_handler(),pm.get_interpolator(),cell);
              valid=valid && std::abs(values[H]-123.)<1e-12 && property.initial_H(cell->center())!=123.;checked=true;break;
            }
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(checked && valid),this->get_mpi_communicator()),ExcMessage("Evolving H history transfer failed."));
          AssertThrow(false,ExcMessage("EVOLVING H TRANSFER PASS: intentional stop before mechanics."));
        });
      });
    }
    std::pair<std::string,std::string> execute(TableHandler &) override {return {};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(EvolvingBirthGuard,"evolving birth guard","Isolated real property-manager transfer guard for evolving H.")
}}
