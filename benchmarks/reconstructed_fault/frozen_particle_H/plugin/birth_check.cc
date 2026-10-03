#include "../../bp3/plugin/particle_initialization.h"
#include "../../bp3/plugin/runtime.h"
#include "../../bp3/plugin/bp3_model.h"
#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/material_model/phase_field_fault.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
  template <int dim>
  class CheckFrozenBirthH : public Interface<dim>, public SimulatorAccess<dim>
  {
    types::particle_index floor=0;
    void check(Particle::Manager<dim> &pm)
    {
      const auto &property=pm.get_property_manager().template get_matching_active_plugin<Particle::Property::BP3FrozenCrackDrivingForce<dim>>();
      const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
      const auto profiles=this->get_phase_field_handler().get_phase_field_profiles(BP3::core_phi);
      const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      const auto path=this->get_output_directory()+"birth_H_rank"+std::to_string(rank)+".csv";
      const bool header=!std::ifstream(path).good();std::ofstream out(path,std::ios::app);out<<std::setprecision(17);
      if(header)out<<"step,time,dt,id,x,y,H,expected,Hc,active,audit_already_present\n";
      bool valid=true;
      for(const auto &p:pm.get_particle_handler())if(p.get_id()>=floor)
        {
          const double fraction=BP3::depth_fraction(p.get_location()[1]);
          const double critical=MaterialModel::MaterialUtilities::average_value(
            {1-fraction,fraction},model.get_critical_crack_driving_forces(),MaterialModel::MaterialUtilities::arithmetic);
          const double expected=property.initial_H(p.get_location());
          const bool active=profiles[0]->value(BP3::normal_distance(p.get_location()[0],p.get_location()[1]))>model.get_phase_field_activation_threshold();
          const bool recorded=BP3Benchmark::work_initial_H.count(p.get_id());
          valid=valid && p.get_properties()[H]==expected && !recorded && (active || expected==critical);
          out<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','<<p.get_properties()[H]<<','<<expected<<','<<critical<<','<<active<<','<<recorded<<'\n';
        }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid),this->get_mpi_communicator()),ExcMessage("Newborn H differs from shared initialization or was audited too early."));
    }
    public:
    void initialize() override
    {
      // This observer registers before BP3's post-simulator setup attaches the
      // birth audit. It checks properties that native insertion has completed,
      // before the maintained audit establishes a new baseline.
      this->get_signals().post_restore_particles.connect([this](auto &pm){floor=pm.get_particle_handler().get_next_free_particle_index();});
      this->get_signals().post_resume_load_user_data.connect([this](auto &){floor=this->get_particle_manager(0).get_particle_handler().get_next_free_particle_index();});
      this->get_signals().post_particle_management.connect([this](auto &pm){check(pm);});
    }
    std::pair<std::string,std::string> execute(TableHandler &) override
    {return {"Frozen newborn H:","verified before birth audit"};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(CheckFrozenBirthH,"check frozen birth H","Check initialized newborn H before audit publication, without writing history.")
}}
