#include "runtime.h"

#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/particle/interpolator/linear_least_squares.h>

#include <cstring>
#include <limits>

namespace aspect
{
  namespace BP3Benchmark
  {
    // Only current owned/ghost particles and one timestep backup are retained.
    // Native MPI transfer carries baselines; checkpoint capture is separate.
    std::map<types::particle_index,double> work_initial_H;
    std::map<types::particle_index,double> checkpoint_particle_H;
    namespace
    {
      std::map<types::particle_index,double> backup_H;
      types::particle_index birth_floor = 0, backup_birth_floor = 0;
      bool initialized = false;

      template <int dim>
      void attach_transfer(Particle::Manager<dim> &pm)
      {
        pm.get_particle_handler().register_additional_store_load_functions(
          []() -> std::size_t { return sizeof(double); },
          [](const auto &particle, void *buffer) -> void *
          {
            const auto found = work_initial_H.find(particle->get_id());
            const double H = found==work_initial_H.end()
                             ? std::numeric_limits<double>::quiet_NaN() : found->second;
            std::memcpy(buffer, &H, sizeof(H));
            return static_cast<char *>(buffer)+sizeof(H);
          },
          [](const auto &particle, const void *buffer) -> const void *
          {
            double H;
            std::memcpy(&H, buffer, sizeof(H));
            if (std::isfinite(H)) work_initial_H[particle->get_id()] = H;
            return static_cast<const char *>(buffer)+sizeof(H);
          });
      }

      template <int dim>
      void prepare_audit(Particle::Manager<dim> &pm)
      {
        const auto H = pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
        std::map<types::particle_index,double> current;
        bool valid = true;
        for (const auto &particle : pm.get_particle_handler())
          {
            const double value = particle.get_properties()[H];
            const auto old = work_initial_H.find(particle.get_id());
            if (old != work_initial_H.end())
              {
                valid = valid && value==old->second;
                current.emplace(*old);
              }
            else
              {
                // A migrated survivor must bring its audit value. Only a new
                // native ID can establish a new baseline after initialization.
                valid = valid && (!initialized || particle.get_id()>=birth_floor)
                        && std::isfinite(value) && value>=0.;
                current.emplace(particle.get_id(), value);
              }
          }
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), pm.get_mpi_communicator()),
                    ExcMessage("BP3 particle audit: changed survivor H, missing migrated baseline, or invalid birth H."));
        work_initial_H.swap(current); // Legitimate removals and old ghost entries are forgotten.
        initialized = true;
        birth_floor = pm.get_particle_handler().get_next_free_particle_index();
      }
    }

    void restore_particle_audit(const types::particle_index next_id)
    {
      // A restart must verify restored survivors before native management can
      // introduce births. Do not treat deserialization as fresh initialization.
      initialized = true;
      birth_floor = next_id;
    }

    template <int dim>
    void connect_particle_history_audit(const SimulatorAccess<dim> &sim)
    {
      // This setup signal exposes const simulator access; registration changes
      // auxiliary particle transport callbacks, not physical particle state.
      auto &pm = const_cast<Particle::Manager<dim> &>(
        sim.get_phase_field_handler().get_associated_particle_manager());
      attach_transfer(pm);
      sim.get_signals().post_set_initial_state.connect([&pm](const SimulatorAccess<dim> &)
      {
        prepare_audit(pm);
        birth_floor = pm.get_particle_handler().get_next_free_particle_index();
      });
      sim.get_signals().post_particle_backup.connect([&pm](Particle::Manager<dim> &manager)
      {
        if (&manager != &pm) return;
        backup_H = work_initial_H;
        birth_floor = pm.get_particle_handler().get_next_free_particle_index();
        backup_birth_floor = birth_floor;
      });
      sim.get_signals().post_particle_restore.connect([&pm](Particle::Manager<dim> &manager)
      {
        if (&manager != &pm) return;
        work_initial_H = backup_H;
        birth_floor = backup_birth_floor;
        attach_transfer(pm);
      });
      sim.get_signals().post_particle_management.connect([&pm](Particle::Manager<dim> &manager)
      {
        if (&manager == &pm) prepare_audit(pm);
      });
      sim.get_signals().pre_checkpoint_store_user_data.connect([&pm](auto &)
      {
        std::map<types::particle_index,double> owned;
        for (const auto &particle : pm.get_particle_handler())
          owned.emplace(particle.get_id(), work_initial_H.at(particle.get_id()));
        // Native additional-data callbacks cover migration/ghost exchange, but
        // deal.II 9.6 mesh checkpoint packing omits that auxiliary payload.
        // Gather only current owned baselines, in the all-rank checkpoint hook.
        const auto parts = Utilities::MPI::gather(pm.get_mpi_communicator(), owned, 0);
        checkpoint_particle_H.clear();
        for (const auto &part : parts)
          checkpoint_particle_H.insert(part.begin(), part.end());
      });
      sim.get_signals().post_checkpoint.connect([](const std::string &)
      {
        checkpoint_particle_H.clear();
      });
      sim.get_signals().post_resume_load_user_data.connect([&pm](auto &)
      {
        // The archive's current-population baseline has already been checked
        // and pruned by the post-management slot, including after repartition.
        birth_floor = pm.get_particle_handler().get_next_free_particle_index();
        attach_transfer(pm);
      });
    }

    template void connect_particle_history_audit(const SimulatorAccess<2> &);
    template void connect_particle_history_audit(const SimulatorAccess<3> &);
  }

  namespace Particle { namespace Interpolator {
    /** BP3 policy only; all fitting, fallback and limiting remain native LLS. */
    template <int dim>
    class BP3HistoryLLS : public LinearLeastSquares<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &) {}
        void parse_parameters(ParameterHandler &prm) override
        {
          const auto &info = this->get_particle_manager(this->get_particle_manager_index())
                             .get_property_manager().get_data_info();
          const unsigned int internal = info.get_position_by_field_name("internal: integrator properties");
          AssertThrow(internal+info.get_components_by_field_name("internal: integrator properties")==info.n_components(),
                      ExcMessage("BP3 LLS requires native integrator properties at the end of the layout."));
          std::vector<bool> mask(internal, false);
          for (const std::string name : {"crack_driving_force", "maxwell stress"})
            {
              AssertThrow(info.fieldname_exists(name), ExcMessage("BP3 LLS requires particle field "+name));
              const unsigned int start = info.get_position_by_field_name(name);
              const unsigned int count = info.get_components_by_field_name(name);
              AssertThrow(count==(name=="crack_driving_force" ? 1 : SymmetricTensor<2,dim>::n_independent_components),
                          ExcMessage("Unexpected BP3 history component count for "+name));
              for (unsigned int k=0; k<count; ++k) mask.at(start+k)=true;
              this->get_pcout() << "BP3 native LLS limiter: " << name << " components " << start
                               << ".." << start+count-1 << "; boundary extrapolation disabled." << std::endl;
            }
          std::string list;
          for (const bool selected : mask) { if (!list.empty()) list += ','; list += selected ? "true" : "false"; }
          prm.enter_subsection("Interpolator");
          prm.enter_subsection("Linear least squares");
          prm.set("Use linear least squares limiter", list);
          prm.set("Use boundary extrapolation", "false");
          prm.leave_subsection(); prm.leave_subsection();
          LinearLeastSquares<dim>::parse_parameters(prm);
        }
    };
    ASPECT_REGISTER_PARTICLE_INTERPOLATOR(BP3HistoryLLS, "BP3 history linear least squares",
                                        "Native LLS with runtime-named H/Maxwell limiting and no boundary extrapolation.")
  }}
}
