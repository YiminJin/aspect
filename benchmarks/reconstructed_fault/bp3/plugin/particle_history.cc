#include "runtime.h"
#include "particle_initialization.h"

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
    types::particle_index population_before_attempt = 0;
    std::set<types::particle_index> attempt_births;
    namespace
    {
      std::map<types::particle_index,double> backup_H;
      std::set<types::particle_index> pending_births;
      bool initialized = false;

      template <int dim>
      void attach_transfer(Particle::Manager<dim> &pm)
      {
        pm.get_particle_handler().register_additional_store_load_functions(
          []() -> std::size_t { return sizeof(double)+sizeof(unsigned char); },
          [](const auto &particle, void *buffer) -> void *
          {
            const auto found = work_initial_H.find(particle->get_id());
            const double H = found==work_initial_H.end()
                             ? std::numeric_limits<double>::quiet_NaN() : found->second;
            std::memcpy(buffer, &H, sizeof(H));
            const unsigned char born = attempt_births.count(particle->get_id()) != 0;
            auto *next = static_cast<char *>(buffer)+sizeof(H);
            std::memcpy(next, &born, sizeof(born));
            return next+sizeof(born);
          },
          [](const auto &particle, const void *buffer) -> const void *
          {
            double H;
            std::memcpy(&H, buffer, sizeof(H));
            unsigned char born;
            const auto *next = static_cast<const char *>(buffer)+sizeof(H);
            std::memcpy(&born, next, sizeof(born));
            if (std::isfinite(H)) work_initial_H[particle->get_id()] = H;
            if (born) attempt_births.insert(particle->get_id());
            else attempt_births.erase(particle->get_id());
            return next+sizeof(born);
          });
      }

      template <int dim>
      void prepare_audit(Particle::Manager<dim> &pm)
      {
        const auto H = pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
        std::map<types::particle_index,double> current;
        bool valid = true;
        std::set<types::particle_index> current_births;
        for (const auto &particle : pm.get_particle_handler())
          {
            const double value = particle.get_properties()[H];
            const auto old = work_initial_H.find(particle.get_id());
            if (!pending_births.count(particle.get_id()) && old != work_initial_H.end())
              {
                valid = valid && value==old->second;
                current.emplace(*old);
              }
            else
              {
                // An actual insertion may reuse a retired ID whose old
                // baseline remains here. Migrated survivors must bring theirs.
                valid = valid && (!initialized || pending_births.count(particle.get_id()))
                        && std::isfinite(value) && value>=0.;
                const auto &properties = pm.get_property_manager();
                if (properties.template has_matching_active_plugin<Particle::Property::BP3FrozenCrackDrivingForce<dim>>())
                  {
                    const auto &initializer = properties.template get_matching_active_plugin<Particle::Property::BP3FrozenCrackDrivingForce<dim>>();
                    if (initializer.uses_stationary_initialization())
                      valid = valid && value==initializer.initial_H(particle.get_location());
                  }
                current.emplace(particle.get_id(), value);
              }
            if (attempt_births.count(particle.get_id()))
              current_births.insert(particle.get_id());
          }
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), pm.get_mpi_communicator()),
                    ExcMessage("BP3 particle audit: changed survivor H, missing migrated baseline, or invalid birth H (including the selected stationary initializer)."));
        work_initial_H.swap(current); // Legitimate removals and old ghost entries are forgotten.
        attempt_births.swap(current_births);
        pending_births.clear();
        initialized = true;
      }
    }

    void restore_particle_audit(const types::particle_index /*legacy_next_id*/)
    {
      // A restart must verify restored survivors before native management can
      // introduce births. Do not treat deserialization as fresh initialization.
      initialized = true;
      // Consume the legacy archive field without treating it as a lifetime ID.
      attempt_births.clear();
      pending_births.clear();
    }

    template <int dim>
    void connect_particle_history_audit(const SimulatorAccess<dim> &sim)
    {
      // This setup signal exposes const simulator access; registration changes
      // auxiliary particle transport callbacks, not physical particle state.
      auto &pm = const_cast<Particle::Manager<dim> &>(
        sim.get_phase_field_handler().get_associated_particle_manager());
      attach_transfer(pm);
      pm.post_particle_creation.connect([](const auto &particle)
      {
        // Native count management does not migrate particles between insertion
        // and post-management. Defer validation/publication to that existing
        // collective audit, retaining observer timing and survivor baselines.
        pending_births.insert(particle->get_id());
        attempt_births.insert(particle->get_id());
      });
      sim.get_signals().post_set_initial_state.connect([&pm](const SimulatorAccess<dim> &)
      {
        prepare_audit(pm);
        attempt_births.clear();
      });
      sim.get_signals().post_particle_backup.connect([&pm](Particle::Manager<dim> &manager)
      {
        if (&manager != &pm) return;
        population_before_attempt=pm.get_particle_handler().n_locally_owned_particles();
        backup_H = work_initial_H;
        attempt_births.clear();
        pending_births.clear();
      });
      sim.get_signals().post_particle_restore.connect([&pm](Particle::Manager<dim> &manager)
      {
        if (&manager != &pm) return;
        work_initial_H = backup_H;
        attempt_births.clear();
        pending_births.clear();
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
        attempt_births.clear();
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
