#include "particle_initialization.h"
#include "bp3_model.h"

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    double stationary_particle_H(const SimulatorAccess<dim> &sim,
                                 const Point<dim> &position,
                                 const PhaseField::PhaseFieldProfile &profile)
    {
      const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
                           sim.get_material_model());
      // The material's mature specialization requires Evolve phase field=false.
      // Do not infer loading history from phi in an evolving/cohesive model.
      AssertThrow(model.is_mature_frictional_fault(),
                  ExcMessage("BP3 stationary H initialization requires the frozen mature specialization."));
      const double phi = profile.value(BP3::normal_distance(position[0], position[1]));
      const double f = BP3::depth_fraction(position[1]);
      if (phi <= model.get_phase_field_activation_threshold())
        // Same arithmetic Hc average as the native startup property. Initial
        // composition plugins are discarded after startup; use BP3's retained
        // stationary material-fraction definition for later births instead.
        return MaterialModel::MaterialUtilities::average_value(
                 {1-f, f}, model.get_critical_crack_driving_forces(),
                 MaterialModel::MaterialUtilities::arithmetic);

      return sim.get_phase_field_handler().stationary_crack_driving_force(
               {1-f, f}, phi, BP3::core_phi);
    }

    template double stationary_particle_H(const SimulatorAccess<2> &, const Point<2> &,
                                           const PhaseField::PhaseFieldProfile &);
    template double stationary_particle_H(const SimulatorAccess<3> &, const Point<3> &,
                                           const PhaseField::PhaseFieldProfile &);
  }

  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      bool BP3FrozenCrackDrivingForce<dim>::uses_stationary_initialization() const
      {
        const auto *model = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
                              &this->get_material_model());
        // This existing material invariant also guarantees the frozen flag.
        return model != nullptr && model->is_mature_frictional_fault();
      }

      template <int dim>
      double BP3FrozenCrackDrivingForce<dim>::initial_H(const Point<dim> &position) const
      {
        if (!uses_stationary_initialization())
          {
            std::vector<double> data;
            CrackDrivingForce<dim>::initialize_one_particle_property(position, data);
            return data[0];
          }

        if (!profile)
          profile = std::move(this->get_phase_field_handler().get_phase_field_profiles(BP3::core_phi)[0]);
        return BP3Benchmark::stationary_particle_H(*this, position, *profile);
      }

      template <int dim>
      void BP3FrozenCrackDrivingForce<dim>::initialize_one_particle_property(
        const Point<dim> &position, std::vector<double> &data) const
      {
        data.push_back(initial_H(position));
      }

      template <int dim>
      InitializationModeForLateParticles BP3FrozenCrackDrivingForce<dim>::late_initialization_mode() const
      {
        // Native property initialization runs before particle insertion. Other
        // properties, including Maxwell stress, keep their own transfer modes.
        return uses_stationary_initialization() ? Property::initialize
               : CrackDrivingForce<dim>::late_initialization_mode();
      }

      ASPECT_REGISTER_PARTICLE_PROPERTY(BP3FrozenCrackDrivingForce,
                                        "BP3 frozen crack driving force",
                                        "Initialize frozen mature BP3 H from its prescribed stationary profile; "
                                        "retain native critical-H initialization and late history interpolation "
                                        "for other models. The field layout is unchanged.")
    }
  }
}
