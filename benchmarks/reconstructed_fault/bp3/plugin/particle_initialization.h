#ifndef ASPECT_BENCHMARK_BP3_PARTICLE_INITIALIZATION_H
#define ASPECT_BENCHMARK_BP3_PARTICLE_INITIALIZATION_H

#include <aspect/particle/property/crack_driving_force.h>
#include <aspect/phase_field.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    /** BP3's frozen stationary profile, including the unchanged exterior Hc. */
    template <int dim>
    double stationary_particle_H(const SimulatorAccess<dim> &sim,
                                 const Point<dim> &position,
                                 const PhaseField::PhaseFieldProfile &profile);
  }

  namespace Particle
  {
    namespace Property
    {
      /** Opt-in BP3 initialization; generic evolving history remains interpolated. */
      template <int dim>
      class BP3FrozenCrackDrivingForce : public CrackDrivingForce<dim>
      {
        public:
          void initialize_one_particle_property(const Point<dim> &position,
                                                std::vector<double> &data) const override;
          InitializationModeForLateParticles late_initialization_mode() const override;
          double initial_H(const Point<dim> &position) const;
          bool uses_stationary_initialization() const;

        private:
          // The supported BP3 material/profile is stationary. Construct lazily
          // after the phase-field handler is initialized, also after restart.
          mutable std::unique_ptr<PhaseField::PhaseFieldProfile> profile;
      };
    }
  }
}
#endif
