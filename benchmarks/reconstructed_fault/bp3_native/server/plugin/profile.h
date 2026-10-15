#ifndef ASPECT_BENCHMARK_BP3_PROFILE_H
#define ASPECT_BENCHMARK_BP3_PROFILE_H
#include <aspect/simulator_access.h>

namespace BP3
{
  /** Transient loading primitive, distinct from the discrete Q1 mechanical Ih.
   * Prepared from the live stationary profile and energetic degradation law.
   * Cubic Hermite interpolation is qualified against quadrature at preparation;
   * no integration, geometric search or communication occurs in cumulative(). */
  struct LoadingProfile
  {
    double support;
    std::vector<double> radius, integral, slope;
    double cumulative(double signed_distance) const;
  };

  template <int dim>
  const LoadingProfile &loading_profile(const aspect::SimulatorAccess<dim> &sim);
}
#endif
