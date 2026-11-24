/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_fracture_h
#define _aspect_midpoint_fracture_h
#include <aspect/simulator_access.h>

namespace aspect::PhaseField
{
  /** Local contributions to the AT1 surface energy divided by Kf=Gc/(c0 ell).
   * Coefficients address the phase block. The quadratic part is symmetric SIPG
   * with natural exterior boundaries and a row for each owned cell test. */
  struct FractureRows
  {
    using Index = types::global_dof_index;
    using Matrix = std::map<std::pair<Index,Index>,double>;
    FractureRows(const SimulatorAccess<2> &, unsigned int composition);
    std::map<Index,double> mass;
    Matrix stiffness;
  };

  /** Stage obstacle. A positive gradient at the lower bound is admissible.
   * The residual is the natural map in Bernstein coefficient coordinates. */
  inline double projected_gradient(double value,double gradient,double mass,double lower,double upper)
  {
    return value-std::clamp(value-gradient/mass,lower,upper);
  }
}
#endif
