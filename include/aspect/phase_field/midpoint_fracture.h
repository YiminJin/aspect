/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_fracture_h
#define _aspect_midpoint_fracture_h

#include <aspect/simulator_access.h>

namespace aspect::PhaseField
{
  /**
   * Local rows of the AT1 surface energy divided by Kf = Gc/(c0 ell).
   * Indices are global within the compositional phase block. Each rank stores
   * the test rows of its owned cells and includes neighbor trial functions on
   * interior faces. The midpoint solver combines the rows across ranks.
   */
  struct FractureRows
  {
    using Index = types::global_dof_index;
    using Matrix = std::map<std::pair<Index,Index>,double>;

    FractureRows(const SimulatorAccess<2> &sim, unsigned int composition);

    /** Integral of each Bernstein test function over its cell (m²). */
    std::map<Index,double> mass;

    /** Symmetric interior-penalty gradient operator, including 2 ell². */
    Matrix stiffness;
  };

  /** Return the difference between the stage coefficient and its projected
   * gradient step. A positive gradient at the lower obstacle is admissible.
   * The caller supplies the gradient and row scaling in the same units. */
  inline double projected_gradient(double value, double gradient, double mass, double lower, double upper)
  {
    return value - std::clamp(value - gradient/mass, lower, upper);
  }
}
#endif
