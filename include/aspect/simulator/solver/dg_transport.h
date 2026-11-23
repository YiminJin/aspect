/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_simulator_solver_dg_transport_h
#define _aspect_simulator_solver_dg_transport_h
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Transport
  {
    /** Physical xx, yy and xy stress in Pa at the scalar DG support points
     * of each locally owned cell, in base-element order. */
    using CellStress = std::map<CellId,std::vector<SymmetricTensor<2,2>>>;

    /** Collectively advect the three DGQ2 compositions with the velocity in
     * accepted, on a fixed, conforming Cartesian box. The stress field indices
     * are ordered xx, yy, xy. At incoming physical faces evaluate exterior
     * (same components, Pa) at the stage time in seconds. Return the candidate
     * coefficients after duration seconds without modifying accepted. */
    CellStress advect_stress(const SimulatorAccess<2> &sim,
                             const std::array<unsigned int,3> &fields,
                             const LinearAlgebra::BlockVector &accepted,
                             Function<2> &exterior,
                             double start_time, double duration);

    /** Scalar coefficients on owned cells in base-element order. The bounded
     * route uses native quadratic Bernstein coefficients and mapped means. */
    using CellScalar = std::map<CellId,std::vector<double>>;

    /** Physical boundary value at an incoming face point and stage time (s). */
    using ScalarInflow = std::function<double(types::boundary_id,const Point<2>&,double)>;

    /** Collectively advance a DGQ2 scalar from the old solution and velocity. */
    CellScalar advect_scalar(const SimulatorAccess<2> &sim, unsigned int field,
                             const ScalarInflow &inflow, bool bounded_bernstein);
  }
}
#endif
