/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_transport_dg_stress_h
#define _aspect_transport_dg_stress_h

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
  }
}

#endif
