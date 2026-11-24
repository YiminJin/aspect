/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_fault_h
#define _aspect_midpoint_fault_h
#include <aspect/material_model/growing_fault.h>

namespace aspect::GrowingFault
{
  /** Assemble and solve the plane-strain stage on the fixed material mesh.
   * The accepted history and old FE fields remain immutable. Trial history
   * is returned to the material owner pending the ordinary repeat decision. */
  void solve_midpoint(const Material<2> &, LinearAlgebra::BlockVector &, History &, Balance &);
}
#endif
