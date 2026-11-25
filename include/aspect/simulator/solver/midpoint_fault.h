/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_fault_h
#define _aspect_midpoint_fault_h
#include <aspect/material_model/growing_fault/midpoint_material.h>
#include <aspect/simulator_access.h>

namespace aspect::GrowingFault
{
  /** Read the wall function for an independent midpoint clock. */
  template <int dim>
  void parse_midpoint_wall_velocity(ParameterHandler &, Parameters<dim> &);

  /** Validate the configured phase method against the top-level material
   * after plugin parsing, before any independent composition can be skipped. */
  template <int dim>
  void validate_midpoint_material(const Parameters<dim> &, const MaterialModel::Interface<dim> &);

  /** Assemble and solve the plane-strain stage on the fixed material mesh.
   * The accepted history and old FE fields remain immutable. Trial history
   * is returned to the material owner pending the ordinary repeat decision. */
  SolverControl advance_midpoint(const SimulatorAccess<2> &, MidpointMaterial &, LinearAlgebra::BlockVector &);
}
#endif
