/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_simulator_solver_explicit_composition_h
#define _aspect_simulator_solver_explicit_composition_h

#include <aspect/global.h>
#include <deal.II/base/parameter_handler.h>
#include <functional>

namespace aspect
{
  template <int dim> struct Parameters;
  template <int dim> class SimulatorAccess;

  namespace Transport
  {
    /** Prescribed Cartesian composition at a point, field index and stage time in seconds. */
    template <int dim>
    using CompositionInflow = std::function<double(const Point<dim> &, unsigned int, double)>;

    /** Validate fields with the explicit DG SSPRK3 method and parse their stage-time
     * inflow function. Return an empty evaluator when no inflow is prescribed. */
    template <int dim>
    CompositionInflow<dim>
    parse_explicit_composition_parameters(ParameterHandler &prm, const Parameters<dim> &parameters);

    /** Advance one selected DG field from its accepted coefficients
     * and accepted velocity. Write only its candidate solution block. */
    template <int dim>
    void advance_composition(const SimulatorAccess<dim> &sim,
                             unsigned int                field,
                             LinearAlgebra::BlockVector &solution);
  }
}
#endif
