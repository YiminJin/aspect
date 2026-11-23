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
    /** Per-composition opt-in. The stage evaluator reads the same prescribed
     * Cartesian boundary function as implicit advection, with time in seconds
     * supplied explicitly by the stage. */
    template <int dim>
    struct ExplicitCompositions
    {
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm, const Parameters<dim> &parameters);
      std::vector<bool> selected;
      std::function<double(const Point<dim> &, unsigned int, double)> inflow;
    };

    /** Advance one selected DG field from its accepted coefficients
     * and accepted velocity. Write only its candidate solution block. */
    void advance_composition(const SimulatorAccess<2> &sim, unsigned int field,
                             LinearAlgebra::BlockVector &solution);
  }
}
#endif
