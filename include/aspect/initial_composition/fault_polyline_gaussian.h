/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_initial_composition_fault_polyline_gaussian_h
#define _aspect_initial_composition_fault_polyline_gaussian_h

#include <aspect/initial_composition/interface.h>

namespace aspect::InitialComposition
{
  /**
   * Initialize the named phase composition with a Gaussian about an open
   * polyline in the Cartesian plane. Distance is measured to the closest
   * point on any finite segment, including the first and last vertices.
   *
   * @ingroup InitialCompositionModels
   */
  template <int dim>
  class FaultPolylineGaussian : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      /** Return the dimensionless phase at a point in metres, or zero for
       * other compositions so their additive initial models are preserved. */
      double initial_composition(const Point<dim> &position, const unsigned int compositional_index) const override;

      static void declare_parameters(ParameterHandler &prm);

      /** Bind the phase by name and read the vertex coordinates collectively
       * at setup, before the simulator interpolates the initial compositions. */
      void parse_parameters(ParameterHandler &prm) override;

    private:
      std::vector<Point<dim>> vertices;
      unsigned int phase_index;
      double background, peak, width;
  };
}

#endif
