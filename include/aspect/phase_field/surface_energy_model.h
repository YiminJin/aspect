/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_phase_field_surface_energy_model_h
#define _aspect_phase_field_surface_energy_model_h

#include <aspect/phase_field/crack_geometry.h>

namespace dealii
{
  class ParameterHandler;
}

namespace aspect
{
  namespace PhaseField
  {
    /**
     * The crack-surface functional configured for one simulation. The
     * composition is identified by name so each FE consumer can bind it
     * through introspection when it starts using the functional.
     */
    struct SurfaceEnergyModel
    {
      static void declare_parameters(dealii::ParameterHandler &prm);
      static SurfaceEnergyModel parse_parameters(dealii::ParameterHandler &prm);

      /** Check the physical constants when a consumer activates the model. */
      void validate() const;

      const CrackGeometry &geometry;
      const std::string composition;
      const double fracture_energy;  ///< Gc in J/m^2.
      const double length_scale;     ///< Regularization length in metres.
    };
  }
}
#endif
