/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/phase_field/surface_energy_model.h>
#include <deal.II/base/parameter_handler.h>

#include <cmath>

namespace aspect
{
  namespace PhaseField
  {
    using namespace dealii;

    void SurfaceEnergyModel::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Phase field surface energy");
      prm.declare_entry("Crack geometry", "AT1", Patterns::Selection(CrackGeometry::selection_pattern()),
                        "Crack-geometric law for the configured phase-field surface functional.");
      prm.declare_entry("Phase composition", "phase_field", Patterns::Anything(),
                        "Name of the compositional field that contains the dimensionless phase.");
      prm.declare_entry("Fracture energy", "1", Patterns::Double(),
                        "Positive fracture energy per unit crack area Gc in J/m^2.");
      prm.declare_entry("Length scale", "1", Patterns::Double(),
                        "Positive regularization length in metres.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    SurfaceEnergyModel SurfaceEnergyModel::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Phase field surface energy");
      SurfaceEnergyModel model {CrackGeometry::select(prm.get("Crack geometry")),
                                prm.get("Phase composition"),
                                prm.get_double("Fracture energy"),
                                prm.get_double("Length scale")
                               };
      prm.leave_subsection();
      prm.leave_subsection();
      return model;
    }

    void SurfaceEnergyModel::validate() const
    {
      AssertThrow(std::isfinite(fracture_energy) && fracture_energy > 0.,
                  ExcMessage("Phase field surface energy requires a finite, positive fracture energy in J/m^2."));
      AssertThrow(std::isfinite(length_scale) && length_scale > 0.,
                  ExcMessage("Phase field surface energy requires a finite, positive length scale in metres."));
    }
  }
}
