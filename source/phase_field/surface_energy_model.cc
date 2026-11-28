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

    namespace
    {
      void declare_functional(ParameterHandler &prm, const bool legacy)
      {
        const std::string location = legacy ? "Compatibility spelling of the entry in Phase field. " : "";
        prm.declare_entry("Crack geometry", "",
                          Patterns::List(Patterns::Selection(CrackGeometry::selection_pattern()),0,1),
                          location + "Crack-geometric law. The default is AT1.");
        prm.declare_entry("Phase composition", "", Patterns::Anything(),
                          location + "Name of the dimensionless phase field. The default is phase_field.");
        prm.declare_entry("Fracture energy", "", Patterns::List(Patterns::Double(),0,1),
                          location + "Positive fracture energy per unit crack area Gc in J/m^2. The default is 1.");
        prm.declare_entry("Length scale", "", Patterns::List(Patterns::Double(),0,1),
                          location + "Positive regularization length in metres. The default is 1.");
      }

      // An empty spelling inherits the other section. Apply the physical
      // default only after checking both, including an explicit default value.
      template <typename T, typename Reader>
      T functional_parameter(const ParameterHandler &prm,
                             const std::string      &name,
                             const T                &fallback,
                             const Reader           &read)
      {
        const std::vector<std::string> phase = {"Phase field"};
        const std::vector<std::string> legacy = {"Postprocess","Phase field surface energy"};
        if (prm.get(phase,name).empty())
          return prm.get(legacy,name).empty() ? fallback : read(legacy,name);

        const T value = read(phase,name);
        AssertThrow(prm.get(legacy,name).empty() || value == read(legacy,name),
                    ExcMessage("Conflicting " + name + " in Phase field and Postprocess / Phase field surface energy."));
        return value;
      }
    }

    void SurfaceEnergyModel::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Phase field");
      declare_functional(prm, false);
      prm.leave_subsection();

      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Phase field surface energy");
      declare_functional(prm, true);
      prm.leave_subsection();
      prm.leave_subsection();
    }

    SurfaceEnergyModel SurfaceEnergyModel::parse_parameters(ParameterHandler &prm)
    {
      const auto text = [&](const auto &path, const auto &name)
      {
        return prm.get(path,name);
      };
      const auto number = [&](const auto &path, const auto &name)
      {
        return prm.get_double(path,name);
      };
      return {CrackGeometry::select(functional_parameter(prm,"Crack geometry",std::string("AT1"),text)),
              functional_parameter(prm,"Phase composition",std::string("phase_field"),text),
              functional_parameter(prm,"Fracture energy",1.,number),
              functional_parameter(prm,"Length scale",1.,number)
             };
    }

    double SurfaceEnergyModel::compact_profile_half_width() const
    {
      const auto factor = geometry.compact_profile_half_width_factor();
      AssertThrow(factor.has_value(),
                  ExcMessage("A finite reference profile is required, but " + geometry.name() +
                             " has infinite support."));
      const double width = length_scale * factor.value();
      AssertThrow(std::isfinite(width) && width > 0.,
                  ExcMessage("The " + geometry.name() +
                             " reference profile requires a finite, positive half-width in metres."));
      return width;
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
