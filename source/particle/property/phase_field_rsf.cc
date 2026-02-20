/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/particle/property/phase_field_rsf.h>
#include <aspect/material_model/phase_field_rsf.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void PhaseFieldRSF<dim>::initialize()
      {
        // Check if the material model is PhaseFieldRSF
        AssertThrow(Plugins::plugin_type_matches<const MaterialModel::PhaseFieldRSF<dim>>(this->get_material_model()),
                    ExcMessage("Particle property `phase field rsf' only works when the material model is also set to "
                               "`phase field rsf'."));

        // Check if the fault normal and the stress components are associated with compositional fields
        AssertThrow(this->get_parameters().mapped_particle_properties.size() > 0,
                    ExcMessage("Particle property `phase field rsf' requires a map between compositional indices and "
                               "particle properties."));
        
        
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      PhaseFieldRSF<dim>::get_property_information() const
      {
        std::vector<std::string, unsigned int>> property_information;

        property_information.emplace_back("crack_driving_force", 1);
        property_information.emplace_back("slip_rate", 1);
        property_information.emplace_back("slip_state", 1);
        property_information.emplace_back("normal", dim);
        property_information.emplace_back("stress", SymmetricTensor<2, dim>::n_independent_components);
      }
    }
  }
}
