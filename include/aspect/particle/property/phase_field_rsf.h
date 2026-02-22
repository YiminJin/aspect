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

#ifndef _aspect_particle_property_phase_field_rsf_h
#define _aspect_particle_property_phase_field_rsf_h

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      class PhaseFieldRSF : public Interface<dim>, 
        public SimulatorAccess<dim>
      {
        public:
          /**
           * Constructor.
           */
          PhaseFieldRSF();

          void initialize() override;

          /**
           * @copydoc aspect::Particle::Property::Interface::initialize_one_particle_property()
           */
          void
          initialize_one_particle_property(const Point<dim> &position,
                                           std::vector<double> &particle_properties) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::update_particle_properties()
           */
          void
          update_particle_properties(const ParticleUpdateInputs<dim> &inputs,
                                     typename ParticleHandler<dim>::particle_iterator_range &particles) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::need_update()
           */
          UpdateTimeFlags
          need_update () const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_update_flags()
           */
          UpdateFlags
          get_update_flags(const unsigned int component) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_property_information()
           */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

          static
          void
          declare_parameters(ParameterHandler &prm);

          void
          parse_parameters(ParameterHandler &prm) override;

        private:
          struct CompositionalIndices
          {
            unsigned int slip_rate;
            std::vector<unsigned int> normal;
            std::vector<unsigned int> stress;

            CompositionalIndices()
              : slip_rate(numbers::invalid_unsigned_int)
              , normal(dim, numbers::invalid_unsigned_int)
              , stress(SymmetricTensor<2, dim>::n_independent_components, numbers::invalid_unsigned_int)
            {}
          };

          CompositionalIndices compositional_indices;
      };
    }
  }
}

#endif
