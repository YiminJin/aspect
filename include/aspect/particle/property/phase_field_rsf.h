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
          void initialize() override;

          void update() override;

          /**
           * @copydoc aspect::Particle::Property::Interface::initialize_one_particle_property()
           */
          void
          initialize_one_particle_property(const Point<dim> &position,
                                           std::vector<double> &particle_properties) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::initialize_late_particle()
           */
          std::vector<double>
          initialize_late_particle(const Point<dim> &particle_location,
                                   const typename Triangulation<dim>::active_cell_iterator &cell) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::need_update()
           */
          UpdateTimeFlags
          need_update () const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::late_initialization_mode()
           */
          InitializationModeForLateParticles
          late_initialization_mode() const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_property_information()
           */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters(ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters(ParameterHandler &prm) override;

        private:
          unsigned int phase_field_base_element;

          bool output_friction_coefficient;

          bool output_slip_increment;

          double maximum_slip_distance_between_outputs;
      };
    }
  }
}

#endif
