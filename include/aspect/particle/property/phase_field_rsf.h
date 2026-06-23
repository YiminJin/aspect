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
           * @copydoc aspect::Particle::Property::Interface::get_update_flags()
           */
          UpdateFlags
          get_update_flags(const unsigned int component) const override;

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

        private:
          void initialize_data_position_cache();

          void do_cleanup_after_initialization();

          double
          get_phase_field_value(const LinearAlgebra::BlockVector &solution,
                                const typename Triangulation<dim>::active_cell_iterator &cell,
                                const Point<dim> &reference_location) const;

          struct CompositionalIndices
          {
            unsigned int crack_driving_force;
            unsigned int slip_rate;
            unsigned int slip_state;
            std::array<unsigned int, dim> normal_direction;
            std::array<unsigned int, dim> slip_direction;
            std::array<unsigned int, SymmetricTensor<2, dim>::n_independent_components> bulk_stress;

            CompositionalIndices()
              : crack_driving_force(numbers::invalid_unsigned_int)
              , slip_rate(numbers::invalid_unsigned_int)
              , slip_state(numbers::invalid_unsigned_int)
            {
              normal_direction.fill(numbers::invalid_unsigned_int);
              slip_direction.fill(numbers::invalid_unsigned_int);
              bulk_stress.fill(numbers::invalid_unsigned_int);
            }
          };

          CompositionalIndices compositional_indices;

          struct DataPositionCache
          {
            unsigned int crack_driving_force;
            unsigned int slip_rate;
            unsigned int slip_state;
            unsigned int normal_direction;
            unsigned int slip_direction;
            unsigned int bulk_stress;
            unsigned int interface_stress;
            std::vector<unsigned int> chemical_fields;
          };

          DataPositionCache data_position_cache;

          unsigned int phase_field_component_index;

          unsigned int phase_field_base_index;

          bool start_with_slip;

          std::unique_ptr<GridTools::Cache<dim>> grid_cache;
      };
    }
  }
}

#endif
