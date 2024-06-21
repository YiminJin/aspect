/*
  Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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

#include <aspect/particle/property/material_orientation.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void
      MaterialOrientation<dim>::
      initialize_one_particle_property(const Point<dim> &position,
                                       std::vector<double> &data) const
      {
        data.push_back(this->get_initial_composition_manager().initial_composition(position, this->introspection().compositional_index_for_name("n_x")));
        data.push_back(this->get_initial_composition_manager().initial_composition(position, this->introspection().compositional_index_for_name("n_y")));
        if (dim == 3)
          data.push_back(this->get_initial_composition_manager().initial_composition(position, this->introspection().compositional_index_for_name("n_z")));
      }


      
      template <int dim>
      void
      MaterialOrientation<dim>::
      update_particle_property(const unsigned int data_position,
                               const Vector<double> &/*solution*/,
                               const std::vector<Tensor<1,dim>> &gradients,
                               typename ParticleHandler<dim>::particle_iterator &particle) const
      {
        // Get the normal vector of the previous time step.
        Tensor<1,dim> n;
        for (unsigned int d=0; d<dim; ++d)
          n[d] = particle->get_properties()[data_position + d];

        Tensor<1,dim> n_new;
        if (n.norm() > 0.5)
          {
            // Compute the time derivative of n (Chaves E. W., 2013, p217)
            // and make sure that n is a unit vector. Handle time step 0 differently.
            if (this->get_timestep_number() == 0)
              n_new = n / n.norm();
            else
              {
                Tensor<2,dim> L;
                for (unsigned int d = 0; d < dim; ++d)
                  L[d] = gradients[d];

                const Tensor<1,dim> n_dot = -n * L + n * (n * L * n);
                n_new = n + (n_dot * this->get_timestep());
                n_new /= n_new.norm();
              }
          }

        for (unsigned int d = 0; d < dim; ++d)
          particle->get_properties()[data_position + d] = n_new[d];
    }


      
      template <int dim>
      UpdateTimeFlags
      MaterialOrientation<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      UpdateFlags
      MaterialOrientation<dim>::get_needed_update_flags() const
      {
        return update_gradients;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      MaterialOrientation<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string, unsigned int>> property_information;

        // Check which fields are used in model and make an output for each.
        if (this->introspection().compositional_name_exists("n_x"))
          property_information.emplace_back("n_x", 1);

        if (this->introspection().compositional_name_exists("n_y"))
          property_information.emplace_back("n_y", 1);

        if (dim == 3 && this->introspection().compositional_name_exists("n_z"))
          property_information.emplace_back("n_z", 1);

        return property_information;
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(MaterialOrientation,
                                        "material orientation",
                                        "A plugin in which the particle property vector is "
                                        "defined as the material orientation. It is useful "
                                        "for transversely anisotropic material models.")
    }
  }
}
