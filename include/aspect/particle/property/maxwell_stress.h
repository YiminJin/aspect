/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#ifndef _aspect_particle_property_maxwell_stress_h
#define _aspect_particle_property_maxwell_stress_h

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>

#include <array>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      /**
       * Store the committed deviatoric stress history of the Maxwell body used
       * by the reconstructed-fault material model. The property contains one
       * symmetric tensor and deliberately performs no automatic time update or
       * objective rotation.
       *
       * Initial values are read from compositional fields mapped to the
       * individual components of the particle property @p maxwell_stress.
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class MaxwellStress : public Interface<dim>, public SimulatorAccess<dim>
      {
        public:
          /** Validate the material model and component mappings. */
          void
          initialize() override;

          /** Initialize all tensor components from the initial composition model. */
          void
          initialize_one_particle_property(const Point<dim> &position,
                                           std::vector<double> &particle_properties) const override;

          /** Return one symmetric-tensor particle property. */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

        private:
          /**
           * Compositional field supplying each component in deal.II's
           * SymmetricTensor unrolled order.
           */
          std::array<unsigned int, SymmetricTensor<2,dim>::n_independent_components>
          initial_stress_field_indices;
      };
    }
  }
}

#endif
