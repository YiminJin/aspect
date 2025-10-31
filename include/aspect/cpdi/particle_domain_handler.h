/*
 Copyright (C) 2023 - 2024 by the authors of the ASPECT code.

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

#ifndef _aspect_cpdi_particle_domain_handler_h
#define _aspect_cpdi_particle_domain_handler_h

#include <deal.II/particles/particle_handler.h>

#include <voro++.hh>

namespace aspect
{
  namespace CPDI
  {
    template <int dim, int spacedim>
    class ParticleDomainAccessor;

    template <int dim, int spacedim>
    class ParticleDomainIterator;

    template <int dim, int spacedim>
    class ParticleDomainHandler : public Subscriptor
    {
      public:
        using pid_type = int;

        using particle_iterator = typename ParticleIterator<dim, spacedim>;
        using pd_iterator       = typename ParticleDomainIterator<dim, spacedim>;

        ParticleDomainHandler();

#if DEAL_II_VERSION_GTE(9,8,0)
        void initialize(const Particles::ParticleHandler<dim, spacedim> &particle_handler);
#else
        void initialize(const Particles::ParticleHandler<dim, spacedim> &particle_handler,
                        const Triangulation<dim, spacedim>              &triangulation);
#endif

        void reinit();

        void clear();

        pd_iterator begin() const;

        pd_iterator end() const;

      private:
        SmartPointer<const Particles::ParticleHandler<dim, spacedim>, ParticleDomainHandler<dim, spacedim>> particle_handler;

        SmartPointer<const Triangulation<dim, spacedim>, ParticleDomainHandler<dim, spacedim>> triangulation;

        voro::container voro_container;

        std::map<pid_type, particle_iterator> pd2p;

        friend class ParticleDomainAccessor<dim, spacedim>;
    };
  }
}

#endif
