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

#ifndef _aspect_cpdi_particle_domain_accessor_h
#define _aspect_cpdi_particle_domain_accessor_h

#include <aspect/cpdi/particle_domain_handler.h>

namespace aspect
{
  namespace CPDI
  {
    template <int dim, int spacedim>
    class ParticleDomainFaceAccessor
    {
      public:
        const std::vector<Point<spacedim>> &get_vertices() const;

        void next();

        bool operator==(const ParticleDomainFaceAccessor<dim, spacedim> &other) const;

      private:
        const std::shared_ptr<ParticleDomainAccessor<dim, spacedim>> cell;

        unsigned int index;

        int handle;

        std::vector<Point<spacedim>> vertices;
    };

    template <int dim, int spacedim>
    class ParticleDomainAccessor
    {
      public:
        SimpleIterator<ParticleDomainFaceAccessor<dim, spacedim>>
        begin_face() const;

        SimpleIterator<ParticleDomainFaceAccessor<dim, spacedim>>
        end_face() const;

        SimpleIterator<ParticleDomainFaceAccessor<dim, spacedim>>
        face(const unsigned int face_index) const;

        void next();

      private:
        const std::shared_ptr<const ParticleDomainHandler<dim, spacedim>> pd_handler;

        voro::c_loop_all voro_loop;

        voro::voronoicell_neighbor voro_cell;

        index_type index;

        double x;
        double y;
        double z;

        std::vector<int> neighbors;

        std::vector<int> face_vertex_info;

        std::vector<double> vertex_coords;
    };



    /*------------------------ inline functions ---------------------------*/

  }
}

#endif
