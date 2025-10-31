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

#include <aspect/cpdi/particle_domain_accessor.h>

namespace aspect
{
  namespace CPDI
  {
    template <int dim, int spacedim>
    void
    ParticleDomainFaceAccessor<dim, spacedim>::next()
    {
      Assert(state() == IteratorState::valid, ExcInternalError());

      vertices.clear();

      const std::vector<int> &info = cell->face_vertex_info;
      handle += info[handle] + 1;
      if (handle < info.size())
        {
          // Advance the face index
          ++index;

          // Reset face vertices
          for (unsigned int v = 0; v < info[handle]; ++v)
            {
              const unsigned int i = voro_dim * _info[handle + 1 + v];
              AssertIndexRange(i + spacedim - 1, cell->vertex_coords.size());

              Point<spacedim> p;
              for (unsigned int d = 0; d < spacedim; ++d)
                p[d] = cell->vertex_coords[i + d];
              vertices.push_back(p);
            }
        }
      else
        {
          index = numbers::invalid_unsigned_int;
          handle = -1;
        }
    }



    template <int dim, int spacedim>
    void
    ParticleDomainAccessor<dim, spacedim>::next()
    {
      Assert(state() == IteratorState::valid, ExcInternalError());

      if (voro_loop.inc())
        {
          // Compute the voronoi cell.
          Assert(pd_handler->voro_container.compute_cell(voro_cell, voro_loop),
                 ExcMessage("Voronoi cell cannot be computed."));

          voro_loop.pos(x, y, z);
          voro_loop.pid(index);

          voro_cell.neighbors(neighbors);
          voro_cell.face_vertices(face_vertex_info);
          voro_cell.vertices(x, y, z, vertex_coords);
        }
      else
        {
          // Set the index to invalid value
          index = -1;

          // Clear the data
          neighbors.clear();
          face_vertex_info.clear();
          vertex_coords.clear();
        }
    }
  }
}
