/*
  Copyright (C) 2016 - 2025 by the authors of the ASPECT code.

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

#ifndef _aspect_postprocess_cumulative_slip_profile_h
#define _aspect_postprocess_cumulative_slip_profile_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class CumulativeSlipProfile : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        std::pair<std::string, std::string>
        execute(TableHandler &statistics) override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        void update_point_locations();

        std::string filename;

        std::vector<Point<dim>> sample_points;

        std::vector<std::vector<Point<dim>>> locally_owned_sample_points;

        std::vector<std::vector<unsigned int>> locally_owned_sample_point_indices;

        std::vector<typename Triangulation<dim>::active_cell_iterator> host_cells;
    };
  }
}

#endif
