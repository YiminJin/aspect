/*
  Copyright (C) 2020 - 2024 by the authors of the ASPECT code.

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

#include "common.h"
#include <aspect/particle/property/interface.h>
#include <aspect/particle/manager.h>
#include <deal.II/base/parameter_handler.h>
#include <aspect/particle/particle_domain.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/distributed/tria.h>

#ifdef ASPECT_WITH_VORO
TEST_CASE("Advected Voronoi domains conserve area", "[particle_domain_area]")
{
  using namespace dealii;
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(4);
  MappingQ1<2> mapping;
  Particles::ParticleHandler<2> particles(triangulation, mapping, 0);
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i=0; i<3; ++i)
        for (unsigned int j=0; j<3; ++j)
          {
            Point<2> reference((i+0.5)/3, (j+0.5)/3);
            Point<2> position = mapping.transform_unit_to_real_cell(cell, reference);
            position[0] += 1.e-6*std::sin(17*position[0]+13*position[1]);
            position[1] += 1.e-6*std::cos(11*position[0]-7*position[1]);
            reference = mapping.transform_real_to_unit_cell(cell, position);
            const types::particle_index id = cell->global_active_cell_index()*9+i*3+j;
            particles.insert_particle(Particles::Particle<2>(position, reference, id), cell);
          }
  particles.update_cached_numbers();
  particles.exchange_ghost_particles();
#if DEAL_II_VERSION_GTE(9,8,0)
  aspect::Particle::ParticleDomainHandler<2> domains(particles, false, true);
#else
  aspect::Particle::ParticleDomainHandler<2> domains(particles, triangulation, mapping, false, true);
#endif
  domains.generate_particle_domains();
  double area = 0;
  for (const auto &particle : particles)
    area += domains.get_particle_domain(particle.get_local_index()).volume();
  REQUIRE(Utilities::MPI::sum(area, MPI_COMM_WORLD) == Approx(1.0).margin(1.e-10));
}
#endif

TEST_CASE("Particle Manager plugin names")
{
  dealii::ParameterHandler prm;
  aspect::Particle::Property::Manager<2> manager;
  // The property manager needs to know about the integrator, which is declared in World
  aspect::Particle::Manager<2>::declare_parameters(prm);

  prm.enter_subsection("Particles");
  manager.declare_parameters(prm);
  prm.set("List of particle properties","composition, position");
  manager.parse_parameters(prm);
  prm.leave_subsection();

  // existing and listed pluring
  REQUIRE(manager.plugin_name_exists("composition") == true);
  REQUIRE(manager.plugin_name_exists("position") == true);

  // existing but not listed plugin
  REQUIRE(manager.plugin_name_exists("pT path") == false);

  // non-existed plugin
  REQUIRE(manager.plugin_name_exists("non-existent plugin name") == false);

  // check that one is before the other
  REQUIRE(manager.check_plugin_order("composition", "position") == true);
  REQUIRE(manager.check_plugin_order("position", "composition") == false);

  // Check the plugin indices
  REQUIRE(manager.get_plugin_index_by_name("composition") == 0);
  REQUIRE(manager.get_plugin_index_by_name("position") == 1);
}
