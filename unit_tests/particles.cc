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
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_tools.h>

#ifdef ASPECT_WITH_VORO
// Test the production implementation without adding a public particle-domain API.
namespace aspect
{
  namespace Particle
  {
    namespace internal
    {
      std::pair<dealii::Triangulation<2>::active_cell_iterator, dealii::Point<2>>
      cpdi_sample_owner(const dealii::Point<2> &,
                        const std::set<dealii::Triangulation<2>::active_cell_iterator> &,
                        const dealii::Mapping<2> &, const double);
    }
  }
}

TEST_CASE("CPDI face samples have one supported owner", "[particle_domain_constants]")
{
  using namespace dealii;
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{2,2},
                                           Point<2>(0,-1.0/64), Point<2>(0.25,1.0/64));
  SECTION("Conforming cells") {}
  SECTION("Coarse and fine neighbors")
  {
    triangulation.begin_active()->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
  }
  std::set<Triangulation<2>::active_cell_iterator> cells;
  for (const auto &cell : triangulation.active_cell_iterators())
    cells.insert(cell);
  MappingQ1<2> mapping;
  FE_Q<2> fe(1);
  // Include the actual K1 gap and exact/next-representable shared-face points.
  for (const double y : {0.0, -8.6736173798840355e-19,
                        std::nextafter(0.0, -1.0), std::nextafter(0.0, 1.0)})
    for (const double x : {0.005, 0.125, std::nextafter(0.125, 0.0),
                          std::nextafter(0.125, 1.0)})
      {
        const auto sample = aspect::Particle::internal::cpdi_sample_owner(
          Point<2>(x,y), cells, mapping, 1.e-12);
        double constant = 0.0;
        double affine = 0.0;
        for (unsigned int i=0; i<4; ++i)
          {
            constant += fe.shape_value(i, sample.second);
            affine += fe.shape_value(i, sample.second) *
                      (sample.first->vertex(i)[0]+sample.first->vertex(i)[1]);
          }
        REQUIRE(constant == Approx(1.0).margin(1.e-14));
        REQUIRE(affine == Approx(x+y).margin(1.e-14));
      }
  REQUIRE_THROWS(aspect::Particle::internal::cpdi_sample_owner(
                   Point<2>(-0.01,0), cells, mapping, 1.e-12));
}

TEST_CASE("CPDI domains reproduce constants across shared faces", "[particle_domain_constants]")
{
  using namespace dealii;
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{1,4},
                                           Point<2>(0,-0.5), Point<2>(0.25,0.5), true);
  std::vector<GridTools::PeriodicFacePair<Triangulation<2>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  triangulation.add_periodicity(periodic_faces);
  triangulation.refine_global(4);
  SECTION("K1 regular layout") {}
  SECTION("Nonuniform neighboring cells")
  {
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[1] > 0 && cell->center()[1] < 0.1)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
  }
  MappingQ1<2> mapping;
  Particles::ParticleHandler<2> particles(triangulation, mapping, 0);
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i=0; i<3; ++i)
        for (unsigned int j=0; j<3; ++j)
          {
            const Point<2> reference((i+0.5)/3, (j+0.5)/3);
            const Point<2> position = mapping.transform_unit_to_real_cell(cell, reference);
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
  double area = 0.0, value_error = 0.0, gradient_error = 0.0;
  for (const auto &particle : particles)
    {
      const auto domain = domains.get_particle_domain(particle.get_local_index());
      double sum = 0.0;
      Tensor<1,2> gradient;
      for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
        {
          sum += domain.weighting_function_value(i);
          gradient += domain.weighting_function_gradient(i);
        }
      value_error = std::max(value_error, std::abs(sum-1.0));
      gradient_error = std::max(gradient_error, gradient.norm());
      area += domain.volume();
    }
  // Reduce before assertions so all ranks follow the same collective lifecycle.
  area = Utilities::MPI::sum(area, MPI_COMM_WORLD);
  value_error = Utilities::MPI::max(value_error, MPI_COMM_WORLD);
  gradient_error = Utilities::MPI::max(gradient_error, MPI_COMM_WORLD);
  REQUIRE(area == Approx(0.25).margin(1.e-12));
  REQUIRE(value_error < 1.e-12);
  REQUIRE(gradient_error < 1.e-10);
}

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
