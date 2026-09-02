/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/material_model/phase_field_rsf.h>
#include <aspect/particle/property/maxwell_stress.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/particles/particle.h>

#include <array>
#include <limits>

namespace
{
  class MaxwellThrowOnDealIIException
  {
    public:
      MaxwellThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::disable_abort_on_exception();
      }

      ~MaxwellThrowOnDealIIException()
      {
        dealii::deal_II_exceptions::enable_abort_on_exception();
      }
  };



  void
  populate_particle_handler(
    dealii::Triangulation<2> &triangulation,
    const dealii::MappingQ1<2> &mapping,
    dealii::Particles::ParticleHandler<2> &particle_handler)
  {
    dealii::GridGenerator::hyper_cube(triangulation);
    particle_handler.initialize(
      triangulation, mapping,
      dealii::SymmetricTensor<2,2>::n_independent_components);

    const auto cell = triangulation.begin_active();
    for (unsigned int particle_id = 0; particle_id < 2; ++particle_id)
      {
        const dealii::Point<2> reference_position(0.25 + 0.5 * particle_id,
                                                   0.5);
        const std::array<double,3> properties = {{1.0 + particle_id,
                                                   2.0 + particle_id,
                                                   3.0 + particle_id}};
        particle_handler.insert_particle(reference_position,
                                         reference_position,
                                         particle_id,
                                         cell,
                                         dealii::make_array_view(properties));
      }

  }
}



TEST_CASE("PhaseFieldRSF Maxwell coefficients use stable relaxation")
{
  using aspect::MaterialModel::internal::compute_maxwell_coefficients;

  const auto zero_step = compute_maxwell_coefficients(8.0, 2.0, 0.0);
  REQUIRE(zero_step.beta == Approx(1.0));
  REQUIRE(zero_step.kappa == Approx(0.0));

  const auto half_relaxed =
    compute_maxwell_coefficients(8.0, 2.0, 4.0 * std::log(2.0));
  REQUIRE(half_relaxed.beta == Approx(0.5));
  REQUIRE(half_relaxed.kappa == Approx(4.0));

  const double viscosity = 1.e30;
  const double shear_modulus = 1.e10;
  const double time_step = 1.e-6;
  const auto small_exponent =
    compute_maxwell_coefficients(viscosity, shear_modulus, time_step);
  REQUIRE(viscosity * (1.0 - std::exp(-time_step * shear_modulus / viscosity))
          == 0.0);
  REQUIRE(small_exponent.kappa
          == Approx(shear_modulus * time_step).epsilon(1.e-12));

  const auto fully_relaxed = compute_maxwell_coefficients(7.0, 1.0, 7000.0);
  REQUIRE(fully_relaxed.beta == Approx(0.0));
  REQUIRE(fully_relaxed.kappa == Approx(7.0));

  const MaxwellThrowOnDealIIException throw_on_dealii_exception;
  REQUIRE_THROWS(compute_maxwell_coefficients(0.0, 1.0, 1.0));
  REQUIRE_THROWS(compute_maxwell_coefficients(1.0, 0.0, 1.0));
  REQUIRE_THROWS(compute_maxwell_coefficients(1.0, 1.0, -1.0));
}



TEST_CASE("PhaseFieldRSF Maxwell stress is componentwise and non-rotational")
{
  const aspect::MaterialModel::internal::MaxwellCoefficients coefficients =
  {0.25, 3.0};

  dealii::SymmetricTensor<2,2> strain_rate;
  strain_rate[0][0] = 1.0;
  strain_rate[1][1] = -2.0;
  strain_rate[0][1] = 4.0;

  dealii::SymmetricTensor<2,2> previous_stress;
  previous_stress[0][0] = 8.0;
  previous_stress[1][1] = 12.0;
  previous_stress[0][1] = -4.0;

  const dealii::SymmetricTensor<2,2> stress =
    aspect::MaterialModel::internal::compute_maxwell_stress<2>(
      coefficients,
      strain_rate,
      previous_stress);

  REQUIRE(stress[0][0] == Approx(8.0));
  REQUIRE(stress[1][1] == Approx(-9.0));
  REQUIRE(stress[0][1] == Approx(23.0));

  const dealii::SymmetricTensor<2,2> zero_rate_stress =
    aspect::MaterialModel::internal::compute_maxwell_stress<2>(
      coefficients,
      dealii::SymmetricTensor<2,2>(),
      previous_stress);
  REQUIRE(zero_rate_stress[0][0] == Approx(2.0));
  REQUIRE(zero_rate_stress[1][1] == Approx(3.0));
  REQUIRE(zero_rate_stress[0][1] == Approx(-1.0));
}



TEST_CASE("MaxwellStress particle property has one symmetric tensor")
{
  aspect::Particle::Property::MaxwellStress<2> property_2d;
  const auto information_2d = property_2d.get_property_information();
  REQUIRE(information_2d.size() == 1);
  REQUIRE(information_2d[0].first == "maxwell stress");
  REQUIRE(information_2d[0].second
          == dealii::SymmetricTensor<2,2>::n_independent_components);

  aspect::Particle::Property::MaxwellStress<3> property_3d;
  const auto information_3d = property_3d.get_property_information();
  REQUIRE(information_3d.size() == 1);
  REQUIRE(information_3d[0].first == "maxwell stress");
  REQUIRE(information_3d[0].second
          == dealii::SymmetricTensor<2,3>::n_independent_components);
}



TEST_CASE("Maxwell stress transaction freezes, commits, and rolls back history")
{
  dealii::Triangulation<2> triangulation;
  const dealii::MappingQ1<2> mapping;
  dealii::Particles::ParticleHandler<2> particle_handler;
  populate_particle_handler(triangulation, mapping, particle_handler);

  aspect::MaterialModel::internal::MaxwellStressUpdateTransaction<2> transaction;
  transaction.begin();

  for (const auto &particle : particle_handler)
    {
      dealii::SymmetricTensor<2,2> stress;
      stress[0][0] = 10.0 + particle.get_id();
      stress[1][1] = 20.0 + particle.get_id();
      stress[0][1] = 30.0 + particle.get_id();
      transaction.stage(particle.get_id(), stress);
    }

  // Staging does not change committed particle history.
  REQUIRE(particle_handler.begin()->get_properties()[0] == Approx(1.0));
  transaction.rollback();
  REQUIRE_FALSE(transaction.is_active());
  REQUIRE(particle_handler.begin()->get_properties()[0] == Approx(1.0));

  transaction.begin();
  for (const auto &particle : particle_handler)
    {
      dealii::SymmetricTensor<2,2> stress;
      stress[0][0] = 10.0 + particle.get_id();
      stress[1][1] = 20.0 + particle.get_id();
      stress[0][1] = 30.0 + particle.get_id();
      transaction.stage(particle.get_id(), stress);
    }
  transaction.commit(particle_handler, 0, MPI_COMM_WORLD);

  REQUIRE_FALSE(transaction.is_active());
  for (const auto &particle : particle_handler)
    {
      REQUIRE(particle.get_properties()[0]
              == Approx(10.0 + particle.get_id()));
      REQUIRE(particle.get_properties()[1]
              == Approx(20.0 + particle.get_id()));
      REQUIRE(particle.get_properties()[2]
              == Approx(30.0 + particle.get_id()));
    }
}



TEST_CASE("Maxwell stress transaction validates before writing")
{
  dealii::Triangulation<2> triangulation;
  const dealii::MappingQ1<2> mapping;
  dealii::Particles::ParticleHandler<2> particle_handler;
  populate_particle_handler(triangulation, mapping, particle_handler);
  aspect::MaterialModel::internal::MaxwellStressUpdateTransaction<2> transaction;
  const MaxwellThrowOnDealIIException throw_on_dealii_exception;

  transaction.begin();
  dealii::SymmetricTensor<2,2> stress;
  stress[0][0] = 10.0;
  transaction.stage(particle_handler.begin()->get_id(), stress);
  REQUIRE_THROWS(transaction.commit(particle_handler, 0, MPI_COMM_WORLD));
  REQUIRE(particle_handler.begin()->get_properties()[0] == Approx(1.0));
  transaction.rollback();

  transaction.begin();
  stress[0][0] = std::numeric_limits<double>::quiet_NaN();
  transaction.stage(particle_handler.begin()->get_id(), stress);
  REQUIRE_THROWS(transaction.commit(particle_handler, 0, MPI_COMM_WORLD));
  REQUIRE(particle_handler.begin()->get_properties()[0] == Approx(1.0));
  transaction.rollback();
}
