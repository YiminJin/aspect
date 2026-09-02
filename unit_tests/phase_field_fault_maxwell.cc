/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/material_model/rheology/fault_friction.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/property/maxwell_stress.h>
#include <aspect/phase_field.h>

namespace
{
  class TestPhaseFieldModel : public aspect::MaterialModel::PhaseFieldModel<2>
  {
    public:
      std::vector<double> get_critical_crack_driving_forces() const override
      {
        return {1.0};
      }

      std::vector<double> get_critical_energy_release_rates() const override
      {
        return {1.0};
      }
  };
}



TEST_CASE("Phase-field physical and activation ranges are distinct")
{
  const TestPhaseFieldModel model;
  REQUIRE(model.get_phase_field_range() == std::make_pair(0.0, 1.0));
  REQUIRE(model.get_phase_field_activation_threshold() == 0.01);
  REQUIRE(model.get_phase_field_upper_admissibility_threshold() == 0.99);
}



TEST_CASE("I_h distinguishes physical phase-field range from singular degradation")
{
  using TestAccess =
    aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;

  REQUIRE(TestAccess::normalization_integrand(0.0, 1.0) == 0.0);
  REQUIRE_THROWS_WITH(TestAccess::normalization_integrand(1.0, 0.0),
                      Catch::Matchers::Contains("I_h singularity"));
  REQUIRE_THROWS_WITH(TestAccess::normalization_integrand(-1.e-6, 1.0),
                      Catch::Matchers::Contains("phase-field invariant"));
  REQUIRE_THROWS_WITH(TestAccess::normalization_integrand(1.0+1.e-6, 1.0),
                      Catch::Matchers::Contains("phase-field invariant"));
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



TEST_CASE("FaultFriction defaults to the implemented rate-state law")
{
  aspect::MaterialModel::Rheology::FaultFriction<2> friction;
  REQUIRE(friction.has_state_variable());

  dealii::ParameterHandler parameters;
  aspect::MaterialModel::Rheology::FaultFriction<2>::declare_parameters(parameters);
  REQUIRE(parameters.get("Friction law") == "rate state");
}
