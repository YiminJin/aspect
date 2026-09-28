/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"
#include <aspect/time_stepping/reconstructed_fault.h>

#include <aspect/material_model/rheology/fault_friction.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/property/maxwell_stress.h>
#include <aspect/phase_field.h>

#include "../tests/phase_field_fault_test_access.h"

#include <sstream>

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
  REQUIRE(model.get_phase_field_upper_admissibility_threshold() == 1.0);

  aspect::MaterialModel::PhaseFieldFault<2> fault_model;
  REQUIRE(fault_model.get_phase_field_upper_admissibility_threshold() == 0.99);
}



TEST_CASE("I_h distinguishes physical phase-field range from singular degradation")
{
  using TestAccess =
    aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;

  const double tolerance =
    TestAccess::normalization_phase_field_undershoot_tolerance();
  REQUIRE(TestAccess::normalization_effective_phase_field(-0.5*tolerance) == 0.0);
  REQUIRE(TestAccess::normalization_effective_phase_field(0.25) == 0.25);
  REQUIRE(TestAccess::normalization_effective_phase_field(1.0) == 1.0);
  REQUIRE_NOTHROW(TestAccess::validate_normalization_phase_field_minimum(-tolerance));
  REQUIRE_THROWS_WITH(
    TestAccess::validate_normalization_phase_field_minimum(-2.0*tolerance),
    Catch::Matchers::Contains("error-detection"));
  REQUIRE(TestAccess::normalization_integrand(0.0, 1.0) == 0.0);
  REQUIRE_THROWS_WITH(TestAccess::normalization_integrand(1.0, 0.0),
                      Catch::Matchers::Contains("I_h singularity"));
  REQUIRE_THROWS_WITH(TestAccess::normalization_effective_phase_field(1.0+1.e-6),
                      Catch::Matchers::Contains("phase-field invariant"));
}



TEST_CASE("Legacy slip-rate normalization requires an interior phase-field interval")
{
  const aspect::PhaseField::GeometricFunction geometric_function(0.1, 1.0, 8.0/3.0);
  const aspect::PhaseField::DegradationFunction degradation_function(0.0, 1.0);
  REQUIRE_THROWS_WITH(
    aspect::PhaseField::SlipRateNormalizer(geometric_function,
                                           degradation_function,
                                           0.01,
                                           1.0),
    Catch::Matchers::Contains("strict interior"));
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



TEST_CASE("FaultFriction declares the Stage E friction-law defaults")
{
  aspect::MaterialModel::Rheology::FaultFriction<2> friction;
  REQUIRE(friction.has_state_variable());

  dealii::ParameterHandler parameters;
  aspect::MaterialModel::Rheology::FaultFriction<2>::declare_parameters(parameters);
  REQUIRE(parameters.get("Friction law") == "rate state");
  REQUIRE(parameters.get("Dynamic friction coefficients") == "0.4");
  REQUIRE(parameters.get("Characteristic weakening slip rates") == "1.e-6");
  std::ostringstream parameter_text;
  parameters.print_parameters(parameter_text,
                              dealii::ParameterHandler::OutputStyle::Text);
  REQUIRE(parameter_text.str().find("Maximum slip rate") == std::string::npos);
}

TEST_CASE("Reconstructed fault state limiter parameter is opt in", "[fault_state_limiter]")
{
  using Limiter = aspect::TimeStepping::ReconstructedFault<2>;
  const std::string disabled = dealii::Utilities::to_string(std::numeric_limits<double>::max());
  for (const std::string &value : std::vector<std::string>{disabled, "infinity", "0.1", "1e-12", "2"})
    {
      dealii::ParameterHandler parameters;
      Limiter::declare_parameters(parameters);
      parameters.enter_subsection("Time stepping");
      parameters.enter_subsection("Reconstructed fault time step");
      REQUIRE(parameters.get_double("Maximum logarithmic state change") == std::numeric_limits<double>::max());
      parameters.set("Maximum logarithmic state change", value);
      parameters.leave_subsection();
      parameters.leave_subsection();
      Limiter limiter;
      REQUIRE_NOTHROW(limiter.parse_parameters(parameters));
    }
  for (const std::string value : {"0", "-0.1", "nan", "-infinity", "invalid", "0.1 junk"})
    {
      dealii::ParameterHandler parameters;
      Limiter::declare_parameters(parameters);
      parameters.enter_subsection("Time stepping");
      parameters.enter_subsection("Reconstructed fault time step");
      parameters.set("Maximum logarithmic state change", value);
      parameters.leave_subsection();
      parameters.leave_subsection();
      Limiter limiter;
      REQUIRE_THROWS(limiter.parse_parameters(parameters));
    }
}



TEST_CASE("PhaseFieldFault declares dynamic fault pressure by default")
{
  dealii::ParameterHandler parameters;
  aspect::MaterialModel::PhaseFieldFault<2>::declare_parameters(parameters);
  parameters.enter_subsection("Material model");
  parameters.enter_subsection("Phase field fault");
  REQUIRE_FALSE(parameters.get_bool("Use adiabatic pressure in fault friction"));
}
