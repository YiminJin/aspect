/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/particle/property/maxwell_stress.h>



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
