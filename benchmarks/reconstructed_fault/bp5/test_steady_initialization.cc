#include "../bp3/bp3_model.h"
#include <iostream>
#include <stdexcept>

#ifndef ASPECT_BP5_STEADY_INITIALIZATION
#error This test must select the steady initialization variant
#endif

// Deliberately provides no inverse-friction operation: that initialization
// cannot be called in this variant, even accidentally before projection.
struct ConfiguredLaw
{
  double get_characteristic_slip_distance() const { return .1; }
};

int main()
{
  const ConfiguredLaw law;
  for (const double xd : {0., 15000., 30000., 31500., 33000., 115470.})
    {
      const double theta = BP3::configured_initial_state(xd, law);
      if (theta != 1e8)
        throw std::runtime_error("Initial state is not spatially uniform Dc/Vinit.");
      for (const double dt : {0., 150., 300., 4e6})
        if (std::abs(BP3::aging_state_reference(BP3::Vinit, theta, dt, .1)/theta-1) > 1e-15)
          throw std::runtime_error("Steady sliding is not a fixed point of exact aging.");
    }
  std::cout << "Steady initialization: six positions, four aging intervals; no inverse-state API. PASS\n";
  const auto steady_identity=BP3::initial_condition_identity();
  if (steady_identity!="steady state/native weak prestress v1")
    throw std::runtime_error("Default checkpoint identity changed.");
  BP3::weakening_length=30000.;
  BP3::weakening_initial_state_ratio=.8;
  for (const double f : {0., .1, .5, .9, 1.})
    if (std::abs(BP3::configured_initial_state(30000.+3000.*f,law)/(1e8*std::pow(.8,1-f))-1)>1e-15)
      throw std::runtime_error("Loading state profile differs from geometric mixture.");
  const auto loading_identity=BP3::initial_condition_identity();
  BP3::weakening_initial_state_ratio=.9;
  if (loading_identity==steady_identity || loading_identity==BP3::initial_condition_identity())
    throw std::runtime_error("Distinct ratios share a restart identity.");
  BP3::weakening_initial_state_ratio=.8;
  const double measure=7.5*std::log(BP3::aging_state_reference(1e-9,8e7,1e6,.1)/8e7);
  if (std::abs(measure-.0186333956437284)>1e-13)
    throw std::runtime_error("Loading predictor scalar screen failed.");
  std::cout << "Loading initialization: five mixture fractions, distinct restart identities, scalar predictor. PASS\n";
}
