/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
#ifndef aspect_reconstructed_fault_cohesive_interface_h
#define aspect_reconstructed_fault_cohesive_interface_h
#include <aspect/global.h>
#include <cmath>

namespace aspect { namespace CohesiveInterface
{
  /** Experimental geometric bond/contact law. Not selected by production models.
   * All rates are interface m/s; damage is a surface variable, not bulk phi.
   */
  struct Parameters
  {
    double G=1e6, eta=1e8, Gc=1., peak=3000., p0=10000.;
    double direct=.005, evolution=.01, mu0=.1, V0=5e-6, Dc=.001;
    double radiation=1e6/(2.*std::sqrt(1e6/2670.));
  };

  struct PointState
  {
    double jump=0., damage=0., theta=1., reference=0., relaxed_jump=0., absolute_slip=0.;
    template<class Archive> void serialize(Archive &ar,const unsigned int)
    { ar &jump &damage &theta &reference &relaxed_jump &absolute_slip; }
  };

  struct PointTrial
  {
    PointState state;
    double q=0., resistance=0., derivative=0., pressure_derivative=0.;
    double storage=0., friction=0., radiation=0., velocity=0.;
  };

  /** Evaluate from an immutable accepted point. Incoming normal pressure is the
   * current mixed mechanical reaction. The derivative includes implicit aging
   * and damage; pressure_derivative is a separate partial derivative.
   * Exact intact points are tied by the caller; sign_hint is their one-sided
   * admission branch. No finite stiffness or phase/rate floor is substituted.
   */
  PointTrial evaluate(const Parameters &, const PointState &, double increment,
                      double pressure, double dt, double sign_hint=1.);
}}
#endif
