/* Copyright (C) 2026 by the authors of ASPECT. GPL-2.0-or-later. */
#ifndef ASPECT_BENCHMARK_PF6_INTERVAL_RAMP_H
#define ASPECT_BENCHMARK_PF6_INTERVAL_RAMP_H
#include <aspect/boundary_velocity/interface.h>

namespace aspect { namespace BoundaryVelocity {
  template<int dim> class PF6aIntervalRamp: public Interface<dim>,public SimulatorAccess<dim>
  {
  public:
    Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
    {
      Tensor<1,dim> velocity;
      velocity[0]=.256*std::max(0.,this->get_time()-.5*this->get_timestep())*(x[1]-.5);
      return velocity;
    }
  };
  ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(PF6aIntervalRamp,"PF6a interval ramp",
    "Fixed verification ramp with its exact interval mean, including native cutbacks.")
}}

#endif
