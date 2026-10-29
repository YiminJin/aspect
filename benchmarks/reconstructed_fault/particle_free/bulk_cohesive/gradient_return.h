/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#pragma once
// Point-only prototype: a prescribed-X inverse is demonstrably not unique
// throughout the intended phase range. No native backend selects this API.
#include <aspect/material_model/phase_field_fault/bulk_cohesive_gradient.h>

namespace aspect { namespace MaterialModel { namespace BulkCohesive { namespace GradientReturn
{
  /** Experimental fixed-phase Coulomb return with a prescribed energetic
   * backstress X. A spatial microforce solve, not this stateless return,
   * supplies X=-div(2G ell_s^2 grad K). Physical normal pressure uses tau.
   * Incoming histories are physical and are never transformed in place. */
  struct Response : CoulombReturn::Response { Tensor backstress; };

  Response update(const CoulombReturn::Parameters &,const TensorReturn::State &,
                  const Tensor &increment,double dt,double pressure,const Tensor &X);

  /** Complete fixed-phase action, including changing directions, both slip
   * systems and variations of pressure and X. Phase evolution is not enabled.
   * At the intact homogeneous control X=dX=0, reuse the retained intact law. */
  CoulombReturn::Variation derivative(const CoulombReturn::Parameters &,
                  const TensorReturn::State &,const Response &,const Tensor &increment_variation,
                  double dt,double pressure_variation,const Tensor &backstress_variation);

  using SlipGradient::Projection;
  using SlipGradient::project;
  using SlipGradient::projection_action;
}}}}
