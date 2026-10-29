/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Experimental closed-contact tensor return; phase-active native use is unqualified.
#pragma once
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <array>
namespace aspect { namespace MaterialModel { namespace BulkCohesive { namespace CoulombReturn
{
  // residual_cohesion is c_r, not the native 2000 Pa strength-drop parameter.
  struct Parameters { BulkCohesive::Parameters material; double mu=0.,residual_cohesion=0.; };
  // shear is (tau-Q):S; defined means a plane maximizer exists, not that W!=0.
  struct Plane { double theta=0.,normal=0.,shear=0.,margin=0.; bool defined=false; };
  struct Response
  {
    TensorReturn::Response point;
    std::array<Plane,2> planes;
    std::array<double,2> gamma{{0.,0.}};
    double pressure=0.,system_speed=0.;
    unsigned int active=0;
  };
  struct Variation
  {
    TensorReturn::Variation point;
    std::array<double,2> theta{{0.,0.}},normal{{0.,0.}},shear{{0.,0.}},gamma{{0.,0.}},margin{{0.,0.}};
    double system_speed=0.;
  };
  std::array<Plane,2> planes(const Tensor &tau,const Tensor &Q,double pressure,double mu,double cohesion);
  Response update(const Parameters &,const TensorReturn::State &,const Tensor &de,double phi,double dt,double pressure);
  Variation derivative(const Parameters &,const TensorReturn::State &,const Response &,const Tensor &de,double dphi,double dt,double dpressure);
}}}}
