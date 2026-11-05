/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Experimental closed-contact tensor return; phase-active native use is unqualified.
#pragma once
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <array>
namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace CoulombReturn
      {
        /** Closed-contact coefficients. residual_cohesion is c_r (Pa), not the
         * native strength-drop parameter; mu is dimensionless. */
        struct Parameters
        {
          BulkCohesive::Parameters material;
          double mu = 0., residual_cohesion = 0.;
        };
        /** Maximizing plane: theta in radians; normal/shear/margin in Pa.
         * shear is (tau-Q):S; defined means a maximizer exists, not W!=0. */
        struct Plane
        {
          double theta = 0., normal = 0., shear = 0., margin = 0.;
          bool defined = false;
        };
        /** Unpublished point response with up to two active slip planes. */
        struct Response
        {
          TensorReturn::Response point;
          std::array<Plane, 2> planes;
          std::array<double, 2> gamma {{0., 0.}};
          double pressure = 0., system_speed = 0.;
          unsigned int active = 0;
        };
        /** Directional response action at fixed incoming history. */
        struct Variation
        {
          TensorReturn::Variation point;
          std::array<double, 2> theta {{0., 0.}}, normal {{0., 0.}}, shear {{0., 0.}}, gamma {{0., 0.}}, margin {{0., 0.}};
          double system_speed = 0.;
        };
        /** Return the two closed-contact plane extrema for physical tau-Q. */
        std::array<Plane, 2> planes(const Tensor &tau, const Tensor &Q, double pressure, double mu, double cohesion);
        /** Backward-Euler point return at supplied phase, dt (s), pressure (Pa). */
        Response update(const Parameters &parameters, const TensorReturn::State &incoming, const Tensor &de, double phi, double dt, double pressure);
        /** Differentiate the selected active set; de/dphi/dpressure are variations. */
        Variation derivative(const Parameters &parameters, const TensorReturn::State &incoming, const Response &response, const Tensor &de, double dphi, double dt, double dpressure);
      }
    }
  }
}
