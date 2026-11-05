/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <deal.II/base/exceptions.h>
#include <algorithm>
#include <cmath>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace TensorReturn
      {
        Tensor in_plane(const Tensor &x)
        {
          Tensor y;
          y[0][0] = .5 * (x[0][0] - x[1][1]);
          y[1][1] = -y[0][0];
          y[0][1] = x[0][1];
          return y;
        }

        double magnitude(const Tensor &x)
        {
          return std::hypot(.5 * (x[0][0] - x[1][1]), x[0][1]);
        }

        namespace
        {
          Tensor project(const Tensor &x, const Tensor *S)
          {
            return S ? 2.*(x * (*S)) * (*S) : in_plane(x);
          }

          void validate(const Parameters &p, const State &s, const Tensor &increment,
                        const double phi, const double dt, const Tensor *S)
          {
            AssertThrow(p.G > 0. && p.eta > 0. && p.m > 0. && p.resistance >= 0.
                        && std::isfinite(dt) && dt > 0. && phi >= s.phi && s.phi >= 0.
                        && phi < 1. && s.L >= 0., dealii::ExcMessage("Invalid tensor point parameters or physical obstacle."));
            const auto planar = [](const Tensor & x)
            {
              return std::isfinite(x.norm()) && std::abs(dealii::trace(x)) + std::abs(x[0][2]) + std::abs(x[1][2]) <= 1e-12 * std::max(1., x.norm());
            };
            AssertThrow(planar(s.T) && planar(increment), dealii::ExcMessage("Tensor point source requires the full plane-strain deviator."));
            if (S)
              AssertThrow((in_plane(*S) - *S).norm() <= 1e-12 && std::abs((*S) * (*S) - .5) <= 1e-12,
                          dealii::ExcMessage("Fixed-direction control requires an in-plane S:S=1/2."));
            for (const auto *x :
            {
              &s.A, &s.K, &s.Q
            })
            AssertThrow(std::isfinite(x->norm()) && (*x - project(*x, S)).norm() <= 1e-12 * std::max(1e-3, x->norm()),
                        dealii::ExcMessage("Incoming tensor history is outside the selected flow space; no projection or rotation is permitted."));
            const double h = compliance(s.phi, p.m).h;
            AssertThrow(h > 0. || s.A == Tensor(), dealii::ExcMessage("Intact point cannot carry finite cohesive memory."));
            if (h > 0.)
              AssertThrow((2 * p.G * s.A - h * s.Q).norm() <= 1e-12 * std::max({1., 2 * p.G * s.A.norm(), h * s.Q.norm()}),
                          dealii::ExcMessage("Incoming physical A/Q/compliance identity is inconsistent."));
          }
        }

        Response update(const Parameters &p, const State &old, const Tensor &increment,
                        const double phi, const double dt, const Tensor *S)
        {
          validate(p, old, increment, phi, dt, S);
          const auto h = compliance(phi, p.m);
          Response r;
          r.state = old;
          r.state.phi = phi;
          r.intact = h.h == 0.;
          r.beta = 1. / (1. + dt * p.G / p.eta);
          r.state.T = r.beta * (old.T + 2 * p.G * increment);
          const auto X = project(r.state.T, S), abar = r.beta * old.A;
          Tensor delta;
          if (r.intact)
            {
              const double v = magnitude(X);
              r.sliding = v > p.resistance;
              r.state.A = 0.;
              r.state.Q = r.sliding ? (1. - p.resistance / v) * X : Tensor();
            }
          else
            {
              // h-scaled trial avoids 1/h at the intact limit. All shear modes
              // share ONE radial norm/direction; older A is not realigned.
              const auto V = h.h * X - 2 * p.G * (1 + h.h) * abar;
              const double v = magnitude(V);
              r.sliding = v > p.resistance * h.h;
              if (r.sliding)
                {
                  const auto E = V / v;
                  r.state.Q = (X - p.resistance * E) / (1 + h.h);
                  r.state.A = (h.h * r.state.Q) / (2 * p.G);
                  // Compute flow from the positive radial excess rather than
                  // subtracting almost equal memories close to sticking.
                  delta = (v - p.resistance * h.h) / (2 * p.G * (1 + h.h)) * E;
                }
              else
                {
                  r.state.A = abar;
                  r.state.Q = (2 * p.G * abar) / h.h;
                }
            }
          r.flow_rate = delta / (r.beta * dt);
          // sqrt(2)||W||_3D = 2*hypot(Wxx,Wxy) in this shear space.
          // hypot retains representable tiny flow without squaring it to zero.
          r.speed = 2 * magnitude(r.flow_rate);
          r.state.K += delta / r.beta;
          r.state.L += dt * r.speed;
          r.stress = r.state.T - 2 * p.G * r.state.A;
          r.driving = h.first * (r.state.Q * r.state.Q) / (4 * p.G);
          r.bulk_energy = (r.stress * r.stress) / (4 * p.G);
          r.cohesive_energy = h.h * (r.state.Q * r.state.Q) / (4 * p.G);
          r.bulk_dissipation = (r.stress * r.stress) / (2 * p.eta);
          r.cohesive_dissipation = h.h * (r.state.Q * r.state.Q) / (2 * p.eta);
          r.friction_dissipation = p.resistance * r.speed;
          AssertThrow(std::isfinite(r.driving + r.bulk_energy + r.cohesive_energy + r.speed),
                      dealii::ExcMessage("Nonfinite experimental tensor return."));
          return r;
        }

        Variation derivative(const Parameters &p, const State &old, const Response &r,
                             const Tensor &de, const double dphi, const double dt, const Tensor *S)
        {
          const auto h = compliance(r.state.phi, p.m);
          Variation d;
          d.T = 2 * p.G * r.beta * de;
          const auto X = project(r.state.T, S), dX = project(d.T, S), abar = r.beta * old.A;
          if (r.intact)
            {
              if (r.sliding)
                {
                  const double v = magnitude(X);
                  const auto E = X / v;
                  const auto dE = (dX - .5 * (E * dX) * E) / v;
                  d.Q = dX - p.resistance * dE - h.first * dphi * r.state.Q;
                  d.A = h.first * dphi / (2 * p.G) * r.state.Q;
                  d.speed = h.first * magnitude(r.state.Q) / (p.G * r.beta * dt) * dphi;
                }
            }
          else if (r.sliding)
            {
              const auto V = h.h * X - 2 * p.G * (1 + h.h) * abar;
              const auto dV = h.h * dX + h.first * dphi * (X - 2 * p.G * abar);
              const double v = magnitude(V);
              const auto E = V / v;
              const auto dE = (dV - .5 * (E * dV) * E) / v;
              d.Q = (dX - p.resistance * dE) / (1 + h.h) - h.first * dphi / (1 + h.h) * r.state.Q;
              d.A = (h.first * dphi * r.state.Q + h.h * d.Q) / (2 * p.G);
            }
          else d.Q = (-h.first * dphi * r.state.Q) / h.h;
          d.flow_rate = d.A / (r.beta * dt);
          d.K = d.A / r.beta;
          if (r.speed > 0.)
            d.speed = 2 * ((r.flow_rate / r.speed) * d.flow_rate);
          d.L = dt * d.speed;
          d.stress = d.T - 2 * p.G * d.A;
          d.driving = h.second * dphi * (r.state.Q * r.state.Q) / (4 * p.G) + h.first * (r.state.Q * d.Q) / (2 * p.G);
          d.bulk_energy = (r.stress * d.stress) / (2 * p.G);
          d.cohesive_energy = h.first * dphi * (r.state.Q * r.state.Q) / (4 * p.G) + h.h * (r.state.Q * d.Q) / (2 * p.G);
          d.bulk_dissipation = (r.stress * d.stress) / p.eta;
          d.cohesive_dissipation = h.first * dphi * (r.state.Q * r.state.Q) / (2 * p.eta) + h.h * (r.state.Q * d.Q) / p.eta;
          d.friction_dissipation = p.resistance * d.speed;
          return d;
        }

        StepAccounting account(const Parameters &p, const State &old, const Response &r,
                               const Tensor &increment, const double dt)
        {
          const double hn = compliance(old.phi, p.m).h, h = compliance(r.state.phi, p.m).h;
          const auto tau_old = old.T - 2 * p.G * old.A;
          const double Ec = hn == 0. ? 0. : p.G * (old.A * old.A) / hn;
          StepAccounting a;
          a.internal_work = r.stress * increment;
          a.storage_change = r.bulk_energy + r.cohesive_energy - (tau_old * tau_old) / (4 * p.G) - Ec;
          a.physical_dissipation = dt * (r.bulk_dissipation + r.cohesive_dissipation + r.friction_dissipation);
          const auto ds = r.stress - tau_old, dq = r.state.Q - old.Q;
          a.stress_loss = (ds * ds) / (4 * p.G);
          a.traction_loss = hn * (dq * dq) / (4 * p.G);
          a.compliance_work = (h - hn) * (r.state.Q * r.state.Q) / (4 * p.G);
          a.error = a.internal_work - a.storage_change - a.physical_dissipation - a.stress_loss - a.traction_loss - a.compliance_work;
          return a;
        }
      }
    }
  }
}
