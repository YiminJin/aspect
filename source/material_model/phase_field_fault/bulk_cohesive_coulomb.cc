/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <deal.II/base/exceptions.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace CoulombReturn
      {
        namespace
        {
          using V = std::array<double, 2>;
          V add(V a, V b)
          {
            return {{a[0] + b[0], a[1] + b[1]}};
          }
          V sub(V a, V b)
          {
            return {{a[0] - b[0], a[1] - b[1]}};
          }
          V mul(double s, V a)
          {
            return {{s *a[0], s *a[1]}};
          }
          double dot(V a, V b)
          {
            return a[0] * b[0] + a[1] * b[1];
          }
          double cross(V a, V b)
          {
            return a[0] * b[1] - a[1] * b[0];
          }
          double norm(V a)
          {
            return std::hypot(a[0], a[1]);
          }
          V J(V a)
          {
            return {{ -a[1], a[0]}};
          }
          V vec(const Tensor &a)
          {
            return {{.5 * (a[0][0] - a[1][1]), a[0][1]}};
          }
          Tensor tensor(V a)
          {
            Tensor t;
            t[0][0] = a[0];
            t[1][1] = -a[0];
            t[0][1] = a[1];
            return t;
          }
          V z(V t, V q, double h, double mu, int k)
          {
            return sub(mul(-k, J(sub(t, mul(1 + h, q)))), mul(mu, sub(t, mul(h, q))));
          }
          V direction(V z, int k)
          {
            return mul(k / norm(z), J(z));
          }
          std::array<double, 4> solve(double a[4][4], std::array<double, 4> b, unsigned int n)
          {
            for (unsigned int i = 0; i < n; ++i)
              {
                unsigned int pivot = i;
                for (unsigned int j = i + 1; j < n; ++j)
                  if (std::abs(a[j][i]) > std::abs(a[pivot][i]))
                    pivot = j;
                AssertThrow(std::isfinite(a[pivot][i]) && a[pivot][i] != 0., dealii::ExcMessage("Singular Coulomb active-set derivative; no regularization."));
                for (unsigned int k = 0; k < n; ++k)
                  std::swap(a[i][k], a[pivot][k]);
                std::swap(b[i], b[pivot]);
                const double scale = a[i][i];
                for (unsigned int k = i; k < n; ++k)
                  a[i][k] /= scale;
                b[i] /= scale;
                for (unsigned int j = 0; j < n; ++j)
                  if (j != i)
                    {
                      const double s = a[j][i];
                      for (unsigned int k = i; k < n; ++k)
                        a[j][k] -= s * a[i][k];
                      b[j] -= s * b[i];
                    }
              }
            return b;
          }
        }

        std::array<Plane, 2> planes(const Tensor &tau, const Tensor &Q, double pressure, double mu, double cohesion)
        {
          std::array<Plane, 2> result;
          const V d = vec(tau), e = vec(tau - Q);
          const double pbar = pressure - .5 * (tau[0][0] + tau[1][1]);
          for (unsigned int i = 0; i < 2; ++i)
            {
              const int k = i == 0 ? 1 : -1;
              const V zz = sub(mul(-k, J(e)), mul(mu, d));
              const double s = norm(zz);
              auto &r = result[i];
              r.margin = s - cohesion - mu * pbar;
              r.defined = s > 0.;
              if (r.defined)
                {
                  const V v = mul(1 / s, zz);
                  r.theta = .5 * std::atan2(v[1], v[0]);
                  r.normal = pbar + dot(d, v);
                  r.shear = dot(e, J(v));
                }
            }
          return result;
        }

        Response update(const Parameters &p, const TensorReturn::State &old, const Tensor &de, double phi, double dt, double pressure)
        {
          AssertThrow(std::isfinite(pressure) && std::isfinite(p.mu) && p.mu >= 0. && p.residual_cohesion >= 0., dealii::ExcMessage("Invalid Coulomb pressure/friction/cohesion."));
          auto validation = p.material;
          validation.resistance = p.residual_cohesion;
          Response r;
          r.point = TensorReturn::update(validation, old, de, phi, dt);
          r.pressure = pressure;
          const double h = compliance(phi, p.material.m).h, hn = compliance(old.phi, p.material.m).h, G = p.material.G;
        const V t = vec(r.point.state.T), ref = hn == 0. ? V {{0., 0.}}:
          mul(r.point.beta * (hn / h), vec(old.Q));
          const V trial = sub(t, mul(h, ref));
          const double pbar = pressure - .5 * (r.point.state.T[0][0] + r.point.state.T[1][1]);
          AssertThrow(pbar > norm(trial), dealii::ExcMessage("Coulomb point requires fully compressive physical trial stress; opening/tension is unsupported, pressure is not clipped."));
          if (p.mu == 0.)
            {
              // A backstress release can increase physical stress even when the trial
              // stress is closed. The identity control retains the same admission rule.
              AssertThrow(pbar > norm(vec(r.point.stress)), dealii::ExcMessage("Coulomb point requires fully compressive returned stress; opening/tension is unsupported, pressure is not clipped."));
              r.planes = planes(r.point.stress, r.point.state.Q, pressure, 0., p.residual_cohesion);
              r.active = r.point.sliding ? 1 : 0;
              r.system_speed = r.point.speed;
              return r;
            }
          const double R = p.residual_cohesion + p.mu * pbar, l = 1 + h, b = p.mu * h;
          V q = ref;
          std::array<double, 2> gam {{0., 0.}};
          unsigned int mask = 0;
          const double scale = std::max({1., norm(t), norm(ref), R}), tol = 1e-12 * scale;
          if (norm(z(t, ref, h, p.mu, 1)) > R || norm(z(t, ref, h, p.mu, -1)) > R)
            {
              struct Candidate
              {
                V q;
                std::array<double, 2> gamma;
                unsigned int active;
              };
              std::vector<Candidate> candidates;
              const auto admit = [&](V candidate, std::array<double, 2> gamma, unsigned int active)
              {
                V flow {{0., 0.}};
                for (unsigned int i = 0; i < 2; ++i)
                  {
                    const auto zz = z(t, candidate, h, p.mu, i == 0 ? 1 : -1);
                    const double f = norm(zz) - R;
                    if (f > tol || gamma[i] < -tol || ((active & (1u << i)) && std::abs(f) > tol))
                      return;
                    if (active & (1u << i))
                      {
                        gamma[i] = std::max(0., gamma[i]);
                        flow = add(flow, mul(gamma[i], direction(zz, i == 0 ? 1 : -1)));
                      }
                  }
                if (norm(sub(sub(candidate, ref), flow)) > tol || !(pbar > norm(sub(t, mul(h, candidate)))))
                  return;
                for (const auto &c : candidates)
                  if (norm(sub(c.q, candidate)) <= tol)
                    return;
                candidates.push_back({candidate, gamma, active});
              };
              for (unsigned int i = 0; i < 2; ++i)
                {
                  const int k = i == 0 ? 1 : -1;
                  const V z0 = z(t, ref, h, p.mu, k);
                  const double s = norm(z0);
                  if (s <= R)
                    continue;
                  const double excess = (s - R) * (s + R), g = excess / (R * l + std::sqrt((R * l) * (R * l) + (l * l + b * b) * excess));
                  const V v = mul(1 / (s * s), add(mul(R + l * g, z0), mul(b * g * k, J(z0))));
                  std::array<double, 2> gamma {{0., 0.}};
                  gamma[i] = g;
                  admit(add(ref, mul(g * k, J(v))), gamma, 1u << i);
                }
              const auto both = [&](V candidate)
              {
                const V gp = direction(z(t, candidate, h, p.mu, 1), 1), gm = direction(z(t, candidate, h, p.mu, -1), -1), d = sub(candidate, ref);
                const double det = cross(gp, gm);
                if (det == 0.)
                  return;
                admit(candidate, {{cross(d, gm) / det, cross(gp, d) / det}}, 3);
              };
              const double s = norm(t), a = l * l + b * b;
              if (s == 0.)
                {
                  if (norm(ref) > 0.)
                    both(mul(R / (std::sqrt(a)*norm(ref)), ref));
                }
              else
                {
                  const double discriminant = a * R * R - p.mu * p.mu * s * s;
                  if (discriminant >= 0.)
                    {
                      const double root = std::sqrt(discriminant), B = s * (l + p.mu * p.mu * h), C = (1 + p.mu * p.mu) * s * s - R * R;
                      both(mul((B + root) / (a * s), t));
                      const double rho = (B + root) == 0. ? (B - root) / a : C / (B + root);
                      both(mul(rho / s, t));
                    }
                }
              AssertThrow(candidates.size() == 1, dealii::ExcMessage("No unique admissible closed-contact Coulomb return (" + std::to_string(candidates.size()) + " roots); incoming history remains unchanged."));
              q = candidates[0].q;
              gam = candidates[0].gamma;
              mask = candidates[0].active;
            }
          auto &a = r.point;
          a.state = old;
          a.beta = 1. / (1. + dt * G / p.material.eta);
          a.state.T = a.beta * (old.T + 2 * G * de);
          a.state.phi = phi;
          a.state.Q = tensor(q);
          a.state.A = mask ? h / (2 * G) * a.state.Q : a.beta * old.A;
          a.flow_rate = mask ? h / (2 * G * a.beta * dt) * tensor(sub(q, ref)) : Tensor();
          a.speed = 2 * norm(vec(a.flow_rate));
          a.state.K = old.K + dt * a.flow_rate;
          a.state.L = old.L + dt * a.speed;
          a.stress = a.state.T - 2 * G * a.state.A;
          a.intact = h == 0.;
          a.sliding = mask != 0;
          a.driving = compliance(phi, p.material.m).first * (a.state.Q * a.state.Q) / (4 * G);
          a.bulk_energy = (a.stress * a.stress) / (4 * G);
          a.cohesive_energy = h * (a.state.Q * a.state.Q) / (4 * G);
          a.bulk_dissipation = (a.stress * a.stress) / (2 * p.material.eta);
          a.cohesive_dissipation = h * (a.state.Q * a.state.Q) / (2 * p.material.eta);
          a.friction_dissipation = (a.stress - a.state.Q) * a.flow_rate;
          r.gamma = gam;
          r.active = mask;
          r.planes = planes(a.stress, a.state.Q, pressure, p.mu, p.residual_cohesion);
          r.system_speed = h * (gam[0] + gam[1]) / (G * a.beta * dt);
          AssertThrow(a.friction_dissipation >= 0. && std::isfinite(a.driving + a.friction_dissipation + a.speed), dealii::ExcMessage("Invalid Coulomb dissipation/response."));
          return r;
        }

        Variation derivative(const Parameters &p, const TensorReturn::State &old, const Response &r, const Tensor &de, double dphi, double dt, double dp)
        {
          Variation result;
          auto &a = result.point;
          const auto &x = r.point;
          const double G = p.material.G;
          if (p.mu == 0.)
            {
              auto c = p.material;
              c.resistance = p.residual_cohesion;
              a = TensorReturn::derivative(c, old, x, de, dphi, dt);
              result.system_speed = a.speed;
              for (unsigned int i = 0; i < 2; ++i)
                if (r.planes[i].defined)
                  {
                    const int k = i == 0 ? 1 : -1;
                    const V zz = mul(-k, J(vec(x.stress - x.state.Q))), v = mul(1 / norm(zz), zz);
                    const V dz = mul(-k, J(vec(a.stress - a.Q))), dv = mul(1 / norm(zz), sub(dz, mul(dot(v, dz), v)));
                    result.theta[i] = .5 * cross(v, dv);
                    result.shear[i] = dot(vec(a.stress - a.Q), J(v)) + dot(vec(x.stress - x.state.Q), J(dv));
                    result.normal[i] = dp - .5 * (a.stress[0][0] + a.stress[1][1]) + dot(vec(a.stress), v) + dot(vec(x.stress), dv);
                    result.margin[i] = dot(v, dz);
                  }
              return result;
            }
          const auto H = compliance(x.state.phi, p.material.m);
          const double h = H.h, dh = H.first * dphi;
          const double hn = compliance(old.phi, p.material.m).h;
          a.T = 2 * G * x.beta * de;
          const V t = vec(x.state.T), q = vec(x.state.Q), dT = vec(a.T);
        const V ref = hn == 0. ? V {{0., 0.}}:
          mul(x.beta * (hn / h), vec(old.Q));
        const V dref = hn == 0. ? V {{0., 0.}}:
          mul(-dh / h, ref);
          const double dP = dp - .5 * (a.T[0][0] + a.T[1][1]);
          V dq = dref;
          if (r.active)
            {
              std::array<unsigned int, 2> indices {};
              unsigned int m = 0;
              for (unsigned int i = 0; i < 2; ++i)
                if (r.active & (1u << i))
                  indices[m++] = i;
              double M[4][4] {};
              M[0][0] = M[1][1] = 1.;
              std::array<double, 4> rhs {{dref[0], dref[1], 0., 0.}};
              for (unsigned int j = 0; j < m; ++j)
                {
                  const unsigned int i = indices[j];
                  const int k = i == 0 ? 1 : -1;
                  const V zz = z(t, q, h, p.mu, k), v = mul(1 / norm(zz), zz), g = mul(k, J(v));
                  const auto dz = [&](V dq)
                  {
                    return add(mul(k * (1 + h), J(dq)), mul(p.mu * h, dq));
                  };
                  const auto dg = [&](V dz)
                  {
                    return mul(k / norm(zz), J(sub(dz, mul(dot(v, dz), v))));
                  };
                  const V input = add(mul(-1, add(mul(k, J(dT)), mul(p.mu, dT))), mul(dh, add(mul(k, J(q)), mul(p.mu, q))));
                  const V gi = dg(input);
                  rhs[0] += r.gamma[i] * gi[0];
                  rhs[1] += r.gamma[i] * gi[1];
                  rhs[2 + j] = p.mu * dP - dot(v, input);
                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      V unit {{0., 0.}};
                      unit[c] = 1.;
                      const V zc = dz(unit), gc = dg(zc);
                      M[0][c] -= r.gamma[i] * gc[0];
                      M[1][c] -= r.gamma[i] * gc[1];
                      M[2 + j][c] = dot(v, zc);
                    }
                  M[0][2 + j] = -g[0];
                  M[1][2 + j] = -g[1];
                }
              const auto d = solve(M, rhs, 2 + m);
              dq = {{d[0], d[1]}};
              for (unsigned int j = 0; j < m; ++j)
                result.gamma[indices[j]] = d[2 + j];
            }
          a.Q = tensor(dq);
          a.A = r.active ? (dh * x.state.Q + h * a.Q) / (2 * G) : Tensor();
          a.stress = a.T - 2 * G * a.A;
          a.flow_rate = a.A / (x.beta * dt);
          a.K = a.A / x.beta;
          if (x.speed > 0.)
            a.speed = 2 * (x.flow_rate * a.flow_rate) / x.speed;
          else if (x.intact && r.active)
            a.speed = H.first * norm(q) / (G * x.beta * dt) * dphi;
          a.L = dt * a.speed;
          result.system_speed = (dh * (r.gamma[0] + r.gamma[1]) + h * (result.gamma[0] + result.gamma[1])) / (G * x.beta * dt);
          a.driving = H.second * dphi * (x.state.Q * x.state.Q) / (4 * G) + H.first * (x.state.Q * a.Q) / (2 * G);
          a.bulk_energy = (x.stress * a.stress) / (2 * G);
          a.cohesive_energy = dh * (x.state.Q * x.state.Q) / (4 * G) + h * (x.state.Q * a.Q) / (2 * G);
          a.bulk_dissipation = (x.stress * a.stress) / p.material.eta;
          a.cohesive_dissipation = dh * (x.state.Q * x.state.Q) / (2 * p.material.eta) + h * (x.state.Q * a.Q) / p.material.eta;
          a.friction_dissipation = (a.stress - a.Q) * x.flow_rate + (x.stress - x.state.Q) * a.flow_rate;
          for (unsigned int i = 0; i < 2; ++i)
            if (r.planes[i].defined)
              {
                const int k = i == 0 ? 1 : -1;
                const V zz = z(t, q, h, p.mu, k), v = mul(1 / norm(zz), zz);
                const V dz = sub(mul(-k, J(sub(vec(a.stress), dq))), mul(p.mu, vec(a.stress)));
                const V dv = mul(1 / norm(zz), sub(dz, mul(dot(v, dz), v)));
                result.theta[i] = .5 * cross(v, dv);
                result.normal[i] = dP + dot(vec(a.stress), v) + dot(vec(x.stress), dv);
                result.shear[i] = dot(vec(a.stress - a.Q), J(v)) + dot(vec(x.stress - x.state.Q), J(dv));
                result.margin[i] = dot(v, dz) - p.mu * dP;
              }
          return result;
        }
      }
    }
  }
}
