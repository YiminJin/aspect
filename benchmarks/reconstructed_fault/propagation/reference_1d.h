// Independent one-dimensional weak integration, using native material
// callbacks. Natural zero-flux boundaries on [0,1]; no prescribed phase at the
// band center.
#ifndef PROPAGATION_REFERENCE_1D_H
#define PROPAGATION_REFERENCE_1D_H
#include <algorithm>
#include <cmath>
#include <deal.II/base/quadrature_lib.h>
#include <functional>
#include <numeric>
#include <vector>

namespace P1Reference
{
struct Result
{
  std::vector<double> x, phi;
  double residual = 0, correction = 0;
  unsigned int iterations = 0;
  double
  value (double y) const
  {
    const auto it = std::upper_bound (x.begin (), x.end (), y);
    const unsigned int i = std::min<unsigned int> (
        std::max<std::ptrdiff_t> (1, it - x.begin ()) - 1, x.size () - 2);
    const double t = (y - x[i]) / (x[i + 1] - x[i]);
    return (1 - t) * phi[i] + t * phi[i + 1];
  }
};

template <class Coefficients>
Result
solve (const unsigned int cells, const unsigned int order,
       const std::vector<double> &breaks,
       const std::function<double (double)> &H,
       const std::function<double (double)> &profile,
       const Coefficients &coefficients, const bool cpdi_midpoints = false)
{
  Result out;
  for (unsigned int i = 0; i <= cells; ++i)
    out.x.push_back (double (i) / cells);
  // Reference mesh resolves the actual discontinuities, without smoothing H.
  if (!cpdi_midpoints && cells > 128)
    {
      out.x.insert (out.x.end (), breaks.begin (), breaks.end ());
      std::sort (out.x.begin (), out.x.end ());
      out.x.erase (std::unique (out.x.begin (), out.x.end (),
                                [] (double a, double b)
                                  { return std::abs (a - b) < 1e-14; }),
                   out.x.end ());
    }
  for (double x : out.x)
    out.phi.push_back (profile (x));
  struct QP
  {
    unsigned int cell;
    double t, w, H;
  };
  std::vector<QP> qps;
  const dealii::QGauss<1> gauss (order);
  for (unsigned int i = 0; i + 1 < out.x.size (); ++i)
    {
      const double a = out.x[i], b = out.x[i + 1];
      if (cpdi_midpoints)
        {
          for (unsigned int j = 0; j < 3; ++j)
            {
              const double t = (j + 0.5) / 3;
              qps.push_back ({ i, t, (b - a) / 3, H (a + t * (b - a)) });
            }
        }
      else
        {
          std::vector<double> cuts = { a };
          for (double x : breaks)
            if (x > a + 1e-14 && x < b - 1e-14)
              cuts.push_back (x);
          cuts.push_back (b);
          std::sort (cuts.begin (), cuts.end ());
          for (unsigned int k = 0; k + 1 < cuts.size (); ++k)
            for (unsigned int q = 0; q < gauss.size (); ++q)
              {
                const double y
                    = cuts[k] + (cuts[k + 1] - cuts[k]) * gauss.point (q)[0];
                qps.push_back ({ i, (y - a) / (b - a),
                                 (cuts[k + 1] - cuts[k]) * gauss.weight (q),
                                 H (y) });
              }
        }
    }
  const unsigned int n = out.x.size ();
  std::vector<double> diag (n), off (n - 1), rhs (n);
  const auto assemble = [&] (const std::vector<double> &phi)
    {
      std::fill (diag.begin (), diag.end (), 0);
      std::fill (off.begin (), off.end (), 0);
      std::fill (rhs.begin (), rhs.end (), 0);
      for (const auto &q : qps)
        {
          const unsigned int i = q.cell;
          const double h = out.x[i + 1] - out.x[i];
          const double N[2] = { 1 - q.t, q.t }, D[2] = { -1 / h, 1 / h };
          const double u = N[0] * phi[i] + N[1] * phi[i + 1],
                       grad = (phi[i + 1] - phi[i]) / h;
          const auto c = coefficients (u, q.H);
          if (!c.valid)
            return false;
          for (unsigned int j = 0; j < 2; ++j)
            {
              rhs[i + j]
                  -= q.w * (N[j] * c.reaction + D[j] * c.diffusion * grad);
              diag[i + j]
                  += q.w
                     * (N[j] * N[j] * c.tangent + D[j] * D[j] * c.diffusion);
            }
          off[i]
              += q.w * (N[0] * N[1] * c.tangent + D[0] * D[1] * c.diffusion);
        }
      return true;
    };
  const auto norm = [] (const std::vector<double> &v)
    {
      double s = 0;
      for (double x : v)
        s += x * x;
      return std::sqrt (s);
    };
  for (unsigned int iteration = 0; iteration < 60; ++iteration)
    {
      AssertThrow (
          assemble (out.phi),
          dealii::ExcMessage ("1D reference left native degradation domain."));
      out.residual = norm (rhs);
      out.iterations = iteration;
      // Direct tridiagonal Newton correction, independent of production AMG.
      auto d = diag, update = rhs;
      for (unsigned int i = 1; i < n; ++i)
        {
          const double m = off[i - 1] / d[i - 1];
          d[i] -= m * off[i - 1];
          update[i] -= m * update[i - 1];
        }
      update[n - 1] /= d[n - 1];
      for (unsigned int i = n - 1; i > 0; --i)
        update[i - 1] = (update[i - 1] - off[i - 1] * update[i]) / d[i - 1];
      out.correction = 0;
      for (double x : update)
        out.correction = std::max (out.correction, std::abs (x));
      // State correction plus absolute weak residual; not a discretization
      // bound.
      if (out.correction < 2e-12 && out.residual < 2e-9)
        return out;
      bool accepted = false;
      for (unsigned int ls = 0; ls < 35; ++ls)
        {
          auto trial = out.phi;
          const double alpha = std::ldexp (1.0, -int (ls));
          for (unsigned int i = 0; i < n; ++i)
            trial[i] += alpha * update[i];
          if (assemble (trial) && norm (rhs) < out.residual)
            {
              out.phi = trial;
              accepted = true;
              break;
            }
        }
      AssertThrow (accepted,
                   dealii::ExcMessage ("1D reference line search stalled."));
    }
  AssertThrow (false,
               dealii::ExcMessage ("1D reference Newton budget exhausted."));
  return out;
}
}
#endif
