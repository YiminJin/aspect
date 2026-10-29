/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_geometry.h>
#include <deal.II/base/quadrature_lib.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace aspect
{
  namespace CohesiveInterface
  {
    Geometry::Geometry (std::vector<Piece> pieces, const double width)
        : curve (std::move (pieces)), support (width)
    {
      rebuild ();
    }

    void
    Geometry::rebuild ()
    {
      AssertThrow (!curve.empty () && support > 0.,
                   ExcMessage ("Invalid exact-jump curve/support."));
      nodes.clear ();
      for (const auto &p : curve)
        {
          AssertThrow (p.end > p.x && p.cells > 0, ExcMessage ("Invalid exact-jump graph piece."));
          if (nodes.empty ())
            nodes.push_back (p.x);
          else
            AssertThrow (std::abs (nodes.back () - p.x) < 1e-11,
                         ExcMessage ("Disconnected exact-jump graph."));
          for (unsigned int j = 1; j <= p.cells; ++j)
            nodes.push_back (p.x + (p.end - p.x) * j / p.cells);
        }
    }

    Geometry::Sample
    Geometry::sample (const double x) const
    {
      const auto it = std::find_if (curve.begin (), curve.end (),
                                    [&] (const Piece &p) { return x <= p.end + 1e-12; });
      AssertThrow (it != curve.end () && x >= curve.front ().x - 1e-12,
                   ExcMessage ("Outside exact-jump graph."));
      const auto &p = *it;
      const double h = p.end - p.x, z = std::clamp ((x - p.x) / h, 0., 1.);
      Sample out;
      if (p.hermite)
        {
          out.y = p.y
                  + h
                        * (p.a * z + .5 * p.c1 * z * z + p.c2 * z * z * z / 3
                           + p.c3 * z * z * z * z / 4);
          out.slope = p.a + p.c1 * z + p.c2 * z * z + p.c3 * z * z * z;
          out.second = (p.c1 + 2 * p.c2 * z + 3 * p.c3 * z * z) / h;
        }
      else
        {
          out.y = p.y + p.a * h * z + (p.b - p.a) * h * (z * z * z - .5 * z * z * z * z);
          out.slope = p.a + (p.b - p.a) * (3 * z * z - 2 * z * z * z);
          out.second = (p.b - p.a) * (6 * z - 6 * z * z) / h;
        }
      out.tangent[0] = 1.;
      out.tangent[1] = out.slope;
      out.tangent /= out.tangent.norm ();
      out.tangent_derivative[0] = -out.slope * out.second;
      out.tangent_derivative[1] = out.second;
      out.tangent_derivative /= std::pow (1 + out.slope * out.slope, 1.5);
      return out;
    }

    std::vector<Geometry::Mode>
    Geometry::lifting (const dealii::Point<2> &p, const int side) const
    {
      if (p[0] < nodes.front () || p[0] > nodes.back ())
        return {};
      const auto g = sample (p[0]);
      const double d = p[1] - g.y;
      if (std::abs (d) >= support)
        return {};
      const double sign = std::abs (d) < 1e-10 && side ? side : (d >= 0 ? 1. : -1.);
      const double rho = .5 * sign * (1 - std::abs (d) / support);
      const auto next = std::upper_bound (nodes.begin (), nodes.end (), p[0]);
      const unsigned int j = std::min<unsigned int> (nodes.size () - 2, next - nodes.begin () - 1);
      const double h = nodes[j + 1] - nodes[j], z = (p[0] - nodes[j]) / h;
      std::vector<Mode> result;
      for (unsigned int v = 0; v < 2; ++v)
        {
          const unsigned int index = j + v;
          if (index == 0 || index + 1 == nodes.size ())
            continue;
          const double b = v ? z : 1 - z, db = (v ? 1. : -1.) / h;
          Mode m;
          m.index = index - 1;
          m.value = b * rho * g.tangent;
          for (unsigned int a = 0; a < 2; ++a)
            {
              m.gradient[a][0] = (db * rho + b * g.slope / (2 * support)) * g.tangent[a]
                                 + b * rho * g.tangent_derivative[a];
              m.gradient[a][1] = -b * g.tangent[a] / (2 * support);
            }
          result.push_back (m);
        }
      return result;
    }

    bool
    Geometry::normal_active (const dealii::Point<2> &p) const
    {
      return p[0] > nodes.front () && p[0] < nodes.back ()
             && std::abs (p[1] - sample (p[0]).y) < support;
    }

    Geometry::Mode
    Geometry::normal_test (const dealii::Point<2> &p) const
    {
      const auto next = std::upper_bound (nodes.begin (), nodes.end (), p[0]);
      const unsigned int j = std::min<unsigned int> (nodes.size () - 2, next - nodes.begin () - 1);
      const auto g = sample (p[0]);
      const double h = nodes[j + 1] - nodes[j], z = (p[0] - nodes[j]) / h, d = p[1] - g.y;
      const double rho = .5 * (d >= 0 ? 1. : -1.) * (1 - std::abs (d) / support),
                   b = 4 * z * (1 - z), db = 4 * (1 - 2 * z) / h;
      Tensor<1, 2> n, dn;
      n[0] = -g.tangent[1];
      n[1] = g.tangent[0];
      dn[0] = -g.tangent_derivative[1];
      dn[1] = g.tangent_derivative[0];
      Mode m;
      m.index = j;
      m.value = b * rho * n;
      for (unsigned int a = 0; a < 2; ++a)
        {
          m.gradient[a][0] = (db * rho + b * g.slope / (2 * support)) * n[a] + b * rho * dn[a];
          m.gradient[a][1] = -b * n[a] / (2 * support);
        }
      return m;
    }

    std::vector<double>
    Geometry::cuts (const double l, const double r, const double bottom, const double top) const
    {
      std::vector<double> result = { l, r };
      const auto add = [&] (const double x)
        {
          if (x > l + 1e-11 && x < r - 1e-11)
            result.push_back (x);
        };
      for (const double x : nodes)
        add (x);
      for (const auto &piece : curve)
        {
          const double a = std::max (l, piece.x), b = std::min (r, piece.end);
          if (b <= a + 1e-12)
            continue;
          std::vector<double> monotone = { a, b };
          if (sample (a).slope * sample (b).slope < 0)
            {
              double u = a, v = b;
              for (unsigned int k = 0; k < 60; ++k)
                {
                  const double m = .5 * (u + v);
                  if (sample (u).slope * sample (m).slope <= 0)
                    v = m;
                  else
                    u = m;
                }
              monotone.insert (monotone.begin () + 1, .5 * (u + v));
            }
          for (const double boundary : { bottom, top })
            for (const int sign : { -1, 0, 1 })
              for (unsigned int j = 1; j < monotone.size (); ++j)
                {
                  double u = monotone[j - 1], v = monotone[j];
                  const double level = boundary + sign * support;
                  double fu = sample (u).y - level, fv = sample (v).y - level;
                  if (fu * fv < 0)
                    {
                      for (unsigned int k = 0; k < 60; ++k)
                        {
                          const double m = .5 * (u + v), fm = sample (m).y - level;
                          if (fu * fm <= 0)
                            {
                              v = m;
                              fv = fm;
                            }
                          else
                            {
                              u = m;
                              fu = fm;
                            }
                        }
                      add (.5 * (u + v));
                    }
                }
        }
      std::sort (result.begin (), result.end ());
      result.erase (std::unique (result.begin (), result.end (),
                                 [] (double a, double b) { return std::abs (a - b) < 1e-11; }),
                    result.end ());
      return result;
    }

    Quadrature<2>
    Geometry::quadrature (const dealii::Point<2> &lo, const dealii::Point<2> &hi,
                          const unsigned int order) const
    {
      AssertThrow (hi[0] > lo[0] && hi[1] > lo[1],
                   ExcMessage ("Cut quadrature requires affine Cartesian cells."));
      const auto xs = cuts (lo[0], hi[0], lo[1], hi[1]);
      const QGauss<1> q (order);
      std::vector<dealii::Point<2>> points;
      std::vector<double> weights;
      for (unsigned int i = 1; i < xs.size (); ++i)
        for (unsigned int u = 0; u < order; ++u)
          {
            const double x = xs[i - 1] + (xs[i] - xs[i - 1]) * q.point (u)[0];
            std::vector<double> ys = { lo[1], hi[1] };
            if (x >= nodes.front () && x <= nodes.back ())
              for (const int sign : { -1, 0, 1 })
                {
                  const double y = sample (x).y + sign * support;
                  if (y > lo[1] + 1e-11 && y < hi[1] - 1e-11)
                    ys.push_back (y);
                }
            std::sort (ys.begin (), ys.end ());
            for (unsigned int j = 1; j < ys.size (); ++j)
              for (unsigned int v = 0; v < order; ++v)
                {
                  const double y = ys[j - 1] + (ys[j] - ys[j - 1]) * q.point (v)[0];
                  points.emplace_back ((x - lo[0]) / (hi[0] - lo[0]),
                                       (y - lo[1]) / (hi[1] - lo[1]));
                  weights.push_back ((xs[i] - xs[i - 1]) * (ys[j] - ys[j - 1]) * q.weight (u)
                                     * q.weight (v) / ((hi[0] - lo[0]) * (hi[1] - lo[1])));
                }
          }
      return { points, weights };
    }

    std::vector<Geometry::Point>
    Geometry::surface_points () const
    {
      const QGauss<1> q (3);
      std::vector<Point> result;
      for (unsigned int j = 0; j + 1 < nodes.size (); ++j)
        for (unsigned int k = 0; k < 3; ++k)
          {
            const double xi = q.point (k)[0], h = nodes[j + 1] - nodes[j], x = nodes[j] + h * xi;
            const auto g = sample (x);
            result.push_back ({ dealii::Point<2> (x, g.y), g.tangent,
                                h * q.weight (k) * std::sqrt (1 + g.slope * g.slope), xi, j, j,
                                j + 1 });
          }
      return result;
    }

    double
    Geometry::preferred_direction (const SymmetricTensor<2, 3> &T, const double previous)
    {
      AssertThrow (std::hypot (.5 * (T[0][0] - T[1][1]), T[0][1]) > 1e-8,
                   ExcMessage ("No resolved shear-driving direction."));
      const double pi = numbers::PI;
      const auto wrap = [&] (double angle)
        {
          while (angle < -pi / 2)
            angle += pi;
          while (angle >= pi / 2)
            angle -= pi;
          return angle;
        };
      const double compression = wrap (.5 * std::atan2 (2 * T[0][1], T[0][0] - T[1][1]) + pi / 2);
      const double a = wrap (compression + pi / 4), b = wrap (compression - pi / 4);
      const double da = std::abs (wrap (a - previous)), db = std::abs (wrap (b - previous));
      AssertThrow (std::abs (da - db) > 1e-8,
                   ExcMessage ("Conjugate path preference is unresolved."));
      return previous + wrap ((da < db ? a : b) - previous);
    }

    std::string
    Geometry::mode_signature (const unsigned int index) const
    {
      AssertIndexRange (index, modes ());
      std::ostringstream out;
      out << std::setprecision (17) << support << ';';
      for (unsigned int j = 0; j < 3; ++j)
        out << nodes[index + j] << ';';
      for (const auto &p : curve)
        if (p.end > nodes[index] && p.x < nodes[index + 2])
          out << p.x << ';' << p.y << ';' << p.end << ';' << p.a << ';' << p.b << ';' << p.c1 << ';'
              << p.c2 << ';' << p.c3 << ';' << p.hermite << ';';
      return out.str ();
    }

    void
    Geometry::append (const double target, const double length, const double R,
                      const unsigned int cells)
    {
      AssertThrow (length > 0 && R > 0 && cells > 0, ExcMessage ("Invalid causal path extension."));
      const auto end = sample (nodes.back ());
      const double theta0 = std::atan (end.slope),
                   theta1 = theta0 + (target - theta0) * (-std::expm1 (-length / R));
      AssertThrow (std::abs (theta1) < 80 * numbers::PI / 180,
                   ExcMessage ("Selected path exceeds graph capability."));
      Piece p;
      p.x = nodes.back ();
      p.y = end.y;
      p.a = end.slope;
      p.b = std::tan (theta1);
      p.cells = cells;
      p.hermite = true;
      const double d0 = (target - theta0) / (R * std::pow (std::cos (theta0), 3));
      const double d1 = (target - theta1) / (R * std::pow (std::cos (theta1), 3));
      const auto coefficients = [&] (const double dx)
        {
          p.c1 = dx * d0;
          p.c2 = 3 * (p.b - p.a) - 2 * dx * d0 - dx * d1;
          p.c3 = -2 * (p.b - p.a) + dx * d0 + dx * d1;
        };
      const QGauss<1> q (20);
      double lo = .05 * length, hi = length;
      for (unsigned int k = 0; k < 60; ++k)
        {
          const double dx = .5 * (lo + hi);
          coefficients (dx);
          double actual = 0.;
          for (unsigned int j = 0; j < q.size (); ++j)
            {
              const double z = q.point (j)[0],
                           slope = p.a + p.c1 * z + p.c2 * z * z + p.c3 * z * z * z;
              actual += dx * q.weight (j) * std::sqrt (1 + slope * slope);
            }
          if (actual > length)
            hi = dx;
          else
            lo = dx;
        }
      p.end = p.x + .5 * (lo + hi);
      coefficients (p.end - p.x);
      curve.push_back (p);
      rebuild ();
    }
  }
}
