/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_onset_score_h
#define _aspect_onset_score_h
#include <deal.II/base/numbers.h>
#include <array>
#include <cmath>
#include <limits>

namespace aspect::GrowingFault
{
  /** Radius (m) of the direction score's compact kernel about the supported
   * mechanical tip. The argument is the phase-field length scale in metres. */
  inline double direction_search_radius(const double length_scale)
  {
    return 2.*length_scale;
  }

  /** Largest magnitude of a prospective plane's angle to the seed tangent,
   * in radians. Together these two directions bound the forward growth sector. */
  inline constexpr double maximum_turn_angle = dealii::numbers::PI / 4.;

  /** Contributions of one native point to a prospective plane's direction.
   * Resolve the seed-frame shear predictor (Pa) as u sin(2 beta) + v cos(2 beta).
   * Entries 0--2 integrate (u, v, 1) against area*fraction on the full strip
   * (Pa m², Pa m², m²). Entries 3--8 integrate (u², v², uv, u, v, 1)
   * on the forward disk with weight area*(1-fraction)²*kernel.
   * Sum the point contributions across ranks before selecting a plane. */
  inline std::array<double, 9> onset_moments(double area,
                                             double kernel,
                                             double fraction,
                                             double u,
                                             double v)
  {
    const double w = area * (1. - fraction) * (1. - fraction) * kernel;
    return {{
        area *fraction*u, area *fraction*v, area*fraction,
        w *u*u, w *v*v, w *u*v, w*u, w*v, w
      }
    };
  }

  struct OnsetScore
  {
    double score, resistance;
  };

  /** Evaluate the unnormalized area integral of squared traction in Pa² m².
   * The force comes from the complete candidate strip at its current arclength.
   * GBh_ds is shear modulus times relaxation, half-duration and strip arclength
   * (Pa s m). The solve returns the signed rate (m/s) and resistance (Pa).
   * If the expanded score cancels in cold creep, use the three globally
   * centered disk moments of (u², uv, v²), in Pa² m². */
  template <typename Resistance>
  inline OnsetScore candidate_onset(const double     *m,
                                    const double     *centered,
                                    const Resistance &solve,
                                    double           GBh_ds,
                                    double           angle)
  {
    const double s = std::sin(2.*angle);
    const double c = std::cos(2.*angle);
    const double mobility = m[2] / GBh_ds;
    const auto root = solve((m[0] * s + m[1] * c) / GBh_ds, mobility);
    const double f = root.resistance;
    const double expanded = m[3]*s*s + m[4]*c*c + 2.*m[5]*s*c
                            - 2.*f*(m[6]*s + m[7]*c) + m[8]*f*f;
    if (expanded > std::sqrt(std::numeric_limits<double>::epsilon()) * (m[3] + m[4] + m[8]*f * f))
      return {expanded, f};
    // The signed section solve gives V + M f = Vfree = M * mean_strip.
    // In cold creep, mean_strip and f can round to the same traction (Pa).
    // Compute mean_disk - f as (mean_disk - mean_strip) + V/M, taking
    // the difference of the two weighted means in the seed frame first.
    double mean_excess_traction;
    if (mobility > 0.)
      {
        const double delta_u = m[6] / m[8] - m[0] / m[2];
        const double delta_v = m[7] / m[8] - m[1] / m[2];
        mean_excess_traction = delta_u * s + delta_v * c + root.rate / mobility;
      }
    else
      // Without mobile area the full-strip mean is undefined.
      mean_excess_traction = (m[6] * s + m[7] * c) / m[8] - f;
    return {centered[0] *s *s + 2.*centered[1] *s *c + centered[2] *c *c
            + m[8] *mean_excess_traction * mean_excess_traction, f
           };
  }
}
#endif
