/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_onset_score_h
#define _aspect_onset_score_h
#include <array>
#include <cmath>
#include <limits>

namespace aspect::GrowingFault
{
  /** Contributions of one native point to a prospective plane's direction.
   * For seed-frame traction u sin(2 beta)+v cos(2 beta), entries 0--2
   * resolve the full-section force. Entries 3--8 localize the direct score. */
  inline std::array<double,9> onset_moments(double area,double kernel,double fraction,double u,double v)
  {
    const double w=area*(1.-fraction)*(1.-fraction)*kernel;
    return {{area *fraction*u,area *fraction*v,area*fraction,w *u*u,w *v*v,w *u*v,w*u,w*v,w}};
  }

  struct OnsetScore
  {
    double score,resistance;
  };

  /** Evaluate the unnormalized area integral of squared traction in Pa² m².
   * The force comes from the complete candidate strip at its current arclength.
   * Centered second moments avoid cancellation in the cold creep limit. */
  template <typename Resistance>
  inline OnsetScore candidate_onset(const double *m,const double *centered,const Resistance &solve,
                                    double GBh_ds,double angle)
  {
    const double s=std::sin(2.*angle),c=std::cos(2.*angle);
    const double f=solve((m[0]*s+m[1]*c)/GBh_ds,m[2]/GBh_ds);
    const double expanded=m[3]*s*s+m[4]*c*c+2.*m[5]*s*c-2.*f*(m[6]*s+m[7]*c)+m[8]*f*f;
    if (expanded>std::sqrt(std::numeric_limits<double>::epsilon())*(m[3]+m[4]+m[8]*f*f))
      return {expanded,f};
    const double mean=(m[6]*s+m[7]*c)/m[8]-f;
    return {centered[0] *s *s+2.*centered[1] *s *c+centered[2] *c *c+m[8] *mean*mean,f};
  }
}
#endif
