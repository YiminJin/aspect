/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_phase_field_bernstein_bounds_h
#define _aspect_phase_field_bernstein_bounds_h

#include <deal.II/base/exceptions.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace aspect
{
  namespace PhaseField
  {
    using BernsteinPatch = std::array<double,9>;

    /** Enclose a tensor Q2 polynomial in its Bernstein coordinates. Tighten a
     * hull crossing [0,1] by de Casteljau subdivision in both directions. */
    inline std::pair<double,double> enclosure(const BernsteinPatch &p, const unsigned int depth = 4)
    {
      const auto extrema = std::minmax_element(p.begin(),p.end());
      if (depth==0 || (*extrema.first >= 0. && *extrema.second <= 1.))
        return {*extrema.first,*extrema.second};
      const auto split = [](const BernsteinPatch &patch, const unsigned int stride)
      {
        std::array<BernsteinPatch,2> halves = {{patch,patch}};
        for (unsigned int i=0; i<9; ++i)
          if ((i/stride)%3==0)
            {
              const double a=patch[i], b=patch[i+stride], c=patch[i+2*stride];
              halves[0][i+stride]=.5*(a+b);
              halves[1][i+stride]=.5*(b+c);
              halves[0][i+2*stride]=halves[1][i]=.25*a+.5*b+.25*c;
            }
        return halves;
      };
      std::pair<double,double> range = {std::numeric_limits<double>::max(),std::numeric_limits<double>::lowest()};
      for (const auto &half : split(p,1))
        for (const auto &quarter : split(half,3))
          {
            const auto child = enclosure(quarter,depth-1);
            range.first=std::min(range.first,child.first);
            range.second=std::max(range.second,child.second);
          }
      return range;
    }

    /** Preserve the mapped mean, scaling the polynomial only when its
     * enclosure exceeds the physical interval. Return false for an invalid
     * mean so the caller can report the failure collectively. */
    inline bool limit_bernstein(std::vector<double> &coefficients, const double mean)
    {
      const double tolerance = 128.*std::numeric_limits<double>::epsilon();
      if (!std::isfinite(mean) || mean < -tolerance || mean > 1.+tolerance)
        return false;
      BernsteinPatch patch;
      AssertThrow(coefficients.size()==patch.size(),dealii::ExcMessage("Expected a tensor Q2 Bernstein patch."));
      std::copy(coefficients.begin(),coefficients.end(),patch.begin());
      const auto range=enclosure(patch);
      double scale=1.;
      if (range.first<0. && range.first<mean)
        scale=std::min(scale,std::max(0.,mean/(mean-range.first)));
      if (range.second>1. && range.second>mean)
        scale=std::min(scale,std::max(0.,(1.-mean)/(range.second-mean)));
      if (scale<1.)
        for (auto &v : coefficients)
          v=mean+scale*(v-mean);
      return true;
    }
  }
}
#endif
