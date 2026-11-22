/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/phase_field/mature_degradation.h>
#include <deal.II/base/exceptions.h>
#include <limits>

namespace aspect
{
  namespace PhaseField
  {
    double excess_compliance(const double phase, const double magnitude)
    {
      AssertThrow(phase >= 0. && phase <= 1. && magnitude > 0.,
                  dealii::ExcMessage("Invalid phase or mature degradation magnitude."));
      if (phase == 1.) return std::numeric_limits<double>::infinity();
      return magnitude*phase*(1.+phase)/((1.-phase)*(1.-phase));
    }
  }
}
