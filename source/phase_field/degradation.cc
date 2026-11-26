/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/phase_field/degradation.h>
#include <deal.II/base/exceptions.h>
#include <limits>

namespace aspect
{
  namespace PhaseField
  {
    CohesiveDegradation::CohesiveDegradation(const double phi, const double m)
      : excess_numerator(m *phi*(1.+phi)), intact_factor((1.-phi)*(1.-phi)), phase(phi), magnitude(m)
    {
      AssertThrow(phi>=0. && phi<=1. && m>0.,dealii::ExcMessage("Invalid phase or degradation magnitude."));
    }

    double CohesiveDegradation::cohesive_energy(const double strain_energy) const
    {
      // Maintain the product order of the discrete compliance-work ledger.
      return strain_energy*(1.-phase)*(1.-phase)/excess_numerator;
    }

    double CohesiveDegradation::fraction() const
    {
      return excess_numerator/(excess_numerator+intact_factor);
    }

    double CohesiveDegradation::fraction_derivative() const
    {
      const double denominator = excess_numerator + intact_factor;
      return (magnitude*(1.+2.*phase)*intact_factor+excess_numerator*2.*(1.-phase))/(denominator*denominator);
    }

    double CohesiveDegradation::driving_factor() const
    {
      const double denominator = excess_numerator + intact_factor;
      return magnitude*(1.+3.*phase)*(1.-phase)/(denominator*denominator);
    }

    double CohesiveDegradation::driving_factor_derivative() const
    {
      const double denominator = excess_numerator + intact_factor;
      return magnitude*(2.-6.*phase)/(denominator*denominator)
             - 2.*driving_factor()*(magnitude*(1.+2.*phase)-2.*(1.-phase))/denominator;
    }

    double excess_compliance(const double phase, const double magnitude)
    {
      const CohesiveDegradation law(phase,magnitude);
      if (phase == 1.)
        return std::numeric_limits<double>::infinity();
      return law.excess_numerator/law.intact_factor;
    }
  }
}
