/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/phase_field/crack_geometry.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <utility>

namespace aspect
{
  namespace PhaseField
  {
    namespace
    {
      // Each named law satisfies alpha(0)=0 and alpha(1)=1. Its slope at
      // zero fixes both coefficients of the donor's quadratic polynomial.
      class PolynomialGeometry final : public CrackGeometry
      {
        public:
          PolynomialGeometry(const double initial_slope, const double c0)
            : initial_slope(initial_slope), c0(c0)
          {}

          double value(const double phase) const override
          {
            return phase * (initial_slope + (1.-initial_slope)*phase);
          }

          double first_derivative(const double phase) const override
          {
            return initial_slope + 2.*(1.-initial_slope)*phase;
          }

          double second_derivative(const double) const override
          {
            return 2.*(1.-initial_slope);
          }

          double normalization() const override
          {
            return c0;
          }

        private:
          const double initial_slope, c0;
      };

      const std::pair<const char *,PolynomialGeometry> laws[] =
      {
        {"AT1", {1., 8./3.}},
        {"AT2", {0., 2.}},
        {"CZM", {2., dealii::numbers::PI}}
      };
    }

    const CrackGeometry &CrackGeometry::select(const std::string &name)
    {
      for (const auto &law : laws)
        if (name == law.first)
          return law.second;
      AssertThrow(false, dealii::ExcMessage("Unknown crack geometry '" + name + "'."));
    }

    std::string CrackGeometry::selection_pattern()
    {
      std::string result;
      for (const auto &law : laws)
        {
          if (!result.empty())
            result += "|";
          result += law.first;
        }
      return result;
    }
  }
}
