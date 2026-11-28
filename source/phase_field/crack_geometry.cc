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
          PolynomialGeometry(std::string                 name,
                             const double                initial_slope,
                             const double                c0,
                             const std::optional<double> half_width_factor)
            : law_name(std::move(name)),
              initial_slope(initial_slope),
              c0(c0),
              half_width_factor(half_width_factor)
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

          const std::string &name() const override
          {
            return law_name;
          }

          std::optional<double> compact_profile_half_width_factor() const override
          {
            return half_width_factor;
          }

        private:
          const std::string law_name;
          const double initial_slope, c0;
          const std::optional<double> half_width_factor;
      };

      // The stationary normal profile obeys ell |dphi/dn| = sqrt(alpha(phi)).
      // AT2 has an exponential tail. The other two integrals end at finite n.
      const PolynomialGeometry laws[] =
      {
        {"AT1", 1., 8./3., 2.},
        {"AT2", 0., 2., std::nullopt},
        {"CZM", 2., dealii::numbers::PI, dealii::numbers::PI/2.}
      };
    }

    const CrackGeometry &CrackGeometry::select(const std::string &name)
    {
      for (const auto &law : laws)
        if (name == law.name())
          return law;
      AssertThrow(false, dealii::ExcMessage("Unknown crack geometry '" + name + "'."));
    }

    std::string CrackGeometry::selection_pattern()
    {
      std::string result;
      for (const auto &law : laws)
        {
          if (!result.empty())
            result += "|";
          result += law.name();
        }
      return result;
    }
  }
}
