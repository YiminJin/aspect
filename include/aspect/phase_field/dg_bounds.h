/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_DG_BOUNDS_H
#define ASPECT_PHASE_FIELD_DG_BOUNDS_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace PhaseField
  {
    /** Change of coordinates for ordinary tensor Q2, not a different FE space.
     * Bernstein coefficients bound the polynomial over the complete cell.
     * This helper owns no physical state, DoFs, or acceptance decisions. */
    template <int dim>
    class Q2Bounds
    {
      public:
        explicit Q2Bounds(const dealii::FiniteElement<dim> &element)
          : expansion(element.n_dofs_per_cell(), element.n_dofs_per_cell()),
            inverse(element.n_dofs_per_cell(), element.n_dofs_per_cell())
        {
          AssertThrow(element.degree == 2 && element.n_components() == 1
                      && element.n_dofs_per_cell() == dealii::Utilities::fixed_power<dim>(3u),
                      dealii::ExcMessage("Phase bounds require scalar tensor Q2."));
          const auto &points = element.get_unit_support_points();
          for (unsigned int i = 0; i < points.size(); ++i)
            for (unsigned int j = 0; j < points.size(); ++j)
              {
                unsigned int index = j;
                double value = 1.;
                for (unsigned int d = 0; d < dim; ++d, index /= 3)
                  {
                    const double x = points[i][d];
                    value *= index % 3 == 0 ? (1 - x) * (1 - x) :
                             index % 3 == 1 ? 2 * x * (1 - x) : x * x;
                  }
                expansion(i, j) = value;
              }
          inverse.invert(expansion);
        }

        dealii::Vector<double> coefficients(const dealii::Vector<double> &nodal) const
        {
          dealii::Vector<double> result(nodal.size());
          inverse.vmult(result, nodal);
          return result;
        }

        /** Return a certified enclosure, tightened by de Casteljau subdivision
         * only where the coefficient hull crosses the requested physical range.
         * Subdivision changes neither the polynomial nor its integral. */
        std::pair<double, double> enclosure(const dealii::Vector<double> &nodal,
                                            const double lower, const double upper,
                                            const unsigned int depth = 4) const
        {
          return enclose(coefficients(nodal), lower, upper, depth);
        }

        /** Mean-preserving scaling. An inadmissible mean is an update failure,
         * never clipped. The tolerance covers floating-point evaluation only;
         * it is not a replacement upper cap for singular constitutive laws. */
        double limit(dealii::Vector<double> &nodal, const double mean,
                     const double lower, const double upper) const
        {
          const double tolerance = 128 * std::numeric_limits<double>::epsilon()
                                   * std::max({1., std::abs(lower), std::abs(upper)});
          AssertThrow(std::isfinite(mean) && mean >= lower - tolerance && mean <= upper + tolerance,
                      dealii::ExcMessage("DG phase has an inadmissible cell mean; reduce the transport substep."));
          const auto range = enclosure(nodal, lower, upper);
          double alpha = 1.;
          if (range.first < lower && range.first < mean)
            alpha = std::min(alpha, std::max(0., (mean - lower) / (mean - range.first)));
          if (range.second > upper && range.second > mean)
            alpha = std::min(alpha, std::max(0., (upper - mean) / (range.second - mean)));
          if (alpha < 1.)
            for (auto &value : nodal) value = mean + alpha * (value - mean);
          return alpha;
        }

        dealii::FullMatrix<double> expansion; ///< Nodal values = E Bernstein coefficients.
        dealii::FullMatrix<double> inverse;   ///< Bernstein coefficients = E^-1 nodal values.

      private:
        static std::pair<double, double> enclose(const dealii::Vector<double> &values,
                                                 const double lower, const double upper,
                                                 const unsigned int depth)
        {
          const auto extrema = std::minmax_element(values.begin(), values.end());
          std::pair<double, double> result {*extrema.first, *extrema.second};
          if (depth == 0 || (result.first >= lower && result.second <= upper)) return result;
          std::vector<dealii::Vector<double>> pieces {values};
          unsigned int stride = 1;
          for (unsigned int d = 0; d < dim; ++d, stride *= 3)
            {
              std::vector<dealii::Vector<double>> next;
              for (const auto &piece : pieces)
                {
                  auto left = piece, right = piece;
                  for (unsigned int i = 0; i < piece.size(); ++i)
                    if ((i / stride) % 3 == 0)
                      {
                        const double a = piece[i], b = piece[i + stride], c = piece[i + 2 * stride];
                        const double mid = .25 * a + .5 * b + .25 * c;
                        left[i + stride] = .5 * (a + b);
                        left[i + 2 * stride] = mid;
                        right[i] = mid;
                        right[i + stride] = .5 * (b + c);
                      }
                  next.push_back(std::move(left));
                  next.push_back(std::move(right));
                }
              pieces = std::move(next);
            }
          result = {std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
          for (const auto &piece : pieces)
            {
              const auto child = enclose(piece, lower, upper, depth - 1);
              result.first = std::min(result.first, child.first);
              result.second = std::max(result.second, child.second);
            }
          return result;
        }
    };
  }
}
#endif
