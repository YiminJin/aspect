/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_phase_field_crack_geometry_h
#define _aspect_phase_field_crack_geometry_h

#include <string>

namespace aspect
{
  namespace PhaseField
  {
    /**
     * Dimensionless crack-geometric law on the damage interval [0,1].
     * Its value enters the surface energy, its first derivative the phase
     * residual, and its second derivative the corresponding linearization.
     */
    class CrackGeometry
    {
      public:
        virtual ~CrackGeometry() = default;

        /** Select an immutable law by name. The reference has static lifetime. */
        static const CrackGeometry &select(const std::string &name);

        /** Names accepted by select(), separated by | for parameter declarations. */
        static std::string selection_pattern();

        /** The crack-geometric function alpha(phi). */
        virtual double value(double phase) const = 0;

        /** First derivative of alpha with respect to phase. */
        virtual double first_derivative(double phase) const = 0;

        /** Second derivative of alpha with respect to phase. */
        virtual double second_derivative(double phase) const = 0;

        /** Surface normalization c_0 = 4 integral_0^1 sqrt(alpha(u)) du. */
        virtual double normalization() const = 0;
    };
  }
}
#endif
