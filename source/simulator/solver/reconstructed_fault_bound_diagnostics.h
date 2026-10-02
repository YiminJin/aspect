/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#ifndef ASPECT_RECONSTRUCTED_FAULT_BOUND_DIAGNOSTICS_H
#define ASPECT_RECONSTRUCTED_FAULT_BOUND_DIAGNOSTICS_H

#include <iosfwd>
#include <string>

namespace dealii
{
  class ConditionalOStream;
}

namespace aspect
{
  namespace internal
  {
    /** Open the caller-owned per-iteration stream, preserving silent I/O failure. */
    void open_fault_bound_diagnostic(std::ofstream &out,
                                     const std::string &output_directory,
                                     const unsigned int timestep_number,
                                     const unsigned int nonlinear_iteration);

    /**
     * Record current Newton values and the separately computed lower-rate probe
     * density. These are not an accepted trial or committed timestep. No inputs
     * are retained; the caller owns admission, evaluation and stream lifetime.
     */
    void write_fault_bound_diagnostic(std::ostream &out,
                                      const unsigned int nonlinear_iteration,
                                      const unsigned int fault,
                                      const unsigned int vertex,
                                      const double slip_rate,
                                      const double slip_rate_direction,
                                      const bool prescribed,
                                      const bool lower_active,
                                      const double density,
                                      const double maximum_step_length,
                                      const double current_bulk_norm,
                                      const double current_surface_norm);

    /** Write to the existing shared stream without changing its formatting state. */
    void report_fault_bound_diagnostic(const dealii::ConditionalOStream &pcout,
                                       const unsigned int free,
                                       const unsigned int lower_active,
                                       const double minimum,
                                       const double minimum_free,
                                       const unsigned int prefers_lower,
                                       const double maximum_step_length);
  }
}

#endif
