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


#include <aspect/simulator/solver/reconstructed_fault_bound_diagnostics.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace internal
  {
    void open_fault_bound_diagnostic(std::ofstream &out,
                                     const std::string &output_directory,
                                     const unsigned int timestep_number,
                                     const unsigned int nonlinear_iteration)
    {
      out.open(output_directory + "nonlinear_bounds_"
               + dealii::Utilities::int_to_string(timestep_number) + ".csv",
               nonlinear_iteration == 0 ? std::ios::out : std::ios::app);
      if (nonlinear_iteration == 0)
        out << "iteration,fault,vertex,V,dV,prescribed,lower_active,Fmin_weak_density,alpha_max,bulk,surface\n";
      out << std::setprecision(17);
    }


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
                                      const double current_surface_norm)
    {
      out << nonlinear_iteration << ',' << fault << ',' << vertex << ',' << slip_rate
          << ',' << slip_rate_direction << ',' << prescribed
          << ',' << lower_active << ',' << density
          << ',' << maximum_step_length << ',' << current_bulk_norm << ','
          << current_surface_norm << '\n';
    }


    void report_fault_bound_diagnostic(const dealii::ConditionalOStream &pcout,
                                       const unsigned int free,
                                       const unsigned int lower_active,
                                       const double minimum,
                                       const double minimum_free,
                                       const unsigned int prefers_lower,
                                       const double maximum_step_length)
    {
      pcout << "      Fault bound audit: free=" << free << ", lower-active=" << lower_active
            << ", min V=" << minimum << ", min free V=" << minimum_free
            << ", negative Fmin=" << prefers_lower << ", alpha_max=" << maximum_step_length
            << std::endl;
    }
  }
}
