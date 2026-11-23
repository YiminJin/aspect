/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/postprocess/phase_field_surface_energy.h>

#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string> PhaseFieldSurfaceEnergy<dim>::execute(TableHandler &statistics)
    {
      const auto &model = this->get_parameters().surface_energy_model;
      const auto &intro = this->introspection();
      // For a Q_p phase, alpha(phi) has coordinate degree at most 2p.
      const auto &quadrature = intro.quadratures.compositional_fields[phase_index];
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), quadrature,
                       update_values | update_gradients | update_JxW_values);
      const auto &phase = fe[intro.extractors.compositional_fields[phase_index]];
      std::vector<double> values(quadrature.size());
      std::vector<Tensor<1,dim>> gradients(quadrature.size());
      double alpha_integral = 0., gradient_integral = 0.;
      double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
      bool invalid_phase = false;
      constexpr double slack = 64 * std::numeric_limits<double>::epsilon();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            phase.get_function_values(this->get_solution(), values);
            phase.get_function_gradients(this->get_solution(), gradients);
            for (unsigned int q=0; q<quadrature.size(); ++q)
              {
                const double phi = values[q];
                if (!std::isfinite(phi))
                  {
                    invalid_phase = true;
                    continue;
                  }
                minimum = std::min(minimum, phi);
                maximum = std::max(maximum, phi);
                if (phi < -slack || phi > 1. + slack)
                  {
                    invalid_phase = true;
                    continue;
                  }
                alpha_integral += model.geometry.value(phi) * fe.JxW(q);
                gradient_integral += gradients[q].norm_square() * fe.JxW(q);
              }
          }

      // All ranks finish sampling before any rank can reject the functional.
      // Empty ranks contribute neutral extrema and zero to both integrals.
      const MPI_Comm comm = this->get_mpi_communicator();
      const auto global = Utilities::MPI::min_max_avg({alpha_integral, gradient_integral,
                                                       minimum, maximum, invalid_phase ? 1. : 0.
                                                      }, comm);
      if (global[4].sum > 0.)
        {
          std::ostringstream message;
          message << "Invalid sampled phase on " << global[4].sum << " of "
                  << Utilities::MPI::n_mpi_processes(comm) << " MPI ranks. ";
          if (global[2].min <= global[3].max)
            message << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << "Finite sampled minimum = " << global[2].min
                    << ", maximum = " << global[3].max << ". ";
          else
            message << "All sampled values are nonfinite. ";
          message << "Crack surface energy requires finite phase in [0,1] "
                  << "with a roundoff allowance of 64 machine epsilons.";
          AssertThrow(false, ExcMessage(message.str()));
        }
      const double prefactor = model.fracture_energy / model.geometry.normalization();
      const double local_energy = prefactor * global[0].sum / model.length_scale;
      const double gradient_energy = prefactor * global[1].sum * model.length_scale;
      const double total_energy = local_energy + gradient_energy;
      const std::string unit = (dim == 2 ? "J/m" : "J");
      const auto add_energy = [&](const std::string &name, const double energy)
      {
        const std::string column = "Crack surface energy " + name + " (" + unit + ")";
        statistics.add_value(column, energy);
        statistics.set_precision(column, 16);
        statistics.set_scientific(column, true);
      };
      add_energy("local", local_energy);
      add_energy("gradient", gradient_energy);
      add_energy("total", total_energy);
      return {"Crack surface energy", Utilities::to_string(total_energy) + " " + unit};
    }

    template <int dim>
    void PhaseFieldSurfaceEnergy<dim>::initialize()
    {
      const auto &model = this->get_parameters().surface_energy_model;
      model.validate();
      phase_index = this->introspection().compositional_index_for_name(model.composition);
    }

    ASPECT_REGISTER_POSTPROCESSOR(PhaseFieldSurfaceEnergy,
                                  "phase field surface energy",
                                  "Measure the local and gradient parts of the crack-surface functional "
                                  "$G_c/c_0 \\int_\\Omega (\\alpha(\\phi)/\\ell + \\ell |\\nabla\\phi|^2)\\,dx$ "
                                  "and their sum. The units are J/m of out-of-plane thickness in 2D and J in 3D. "
                                  "Sampled phase must be finite and in [0,1], allowing 64 double-precision machine "
                                  "epsilons for roundoff at either endpoint. Admitted values are used as sampled. "
                                  "For DG fields the gradient is elementwise (broken). The functional omits face jumps and fluxes.")
  }
}
