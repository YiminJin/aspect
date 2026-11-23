/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/postprocess/phase_field_surface_energy.h>

#include <deal.II/fe/fe_values.h>

#include <cmath>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string> PhaseFieldSurfaceEnergy<dim>::execute(TableHandler &statistics)
    {
      const auto &intro = this->introspection();
      // For a Q_p phase, alpha(phi) has coordinate degree at most 2p.
      const auto &quadrature = intro.quadratures.compositional_fields[phase_index];
      FEValues<dim> fe(this->get_mapping(), this->get_fe(), quadrature,
                       update_values | update_gradients | update_JxW_values);
      const auto &phase = fe[intro.extractors.compositional_fields[phase_index]];
      std::vector<double> values(quadrature.size());
      std::vector<Tensor<1,dim>> gradients(quadrature.size());
      double local[2] = {}, global[2];
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            phase.get_function_values(this->get_solution(), values);
            phase.get_function_gradients(this->get_solution(), gradients);
            for (unsigned int q=0; q<quadrature.size(); ++q)
              {
                local[0] += geometry->value(values[q]) * fe.JxW(q);
                local[1] += gradients[q].norm_square() * fe.JxW(q);
              }
          }

      Utilities::MPI::sum(local, this->get_mpi_communicator(), global);
      const double prefactor = fracture_energy / geometry->normalization();
      const double local_energy = prefactor * global[0] / length_scale;
      const double gradient_energy = prefactor * global[1] * length_scale;
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
    void PhaseFieldSurfaceEnergy<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Phase field surface energy");
      prm.declare_entry("Crack geometry", "AT1", Patterns::Selection(PhaseField::CrackGeometry::selection_pattern()),
                        "Crack-geometric law for the measured surface functional.");
      prm.declare_entry("Phase composition", "phase_field", Patterns::Anything(),
                        "Name of the compositional field that contains the dimensionless phase.");
      prm.declare_entry("Fracture energy", "1", Patterns::Double(),
                        "Positive fracture energy per unit crack area Gc in J/m^2.");
      prm.declare_entry("Length scale", "1", Patterns::Double(),
                        "Positive regularization length in metres.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    template <int dim>
    void PhaseFieldSurfaceEnergy<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Phase field surface energy");
      geometry = &PhaseField::CrackGeometry::select(prm.get("Crack geometry"));
      phase_index = this->introspection().compositional_index_for_name(prm.get("Phase composition"));
      fracture_energy = prm.get_double("Fracture energy");
      length_scale = prm.get_double("Length scale");
      AssertThrow(std::isfinite(fracture_energy) && fracture_energy > 0.,
                  ExcMessage("Phase field surface energy requires a finite, positive fracture energy in J/m^2."));
      AssertThrow(std::isfinite(length_scale) && length_scale > 0.,
                  ExcMessage("Phase field surface energy requires a finite, positive length scale in metres."));
      prm.leave_subsection();
      prm.leave_subsection();
    }

    ASPECT_REGISTER_POSTPROCESSOR(PhaseFieldSurfaceEnergy,
                                  "phase field surface energy",
                                  "Measure the local and gradient parts of the crack-surface functional "
                                  "$G_c/c_0 \\int_\\Omega (\\alpha(\\phi)/\\ell + \\ell |\\nabla\\phi|^2)\\,dx$ "
                                  "and their sum. The units are J/m of out-of-plane thickness in 2D and J in 3D. "
                                  "For DG fields the gradient is elementwise (broken). The functional omits face jumps and fluxes.")
  }
}
