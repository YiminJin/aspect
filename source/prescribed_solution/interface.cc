/*
  Copyright (C) 2011 - 2026 by the authors of the ASPECT code.

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

#include <aspect/prescribed_solution/interface.h>
#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace PrescribedSolution
  {

    /**
    * A set of helper functions that either return the point passed to it (if
    * the current dimension is the same) or return a dummy value (otherwise).
    */
    namespace
    {
      std::tuple
      <aspect::internal::Plugins::UnusablePluginList,
      aspect::internal::Plugins::UnusablePluginList,
      aspect::internal::Plugins::PluginList<Interface<2>>,
      aspect::internal::Plugins::PluginList<Interface<3>>> registered_plugins;
    }


    template <int dim>
    void
    Manager<dim>::register_prescribed_solution_plugin (const std::string &name,
                                                       const std::string &description,
                                                       void (*declare_parameters_function) (ParameterHandler &),
                                                       std::unique_ptr<Interface<dim>> (*factory_function) ())
    {
      std::get<dim>(registered_plugins).register_plugin (name,
                                                         description,
                                                         declare_parameters_function,
                                                         factory_function);
    }



    template <int dim>
    void
    Manager<dim>::constrain_solution (AffineConstraints<double> &current_constraints) const
    {
      // If there are no active plugins, return without doing anything
      if (this->plugin_objects.size() == 0)
        return;

      this->get_computing_timer().enter_subsection("Prescribe solution");

      const auto &finite_element = this->get_fe();
      const unsigned int n_dofs = finite_element.n_dofs_per_cell();
      auto points = aspect::Utilities::get_unit_support_points(*this);
      std::vector<bool> pointwise_dofs(n_dofs, true);
      bool has_non_nodal_prescription_points = false;
      if (!finite_element.has_support_points())
        {
          // A mixed element can still prescribe its nodal components. For a
          // non-nodal field ask the plugin at its interpolation points solely
          // to detect unsupported prescriptions. They are not coefficient values.
          const bool has_default_points = points.size() == n_dofs;
          points.resize(n_dofs);
          for (unsigned int i=0; i<n_dofs; ++i)
            {
              const auto index = finite_element.system_to_base_index(i);
              const auto &base = finite_element.base_element(index.first.first);
              if (base.has_support_points())
                points[i] = base.unit_support_point(index.second);
              else if (base.has_generalized_support_points())
                {
                  AssertThrow(base.get_generalized_support_points().size() == base.n_dofs_per_cell(),
                              ExcMessage("Prescribed solution requires one evaluation point per local DoF for " + base.get_name()));
                  points[i] = base.get_generalized_support_points()[index.second];
                  pointwise_dofs[i] = false;
                  has_non_nodal_prescription_points = true;
                }
              else
                // Keep the release evaluation point for locally conservative pressure.
                AssertThrow(has_default_points, ExcMessage("Missing prescription points for " + base.get_name()));
            }
        }
      FEValues<dim> fe_values (finite_element, Quadrature<dim>(points), update_quadrature_points);
      unsigned int unsupported_component = numbers::invalid_unsigned_int;
      std::vector<unsigned int> component_indices(n_dofs);
      std::vector<bool> should_be_constrained(n_dofs);
      std::vector<double> solution(n_dofs);
      std::vector<types::global_dof_index> local_dof_indices(n_dofs);

      // Loop over all cells
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            fe_values.reinit (cell);
            cell->get_dof_indices (local_dof_indices);

            for (unsigned int q=0; q<n_dofs; q++)
              {
                should_be_constrained[q] = false;
                solution[q] = 0.0;
                component_indices[q] = finite_element.system_to_component_index(q).first;
              }

            for (auto &p: this->plugin_objects)
              {
                p->constrain_solution(cell,
                                      fe_values.get_quadrature_points(),
                                      component_indices,
                                      should_be_constrained,
                                      solution);
              }

            for (unsigned int q=0; q<n_dofs; q++)
              {
                if (should_be_constrained[q] && !pointwise_dofs[q])
                  {
                    unsupported_component = std::min(unsupported_component, component_indices[q]);
                    continue;
                  }

                // If it's okay to constrain this DOF
                if (current_constraints.can_store_line(local_dof_indices[q]) &&
                    !current_constraints.is_constrained(local_dof_indices[q]) &&
                    should_be_constrained[q] == true)
                  {
                    // Add a constraint of the form dof[q] = u_i
                    // to the list of constraints.
                    current_constraints.add_constraint(local_dof_indices[q],
                                                       {},
                                                       solution[q]);

                  }
              }
          }
      if (has_non_nodal_prescription_points)
        {
          unsupported_component = Utilities::MPI::min(unsupported_component, this->get_mpi_communicator());
          AssertThrow(unsupported_component == numbers::invalid_unsigned_int,
                      ExcMessage("Cannot prescribe point values on non-nodal component " +
                                 std::to_string(unsupported_component) + "."));
        }
      this->get_computing_timer().leave_subsection("Prescribe solution");
    }



    template <int dim>
    void
    Manager<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Prescribed solution");
      {
        const std::string pattern_of_names
          = std::get<dim>(registered_plugins).get_pattern_of_names ();

        prm.declare_entry("List of model names",
                          "",
                          Patterns::MultipleSelection(pattern_of_names),
                          "A comma-separated list of prescribed solution models that "
                          "will be used to compute the solution in certain regions. "
                          "These plugins are loaded in the order given, and are combined "
                          "via the operators listed "
                          "in 'List of model operators'.\n\n"
                          "The following prescribed solution models are available:\n\n"
                          +
                          std::get<dim>(registered_plugins).get_description_string());
      }
      prm.leave_subsection ();

      std::get<dim>(registered_plugins).declare_parameters (prm);
    }



    template <int dim>
    void
    Manager<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Prescribed solution");
      {
        this->plugin_names
          = Utilities::split_string_list(prm.get("List of model names"));

        AssertThrow(Utilities::has_unique_entries(this->plugin_names),
                    ExcMessage("The list of strings for the parameter "
                               "'Prescribed solution/List of model names' contains entries more than once. "
                               "This is not allowed. Please check your parameter file."));
      }
      prm.leave_subsection ();

      // go through the list, create objects and let them parse
      // their own parameters
      for (auto &model_name : this->plugin_names)
        {
          // create initial temperature objects
          this->plugin_objects.emplace_back (std::get<dim>(registered_plugins)
                                             .create_plugin (model_name,
                                                             "Prescribed solution::Model names"));

          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(this->plugin_objects.back().get()))
            sim->initialize_simulator (this->get_simulator());

          this->plugin_objects.back()->parse_parameters (prm);
          this->plugin_objects.back()->initialize ();
        }
    }



    template <int dim>
    std::string
    get_valid_model_names_pattern ()
    {
      return std::get<dim>(registered_plugins).get_pattern_of_names ();
    }



    template <int dim>
    void
    Manager<dim>::write_plugin_graph (std::ostream &out)
    {
      std::get<dim>(registered_plugins).write_plugin_graph ("Prescribed solution interface",
                                                            out);
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace PrescribedSolution
  {
#define INSTANTIATE(dim) \
  template class Interface<dim>;\
  template class Manager<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }

}
