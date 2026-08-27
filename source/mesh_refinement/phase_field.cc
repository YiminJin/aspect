/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

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


#include <aspect/mesh_refinement/phase_field.h>
#include <aspect/phase_field.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void PhaseField<dim>::initialize()
    {
      AssertThrow(this->get_parameters().enable_phase_field,
                  ExcMessage("Mesh refinement model 'phase field' only works when "
                             "the phase field method is enabled."));
    }



    template <int dim>
    void
    PhaseField<dim>::tag_additional_cells() const
    {
      // Get the phase field activation threshold
      const double phi0 = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>&>(
        this->get_material_model()).get_phase_field_range().first;

      // Get the local dof indices of phase field
      const auto &variable = this->introspection().variable("phase_field");
      const unsigned int component_index = variable.first_component_index;
      const FiniteElement<dim> &base_fe = this->get_fe().base_element(variable.base_index);
      std::vector<unsigned int> local_dof_indices(base_fe.dofs_per_cell);
      for (unsigned int i = 0; i < base_fe.dofs_per_cell; ++i)
        local_dof_indices[i] = this->get_fe().component_to_system_index(component_index, i);

      std::vector<double> cell_dof_values(this->get_fe().dofs_per_cell);

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_values(this->get_solution(), 
                                 cell_dof_values.begin(),
                                 cell_dof_values.end());

            // The cell is marked for refinement if any of its vertices hold
            // phase field value greater than the activation threshold
            for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
              if (cell_dof_values[local_dof_indices[i]] > phi0)
                {
                  cell->clear_coarsen_flag();
                  cell->set_refine_flag();
                  break;
                }
          }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(PhaseField,
                                              "phase field",
                                              "A mesh refinement criterion based on the phase field. A cell "
                                              "is marked for refinement if any of its vertices hold phase field "
                                              "value greater than the activation threshold.")
  }
}
