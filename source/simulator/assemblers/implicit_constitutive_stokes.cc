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

#include <aspect/simulator/assemblers/implicit_constitutive_stokes.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void
    ImplicitConstitutiveStokesPreconditioner<dim>::
    execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
            internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>&>(scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>&>(data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const std::shared_ptr<const MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();
      Assert(implicit_constitutive_outputs != nullptr, ExcInternalError());

      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i, q);
                  scratch.phi_p[i_stokes]       = scratch.finite_element_values[introspection.extractors.pressure].value(i, q);

                  ++i_stokes;
                }
              ++i;
            }

          const double one_over_eta = 1. / implicit_constitutive_outputs->equivalent_viscosities[q];

          const double JxW = scratch.finite_element_values.JxW(q);
        
          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += ( scratch.grads_phi_u[i]
                                             * implicit_constitutive_outputs->linearized_stress_terms[q][j]
                                             // bottom right block: approximate the
                                             // pressure Schur complement by the
                                             // pressure mass matrix.
                                             + one_over_eta
                                             * pressure_scaling
                                             * pressure_scaling
                                             * (scratch.phi_p[i] * scratch.phi_p[j])
                                           ) * JxW;
        }
    }



    template <int dim>
    void
    ImplicitConstitutiveStokesPreconditioner<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.densities.size();
      const unsigned int n_stokes_dofs = this->get_fe().base_element(this->introspection().base_elements.velocities).dofs_per_cell * dim +
                                         this->get_fe().base_element(this->introspection().base_elements.pressure).dofs_per_cell;
      if (outputs.template has_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>() == false)
        outputs.additional_outputs.push_back(std::make_unique<MaterialModel::ImplicitConstitutiveOutputs<dim>>(n_points, n_stokes_dofs, true));
    }



    template <int dim>
    void
    ImplicitConstitutiveStokesSystem<dim>::
    execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
            internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const GravityModel::Interface<dim> &gravity_model = this->get_gravity_model();

      const std::shared_ptr<const MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();
      Assert(implicit_constitutive_outputs != nullptr, ExcInternalError());

      const std::shared_ptr<const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> force
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>();

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Get the values and gradients of the shape functions
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value(i, q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value(i, q);

                  if (scratch.rebuild_stokes_matrix)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i, q);
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence(i, q);
                    }
                  ++i_stokes;
                }
              ++i;
            }

          const SymmetricTensor<2, dim> &deviatoric_stress = implicit_constitutive_outputs->deviatoric_stresses[q];

          const Tensor<1, dim> gravity = gravity_model.gravity_vector(scratch.finite_element_values.quadrature_point(q));
          const double density = scratch.material_model_outputs.densities[q];

          const double pressure = scratch.material_model_inputs.pressure[q];
          const double velocity_divergence = scratch.velocity_divergence[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) -= ( scratch.grads_phi_u[i] * deviatoric_stress
                                     - scratch.div_phi_u[i] * pressure
                                     - scratch.phi_p[i] * pressure_scaling * velocity_divergence
                                     - (scratch.phi_u[i] * gravity) * density
                                   ) * JxW;
                
              if (force != nullptr)
                data.local_rhs(i) += ( scratch.phi_u[i] * force->rhs_u[q]
                                       + scratch.phi_p[i] * pressure_scaling * force->rhs_p[q] 
                                     ) * JxW;
              
              if (scratch.rebuild_stokes_matrix)
                for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i, j) += ( scratch.grads_phi_u[i] 
                                                 * implicit_constitutive_outputs->linearized_stress_terms[q][j]
                                                 // assemble \nabla p as -(p, div v):
                                                 - (pressure_scaling *
                                                    scratch.div_phi_u[i] * scratch.phi_p[j])
                                                 // assemble the term -div(u) as -(div u, q).
                                                 // Note the negative sign to make this
                                                 // operator adjoint to the grad p term:
                                                 - (pressure_scaling *
                                                    scratch.phi_p[i] * scratch.div_phi_u[j])
                                               ) * JxW;
                  }
            }
        }
    }



    template <int dim>
    void
    ImplicitConstitutiveStokesSystem<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.densities.size();

      if (outputs.template has_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>() == false)
        {
          const unsigned int n_stokes_dofs = this->get_fe().base_element(this->introspection().base_elements.velocities).dofs_per_cell +
                                             this->get_fe().base_element(this->introspection().base_elements.pressure).dofs_per_cell;
          outputs.additional_outputs.push_back(std::make_unique<MaterialModel::ImplicitConstitutiveOutputs<dim>>(n_points, n_stokes_dofs, false));
        }

      if (this->get_parameters().enable_additional_stokes_rhs &&
          outputs.template has_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>() == false)
        outputs.additional_outputs.push_back(std::make_unique<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>(n_points));
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
    template class ImplicitConstitutiveStokesPreconditioner<dim>; \
    template class ImplicitConstitutiveStokesSystem<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
