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
#include <aspect/newton.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Assemblers
  {
    namespace
    {
      /**
       * Compute the linear MLS coefficients. The linear MLS method 
       * approximates quantity $A$ at quadrature point q by
       * @f[
       *   A_q = \sum_p\alpha_{qp}A_p,
       * @f]
       * where $A_p$ is the value of $A$ at particle $p$, and the coefficients
       * $\alpha_{qp}$ are given by
       * @f[
       *   \alpha_{qp} = w_p\mathbf\phi_q^T\mathbf M^{-1}\mathbf\phi_p.
       * @f]
       * Here, $w(\boldsymbol x)$ is a weight function, 
       * $\mathbf\phi(\boldsymbol x)$ is the monomial basis ($[1,x,y]^T$ in 2D
       * and $[1,x,y,z]^T$ in 3D), and matrix $\mathbf M$ is given by
       * @f[
       *   \mathbf M = \sum_p w_p\mathbf p_p\mathbf p_p^T.
       * @f]
       * In practice, we use the Wendland $C^2$ function as weight function:
       * @f[
       *   w_p = (1 - s_p)^4 (1 + 4s_p).
       * @f]
       * Here, $s_p = d_p / h$, where $d_p$ is the distance between points $q$
       * and $p$, and $h$ is the diameter of the host cell. Moreover, we let
       * @f[
       *   \mathbf p(\boldsymbol x) = [1, x_p - x_q, y_p - y_q, z_p - z_q]^T,
       * @f]
       * which simplifies the computation due to the fact that
       * $\mathbf p_q = [1,0,0,0]^T$.
       */
      template <int dim>
      std::vector<double>
      compute_MLS_coefficients(const std::vector<Point<dim>> &particle_points,
                               const Point<dim>               quadrature_point,
                               const double                   cell_diameter)
      {
        static constexpr unsigned int n_basis = dim + 1;

        const unsigned int n_p = particle_points.size();
        AssertThrow(n_p >= n_basis,
                    ExcMessage("The linear MLS approximation requires at least "
                               + Utilities::int_to_string(n_basis)
                               + " particles in each cell."));

        small_vector<double> w(n_p);
        small_vector<Vector<double>> phi(n_p, Vector<double>(n_basis));
        FullMatrix<double> M(n_basis, n_basis), Mp(n_basis, n_basis);

        std::vector<double> MLS_coefficients(n_p, numbers::signaling_nan<double>());

        for (unsigned int p = 0; p < n_p; ++p)
          {
            const Tensor<1, dim> reference_location = particle_points[p] - quadrature_point;

            phi[p][0] = 1.;
            for (unsigned int d = 0; d < dim; ++d)
              phi[p][d + 1] = reference_location[d];

            const double s = reference_location.norm() / cell_diameter;
            w[p] = Utilities::fixed_power<4, double>(1. - s) * (1. + 4. * s);

            Mp.outer_product(phi[p], phi[p]);
            M.add(w[p], Mp);
          }

        M.gauss_jordan();

        for (unsigned int p = 0; p < n_p; ++p)
          {
            double alpha = 0;
            for (unsigned int m = 0; m < n_basis; ++m)
              alpha += w[p] * M[0][m] * phi[p][m];

            MLS_coefficients[p] = alpha;
          }

        return MLS_coefficients;
      }
    }



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
      const typename Newton::Parameters::Stabilization stabilization = this->get_newton_handler().parameters.preconditioner_stabilization;

      const std::shared_ptr<const MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();
      Assert(implicit_constitutive_outputs != nullptr, ExcInternalError());

      const auto &cell = scratch.material_model_inputs.current_cell;
      const double cell_diameter = cell->diameter();

      const Particles::ParticleHandler<dim> &particle_handler = this->get_particle_manager(0).get_particle_handler();
      const auto particles_in_cell = particle_handler.particles_in_cell(cell);
      const unsigned int n_particles = particle_handler.n_particles_in_cell(cell);
      AssertDimension(implicit_constitutive_outputs->tangent_operators.size(), n_particles);
      AssertDimension(implicit_constitutive_outputs->equivalent_viscosities.size(), n_particles);

      std::vector<Point<dim>> particle_points(n_particles);
      auto particle = particles_in_cell.begin();
      for (unsigned int p = 0; p < n_particles; ++p, ++particle)
        particle_points[p] = particle->get_location();

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

          const std::vector<double> alpha = compute_MLS_coefficients(particle_points,
                                                                     scratch.finite_element_values.quadrature_point(q),
                                                                     cell_diameter);

          small_vector<SymmetricTensor<2, dim>> linearized_stresses(stokes_dofs_per_cell);
          double equivalent_viscosity = 0;

          particle = particles_in_cell.begin();
          for (unsigned int p = 0; p < n_particles; ++p, ++particle)
            {
              equivalent_viscosity += alpha[p] * implicit_constitutive_outputs->equivalent_viscosities[p];
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                linearized_stresses[i] += alpha[p] * (implicit_constitutive_outputs->tangent_operators[p] * scratch.grads_phi_u[i]);
            }

          const double one_over_eta = 1. / equivalent_viscosity;

          const double JxW = scratch.finite_element_values.JxW(q);
        
          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += ( (stabilization & Newton::Parameters::Stabilization::symmetric 
                                              ?
                                              (scratch.grads_phi_u[i] * linearized_stresses[j] +
                                               scratch.grads_phi_u[j] * linearized_stresses[i]) * 0.5
                                              :
                                              scratch.grads_phi_u[i] * linearized_stresses[j])
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
      if (outputs.template has_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>() == false)
        outputs.additional_outputs.push_back(std::make_unique<MaterialModel::ImplicitConstitutiveOutputs<dim>>(true));
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
      const typename Newton::Parameters::Stabilization stabilization = this->get_newton_handler().parameters.velocity_block_stabilization;

      const GravityModel::Interface<dim> &gravity_model = this->get_gravity_model();

      const std::shared_ptr<const MaterialModel::ImplicitConstitutiveOutputs<dim>> implicit_constitutive_outputs
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>();
      Assert(implicit_constitutive_outputs != nullptr, ExcInternalError());

      const std::shared_ptr<const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> force
        = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>();

      const auto &cell = scratch.material_model_inputs.current_cell;
      const double cell_diameter = cell->diameter();

      const Particles::ParticleHandler<dim> &particle_handler = this->get_particle_manager(0).get_particle_handler();
      const auto particles_in_cell = particle_handler.particles_in_cell(cell);
      const unsigned int n_particles = particle_handler.n_particles_in_cell(cell);
      AssertDimension(implicit_constitutive_outputs->tangent_operators.size(), n_particles);
      AssertDimension(implicit_constitutive_outputs->deviatoric_stresses.size(), n_particles);

      std::vector<Point<dim>> particle_points(n_particles);
      auto particle = particles_in_cell.begin();
      for (unsigned int p = 0; p < n_particles; ++p, ++particle)
        particle_points[p] = particle->get_location();

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
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

          const std::vector<double> alpha = compute_MLS_coefficients(particle_points,
                                                                     scratch.finite_element_values.quadrature_point(q),
                                                                     cell_diameter);

          small_vector<SymmetricTensor<2, dim>> linearized_stresses(stokes_dofs_per_cell);
          SymmetricTensor<2, dim> deviatoric_stress;

          particle = particles_in_cell.begin();
          for (unsigned int p = 0; p < n_particles; ++p, ++particle)
            {
              deviatoric_stress += alpha[p] * implicit_constitutive_outputs->deviatoric_stresses[p];
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                linearized_stresses[i] += alpha[p] * (implicit_constitutive_outputs->tangent_operators[p] * scratch.grads_phi_u[i]);
            }

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
                    data.local_matrix(i, j) += ( (stabilization & Newton::Parameters::Stabilization::symmetric
                                                  ?
                                                  (scratch.grads_phi_u[i] * linearized_stresses[j] +
                                                   scratch.grads_phi_u[j] * linearized_stresses[i]) * 0.5
                                                  :
                                                  scratch.grads_phi_u[i] * linearized_stresses[j])
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
      if (outputs.template has_additional_output_object<MaterialModel::ImplicitConstitutiveOutputs<dim>>() == false)
        outputs.additional_outputs.push_back(std::make_unique<MaterialModel::ImplicitConstitutiveOutputs<dim>>(false));

      if (this->get_parameters().enable_additional_stokes_rhs &&
          outputs.template has_additional_output_object<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>() == false)
        outputs.additional_outputs.push_back(std::make_unique<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>(outputs.densities.size()));
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
