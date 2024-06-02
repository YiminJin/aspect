/*
  Copyright (C) 2019 - 2023 by the authors of the ASPECT code.

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


#include <aspect/material_model/rheology/pariseau.h>
#include <aspect/material_model/utilities.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      namespace
      {
        /**
         * This helper function is used for computing the rotation matrix 
         * $\mathbf{Q}$. If we rotate the MPF first around the Y-axis
         * with angle $\beta$ clockwise and then around the Z-axis
         * with angle $\alpha$ counterclockwise, then the rotation matrix 
         * can be represented by
         * \f[
         *   \mathbf{Q} = \mathbf{Q}_Z\mathbf{Q}_Y =
         *   \begin{bmatrix}
         *   \cos\alpha & -\sin\alpha & 0 \\
         *   \sin\alpha & \cos\alpha  & 0 \\
         *   0          &             & 1
         *   \end{bmatrix}
         *   \begin{bmatrix}
         *   \cos\beta  & 0 & \sin\beta \\
         *   0          & 1 & 0         \\
         *   -\sin\beta & 0 & \cos\beta
         *   \end{bmatrix}.
         * \f]
         * From $\mathbf{Q}\bm{e}_1 = \bm{n}$ we can get the relationship
         * bewteen $\bm{n}$ and $(\alpha,\beta)$, through which we can 
         * express $\mathbf{Q}$ in terms of $\{n_1,n_2,n_3\}$ as
         * \f[
         *   \begin{bmatrix}
         *   n_1 & -\frac{n_2}{\sqrt{1-n_3^2}} & -\frac{n_1n_3}{\sqrt{1-n_3^2}} \\
         *   n_2 & \frac{n_1}{\sqrt{1-n_3^2}}  & -\frac{n_2n_3}{\sqrt{1-n_3^2}} \\
         *   n_3 & 0                           & \sqrt{1-n_3^2}
         *   \end{bmatrix}.
         * \f]
         */
        Tensor<2,3>
        compute_rotation_matrix(const Tensor<1,3> &n)
        {
          const double cos_beta = std::sqrt(1. - n[2] * n[2]);

          Tensor<2,3> Q;
          Q[0][0] = n[0];
          Q[0][1] = -n[1] / cos_beta;
          Q[0][2] = -n[0] * n[2] / cos_beta;
          Q[1][0] = n[1];
          Q[1][1] = n[0] / cos_beta;
          Q[1][2] = -n[1] * n[2] / cos_beta;
          Q[2][0] = n[2];
          Q[2][1] = 0;
          Q[2][2] = cos_beta;

          return Q;
        }


        /**
         * This helper function is used to transform the stress tensor from 
         * the lab frame to the material principal frame (MPF). 
         */
        SymmetricTensor<2,3>
        compute_stress_in_material_principal_frame(const SymmetricTensor<2,3> &stress,
                                                   const Tensor<2,3>          &Q)
        {
          SymmetricTensor<2,3> result;
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = i; j < 3; ++j)
              for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                  result[i][j] += Q[k][i] * Q[l][j] * stress[k][l];

          return result;
        }
      }

      template <int dim>
      Pariseau<dim>::Pariseau() = default;

      template <int dim>
      const PariseauParameters
      Pariseau<dim>::
      compute_pariseau_parameters(const unsigned int composition,
                                  const bool isotropization,
                                  const std::vector<double> &phase_function_values,
                                  const std::vector<unsigned int> &n_phase_transitions_per_composition) const
      {
        PariseauParameters pariseau_parameters;

        double F_avg, G_avg, M_avg, U_avg, V_avg;
        if (phase_function_values == std::vector<double>())
          {
            // no phases
            F_avg = F[composition];
            G_avg = G[composition];
            M_avg = M[composition];
            U_avg = U[composition];
            V_avg = V[composition];
            pariseau_parameters.n = exponent_coefficients[composition];
          }
        else
          {
            // Average among phases
            F_avg = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, F, composition);
            G_avg = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, G, composition);
            M_avg = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, M, composition);
            U_avg = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, U, composition);
            V_avg = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, V, composition);
            pariseau_parameters.n = MaterialModel::MaterialUtilities::phase_average_value(phase_function_values, n_phase_transitions_per_composition, 
                                    exponent_coefficients, composition);
          }

        if (isotropization)
          {
            // The isotropic parameters are calculated under the constraint
            // that the criterion remains the same when tau_xx = tau_yy = tau_zz = 0
            // and tau_xy = tau_xz = tau_yz.
            const double c1 = (2 * G_avg + 4 * F_avg + 2 * M_avg) / 18.;
            const double c2 = (U_avg + 2 * V_avg) / 3.;

            pariseau_parameters.F = c1;
            pariseau_parameters.G = c1;
            pariseau_parameters.M = c1 * 6;
            pariseau_parameters.U = c2;
            pariseau_parameters.V = c2;
          }
        else
          {
            pariseau_parameters.F = F_avg;
            pariseau_parameters.G = G_avg;
            pariseau_parameters.M = M_avg;
            pariseau_parameters.U = U_avg;
            pariseau_parameters.V = V_avg;
          }

        return pariseau_parameters;
      }

      template <int dim>
      std::pair<double,bool>
      Pariseau<dim>::
      compute_viscosity(const SymmetricTensor<2,dim>    &effective_strain_rate,
                        const Tensor<1,dim>             &normal_vector,
                        const double                     pressure,
                        const double                     non_yielding_viscosity,
                        const unsigned int               composition,
                        const std::array<double,3>      &weakening_factors,
                        const std::vector<double>       &phase_function_values,
                        const std::vector<unsigned int> &n_phase_transitions_per_composition) const
      {
        // We first compute the deviatoric stress in 3D.
        // The strict form of deviatoric strain rate is (see the manual):
        // $\bm{\varepsilon} - \frac{1}{3}(\nabla\cdot\bm{u})\bm{1}$.
        // However, since the upstream data is calculated through
        // deviator(effective_strain_rate) (e.g. the dislocation creep viscosity),
        // we use the same form to guarantee that all the strain rates used in
        // computation are parallel.
        const SymmetricTensor<2,dim> deviatoric_strain_rate = deviator(effective_strain_rate);

        SymmetricTensor<2,3> deviatoric_stress;

        deviatoric_stress[0][0] = 2. * non_yielding_viscosity * deviatoric_strain_rate[0][0];
        deviatoric_stress[1][1] = 2. * non_yielding_viscosity * deviatoric_strain_rate[1][1];
        deviatoric_stress[0][1] = 2. * non_yielding_viscosity * deviatoric_strain_rate[0][1];
        if (dim > 2)
          {
            deviatoric_stress[2][2] = 2. * non_yielding_viscosity * deviatoric_strain_rate[2][2];
            deviatoric_stress[0][2] = 2. * non_yielding_viscosity * deviatoric_strain_rate[0][2];
            deviatoric_stress[1][2] = 2. * non_yielding_viscosity * deviatoric_strain_rate[1][2];
          }

        // Transform the deviatoric stress from the lab frame to the 
        // material principal frame (MPF) if the normal vector is 
        // valid (i.e., the material is anisotropic).
        const bool anisotropic = (normal_vector.norm() > 0.5);

        SymmetricTensor<2,3> deviatoric_stress_mpf = deviatoric_stress;
        if (anisotropic)
          {
            Tensor<1,3> normal_vector_3d;
            normal_vector_3d[0] = normal_vector[0];
            normal_vector_3d[1] = normal_vector[1];
            if (dim > 2)
              normal_vector_3d[2] = normal_vector_3d[2];

            const Tensor<2,3> Q = compute_rotation_matrix(normal_vector_3d);
            deviatoric_stress_mpf = compute_stress_in_material_principal_frame(deviatoric_stress, Q);
          }

        // Compute the material constants
        const PariseauParameters p = compute_pariseau_parameters(composition,
                                                                 anisotropic == false,
                                                                 phase_function_values,
                                                                 n_phase_transitions_per_composition);

        // Now compute the apparent viscosity. Assume that 
        // $\eta^{apparent} = a\eta^{non-yielding}$. Define
        // $A := F(\tau_{yy}-\tau_{zz})^2 + 
        // G\left[(\tau_{zz}-\tau_{xx})^2 + (\tau_{xx}-\tau_{yy})^2\right] +
        // (2G+4F)\tau_{yz}^2 + M(\tau_{zx}^2 + \tau_{xy}^2)$,
        // $B := U\tau_{xx} + V(\tau_{yy} + \tau_{zz})$ and $C := -(U + 2V)p$,
        // then we have $(A a^2)^{n/2} - w_2 B a - w_2 C = w_1$, where $w_1$ and $w_2$
        // represent the weakening factors applied to the cohesion and
        // the friction angle, respectively. If $n=1$, then we simply have
        // $a = \frac{w_1 + w_2 C}{\sqrt{A} - w_2 B}$; otherwise, the equation will
        // be solved with Newton-Raphson method.
        double apparent_viscosity = non_yielding_viscosity;

        const double A = p.F * Utilities::fixed_power<2,double>(deviatoric_stress_mpf[1][1] - deviatoric_stress_mpf[2][2]) +
                         p.G * (Utilities::fixed_power<2,double>(deviatoric_stress_mpf[2][2] - deviatoric_stress_mpf[0][0]) +
                                Utilities::fixed_power<2,double>(deviatoric_stress_mpf[0][0] - deviatoric_stress_mpf[1][1])) +
                         (2 * p.G + 4 * p.F) * Utilities::fixed_power<2,double>(deviatoric_stress_mpf[1][2]) +
                         2 * p.M * (Utilities::fixed_power<2,double>(deviatoric_stress_mpf[2][0]) +
                                    Utilities::fixed_power<2,double>(deviatoric_stress_mpf[0][1]));
        const double B = -p.U * deviatoric_stress_mpf[0][0] - p.V * (deviatoric_stress_mpf[1][1] + deviatoric_stress_mpf[2][2]);
        const double C = (p.U + 2 * p.V) * pressure;

        const bool yield = (std::pow(A, p.n * 0.5) - weakening_factors[1] * (B + C) > weakening_factors[0]);
        if (yield)
          {
            double a = 1;
            if (p.n == 1)
              {
                a = (weakening_factors[0] + weakening_factors[1] * C) / (std::sqrt(A) - weakening_factors[1] * B);
              }
            else
              {
                const double c1 = std::pow(A, p.n * 0.5);
                const double c2 = weakening_factors[1] * B;
                const double c3 = weakening_factors[0] + weakening_factors[1] * C;

                const double initial_residual = c1 * std::pow(a, p.n) - c2 * a - c3;
                const double tolerance = initial_residual * 1.e-8;
                const unsigned int max_iterations = 10;

                double residual = initial_residual;
                unsigned int iteration = 0;
                while (residual > tolerance)
                  {
                    a -= residual / (p.n * c1 * std::pow(a, p.n - 1) - c2);
                    residual = c1 * std::pow(a, p.n) - c2 * a - c3;

                    iteration++;

                    if (iteration > max_iterations)
                      break;
                  }
                AssertThrow(iteration <= max_iterations,
                            ExcMessage("The Newton method failed to converge within 10 iterations. "
                                       "The final relative residual is " 
                                       + Utilities::to_string(residual / initial_residual)));
              }

            AssertThrow(a > 0 && a <= 1, ExcInternalError());
            apparent_viscosity *= a;

            // If the plastic damper is used, the effective strain rate is partitioned between the
            // viscoelastic and damped plastic (Bingham) elements. Assuming that the viscoelastic
            // elements have viscosities that are not strain rate dependent, we have:
            // edot_eff = tau_T / (2 * eta_ve) + (tau_T - tau_yield) / (2 * eta_d)
            // The apparent viscosity is defined such that:
            // tau_T = 2 * eta_app * edot_eff.
            // Substituting one equation into the other and rearranging yields the expression
            // eta_app = ((1 + tau_yield / (2 * eta_d * edot_eff)) / (1 / eta_d + 1 / eta_ve)).

            if (use_plastic_damper)
              apparent_viscosity = (damper_viscosity + apparent_viscosity) /
                                   (1. + damper_viscosity / non_yielding_viscosity);
          }

        return std::make_pair(apparent_viscosity, yield);
      }



      template <int dim>
      void
      Pariseau<dim>::
      fill_reaction_outputs(const MaterialModel::MaterialModelInputs<dim> &in,
                            MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        if (!(this->simulator_is_past_initialization() &&
              in.current_cell.state() == IteratorState::valid &&
              in.requests_property(MaterialProperties::reaction_terms)))
          return;

        // Get old (previous time step) velocity gradients
        std::vector<Point<dim>> quadrature_positions(in.n_evaluation_points());
        for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
          quadrature_positions[i] = this->get_mapping().transform_real_to_unit_cell(in.current_cell, in.position[i]);

        std::vector<double> solution_values(this->get_fe().dofs_per_cell);
        in.current_cell->get_dof_values(this->get_old_solution(),
                                        solution_values.begin(),
                                        solution_values.end());

        // Only create the evaluator the first time we get here
        if (!evaluator)
          evaluator = std::make_unique<FEPointEvaluation<dim,dim>>(this->get_mapping(),
                                                                   this->get_fe(),
                                                                   update_gradients,
                                                                   this->introspection().component_indices.velocities[0]);

        // Initialize the evaluator for the old velocity gradients
        evaluator->reinit(in.current_cell, quadrature_positions);
        evaluator->evaluate(solution_values,
                            EvaluationFlags::gradients);

        // Get the indices of the normal vector components in the 
        // compositional fields
        std::vector<unsigned int> c_idx_n;
        c_idx_n.push_back(this->introspection().compositional_index_for_name("n_x"));
        c_idx_n.push_back(this->introspection().compositional_index_for_name("n_y"));
        if (dim > 2)
          c_idx_n.push_back(this->introspection().compositional_index_for_name("n_z"));

        for (unsigned int q = 0; q < in.n_evaluation_points(); ++q)
          {
            // Get the normal vector
            Tensor<1,dim> n;
            for (unsigned int i = 0; i < dim; ++i)
              n[i] = in.composition[q][c_idx_n[i]];

            Tensor<1,dim> n_dot;
            if (n.norm() > 0.5)
              {
                // Compute the time derivative of n (Chaves E. W., 2013, p217)
                // and make sure that n is a unit vector. Handle time step 0 differently.
                if (this->get_timestep_number() == 0)
                  {
                    n_dot = (n / n.norm()) - n;
                    for (unsigned int i = 0; i < dim; ++i)
                      out.reaction_terms[q][c_idx_n[i]] = n_dot[i];
                  }
                else
                  {
                    const Tensor<2,dim> L = evaluator->get_gradient(q);
                    n_dot = -n * L + n * (n * L * n);

                    Tensor<1,dim> n_new = n + (n_dot * this->get_timestep());
                    n_new /= n_new.norm();
                    n_dot = (n_new - n) / this->get_timestep();

                    for (unsigned int i = 0; i < dim; ++i)
                      out.reaction_terms[q][c_idx_n[i]] = n_dot[i] * this->get_timestep();
                  }
              }
            else
              {
                n_dot = 0;
              }
          }
      }



      template <int dim>
      void
      Pariseau<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.declare_entry ("Pariseau F", "8.8e-16",
                           Patterns::Anything(),
                           "List of parameter $F$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "The default value fits the experimental data of the middle Ordovician schist from Angers "
                           "(Duveau et al., 1998). Units: \\si{\\pascal}$^{-2}$.");
        prm.declare_entry ("Pariseau G", "9.6e-18",
                           Patterns::Anything(),
                           "List of parameter $G$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "Units: \\si{\\pascal}$^{-2}$.");
        prm.declare_entry ("Pariseau M", "2.37e-14",
                           Patterns::Anything(),
                           "List of parameter $M$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "The default value fits the experimental data of the middle Ordovician schist from Angers "
                           "(Duveau et al., 1998). Units: \\si{\\pascal}$^{-2}$.");
        prm.declare_entry ("Pariseau U", "-1.2e-8",
                           Patterns::Anything(),
                           "List of parameter $U$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "The default value fits the experimental data of the middle Ordovician schist from Angers "
                           "(Duveau et al., 1998). Units: \\si{\\pascal}$^{-1}$.");
        prm.declare_entry ("Pariseau V", "2.12e-8",
                           Patterns::Anything(),
                           "List of parameter $V$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "The default value fits the experimental data of the middle Ordovician schist from Angers "
                           "(Duveau et al., 1998). Units: \\si{\\pascal}$^{-1}$.");
        prm.declare_entry ("Pariseau exponent coefficient", "1",
                           Patterns::Anything(),
                           "List of the exponent coefficient $n$ in the criterion of Pariseau, for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "If only one value is given, then all use the same value. "
                           "The default value is 1. Units: None.");
        prm.declare_entry ("Use plastic damper","false",
                           Patterns::Bool (),
                           "Whether to use a plastic damper when computing the plastic viscosity. "
                           "The damper acts to stabilize the plastic shear "
                           "band width and remove associated mesh-dependent behavior at "
                           "sufficient resolutions.");
        prm.declare_entry ("Plastic damper viscosity", "0.0", Patterns::Double(0),
                           "Viscosity of the damper that acts in parallel with the plastic viscosity "
                           "to produce mesh-independent behavior at sufficient resolutions. Units: \\si{\\pascal\\second}");
      }



      template <int dim>
      void
      Pariseau<dim>::parse_parameters(ParameterHandler &prm,
                                      const std::unique_ptr<std::vector<unsigned int>> &expected_n_phases_per_composition)
      {
        // Retrieve the list of composition names
        std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();

        // Retrieve the list of names of fields that represent chemical compositions, and not, e.g.,
        // plastic strain
        std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();

        // Establish that a background field is required here
        compositional_field_names.insert(compositional_field_names.begin(), "background");
        chemical_field_names.insert(chemical_field_names.begin(), "background");

        // Make options file for parsing maps to double arrays
        Utilities::MapParsing::Options options(chemical_field_names, "Pariseau exponent coefficient");
        options.list_of_allowed_keys = compositional_field_names;
        options.allow_multiple_values_per_key = true;
        if (expected_n_phases_per_composition)
          {
            options.n_values_per_key = *expected_n_phases_per_composition;

            // check_values_per_key is required to be true to duplicate single values
            // if they are to be used for all phases associated with a given key.
            options.check_values_per_key = true;
          }

        // Read the exponent coeffiicents.
        exponent_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau exponent coefficient"),
                                                                                 options);

        // Check that the exponent coefficients are greater than or equal to 1.
        for (const double n : exponent_coefficients)
          AssertThrow(n >= 1, ExcMessage("The exponent coefficient of Pariseau criterion should be "
                                         "equal to 1."));

        // Read the Pariseau parameters.
        options.property_name = "Pariseau F";
        F = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau F"),
                                                                     options);

        options.property_name = "Pariseau G";
        G = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau G"),
                                                                     options);

        options.property_name = "Pariseau M";
        M = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau M"),
                                                                     options);

        options.property_name = "Pariseau U";
        U = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau U"),
                                                                     options);

        options.property_name = "Pariseau V";
        V = Utilities::MapParsing::parse_map_to_double_array(prm.get("Pariseau V"),
                                                                     options);

        // Whether to include a plastic damper when computing the plastic viscosity
        use_plastic_damper = prm.get_bool("Use plastic damper");

        // Stabilize plasticity through a viscous damper.
        // The viscosity of the damper is implicitly zero if it is not used
        if (use_plastic_damper)
          damper_viscosity = prm.get_double("Plastic damper viscosity");
        else
          damper_viscosity = 0.;

        // Check the validity of the input material constants.
        for (unsigned int i = 0; i < chemical_field_names.size(); ++i)
          {
            AssertThrow(G[i] > 0, 
                        ExcMessage("The material constant G in Pariseau criterion must be positive."));
            AssertThrow(2 * F[i] + G[i] > 0, 
                        ExcMessage("The material constants F and G in Pariseau criterion must satisfty 2F + G > 0."));
            AssertThrow(M[i] > 0,
                        ExcMessage("The material constant M in Pariseau criterion must be positive."));
          }

        // Check if the compositional fields contain the components of the normal vector.
        AssertThrow(this->introspection().compositional_name_exists("n_x"),
                    ExcMessage("The Pariseau model only works if there is a compositional field named n_x"));
        AssertThrow(this->introspection().compositional_name_exists("n_y"),
                    ExcMessage("The Pariseau model only works if there is a compositional field named n_y"));
        if (dim > 2)
          AssertThrow(this->introspection().compositional_name_exists("n_z"),
                      ExcMessage("The Pariseau model only works if there is a compositional field named n_z"));
      } 
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  namespace Rheology \
  { \
    template class Pariseau<dim>; \
  }

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
