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

#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/particle/particle_domain.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/solver_cg.h>

#include <iomanip>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    std::pair<double, double>
    PhaseFieldModel<dim>::get_phase_field_range() const
    {
      return std::make_pair(0.01, 0.99);
    }
  }



  namespace PhaseField
  {
    /*------------------------ GeometricFunction ---------------------------*/

    GeometricFunction::GeometricFunction(const double l_,
                                         const double xi_,
                                         const double c0_)
      : l(l_)
      , xi(xi_)
      , c0(c0_)
    {
      AssertThrow(xi >= 0.0 && xi <= 2.0,
                  ExcMessage("Parameter $\\xi$ in the geometric function must be "
                             "in the range of [0,2]."));
    }


    
    double
    GeometricFunction::value(const double d) const
    {
      return d * (xi + (1.0 - xi) * d);
    }



    double
    GeometricFunction::first_derivative(const double d) const
    {
      return xi + 2.0 * (1.0 - xi) * d;
    }



    double
    GeometricFunction::second_derivative(const double /*d*/) const
    {
      return 2.0 * (1.0 - xi);
    }



    /*------------------------ DegradationFunction ---------------------------*/

    DegradationFunction::
    DegradationFunction(const double p_,
                        const double m_)
      : p(p_)
      , m(m_)
    {
      AssertThrow(m >= 0.0,
                  ExcMessage("Parameter $m$ in the degradation function must be non-negative."));
    }



    double
    DegradationFunction::value(const double d) const
    {
      const double numerator   = (1. - d) * (1. - d);
      const double denominator = numerator + m * d * (1. + p * d);
      return numerator / denominator;
    }



    double
    DegradationFunction::first_derivative(const double d) const
    {
      const double numerator   = -m * (1. - d) * (1. + d + 2. * p * d);
      const double denominator = Utilities::fixed_power<2, double>((1. - d) * (1. - d) + m * d * (1. + p * d));
      return numerator / denominator;
    }



    double
    DegradationFunction::second_derivative(const double d) const
    {
      const double A     = (1. - d) * (1. - d) + m * d * (1. + p * d);
      const double dA_dd = 2. * (d - 1.) + m * (1. + 2. * p * d);
      const double B     = (1. - d) * (1. + d + 2. * p * d);
      const double dB_dd = 2. * (p - d - 2. * p * d);
      const double numerator   = m * (2. * B * dA_dd - dB_dd * A);
      const double denominator = A * A * A;
      return numerator / denominator;
    }



    /*------------------------ PhaseFieldProfile ---------------------------*/

    PhaseFieldProfile::
    PhaseFieldProfile(const GeometricFunction   &a_func,
                      const DegradationFunction &g_func,
                      const double               d_hat,
                      const unsigned int         n_points)
      : N(n_points)
      , coordinate_values(N)
      , phase_field_values(N)
    {
      AssertThrow(d_hat > 0 && d_hat < 1,
                  ExcMessage("The peak value of the phase-field must be "
                             "greater than 0 and smaller than 1."));

      const double a_hat = a_func.value(d_hat);
      const double g_hat = g_func.value(d_hat);
      const double h_hat = 1. / g_hat - 1.;

      const double l = a_func.get_length_scale();

      std::vector<double> integrand;

      // Use the change of variables:
      //    d = d_hat cos^2(theta), theta in [0, pi/2]
      const double dtheta = numbers::PI_2 / (N - 1);
      for (unsigned int i = 0; i < N; ++i)
        {
          const double theta = i * dtheta;
          const double cos_theta = std::cos(theta);
          const double d = d_hat * cos_theta * cos_theta;

          phase_field_values[i] = d;
          if (i == 0 || i == N - 1)
            continue;

          const double a = a_func.value(d);
          const double g = g_func.value(d);
          const double h = 1. / g - 1.;
          
          const double D = h_hat * a - a_hat * h;
          Assert(D > 0, ExcInternalError());

          // dzeta / dtheta = 2l * d_hat * sin(theta) * cos(theta) * sqrt(h_hat / D)
          const double sin_theta = std::sin(theta);
          integrand[i] = 2. * l * d_hat * sin_theta * cos_theta * std::sqrt(h_hat / D);
        }

      // Endpoint limits:
      // At theta = 0, dzeta/dtheta ~ 2l * sqrt(d_hat * h_hat / -D'(d_hat))
      const double a_hat_prime = a_func.first_derivative(d_hat);
      const double g_hat_prime = g_func.first_derivative(d_hat);
      const double D_hat_prime = h_hat * a_hat_prime + a_hat * g_hat_prime / (g_hat * g_hat);
      integrand[0] = 2. * l * std::sqrt(d_hat * h_hat / -D_hat_prime);

      // At theta = pi/2, dzeta/dtheta ~ 2l * sqrt(d_hat * h_hat / D'(0))
      const double a0_prime = a_func.first_derivative(0);
      const double g0_prime = g_func.first_derivative(0);
      const double D0_prime = h_hat * a0_prime + a_hat * g0_prime; // g(0) = 1
      integrand[N-1] = 2. * l * std::sqrt(d_hat * h_hat / D0_prime);

      // Cumulatively integrate dzeta/dtheta using the composite trapezoidal rule
      coordinate_values[0] = 0;
      for (unsigned int i = 1; i < N; ++i)
        coordinate_values[i] = coordinate_values[i-1] + (integrand[i-1] + integrand[i]) * dtheta * 0.5;
    }



    double PhaseFieldProfile::value(const double zeta) const
    {
      AssertThrow(zeta >= 0,
                  ExcMessage("The distance from crack center must be non-negative."));

      // Get the index of the first element of the coordinate array that is
      // greater than or equal to zeta
      const unsigned int idx = std::lower_bound(coordinate_values.begin(), coordinate_values.end(), zeta)
                               - coordinate_values.begin();

      if (idx == N)
        return 0.;

      if (idx == 0)
        return phase_field_values[0];

      const double xi = std::max(std::min((zeta - coordinate_values[idx-1]) / 
                                          (coordinate_values[idx] - coordinate_values[idx-1]),
                                          1.),
                                 0.);
      
      return (1. - xi) * phase_field_values[idx-1] + xi * phase_field_values[idx];
    }



    /*------------------------ SlipRateNormalizer ---------------------------*/

    SlipRateNormalizer::
    SlipRateNormalizer(const GeometricFunction   &a_func,
                       const DegradationFunction &g_func,
                       const double               d_min_,
                       const double               d_max_)
      : d_min(d_min_)
      , d_max(d_max_)
    {
      // Distribute the sample points on a uniform logit grid
      logit_d_hat[0]   = std::log(d_min / (1. - d_min));
      logit_d_hat[M-1] = std::log(d_max / (1. - d_max));

      const double dx = (logit_d_hat[M-1] - logit_d_hat[0]) / (M - 1);
      for (unsigned int i = 1; i < M - 1; ++i)
        logit_d_hat[i] = logit_d_hat[0] + i * dx;

      for (unsigned int i = 0; i < M; ++i)
        {
          const double d_hat = 1. / (1. + std::exp(-logit_d_hat[i]));

          // Compute the phase-field profile with d_hat
          const PhaseFieldProfile profile(a_func, g_func, d_hat, N);

          const std::vector<double> &zeta = profile.get_coordinate_values();
          const std::vector<double> &d    = profile.get_phase_field_values();

          // Integrate h(d) over the profile using the trapezoidal rule
          std::vector<double> h(N);
          for (unsigned int j = 0; j < N; ++j)
            {
              const double g = g_func.value(d[j]);
              h[j] = 1. / g - 1.;
            }
          
          double Ih = 0;
          for (unsigned int j = 0; j < N - 1; ++j)
            // We don't need to divide Ih by 2 here, since we only integrate
            // over one half of the profile
            Ih += (h[j] + h[j+1]) * (zeta[j+1] - zeta[j]);

          log_Ih[i] = std::log(Ih);
        }
    }



    double 
    SlipRateNormalizer::normalization_factor(const double d_hat) const
    {
      AssertThrow(d_hat >= d_min && d_hat <= d_max,
                  ExcMessage("The peak value of the phase-field exceeds the range "
                             "that can be handled by SlipRateNormalizer."));

      const double x = std::log(d_hat / (1. - d_hat));

      // Get the index of the first element of the coordinate array that is
      // greater than or equal to x
      const unsigned int idx = std::lower_bound(logit_d_hat.begin(), logit_d_hat.end(), x)
                               - logit_d_hat.begin();

      if (idx == 0)
        return std::exp(log_Ih[0]);

      const double xi = std::max(std::min((x - logit_d_hat[idx-1]) /
                                          (logit_d_hat[idx] - logit_d_hat[idx-1]),
                                          1.),
                                 0.);

      return std::exp((1. - xi) * log_Ih[idx-1] + xi * log_Ih[idx]);
    }
  }


  /*--------------------------- PhaseFieldHandler ---------------------------*/

  template <int dim>
  void
  PhaseFieldHandler<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Phase field model");
    {
      prm.declare_entry("Length scale", "1000",
                        Patterns::Double(0.),
                        "The length scale that characterizes the width of the fracture zone. "
                        "Units: \\si{meter}.");

      prm.declare_entry("Geometric function type", "AT1",
                        Patterns::Selection("AT1|AT2|CZM"),
                        "Type of the geometric function $\\alpha(d)$. The options are:\n"
                        "AT1: $\\alpha(d) = d$;\n"
                        "AT2: $\\alpha(d) = d^2;\n$"
                        "CZM: $\\alpha(d) = 2d - d^2$.");

      prm.declare_entry("Degradation curvature parameter", "1",
                        Patterns::Double(0),
                        "The curvature parameter $p$ for the Lorentz-type degradation function.\n"
                        "Units: none.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver parameters");
    {
      prm.enter_subsection("Phase field solver parameters");
      {
        prm.declare_entry ("Linear solver tolerance", "1e-8",
                           Patterns::Double(0., 1.),
                           "A relative tolerance up to which the linearized phase field system "
                           "in each nonlinear step should be solved.");

        prm.declare_entry ("Max linear solver iterations", "1000",
                           Patterns::Integer(0),
                           "The maximum number of iteration steps for solving the linearized "
                           "phase field system. If the linear solver fails to converge to the "
                           "relative tolerance when reaching the maximum step number, the "
                           "program will be terminated with an error message.");

        prm.declare_entry ("Nonlinear solver tolerance", "1e-5",
                           Patterns::Double(0., 1.),
                           "A relative tolerance up to which the Nonlinear solver for the "
                           "phase field system will iterate.");

        prm.declare_entry ("Max nonlinear iterations", "10",
                           Patterns::Integer(1),
                           "The maximal number of nonlinear iterations to be performed "
                           "for solving the phase field system.");

        prm.declare_entry ("Max Newton line search iterations", "3",
                           Patterns::Integer(0),
                           "The maximum number of line search iterations allowed for the "
                           "phase field system. If the criterion is not reached after "
                           "this number of iterations, we apply the increment even though "
                           "it does not satisfy the necessary criteria and simply continue "
                           "with the next Newton iteration.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Phase field model");
    {
      // Initialize the geometric function
      const double l = prm.get_double("Length scale");
      double xi = numbers::signaling_nan<double>();
      double c0 = numbers::signaling_nan<double>();

      const std::string type = prm.get("Geometric function type");
      if (type == "AT2")
        {
          xi = 0.;
          c0 = 2.;
        }
      else if (type == "AT1")
        {
          xi = 1.;
          c0 = 2.666666666666666667;
        }
      else if (type == "CZM")
        {
          xi = 2.;
          c0 = numbers::PI;
        }
      else
        AssertThrow(false, ExcNotImplemented());

      geometric_function = std::make_unique<PhaseField::GeometricFunction>(l, xi, c0);

      const double p = prm.get_double("Degradation curvature parameter");

      // Get the critical energy release rate and the threshold crack driving force
      // from the material model
      const MaterialModel::PhaseFieldModel<dim> *phase_field_model 
        = dynamic_cast<const MaterialModel::PhaseFieldModel<dim>*>(&this->get_material_model());
      AssertThrow(phase_field_model != nullptr,
                  ExcMessage("The phase field method requires the material model to be derived from "
                             "MaterialModel::PhaseFieldModel."));
      const std::vector<double> &Gc = phase_field_model->get_critical_energy_release_rates();
      const std::vector<double> &Hc = phase_field_model->get_critical_crack_driving_forces();

      degradation_functions.clear();
      slip_rate_normalizers.clear();
      critical_energy_densities.clear();

      for (unsigned int j = 0; j < Gc.size(); ++j)
        {
          // Initialize the degradation function
          const double m = Gc[j] * geometric_function->first_derivative(0.) / (c0 * l * Hc[j]);
          degradation_functions.push_back(std::make_unique<PhaseField::DegradationFunction>(p, m));

          // Initialize the slip rate normalizer
          const std::pair<double, double> min_max = phase_field_model->get_phase_field_range();
          slip_rate_normalizers.push_back(
            std::make_unique<PhaseField::SlipRateNormalizer>(*geometric_function, 
                                                             *degradation_functions.back(), 
                                                             min_max.first,
                                                             min_max.second));

          // Compute the critical energy density
          critical_energy_densities.push_back(Gc[j] / (c0 * l));
        }

    }
    prm.leave_subsection();

    prm.enter_subsection("Solver parameters");
    {
      prm.enter_subsection("Phase field solver parameters");
      {
        solver_parameters.linear_solver_tolerance            = prm.get_double("Linear solver tolerance");
        solver_parameters.max_linear_solver_iterations       = prm.get_integer("Max linear solver iterations");
        solver_parameters.nonlinear_solver_tolerance         = prm.get_double("Nonlinear solver tolerance");
        solver_parameters.max_nonlinear_iterations           = prm.get_integer("Max nonlinear iterations");
        solver_parameters.max_newton_line_search_iterations  = prm.get_integer("Max Newton line search iterations");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }



  template <int dim>
  PhaseFieldHandler<dim>::PhaseFieldHandler(const Simulator<dim> &sim)
    : particle_manager(nullptr)
  {
    this->initialize_simulator(sim);

    this->get_signals().edit_finite_element_variables.connect(
      [&](std::vector<VariableDeclaration<dim>> &vars)
    {
      this->edit_finite_element_variables(vars);
    });
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::
  edit_finite_element_variables(std::vector<VariableDeclaration<dim>> &variables)
  {
    variables.push_back(VariableDeclaration<dim>("phase_field",
                                                 std::make_shared<FE_Q<dim>>(1), 
                                                 1, 
                                                 1));

    if (this->get_parameters().need_slip_rate)
      variables.push_back(VariableDeclaration<dim>("peak_phase_field",
                                                   std::make_shared<FE_Q<dim>>(1),
                                                   1,
                                                   1));
  }



  template <int dim>
  void PhaseFieldHandler<dim>::initialize()
  {
    // Find the particle manager handling the crack driving force
    AssertThrow(this->n_particle_managers() > 0,
                ExcMessage("The phase field method requires particles to be included in the model. "
                           "Please add 'particles' to the list of postprocessors."));

    for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
      {
        const Particle::Manager<dim> &manager = this->get_particle_manager(i);
        if (manager.get_property_manager().get_data_info().fieldname_exists("crack_driving_force"))
          {
            particle_manager = &manager;
            break;
          }
      }

    AssertThrow(particle_manager != nullptr, 
                ExcMessage("The phase field method requires one of the particle sets to include a "
                           "particle property named 'crack_driving_force'."));

    AssertThrow(particle_manager->particle_domains_requested(),
                ExcMessage("The phase field method requires the particle manager to generate "
                           "particle domains."));

    AssertThrow(particle_manager->get_particle_domain_handler().cpdi_data_requested(),
                ExcMessage("The phase field method requires the particle domain handler to "
                           "generate CPDI data."));

    if (this->get_parameters().need_slip_rate)
      {
        const auto &particle_data_info = particle_manager->get_property_manager().get_data_info();
        AssertThrow(particle_data_info.fieldname_exists("normal_direction") &&
                    particle_data_info.get_components_by_field_name("normal_direction") == dim,
                    ExcMessage("The phase field method requires a particle property named "
                               "'normal_direction', which should have " 
                               + Utilities::int_to_string(dim) + " components."));
      }

    const auto &advection_methods = this->get_parameters().compositional_field_methods;
    if (std::find(advection_methods.begin(), advection_methods.end(), 
                  Parameters<dim>::AdvectionFieldMethod::particles)
        != advection_methods.end())
      {
        AssertThrow(this->get_parameters().mapped_particle_properties.size() > 0,
                    ExcMessage("The phase field method requires a map between compositional fields and particle properties"));

        for (const auto &key_and_value : this->get_parameters().mapped_particle_properties)
          if (key_and_value.second.first == "crack_driving_force")
            AssertThrow(this->get_parameters().composition_descriptions[key_and_value.first].type == CompositionalFieldDescription::generic,
                        ExcMessage("If the particle property 'crack_driving_force' is associated with a compositional field, "
                                   "then its field type must be set to 'generic'."));
      }

    // Check the rationality of the parameters provided by the material model:
    // for each material composition, the second derivative of the degradation
    // function at d=0 must be positive, otherwise the nonlinear system would be
    // non-convex
    for (unsigned int j = 0; j < degradation_functions.size(); ++j)
      AssertThrow(degradation_functions[j]->second_derivative(0) > 0,
                  ExcMessage("The second derivative of the energetic degradation function for the "
                             + Utilities::int_to_string(j) + "-th material composition is not positive "
                             "at d = 0. In this case, the nonlinear system becomes non-convex, which "
                             "makes it extremely difficult to find a global minimum. Please change "
                             "the parameters (for example, the critical energy release rate, the "
                             "threshold driving force, the length scale, etc) to ensure that the "
                             "second derivative of the energetic degradation function is positive."));
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  crack_surface_density(const double          d,
                        const Tensor<1, dim> &grad_d) const
  {
    const double a  = geometric_function->value(d);
    const double l  = geometric_function->get_length_scale();
    const double c0 = geometric_function->get_normalization_factor();
    return (a / l + (grad_d * grad_d) * l) / c0;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  energetic_degradation(const std::vector<double> &volume_fractions,
                        const double               d) const
  {
    AssertDimension(volume_fractions.size(), degradation_functions.size());

    double g = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        g += degradation_functions[j]->value(d) * volume_fractions[j];

    return g;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  slip_rate_localization_factor(const std::vector<double> &volume_fractions,
                                const double               g,
                                const double               d_hat) const
  {
    AssertDimension(volume_fractions.size(), slip_rate_normalizers.size());
    AssertThrow(g > 0, ExcMessage("The degradation function must be positive."));
    
    const double h = 1. / g - 1.;

    double Ih = 0;
    for (unsigned int i = 0; i < volume_fractions.size(); ++i)
      if (volume_fractions[i] > 0)
        Ih += volume_fractions[i] * slip_rate_normalizers[i]->normalization_factor(d_hat);

    return h / Ih;
  }



  template <int dim>
  std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>>
  PhaseFieldHandler<dim>::get_phase_field_profiles(const double d_hat) const
  {
    std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>> profiles;
    for (const auto &degradation_function : degradation_functions)
      profiles.push_back(std::make_unique<PhaseField::PhaseFieldProfile>(*geometric_function, 
                                                                         *degradation_function,
                                                                         d_hat));

    return profiles;
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::
  assemble_phase_field_system(LinearAlgebra::BlockSparseMatrix &system_matrix,
                              LinearAlgebra::BlockVector       &system_rhs,
                              const LinearAlgebra::BlockVector &current_solution,
                              const bool assemble_system_jacobian) const
  {
    const Particles::ParticleHandler<dim> &particle_handler = particle_manager->get_particle_handler();
    const Particle::ParticleDomainHandler<dim> &particle_domain_handler = particle_manager->get_particle_domain_handler();

    // Initialize the corresponding block of the system matrix and the system rhs
    const unsigned int block_index = this->introspection().variable("phase_field").block_index;
    system_matrix.block(block_index, block_index) = 0;
    system_rhs.block(block_index) = 0;

    // We need to retrieve the crack driving force and chemical composition values
    // from particle properties
    const auto &particle_data_info = particle_manager->get_property_manager().get_data_info();
    const unsigned int H_property_index = particle_data_info.get_position_by_field_name("crack_driving_force");

    std::vector<unsigned int> C_property_indices;
    for (unsigned int index : this->introspection().chemical_composition_field_indices())
      C_property_indices.push_back(particle_data_info.get_position_by_field_name(
        this->get_parameters().mapped_particle_properties.find(index)->second.first));

    std::vector<double> chemical_composition_values(C_property_indices.size());

    // Vector storing the phase field DoF indices associated with a particle domain
    std::vector<types::global_dof_index> particle_dof_indices;

    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &particle : particle_handler.particles_in_cell(cell))
          {
            // Get access to the CPDI data
            const auto particle_domain = particle_domain_handler.get_particle_domain(particle.get_local_index());

            // Stuff for local assembly
            const unsigned int n_dofs = particle_domain.n_relevant_vertices();
            FullMatrix<double> particle_matrix(n_dofs, n_dofs);
            Vector<double> particle_rhs(n_dofs);
            particle_dof_indices.resize(n_dofs);

            small_vector<double>         weighting_function_values(n_dofs);
            small_vector<Tensor<1, dim>> weighting_function_gradients(n_dofs);

            double d = 0;
            Tensor<1, dim> grad_d;

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                // Collect the weighting function values and gradients
                weighting_function_values[i]    = particle_domain.weighting_function_value(i);
                weighting_function_gradients[i] = particle_domain.weighting_function_gradient(i);

                // Collect the DoF indices
                const unsigned int vertex_index = particle_domain.relevant_vertex_index(i);
                const types::global_dof_index dof_index = vertex_to_dof_indices[vertex_index];
                particle_dof_indices[i] = dof_index;

                // Compute the value and gradient of the phase field in this particle domain
                const double dof_value = current_solution[dof_index];
                d      += dof_value * weighting_function_values[i];
                grad_d += dof_value * weighting_function_gradients[i];
              }

            // Get the crack driving force
            const ArrayView<const double> particle_properties = particle.get_properties();
            const double H = particle_properties[H_property_index];

            // Compute the volume fractions
            for (unsigned int c = 0; c < chemical_composition_values.size(); ++c)
              chemical_composition_values[c] = particle_properties[C_property_indices[c]];
            const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_composition_fractions(chemical_composition_values);

            const double da_dd   = geometric_function->first_derivative(d);
            const double d2a_dd2 = geometric_function->second_derivative(d);
            const double l       = geometric_function->get_length_scale();

            double F = 0;
            double K = 0;
            double dK_dd = 0;
            for (unsigned int j = 0; j < volume_fractions.size(); ++j)
              if (volume_fractions[j] > 0)
                {
                  const double Gc_over_c0l = critical_energy_densities[j];
                  const double dg_dd   = degradation_functions[j]->first_derivative(d);
                  const double d2g_dd2 = degradation_functions[j]->second_derivative(d);

                  F += volume_fractions[j] * (2. * Gc_over_c0l * l * l);
                  K += volume_fractions[j] * (H * dg_dd + Gc_over_c0l * da_dd);
                  if (assemble_system_jacobian)
                    dK_dd += volume_fractions[j] * (H * d2g_dd2 + Gc_over_c0l * d2a_dd2);
                }

            const double V_p = particle_domain.volume();

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const double phi_ip               = weighting_function_values[i];
                const Tensor<1, dim> &grad_phi_ip = weighting_function_gradients[i];
                particle_rhs(i) -= (phi_ip * K + F * (grad_phi_ip * grad_d)) * V_p;

                if (assemble_system_jacobian)
                  {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                      {
                        const double phi_jp               = weighting_function_values[j];
                        const Tensor<1, dim> &grad_phi_jp = weighting_function_gradients[j];
                        particle_matrix(i, j) += (dK_dd * (phi_ip * phi_jp) + F * (grad_phi_ip * grad_phi_jp)) * V_p;
                      }
                  }
              }

            if (assemble_system_jacobian)
              this->get_current_constraints().distribute_local_to_global(particle_matrix,
                                                                         particle_rhs,
                                                                         particle_dof_indices,
                                                                         system_matrix,
                                                                         system_rhs);
            else
              this->get_current_constraints().distribute_local_to_global(particle_rhs,
                                                                         particle_dof_indices,
                                                                         system_rhs);
          }

    system_rhs.compress(VectorOperation::add);
    if (assemble_system_jacobian)
      system_matrix.compress(VectorOperation::add);
  }



  template <int dim>
  unsigned int
  PhaseFieldHandler<dim>::
  solve_phase_field_system(const LinearAlgebra::BlockSparseMatrix &system_matrix,
                           const LinearAlgebra::BlockVector       &system_rhs,
                           LinearAlgebra::BlockVector             &solution_vector) const
  {
    const auto &variable = this->introspection().variable("phase_field");
    const unsigned int component_index = variable.first_component_index;
    const unsigned int block_index     = variable.block_index;

    // Set the preconditioner
    LinearAlgebra::PreconditionAMG preconditioner;
    LinearAlgebra::PreconditionAMG::AdditionalData amg_data;

    std::vector<bool> phase_field_component_mask(this->introspection().n_components, false);
    phase_field_component_mask[component_index] = true;
#if DEAL_II_VERSION_GTE(9,7,0)
    amg_data.constant_modes = DoFTools::extract_constant_modes(
                                this->get_dof_handler(),
                                ComponentMask(phase_field_component_mask));
#else
    std::vector<std::vector<bool>> constant_modes;
    DoFTools::extract_constant_modes(
      this->get_dof_handler(),
      ComponentMask(phase_field_component_mask),
      constant_modes);
    amg_data.constant_modes = constant_modes;
#endif

    amg_data.elliptic = true;
    amg_data.higher_order_elements = false;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    preconditioner.initialize(system_matrix.block(block_index, block_index), amg_data);

    this->get_current_constraints().set_zero(solution_vector);

    SolverControl solver_control(solver_parameters.max_linear_solver_iterations,
                                 solver_parameters.linear_solver_tolerance * system_rhs.block(block_index).l2_norm());

    SolverCG<LinearAlgebra::Vector> solver(solver_control);

    try
      {
        solver.solve(system_matrix.block(block_index, block_index),
                     solution_vector.block(block_index),
                     system_rhs.block(block_index),
                     preconditioner);
      }
    catch (const std::exception &exc)
      {
        // if the solver fails, report the error from processor 0 with some additional
        // information about its location, and throw a quiet exception on all other
        // processors
        Utilities::throw_linear_solver_failure_exception("iterative solver for phase field",
                                                         "PhaseFieldHandler::solve_phase_field_system",
                                                         std::vector<SolverControl> {solver_control},
                                                         exc,
                                                         this->get_mpi_communicator());
      }

    this->get_current_constraints().distribute(solution_vector);

    return solver_control.last_step();
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::evolve_phase_field(LinearAlgebra::BlockSparseMatrix &system_matrix,
                                             LinearAlgebra::BlockVector       &system_rhs,
                                             LinearAlgebra::BlockVector       &solution)
  {
    const unsigned int block_index = this->introspection().variable("phase_field").block_index;

    // Compute the initial residual
    assemble_phase_field_system(system_matrix, system_rhs, solution, false);
    const double initial_residual = system_rhs.block(block_index).l2_norm();

    // Skip solving the phase field system if the initial residual is too small
    if (initial_residual < 1e-50)
      {
        this->get_pcout() << "   Skipping phase field solve because the nonlinear residual is 0." << std::endl;
        return;
      }

    this->get_pcout() << "   Solving phase field system:" << std::endl;

    // Create a test solution vector for computing the system residual in line search
    LinearAlgebra::BlockVector test_solution(this->introspection().index_sets.system_partitioning,
                                             this->introspection().index_sets.system_relevant_partitioning,
                                             this->get_mpi_communicator());

    // Create distributed vectors for the Newton update and the test solution
    LinearAlgebra::BlockVector newton_update(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
    LinearAlgebra::BlockVector distributed_solution(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());

    unsigned int nonlinear_iteration = 0;
    double relative_residual = 1;

    SolverControl nonlinear_solver_control(solver_parameters.max_nonlinear_iterations,
                                           solver_parameters.nonlinear_solver_tolerance);
    do
      {
        // Assemble and solve for the Newton update
        assemble_phase_field_system(system_matrix, system_rhs, solution, true);

        newton_update.block(block_index) = 0;
        const unsigned int linear_solver_iterations = solve_phase_field_system(system_matrix,
                                                                               system_rhs,
                                                                               newton_update);

        // Perform line search
        const double residual_old = system_rhs.block(block_index).l2_norm();
        const double alpha = 1e-4;

        double step_length = 1;
        double residual = numbers::signaling_nan<double>();
        unsigned int line_search_iteration = 0;

        while (line_search_iteration <= solver_parameters.max_newton_line_search_iterations)
          {
            distributed_solution.block(block_index) = solution.block(block_index);
            distributed_solution.block(block_index).sadd(1.0, step_length, newton_update.block(block_index));
            test_solution.block(block_index) = distributed_solution.block(block_index);

            ++line_search_iteration;

            assemble_phase_field_system(system_matrix, system_rhs, test_solution, false);
            residual = system_rhs.block(block_index).l2_norm();
            if (residual < (1. - alpha * step_length) * residual_old)
              break;

            step_length *= 0.5;
          }

        // Update the solution vector
        solution.block(block_index) = test_solution.block(block_index);

        relative_residual = residual / initial_residual;

        this->get_pcout() << "      Nonlinear iteration " 
                          << std::setw(Utilities::int_to_string(solver_parameters.max_nonlinear_iterations).size()) 
                          << nonlinear_iteration
                          << ": linear solver iterations = " 
                          << std::setw(Utilities::int_to_string(solver_parameters.max_linear_solver_iterations).size()) 
                          << linear_solver_iterations
                          << ", line search iterations = " 
                          << std::setw(Utilities::int_to_string(solver_parameters.max_newton_line_search_iterations).size()) 
                          << line_search_iteration - 1
                          << ", relative residual = " << std::scientific << std::setprecision(3) 
                          << relative_residual
                          << std::endl;

        ++nonlinear_iteration;
      }
    while (nonlinear_solver_control.check(nonlinear_iteration, relative_residual) == SolverControl::iterate);
  }



  template <int dim>
  void
  PhaseFieldHandler<dim>::
  make_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &sp)
  {
    // Create the vertex-to-cell map
    GridTools::Cache<dim> grid_cache(this->get_triangulation(), this->get_mapping());
    const auto &vertex_to_cell_map = grid_cache.get_vertex_to_cell_map();

    const AffineConstraints<double> &current_constraints = this->get_current_constraints();

    const unsigned int component_index = this->introspection().variable("phase_field").first_component_index;

    // Loop over the locally owned cells and add the nonzero entries of CPDI system
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // All the phase-field DoFs in the one-layer-patch around a cell are
          // possible to be coupled
          std::set<typename Triangulation<dim>::active_cell_iterator> neighboring_cells;
          for (const unsigned int v : cell->vertex_indices())
            {
              const unsigned int vertex_index = cell->vertex_index(v);
              neighboring_cells.insert(vertex_to_cell_map[vertex_index].begin(),
                                       vertex_to_cell_map[vertex_index].end());
            }

          // Since the CPDI method requires the fields to be discretized by FE_Q(1) element,
          // we only need to loop over the vertices and extract the DoFs corresponding to
          // the first CPDI field
          std::set<types::global_dof_index> coupled_dofs;
          for (const auto &neighbor : neighboring_cells)
            {
              typename DoFHandler<dim>::active_cell_iterator dof_cell(&this->get_triangulation(),
                                                                      neighbor->level(),
                                                                      neighbor->index(),
                                                                      &this->get_dof_handler());

              for (const unsigned int v : dof_cell->vertex_indices())
                coupled_dofs.insert(dof_cell->vertex_dof_index(v, component_index));
            }

          current_constraints.add_entries_local_to_global(std::vector<types::global_dof_index>(coupled_dofs.begin(),
                                                                                               coupled_dofs.end()),
                                                          sp, false);
        }
    
    // Update the vertex-to-dof map
    vertex_to_dof_indices.clear();
    vertex_to_dof_indices.resize(this->get_triangulation().n_vertices(), numbers::invalid_dof_index);
    for (const auto &cell : this->get_dof_handler().active_cell_iterators())
      if (!cell->is_artificial())
        for (const unsigned int v : cell->vertex_indices())
          {
            const unsigned int vertex_index = cell->vertex_index(v);
            if (vertex_to_dof_indices[vertex_index] == numbers::invalid_dof_index)
              vertex_to_dof_indices[vertex_index] = cell->vertex_dof_index(v, component_index);
          }
  }



  template <int dim>
  const Particle::Manager<dim> &
  PhaseFieldHandler<dim>::get_associated_particle_manager() const
  {
    Assert(particle_manager != nullptr, 
           ExcMessage("The pointer to the associated particle manager has not been initiated."));

    return *particle_manager;
  }
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  namespace MaterialModel \
  { \
    template class PhaseFieldModel<dim>; \
  } \
  namespace PhaseField \
  { \
    template struct SolverParameters<dim>; \
  } \
  template class PhaseFieldHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
