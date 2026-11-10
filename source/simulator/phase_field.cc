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
#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/particle/manager.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iomanip>
#include <fstream>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    std::pair<double, double>
    PhaseFieldModel<dim>::get_phase_field_range() const
    {
      return std::make_pair(0.0, 1.0);
    }



    template <int dim>
    double
    PhaseFieldModel<dim>::get_phase_field_activation_threshold() const
    {
      return 0.01;
    }



    template <int dim>
    double
    PhaseFieldModel<dim>::get_phase_field_upper_admissibility_threshold() const
    {
      return 1.0;
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
    GeometricFunction::value(const double phi) const
    {
      return phi * (xi + (1.0 - xi) * phi);
    }



    double
    GeometricFunction::first_derivative(const double phi) const
    {
      return xi + 2.0 * (1.0 - xi) * phi;
    }



    double
    GeometricFunction::second_derivative(const double /*phi*/) const
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
    DegradationFunction::value(const double phi) const
    {
      const double numerator   = (1. - phi) * (1. - phi);
      const double denominator = numerator + m * phi * (1. + p * phi);
      return numerator / denominator;
    }



    double
    DegradationFunction::first_derivative(const double phi) const
    {
      const double numerator   = -m * (1. - phi) * (1. + phi + 2. * p * phi);
      const double denominator = Utilities::fixed_power<2, double>((1. - phi) * (1. - phi) + m * phi * (1. + p * phi));
      return numerator / denominator;
    }



    double
    DegradationFunction::second_derivative(const double phi) const
    {
      const double A       = (1. - phi) * (1. - phi) + m * phi * (1. + p * phi);
      const double dA_dphi = 2. * (phi - 1.) + m * (1. + 2. * p * phi);
      const double B       = (1. - phi) * (1. + phi + 2. * p * phi);
      const double dB_dphi = 2. * (p - phi - 2. * p * phi);
      const double numerator   = m * (2. * B * dA_dphi - dB_dphi * A);
      const double denominator = A * A * A;
      return numerator / denominator;
    }



    bool
    DegradationFunction::is_in_domain(const double phi) const
    {
      // Newton undershoots may be evaluated without clipping, but never across
      // a pole. For negative phi, exclude the second branch beyond both roots
      // of A=(1+m*p)phi^2+(m-2)phi+1, even when A is positive there again.
      const double A = (1.0-phi)*(1.0-phi) + m*phi*(1.0+p*phi);
      const double dA = 2.0*(1.0+m*p)*phi + m-2.0;
      const double discriminant = m*(m-4.0*(1.0+p));
      return std::isfinite(phi) && phi <= 1.0 && A > 0.0
             && (phi >= 0.0 || dA >= 0.0 || discriminant < 0.0);
    }



    /*------------------------ PhaseFieldProfile ---------------------------*/

    PhaseFieldProfile::
    PhaseFieldProfile(const GeometricFunction   &a_func,
                      const DegradationFunction &g_func,
                      const double               phi_hat,
                      const unsigned int         n_points)
      : N(n_points)
      , coordinate_values(N)
      , phase_field_values(N)
    {
      AssertThrow(phi_hat > 0 && phi_hat < 1,
                  ExcMessage("The core phase-field must be in the range of (0, 1)."));

      const double a_hat = a_func.value(phi_hat);
      const double g_hat = g_func.value(phi_hat);
      const double h_hat = 1. / g_hat - 1.;

      const double l = a_func.get_length_scale();

      std::vector<double> integrand(N);

      // Use the change of variables:
      //    d = phi_hat cos^2(theta), theta in [0, pi/2]
      const double dtheta = numbers::PI_2 / (N - 1);
      for (unsigned int i = 0; i < N; ++i)
        {
          const double theta = i * dtheta;
          const double cos_theta = std::cos(theta);
          const double phi = phi_hat * cos_theta * cos_theta;

          phase_field_values[i] = phi;
          if (i == 0 || i == N - 1)
            continue;

          const double a = a_func.value(phi);
          const double g = g_func.value(phi);
          const double h = 1. / g - 1.;
          
          const double D = h_hat * a - a_hat * h;
          Assert(D > 0, ExcInternalError());

          // dzeta / dtheta = 2l * phi_hat * sin(theta) * cos(theta) * sqrt(h_hat / D)
          const double sin_theta = std::sin(theta);
          integrand[i] = 2. * l * phi_hat * sin_theta * cos_theta * std::sqrt(h_hat / D);
        }

      // Endpoint limits:
      // At theta = 0, dzeta/dtheta ~ 2l * sqrt(phi_hat * h_hat / -D'(phi_hat))
      const double a_hat_prime = a_func.first_derivative(phi_hat);
      const double g_hat_prime = g_func.first_derivative(phi_hat);
      const double phi_hat_prime = h_hat * a_hat_prime + a_hat * g_hat_prime / (g_hat * g_hat);
      integrand[0] = 2. * l * std::sqrt(phi_hat * h_hat / -phi_hat_prime);

      // At theta = pi/2, dzeta/dtheta ~ 2l * sqrt(phi_hat * h_hat / D'(0))
      const double a0_prime = a_func.first_derivative(0);
      const double g0_prime = g_func.first_derivative(0);
      const double D0_prime = h_hat * a0_prime + a_hat * g0_prime; // g(0) = 1
      integrand[N-1] = 2. * l * std::sqrt(phi_hat * h_hat / D0_prime);

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
      
      const double value = (1. - xi) * phase_field_values[idx-1] + xi * phase_field_values[idx];
      // A convex interpolation can overshoot its endpoints by one ulp near
      // the peak. Preserve the tabulated bounds required by stationary H.
      return std::clamp(value,
                        std::min(phase_field_values[idx-1], phase_field_values[idx]),
                        std::max(phase_field_values[idx-1], phase_field_values[idx]));
    }



    /*------------------------ SlipRateNormalizer ---------------------------*/

    SlipRateNormalizer::
    SlipRateNormalizer(const GeometricFunction   &a_func,
                       const DegradationFunction &g_func,
                       const double               phi_min_,
                       const double               phi_max_)
      : phi_min(phi_min_)
      , phi_max(phi_max_)
    {
      AssertThrow(std::isfinite(phi_min) && std::isfinite(phi_max)
                  && phi_min > 0.0 && phi_min < phi_max && phi_max < 1.0,
                  ExcMessage("The legacy phase-field slip-rate normalizer requires "
                             "finite bounds satisfying 0 < phi_min < phi_max < 1. "
                             "A phase-field material model that uses this normalizer "
                             "must provide a strict interior upper admissibility threshold."));

      // Distribute the sample points on a uniform logit grid
      logit_phi_hat[0]   = std::log(phi_min / (1. - phi_min));
      logit_phi_hat[M-1] = std::log(phi_max / (1. - phi_max));

      const double dx = (logit_phi_hat[M-1] - logit_phi_hat[0]) / (M - 1);
      for (unsigned int i = 1; i < M - 1; ++i)
        logit_phi_hat[i] = logit_phi_hat[0] + i * dx;

      for (unsigned int i = 0; i < M; ++i)
        {
          const double phi_hat = 1. / (1. + std::exp(-logit_phi_hat[i]));

          // Compute the phase-field profile with phi_hat
          const PhaseFieldProfile profile(a_func, g_func, phi_hat, N);

          const std::vector<double> &zeta = profile.get_coordinate_values();
          const std::vector<double> &phi  = profile.get_phase_field_values();

          // Integrate h(d) over the profile using the trapezoidal rule
          std::vector<double> h(N);
          for (unsigned int j = 0; j < N; ++j)
            {
              const double g = g_func.value(phi[j]);
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
    SlipRateNormalizer::normalization_factor(const double phi_hat_raw) const
    {
      AssertThrow(phi_hat_raw > 0 && phi_hat_raw < 1,
                  ExcMessage("The core phase-field must be in the range of (0, 1)."));

      const double phi_hat = std::clamp(phi_hat_raw, phi_min, phi_max);

      const double x = std::log(phi_hat / (1. - phi_hat));

      // Get the index of the first element of the coordinate array that is
      // greater than or equal to x
      const unsigned int idx = std::lower_bound(logit_phi_hat.begin(), logit_phi_hat.end(), x)
                               - logit_phi_hat.begin();

      if (idx == 0)
        return std::exp(log_Ih[0]);

      const double xi = std::max(std::min((x - logit_phi_hat[idx-1]) /
                                          (logit_phi_hat[idx] - logit_phi_hat[idx-1]),
                                          1.),
                                 0.);

      return std::exp((1. - xi) * log_Ih[idx-1] + xi * log_Ih[idx]);
    }
  }



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
                        "Type of the geometric function $\\alpha(\\phi)$. The options are:\n"
                        "AT1: $\\alpha(\\phi) = \\phi$;\n"
                        "AT2: $\\alpha(\\phi) = \\phi^2;\n$"
                        "CZM: $\\alpha(\\phi) = 2\\phi - \\phi^2$.");

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
        prm.declare_entry("Use bound constraints", "false", Patterns::Bool(),
                          "Solve the phase variational inequality on the material phase range. "
                          "This imposes no previous-phase lower bound and does not change H. "
                          "The default retains the equality solver.");
        prm.declare_entry("Enforce phase irreversibility", "false", Patterns::Bool(),
                          "With bound constraints, use an immutable copy of the last accepted phase "
                          "as the lower bound during physical evolution. Initial relaxation remains "
                          "on the material phase range; H is unchanged.");
        prm.declare_entry("Linear solver tolerance", "1e-8",
                          Patterns::Double(0., 1.),
                          "A relative tolerance up to which the linearized phase field system "
                          "in each nonlinear step should be solved.");
        prm.declare_entry("Max linear solver iterations", "1000",
                          Patterns::Integer(0),
                          "The maximum number of iteration steps for solving the linearized "
                          "phase field system. If the linear solver fails to converge to the "
                          "relative tolerance when reaching the maximum step number, the "
                          "program will be terminated with an error message.");

        prm.declare_entry("Nonlinear solver tolerance", "1e-5",
                          Patterns::Double(0., 1.),
                          "A relative tolerance up to which the Nonlinear solver for the "
                          "phase field system will iterate. The absolute residual target is "
                          "the larger of this tolerance times the initial residual and an "
                          "internal roundoff allowance computed from the initial phase "
                          "residual-term magnitudes. The allowance does not depend on "
                          "nonlinear stagnation or increase the iteration budget.");

        prm.declare_entry("Max nonlinear iterations", "10",
                          Patterns::Integer(1),
                          "The maximal number of nonlinear iterations to be performed "
                          "for solving the phase field system.");

        prm.declare_entry("Max Newton line search iterations", "3",
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
    ++degradation_revision;
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
          const double minimum_phase_field =
            phase_field_model->get_phase_field_activation_threshold();
          const double maximum_phase_field =
            phase_field_model->get_phase_field_upper_admissibility_threshold();
          slip_rate_normalizers.push_back(
            std::make_unique<PhaseField::SlipRateNormalizer>(*geometric_function, 
                                                             *degradation_functions.back(), 
                                                             minimum_phase_field,
                                                             maximum_phase_field));

          // Compute the critical energy density
          critical_energy_densities.push_back(Gc[j] / (c0 * l));
        }
    }
    prm.leave_subsection();

    // Parse the solver parameters
    prm.enter_subsection("Solver parameters");
    {
      prm.enter_subsection("Phase field solver parameters");
      {
        solver_parameters.use_bound_constraints = prm.get_bool("Use bound constraints");
        solver_parameters.enforce_irreversibility = prm.get_bool("Enforce phase irreversibility");
        AssertThrow(!solver_parameters.enforce_irreversibility || solver_parameters.use_bound_constraints,
                    ExcMessage("Phase irreversibility requires Use bound constraints."));
        AssertThrow(!solver_parameters.enforce_irreversibility
                    || this->get_parameters().nonlinear_solver_failure_strategy
                       != Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep,
                    ExcMessage("Phase irreversibility cannot accept a failed physical step: "
                               "do not use continue with next timestep."));
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

    this->get_signals().pre_refinement_store_user_data.connect(
      [this](auto &) { native_spatial_operator.reset(); native_fracture_matrix.reset(); });
    this->get_signals().post_mesh_deformation.connect(
      [this](const auto &) { native_spatial_operator.reset(); native_fracture_matrix.reset(); });

  }


  template <int dim>
  void PhaseFieldHandler<dim>::initialize()
  {
    const auto *fault = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(&this->get_material_model());
    const bool mesh_data = fault && fault->uses_homogeneous_mesh_data();
    const bool registered = this->introspection().composition_type_exists(CompositionalFieldDescription::phase_field);
    AssertThrow(mesh_data && registered && uses_dg_phase() && this->get_parameters().use_bernstein_phase_basis,
                ExcMessage("The active phase model requires mesh data and the native DGQ2 Bernstein composition."));
    for (unsigned int i = 0; i < this->n_particle_managers(); ++i)
      {
        auto &manager = this->get_particle_manager(i);
        if (manager.get_property_manager().get_data_info().fieldname_exists("crack_driving_force")
            && (!mesh_data || manager.get_property_manager().get_data_info().fieldname_exists("maxwell stress")))
          {
            particle_manager = &manager;
            break;
          }
      }
    if (particle_manager)
      AssertThrow(!particle_manager->particle_domains_requested(),
                  ExcMessage("Native diagnostic particles must not own CPDI domains."));

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

    // Initialize the grid cache
    grid_cache = std::make_unique<GridTools::Cache<dim>>(this->get_triangulation(), this->get_mapping());
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  crack_surface_density(const double          phi,
                        const Tensor<1, dim> &grad_phi) const
  {
    const double a  = geometric_function->value(phi);
    const double l  = geometric_function->get_length_scale();
    const double c0 = geometric_function->get_normalization_factor();
    return (a / l + (grad_phi * grad_phi) * l) / c0;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  energetic_degradation(const std::vector<double> &volume_fractions,
                        const double               phi) const
  {
    AssertDimension(volume_fractions.size(), degradation_functions.size());

    double g = 0.0;
    for (unsigned int j = 0; j < volume_fractions.size(); ++j)
      if (volume_fractions[j] > 0.0)
        g += degradation_functions[j]->value(phi) * volume_fractions[j];

    return g;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  slip_rate_localization_factor(const std::vector<double> &volume_fractions,
                                const double               g,
                                const double               phi_hat) const
  {
    AssertDimension(volume_fractions.size(), slip_rate_normalizers.size());
    AssertThrow(g > 0, ExcMessage("The degradation function must be positive."));
    
    const double h = 1. / g - 1.;

    double Ih = 0;
    for (unsigned int i = 0; i < volume_fractions.size(); ++i)
      if (volume_fractions[i] > 0)
        Ih += volume_fractions[i] * slip_rate_normalizers[i]->normalization_factor(phi_hat);

    return h / Ih;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::
  stationary_crack_driving_force(const std::vector<double> &volume_fractions,
                                 const double               phi,
                                 const double               phi_hat) const
  {
    AssertDimension(volume_fractions.size(), degradation_functions.size());
    AssertThrow(phi <= phi_hat && phi_hat < 1, 
                ExcMessage("Invalid input parameters for function PhaseFieldHandler::crack_driving_force_of_stationary_profile: "
                           "phase_field = " + Utilities::to_string(phi) + "; core_phase_field = " + Utilities::to_string(phi_hat)));

    double g = 0, g_hat = 0;
    double Gc_over_c0l = 0;
    for (unsigned int i = 0; i < volume_fractions.size(); ++i)
      if (volume_fractions[i] > 0)
        {
          g           += volume_fractions[i] * degradation_functions[i]->value(phi);
          g_hat       += volume_fractions[i] * degradation_functions[i]->value(phi_hat);
          Gc_over_c0l += volume_fractions[i] * critical_energy_densities[i];
        }

    const double h_hat = 1. / g_hat - 1.;
    const double a_hat = geometric_function->value(phi_hat);

    return Gc_over_c0l * a_hat / (h_hat * g * g);
  }



  template <int dim>
  std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>>
  PhaseFieldHandler<dim>::get_phase_field_profiles(const double phi_hat) const
  {
    std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>> profiles;
    for (const auto &degradation_function : degradation_functions)
      profiles.push_back(std::make_unique<PhaseField::PhaseFieldProfile>(*geometric_function, 
                                                                         *degradation_function,
                                                                         phi_hat));

    return profiles;
  }



  template <int dim>
  typename PhaseFieldHandler<dim>::CellFractureEnergy
  PhaseFieldHandler<dim>::evaluate_cell_fracture(
    const LinearAlgebra::BlockVector &trial,const bool tangent) const
  {
    return evaluate_cell_fracture(trial,tangent,nullptr);
  }

  template <int dim>
  typename PhaseFieldHandler<dim>::CellFractureEnergy
  PhaseFieldHandler<dim>::evaluate_cell_fracture(
    const LinearAlgebra::BlockVector &trial,const bool tangent,
    std::map<std::pair<unsigned int,unsigned int>,double> *sparse_hessian) const
  {
    return evaluate_dg_fracture(trial,tangent,sparse_hessian);
  }


  template <int dim>
  void
  PhaseFieldHandler<dim>::evolve_phase_field(LinearAlgebra::BlockSparseMatrix &,
                                             LinearAlgebra::BlockVector       &,
                                             LinearAlgebra::BlockVector       &solution)
  {
    const unsigned int block_index = this->introspection().phase_field().block_index;

    const auto *history_material = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(&this->get_material_model());
    if (history_material && history_material->uses_transported_mature_history())
      {
        // Growth is disabled. The physical obstacle and profile are one
        // material field, staged with the tensor and surface histories.
        if (this->get_timestep_number()==0 && !history_material->has_fe_strain_history())
          history_material->initialize_fe_shared_phase(solution);
        else
          {
            AssertThrow(history_material->has_fe_strain_history(),
                        ExcMessage("Missing transported mature phase anchor."));
            solution.block(block_index)=history_material->accepted_fe_strain_state().block(block_index);
          }
        return;
      }
    AssertThrow(false, ExcMessage("Phase evolution is owned by the integrated bulk-tip solver or mature material transaction."));
  }


  template <int dim>
  void
  PhaseFieldHandler<dim>::
  make_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &sp)
  {
    Table<2,DoFTools::Coupling> coupling(this->introspection().n_components,
                                       this->introspection().n_components);
    coupling.fill(DoFTools::none);
    const auto c = this->introspection().phase_field().first_component_index;
    coupling[c][c] = DoFTools::always;
    DoFTools::make_flux_sparsity_pattern(this->get_dof_handler(), sp,
                                        this->get_current_constraints(), false,
                                        coupling, coupling,
                                        Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
  }


  template <int dim>
  const Particle::Manager<dim> &
  PhaseFieldHandler<dim>::get_associated_particle_manager() const
  {
    Assert(particle_manager != nullptr, 
           ExcMessage("The pointer to the associated particle manager has not been initiated."));

    return *particle_manager;
  }


  template <int dim>
  Particle::Manager<dim> &
  PhaseFieldHandler<dim>::get_associated_particle_manager()
  {
    Assert(particle_manager != nullptr,
           ExcMessage("The pointer to the associated particle manager has not been initiated."));

    return *particle_manager;
  }



  template <int dim>
  const GridTools::Cache<dim> &
  PhaseFieldHandler<dim>::get_grid_cache() const
  {
    Assert(grid_cache != nullptr,
           ExcMessage("The pointer to the grid cache has not been initiated."));

    return *grid_cache;
  }



  template <int dim>
  double
  PhaseFieldHandler<dim>::get_length_scale() const
  {
    Assert(geometric_function != nullptr, ExcInternalError());
    return geometric_function->get_length_scale();
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
  template class PhaseFieldHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
