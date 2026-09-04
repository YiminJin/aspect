/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "phase_field_fault_test_access.h"

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace
    {
      void
      assert_close(const double actual,
                   const double expected,
                   const double tolerance,
                   const std::string &quantity)
      {
        const double scale = std::max(1.0, std::abs(expected));
        AssertThrow(std::abs(actual - expected) <= tolerance * scale,
                    ExcMessage(quantity + " differs from its independent reference value."));
      }
    }



    template <int dim>
    class VerifyFaultFriction : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string>
        execute(TableHandler &) override
        {
          const auto *phase_field_fault =
            dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(
              &this->get_material_model());
          AssertThrow(phase_field_fault != nullptr,
                      ExcMessage("The fault-friction test requires the phase field fault material model."));

          const auto &friction =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::
            fault_friction(*phase_field_fault);

          if (friction.has_state_variable())
            verify_rate_state(friction);
          else
            verify_rate_dependent(friction);

          return {"Fault friction:", "verified"};
        }

      private:
        static void
        verify_rate_dependent(
          const MaterialModel::Rheology::FaultFriction<dim> &friction)
        {
          const std::vector<double> fractions = {0.2, 0.3, 0.5};
          const double mu_s = 0.2 * 0.6 + 0.3 * 0.7 + 0.5 * 0.8;
          const double mu_d = 0.2 * 0.6 + 0.3 * 0.4 + 0.5 * 0.5;
          const double Vc = 0.2 * 1.e-6 + 0.3 * 2.e-6 + 0.5 * 4.e-6;

          const auto coefficient = [mu_s, mu_d, Vc](const double V)
          {
            return mu_d + (mu_s - mu_d) * Vc / (Vc + V);
          };
          const auto derivative = [mu_s, mu_d, Vc](const double V)
          {
            return -(mu_s - mu_d) * Vc / ((Vc + V) * (Vc + V));
          };

          for (const double V : {1.e-10, Vc, 1.e-2})
            assert_close(friction.friction_coefficient(fractions, V),
                         coefficient(V), 1.e-13,
                         "Rate-dependent friction coefficient");

          assert_close(friction.friction_coefficient_derivative_wrt_slip_rate(fractions, Vc),
                       derivative(Vc), 1.e-13,
                       "Rate-dependent analytic derivative");

          const double dV = 1.e-5 * Vc;
          const double finite_difference =
            (friction.friction_coefficient(fractions, Vc + dV)
             - friction.friction_coefficient(fractions, Vc - dV)) / (2.0 * dV);
          assert_close(friction.friction_coefficient_derivative_wrt_slip_rate(fractions, Vc),
                       finite_difference, 1.e-9,
                       "Rate-dependent finite-difference derivative");

          const std::vector<double> background = {1.0, 0.0, 0.0};
          assert_close(friction.friction_coefficient(background, 1.e-10),
                       0.6, 1.e-13, "Rate-dependent zero-weakening coefficient");
          assert_close(friction.friction_coefficient_derivative_wrt_slip_rate(background, 1.e-10),
                       0.0, 1.e-13, "Rate-dependent zero-weakening derivative");

          AssertThrow(friction.compute_time_step(fractions, Vc, 0.5, false)
                      == std::numeric_limits<double>::max(),
                      ExcMessage("Stateless fault friction imposed a timestep restriction."));
          AssertThrow(friction.compute_time_step(fractions, Vc, 0.5, true)
                      == std::numeric_limits<double>::max(),
                      ExcMessage("Stateless fault friction imposed an operator-split timestep restriction."));

          bool rejected_stateful_interface = false;
          try
            {
              friction.friction_coefficient(fractions, Vc, 1.0);
            }
          catch (const std::exception &)
            {
              rejected_stateful_interface = true;
            }
          AssertThrow(rejected_stateful_interface,
                      ExcMessage("The stateless law accepted the stateful coefficient interface."));

          bool rejected_stateful_derivative = false;
          try
            {
              friction.friction_coefficient_derivative_wrt_slip_rate(
                fractions, Vc, 1.0);
            }
          catch (const std::exception &)
            {
              rejected_stateful_derivative = true;
            }
          AssertThrow(rejected_stateful_derivative,
                      ExcMessage("The stateless law accepted the stateful derivative interface."));

          bool rejected_state_update = false;
          try
            {
              friction.update_state(Vc, 1.0, 1.0);
            }
          catch (const std::exception &)
            {
              rejected_state_update = true;
            }
          AssertThrow(rejected_state_update,
                      ExcMessage("The stateless law accepted a state update."));

          bool rejected_stateful_getter = false;
          try
            {
              friction.get_reference_slip_rate();
            }
          catch (const std::exception &)
            {
              rejected_stateful_getter = true;
            }
          AssertThrow(rejected_stateful_getter,
                      ExcMessage("The stateless law exposed a rate-and-state parameter."));
        }



        static void
        verify_rate_state(
          const MaterialModel::Rheology::FaultFriction<dim> &friction)
        {
          const std::vector<double> fractions = {0.2, 0.3, 0.5};
          const double mu0 = 0.2 * 0.6 + 0.3 * 0.7 + 0.5 * 0.8;
          const double a = 0.2 * 0.02 + 0.3 * 0.03 + 0.5 * 0.04;
          const double b = 0.2 * 0.04 + 0.3 * 0.05 + 0.5 * 0.06;
          const double V0 = 1.e-6;
          const double Dc = 0.04;
          const double V = 2.e-6;
          const double theta = 2.e4;

          const double Z = V / (2.0 * V0)
                           * std::exp((mu0 + b * std::log(theta * V0 / Dc)) / a);
          const double expected_mu = a * std::asinh(Z);
          double expected_derivative = a / V;
          if (Z < 1.e6)
            expected_derivative *= Z / std::sqrt(1.0 + Z * Z);

          assert_close(friction.friction_coefficient(fractions, V, theta),
                       expected_mu, 1.e-13,
                       "Rate-and-state friction coefficient");
          assert_close(friction.friction_coefficient_derivative_wrt_slip_rate(
                         fractions, V, theta),
                       expected_derivative, 1.e-13,
                       "Rate-and-state analytic derivative");

          const double dV = 1.e-5 * V;
          const double finite_difference =
            (friction.friction_coefficient(fractions, V + dV, theta)
             - friction.friction_coefficient(fractions, V - dV, theta)) / (2.0 * dV);
          assert_close(friction.friction_coefficient_derivative_wrt_slip_rate(
                         fractions, V, theta),
                       finite_difference, 1.e-8,
                       "Rate-and-state finite-difference derivative");

          const double dt = 5.0;
          const double x = V * dt / Dc;
          const double expected_state =
            -Dc / V * std::expm1(-x) + theta * std::exp(-x);
          assert_close(friction.update_state(V, theta, dt),
                       expected_state, 1.e-13,
                       "Rate-and-state aging update");

          const double cfl = 0.5;
          assert_close(friction.compute_time_step(fractions, V, cfl, false),
                       cfl * Dc / V, 1.e-13,
                       "Rate-and-state timestep");
          assert_close(friction.compute_time_step(fractions, V, cfl, true),
                       cfl * a * Dc / (b * V), 1.e-13,
                       "Rate-and-state operator-split timestep");

          bool rejected_stateless_interface = false;
          try
            {
              friction.friction_coefficient(fractions, V);
            }
          catch (const std::exception &)
            {
              rejected_stateless_interface = true;
            }
          AssertThrow(rejected_stateless_interface,
                      ExcMessage("The rate-and-state law accepted the stateless interface."));

          bool rejected_stateless_derivative = false;
          try
            {
              friction.friction_coefficient_derivative_wrt_slip_rate(fractions, V);
            }
          catch (const std::exception &)
            {
              rejected_stateless_derivative = true;
            }
          AssertThrow(rejected_stateless_derivative,
                      ExcMessage("The rate-and-state law accepted the stateless derivative interface."));
        }
    };



    ASPECT_REGISTER_POSTPROCESSOR(VerifyFaultFriction,
                                  "verify fault friction",
                                  "Verify the Stage E generic fault-friction laws.")
  }
}
