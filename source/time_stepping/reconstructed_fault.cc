/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/time_stepping/reconstructed_fault.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>

namespace aspect
{
  namespace TimeStepping
  {
    template <int dim>
    void
    ReconstructedFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time stepping");
      prm.enter_subsection("Reconstructed fault time step");
      prm.declare_entry("Maximum logarithmic state change",
                        Utilities::to_string(std::numeric_limits<double>::max()),
                        Patterns::Double(0.),
                        "Positive finite dimensionless bound on max abs(ln(Theta_predicted/Theta)) "
                        "over all reconstructed-fault vertices. Predict with the exponential "
                        "aging law at the latest committed slip rate, without advancing history. "
                        "The default, the largest finite double, disables this additional restriction; "
                        "the literal infinity is not accepted. Zero is not allowed. "
                        "The bound is unweighted and does not include b/a. "
                        "It has no effect for a friction law without state.");
      prm.leave_subsection();
      prm.leave_subsection();
    }



    template <int dim>
    void
    ReconstructedFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time stepping");
      prm.enter_subsection("Reconstructed fault time step");
      maximum_logarithmic_state_change = prm.get_double("Maximum logarithmic state change");
      AssertThrow(maximum_logarithmic_state_change > 0.0,
                  ExcMessage("Maximum logarithmic state change must be strictly positive."));
      prm.leave_subsection();
      prm.leave_subsection();
    }



    template <int dim>
    void
    ReconstructedFault<dim>::initialize()
    {
      phase_field_fault =
        &Plugins::get_plugin_as_type<
          const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
    }



    template <int dim>
    double
    ReconstructedFault<dim>::execute()
    {
      Assert(phase_field_fault != nullptr, ExcInternalError());
      double time_step = phase_field_fault->compute_reconstructed_fault_time_step(
        this->get_parameters().CFL_number);
      const auto &law = phase_field_fault->get_fault_friction();
      if (maximum_logarithmic_state_change == std::numeric_limits<double>::max()
          || !law.has_state_variable())
        return time_step;

      // The global cap already bounds the manager's eventual proposal. Avoid
      // searching above it, but leave the disabled path exactly as before.
      time_step = std::min(time_step, this->get_parameters().maximum_time_step);
      const auto &manager = this->get_reconstructed_fault_manager();
      const unsigned int state_position = manager.get_property_information()[
        manager.get_property_index("phase field fault state")].position;
      struct Node
      {
        double velocity, theta;
      };
      std::vector<Node> nodes;
      for (unsigned int f = 0; f < manager.get_faults().size(); ++f)
        {
          const auto &fault = manager.get_fault(f);
          const auto &velocity = manager.get_timestep_committed_slip_rate(f);
          for (unsigned int i = 0; i < fault.n_vertices(); ++i)
            {
              const double theta = fault.get_properties(i)[state_position];
              AssertThrow(std::isfinite(theta) && theta > 0.0,
                          ExcMessage("The reconstructed-fault state limiter requires positive finite Theta."));
              nodes.push_back({velocity[i], theta});
            }
        }
      const auto admissible = [&](const double dt)
      {
        for (const auto &node : nodes)
          {
            // Dc is global; this is also the implementation called by the
            // mixture overload when committing the nodal aging-law update.
            const double predicted = law.update_state(node.velocity, node.theta, dt);
            AssertThrow(std::isfinite(predicted) && predicted > 0.0,
                        ExcMessage("The reconstructed-fault state prediction is inadmissible."));
            const double ratio = predicted / node.theta;
            // The logarithm remains finite even if the ratio over/underflows.
            const double log_change = std::isfinite(ratio) && ratio > 0.0
                                      ? std::log(ratio)
                                      : std::log(predicted) - std::log(node.theta);
            if (std::abs(log_change) > maximum_logarithmic_state_change)
              return false;
          }
        return true;
      };
      if (admissible(time_step))
        return time_step;

      // At fixed V, Theta moves monotonically toward Dc/V. Thus each absolute
      // log-change and their maximum are monotone in dt. First find a positive
      // safe lower bound, even when it is many orders below the proposal, then
      // bisect while retaining the safe end. No history is changed here.
      double upper = time_step, lower = 0.5 * time_step;
      while (!admissible(lower))
        {
          upper = lower;
          lower *= 0.5;
        }
      AssertThrow(lower > 0.0, ExcMessage("The state limiter requires a timestep below floating-point range."));
      for (unsigned int i = 0; i < std::numeric_limits<double>::digits; ++i)
        {
          const double middle = lower + 0.5 * (upper - lower);
          if (middle == lower || middle == upper)
            break;
          if (admissible(middle))
            lower = middle;
          else
            upper = middle;
        }
      // The time-stepping manager performs the global MPI minimum along with
      // the other restrictions. Fault state is replicated on every rank.
      return lower;
    }



    ASPECT_REGISTER_TIME_STEPPING_MODEL(
      ReconstructedFault,
      "reconstructed fault time step",
      "Compute the operator-split law-specific timestep restriction from "
      "the timestep-committed reconstructed-fault slip rate and surface "
      "material mixture. This model is opt-in and uses ASPECT's global CFL "
      "number. An optional unweighted logarithmic state-change bound predicts "
      "the exponential aging update at the latest committed velocity; its "
      "default disabled value leaves the existing restriction unchanged.")
  }
}
