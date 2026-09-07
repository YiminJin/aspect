/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/simulator/solver/reconstructed_fault_nonlinear.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect
{
  namespace internal
  {
    bool
    reconstructed_fault_locally_at_lower_bound(const double value,
                                                const double minimum)
    {
      constexpr double tolerance_factor = 100.0;
      return value-minimum
             <= tolerance_factor*std::numeric_limits<double>::epsilon()
                * std::max(minimum, std::abs(value));
    }


    ReconstructedFaultActiveSet
    make_reconstructed_fault_inactive_set(
      const ReconstructedFaultVector &slip_rate)
    {
      ReconstructedFaultActiveSet active_set(slip_rate.size());
      for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
        active_set[fault].assign(slip_rate[fault].size(), false);
      return active_set;
    }


    unsigned int
    update_reconstructed_fault_active_set(
      const ReconstructedFaultVector &slip_rate,
      const ReconstructedFaultVector &direction,
      const double minimum,
      ReconstructedFaultActiveSet &active_set)
    {
      AssertDimension(direction.size(), slip_rate.size());
      AssertDimension(active_set.size(), slip_rate.size());

      unsigned int n_new_active_vertices = 0;
      for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
        {
          AssertDimension(direction[fault].size(), slip_rate[fault].size());
          AssertDimension(active_set[fault].size(), slip_rate[fault].size());
          for (unsigned int vertex = 0; vertex < slip_rate[fault].size(); ++vertex)
            if (!active_set[fault][vertex]
                && reconstructed_fault_locally_at_lower_bound(
                  slip_rate[fault][vertex], minimum)
                && direction[fault][vertex] < 0.0)
              {
                active_set[fault][vertex] = true;
                ++n_new_active_vertices;
              }
        }
      return n_new_active_vertices;
    }


    double
    reconstructed_fault_maximum_step_length(
      const ReconstructedFaultVector &slip_rate,
      const ReconstructedFaultVector &direction,
      const ReconstructedFaultActiveSet &active_set,
      const double minimum)
    {
      AssertDimension(direction.size(), slip_rate.size());
      AssertDimension(active_set.size(), slip_rate.size());

      double maximum_step_length = 1.0;
      for (unsigned int fault = 0; fault < slip_rate.size(); ++fault)
        {
          AssertDimension(direction[fault].size(), slip_rate[fault].size());
          AssertDimension(active_set[fault].size(), slip_rate[fault].size());
          for (unsigned int vertex = 0; vertex < slip_rate[fault].size(); ++vertex)
            if (!active_set[fault][vertex] && direction[fault][vertex] < 0.0)
              maximum_step_length = std::min(
                maximum_step_length,
                (slip_rate[fault][vertex]-minimum)
                / (-direction[fault][vertex]));
        }

      AssertThrow(std::isfinite(maximum_step_length)
                  && maximum_step_length > 0.0,
                  ExcMessage("The reconstructed-fault fraction-to-boundary "
                             "calculation produced an invalid step length."));
      return maximum_step_length;
    }


    double
    reconstructed_fault_residual_scale(const double initial_norm,
                                       const double reference_norm,
                                       const double floor_factor)
    {
      Assert(initial_norm >= 0.0 && std::isfinite(initial_norm),
             ExcInternalError());
      Assert(reference_norm >= 0.0 && std::isfinite(reference_norm),
             ExcInternalError());
      Assert(floor_factor > 0.0 && std::isfinite(floor_factor),
             ExcInternalError());
      return std::max(initial_norm, floor_factor*reference_norm);
    }


    double
    normalized_reconstructed_fault_residual(const double residual_norm,
                                            const double scale,
                                            const std::string &block_name)
    {
      if (scale > 0.0)
        return residual_norm/scale;
      AssertThrow(residual_norm == 0.0,
                  ExcMessage("The initial " + block_name
                             + " residual and its reference scale were exactly zero, "
                             "but a later residual is nonzero."));
      return 0.0;
    }


    ReconstructedFaultLineSearchResult
    reconstructed_fault_armijo_line_search(
      const double maximum_step_length,
      const unsigned int maximum_reductions,
      const double current_merit,
      const std::function<double (double)> &evaluate_trial_merit,
      const std::function<void (double)> &accept_trial)
    {
      Assert(maximum_step_length > 0.0
             && maximum_step_length <= 1.0
             && std::isfinite(maximum_step_length),
             ExcInternalError());
      Assert(current_merit >= 0.0 && std::isfinite(current_merit),
             ExcInternalError());

      constexpr double armijo_coefficient = 1e-4;
      constexpr double reduction_factor = 2.0/3.0;
      ReconstructedFaultLineSearchResult result;
      result.step_length = maximum_step_length;

      for (unsigned int reduction = 0;
           reduction <= maximum_reductions; ++reduction)
        {
          const double trial_merit = evaluate_trial_merit(result.step_length);
          AssertThrow(std::isfinite(trial_merit) && trial_merit >= 0.0,
                      ExcMessage("The reconstructed-fault line search produced "
                                 "a non-finite or negative merit value."));
          if (trial_merit
              <= (1.0-armijo_coefficient*result.step_length)*current_merit)
            {
              accept_trial(result.step_length);
              result.accepted = true;
              return result;
            }

          ++result.rejected_candidates;
          if (reduction < maximum_reductions)
            result.step_length *= reduction_factor;
        }

      return result;
    }
  }
}
