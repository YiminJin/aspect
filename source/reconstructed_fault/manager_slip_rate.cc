/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>

#include <algorithm>
#include <cmath>

namespace aspect
{
  // -----------------------------------------------------------------------------
  // Slip-rate nonlinear state
  // -----------------------------------------------------------------------------

  template <int dim>
  bool
  ReconstructedFaultManager<dim>::slip_rates_are_initialized() const
  {
    return !reconstructed_faults.empty()
           && slip_rate_initialized.size() == reconstructed_faults.size()
           && std::all_of(slip_rate_initialized.begin(), slip_rate_initialized.end(),
                          [](const bool initialized)
    {
      return initialized;
    });
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::initialize_slip_rate(
    const unsigned int fault_index,
    const std::vector<double> &values)
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(!slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcMessage("Slip rates cannot be initialized during a nonlinear solve."));
    Assert(!slip_rate_initialized[fault_index], ExcInternalError());
    Assert(values.size() == reconstructed_faults[fault_index].n_vertices(),
           ExcInternalError());
    for (const double value : values)
      AssertThrow(std::isfinite(value) && value >= 0.0,
                  ExcMessage("Reconstructed-fault slip rates must be finite and nonnegative."));

    timestep_committed_slip_rates[fault_index] = values;
    current_newton_slip_rates[fault_index] = values;
    slip_rate_initialized[fault_index] = true;
  }


  template <int dim>
  const std::vector<double> &
  ReconstructedFaultManager<dim>::get_slip_rate(const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(slip_rate_initialized[fault_index],
           ExcMessage("The reconstructed-fault slip rate has not been initialized."));
    const std::vector<double> &values = slip_rate_trial_active
                                        ? trial_slip_rates[fault_index]
                                        : current_newton_slip_rates[fault_index];
    return values;
  }


  template <int dim>
  const std::vector<double> &
  ReconstructedFaultManager<dim>::get_timestep_committed_slip_rate(
    const unsigned int fault_index) const
  {
    AssertIndexRange(fault_index, reconstructed_faults.size());
    Assert(slip_rate_initialized[fault_index],
           ExcMessage("The reconstructed-fault slip rate has not been initialized."));
    return timestep_committed_slip_rates[fault_index];
  }


  template <int dim>
  double
  ReconstructedFaultManager<dim>::interpolate_slip_rate(
    const unsigned int fault_index,
    const unsigned int segment_index,
    const double xi) const
  {
    const std::vector<double> &values = get_slip_rate(fault_index);
    AssertIndexRange(segment_index, reconstructed_faults[fault_index].n_cells());
    Assert(std::isfinite(xi) && xi >= 0.0 && xi <= 1.0, ExcInternalError());
    return ReconstructedFaultUtilities::interpolate_slip_rate(
      values[segment_index], values[segment_index+1], xi);
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::begin_slip_rate_nonlinear_solve()
  {
    Assert(slip_rates_are_initialized(),
           ExcMessage("A slip-rate nonlinear solve requires initialized slip rates."));
    Assert(!slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcInternalError());
    current_newton_slip_rates = timestep_committed_slip_rates;
    for (unsigned int fault = 0; fault < prescribed_slip_rates.size(); ++fault)
      for (const auto &entry : prescribed_slip_rates[fault])
        current_newton_slip_rates[fault][entry.first] = entry.second;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_nonlinear_solve_active = true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::set_prescribed_slip_rates(
    const std::vector<std::map<unsigned int, double>> &values)
  {
    Assert(!slip_rate_nonlinear_solve_active, ExcInternalError());
    AssertThrow(values.size() == reconstructed_faults.size(),
                ExcMessage("Prescribed slip rates require one map per fault."));
    for (unsigned int fault = 0; fault < values.size(); ++fault)
      for (const auto &entry : values[fault])
        AssertThrow(entry.first < reconstructed_faults[fault].n_vertices()
                    && std::isfinite(entry.second) && entry.second > 0.0,
                    ExcMessage("A prescribed fault slip rate needs a valid vertex and positive finite value."));
    prescribed_slip_rates = values;
  }


  template <int dim>
  std::vector<std::vector<bool>>
  ReconstructedFaultManager<dim>::prescribed_slip_rate_mask() const
  {
    std::vector<std::vector<bool>> mask(reconstructed_faults.size());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        mask[fault].assign(reconstructed_faults[fault].n_vertices(), false);
        if (fault < prescribed_slip_rates.size())
          for (const auto &entry : prescribed_slip_rates[fault])
            mask[fault][entry.first] = true;
      }
    return mask;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::validate_slip_rate_nonlinear_commit() const
  {
    AssertThrow(slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
                ExcMessage("A slip-rate commit requires an active nonlinear "
                           "solve and no active trial."));
    AssertDimension(current_newton_slip_rates.size(),
                    timestep_committed_slip_rates.size());
    for (unsigned int fault = 0; fault < current_newton_slip_rates.size(); ++fault)
      {
        AssertDimension(current_newton_slip_rates[fault].size(),
                        timestep_committed_slip_rates[fault].size());
        for (const double value : current_newton_slip_rates[fault])
          AssertThrow(std::isfinite(value) && value >= 0.0,
                      ExcMessage("A committed slip rate must be finite and nonnegative."));
      }
  }



  template <int dim>
  void
  ReconstructedFaultManager<dim>::commit_slip_rate_nonlinear_solve() noexcept
  {
    for (unsigned int fault = 0; fault < current_newton_slip_rates.size(); ++fault)
      std::copy(current_newton_slip_rates[fault].begin(),
                current_newton_slip_rates[fault].end(),
                timestep_committed_slip_rates[fault].begin());
    slip_rate_nonlinear_solve_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rollback_slip_rate_nonlinear_solve()
  {
    Assert(slip_rate_nonlinear_solve_active, ExcInternalError());
    current_newton_slip_rates = timestep_committed_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
    slip_rate_nonlinear_solve_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::begin_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && !slip_rate_trial_active,
           ExcInternalError());
    trial_slip_rates = current_newton_slip_rates;
    slip_rate_trial_active = true;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::set_slip_rate_trial(
    const std::vector<std::vector<double>> &delta_V,
    const double step_length)
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    AssertThrow(std::isfinite(step_length),
                ExcMessage("The slip-rate trial step length must be finite."));
    Assert(delta_V.size() == reconstructed_faults.size(), ExcInternalError());
    for (unsigned int fault = 0; fault < reconstructed_faults.size(); ++fault)
      {
        Assert(delta_V[fault].size() == reconstructed_faults[fault].n_vertices(),
               ExcInternalError());
        for (const double value : delta_V[fault])
          AssertThrow(std::isfinite(value),
                      ExcMessage("Reconstructed-fault slip-rate updates must be finite."));
      }

    trial_slip_rates = current_newton_slip_rates;
    for (unsigned int fault = 0; fault < trial_slip_rates.size(); ++fault)
      for (unsigned int vertex = 0; vertex < trial_slip_rates[fault].size(); ++vertex)
        {
          const double value = current_newton_slip_rates[fault][vertex]
                               + step_length * delta_V[fault][vertex];
          AssertThrow(std::isfinite(value) && value >= 0.0,
                      ExcMessage("A reconstructed-fault slip-rate trial value must be finite "
                                 "and nonnegative."));
          trial_slip_rates[fault][vertex] = value;
        }
    for (unsigned int fault = 0; fault < prescribed_slip_rates.size(); ++fault)
      for (const auto &entry : prescribed_slip_rates[fault])
        AssertThrow(trial_slip_rates[fault][entry.first] == entry.second,
                    ExcMessage("A Newton trial changed a prescribed fault slip rate."));
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::set_slip_rate_trial_values(
    const std::vector<std::vector<double>> &values)
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active, ExcInternalError());
    AssertDimension(values.size(), reconstructed_faults.size());
    // Validate the complete candidate before replacing any trial entries. The
    // material's lower bound is checked by the caller, not owned by geometry.
    for (unsigned int f = 0; f < values.size(); ++f)
      {
        AssertDimension(values[f].size(), reconstructed_faults[f].n_vertices());
        for (const double value : values[f])
          AssertThrow(std::isfinite(value) && value >= 0.,
                      ExcMessage("Absolute fault trial rates must be finite and nonnegative."));
        for (const auto &entry : prescribed_slip_rates[f])
          AssertThrow(values[f][entry.first] == entry.second,
                      ExcMessage("An absolute trial changed a prescribed fault slip rate."));
      }
    trial_slip_rates = values;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::accept_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    current_newton_slip_rates = trial_slip_rates;
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
  }


  template <int dim>
  void
  ReconstructedFaultManager<dim>::rollback_slip_rate_trial()
  {
    Assert(slip_rate_nonlinear_solve_active && slip_rate_trial_active,
           ExcInternalError());
    trial_slip_rates.assign(reconstructed_faults.size(), {});
    slip_rate_trial_active = false;
  }


}


// Instantiate only these moved definitions. The remaining class definitions
// and archive reconstruction stay in manager.cc.
namespace aspect
{
#define INSTANTIATE(dim) \
  template bool ReconstructedFaultManager<dim>::slip_rates_are_initialized() const; \
  template void ReconstructedFaultManager<dim>::initialize_slip_rate(const unsigned int, const std::vector<double> &); \
  template const std::vector<double> &ReconstructedFaultManager<dim>::get_slip_rate(const unsigned int) const; \
  template const std::vector<double> &ReconstructedFaultManager<dim>::get_timestep_committed_slip_rate(const unsigned int) const; \
  template double ReconstructedFaultManager<dim>::interpolate_slip_rate(const unsigned int, const unsigned int, const double) const; \
  template void ReconstructedFaultManager<dim>::begin_slip_rate_nonlinear_solve(); \
  template void ReconstructedFaultManager<dim>::set_prescribed_slip_rates(const std::vector<std::map<unsigned int, double>> &); \
  template std::vector<std::vector<bool>> ReconstructedFaultManager<dim>::prescribed_slip_rate_mask() const; \
  template void ReconstructedFaultManager<dim>::validate_slip_rate_nonlinear_commit() const; \
  template void ReconstructedFaultManager<dim>::commit_slip_rate_nonlinear_solve() noexcept; \
  template void ReconstructedFaultManager<dim>::rollback_slip_rate_nonlinear_solve(); \
  template void ReconstructedFaultManager<dim>::begin_slip_rate_trial(); \
  template void ReconstructedFaultManager<dim>::set_slip_rate_trial(const std::vector<std::vector<double>> &, const double); \
  template void ReconstructedFaultManager<dim>::set_slip_rate_trial_values(const std::vector<std::vector<double>> &); \
  template void ReconstructedFaultManager<dim>::accept_slip_rate_trial(); \
  template void ReconstructedFaultManager<dim>::rollback_slip_rate_trial();

  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
