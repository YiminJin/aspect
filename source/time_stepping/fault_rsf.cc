/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/time_stepping/fault_rsf.h>
#include <aspect/material_model/phase_field_fault.h>

namespace aspect::TimeStepping
{
  template <int dim>
  double FaultRSF<dim>::execute()
  {
    const auto &material = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
    const auto rates = material.timestep_rates();
    const auto &ages = rates.law.ages_in_seconds();
    const bool check_age = log_age_budget > 0. && !ages.empty();
    const double distance = rates.law.characteristic_slip_distance();
    AssertThrow(distance > 0., ExcMessage("Fault slip distance must be positive."));
    AssertThrow(ages.empty() || ages.size() == rates.parent.size(), ExcMessage("Fault age and rate identities differ."));
    const bool check_slip = slip_fraction > 0. && std::isfinite(distance);
    const double dt = this->get_timestep();
    const bool completed = !rates.completed.empty();
    Assert(!completed || rates.completed.size() == rates.parent.size(), ExcInternalError());
    double next = std::numeric_limits<double>::max();
    retry_duration = std::numeric_limits<double>::max();
    for (unsigned int i=0; i<rates.parent.size(); ++i)
      {
        const double rate = completed ? rates.completed[i] : rates.parent[i];
        AssertThrow(std::isfinite(rate), ExcMessage("Nonfinite fault rate for timestep selection."));
        const double slip_limit = check_slip && rate != 0. ? slip_fraction*distance/std::abs(rate) :
                                  std::numeric_limits<double>::infinity();
        next = std::min(next, slip_limit);
        double parent_limit = slip_limit;
        bool reject = completed && dt > slip_limit;
        if (check_age)
          {
            const double age = ages[i];
            AssertThrow(std::isfinite(age) && age > 0., ExcMessage("Nonpositive or nonfinite accepted fault age."));
            const double endpoint = completed && dt > 0. ? rates.law.predict_age(rate, age, dt) : age;
            AssertThrow(std::isfinite(endpoint) && endpoint > 0., ExcMessage("Invalid predicted fault age."));
            // log1p preserves small increments; separate logs avoid ratio overflow for large ones.
            const double change = std::abs(endpoint-age) < .5*age ? std::log1p((endpoint-age)/age) :
                                  std::log(endpoint)-std::log(age);
            if (std::abs(change) > log_age_budget + 4.*std::numeric_limits<double>::epsilon())
              {
                reject = true;
                parent_limit = std::min(parent_limit, rates.law.state_time_limit(rate, age, log_age_budget));
              }
            next = std::min(next, rates.law.state_time_limit(rate, endpoint, log_age_budget));
          }
        if (reject)
          retry_duration = std::min(retry_duration, .9*std::min(dt, parent_limit));
      }
    AssertThrow(next > 0. && retry_duration > 0., ExcMessage("Fault timestep is below floating point range."));
    return next;
  }

  template <int dim>
  std::pair<Reaction,double> FaultRSF<dim>::determine_reaction(const TimeStepInfo &info)
  {
    if (retry_duration < this->get_timestep())
      {
        this->get_pcout() << "   Fault slip/state budget exceeded; retry duration " << retry_duration << " s." << std::endl;
        return {Reaction::repeat_step, retry_duration};
      }
    return Interface<dim>::determine_reaction(info);
  }

  template <int dim>
  void FaultRSF<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time stepping");
    prm.enter_subsection("Fault RSF");
    prm.declare_entry("Maximum slip fraction", "0.1", Patterns::Double(0.),
                      "Maximum absolute slip increment in units of the law's slip distance. Zero disables this limit.");
    prm.declare_entry("Maximum logarithmic state change", "0.1", Patterns::Double(0.),
                      "Maximum absolute change in the natural logarithm of age. Zero disables this limit.");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void FaultRSF<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time stepping");
    prm.enter_subsection("Fault RSF");
    slip_fraction = prm.get_double("Maximum slip fraction");
    log_age_budget = prm.get_double("Maximum logarithmic state change");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  ASPECT_REGISTER_TIME_STEPPING_MODEL(FaultRSF, "fault RSF",
                                      "Predict and enforce independent slip and age resolution on material fault IDs.")
}
