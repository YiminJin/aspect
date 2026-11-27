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
    const double uniform_distance = rates.law.characteristic_slip_distance();
    AssertThrow(uniform_distance > 0., ExcMessage("Fault slip distance must be positive."));
    AssertThrow(ages.empty() || ages.size() == rates.parent.size(), ExcMessage("Fault age and rate identities differ."));

    const double dt = this->get_timestep();
    const bool completed = !rates.completed.empty();
    Assert(!completed || rates.completed.size() == rates.parent.size(), ExcInternalError());
    double next = std::numeric_limits<double>::max();
    double limiting_budget=0., retry_slip_budget=0., retry_slip=0., retry_age=0.;
    double max_slip = 0., max_age = 0.;
    unsigned int limiting = 0, max_slip_id = 0, max_age_id = 0, retry_id = 0;
    const char *retry_reason = "none";
    const char *criterion = "none";
    retry_duration = std::numeric_limits<double>::max();
    for (unsigned int i=0; i<rates.parent.size(); ++i)
      {
        const double rate = completed ? rates.completed[i] : rates.parent[i];
        AssertThrow(std::isfinite(rate), ExcMessage("Nonfinite fault rate for timestep selection."));
        double distance = uniform_distance;
        double budget = slip_fraction;
        if (weakening_policy)
          {
            const auto local = rates.law.slip_resolution_parameters(i);
            AssertThrow(local.has_value(),
                        ExcMessage("Weakening-dependent slip resolution requires local RSF coefficients from the resistance law."));
            const auto [a,b,dc] = *local;
            AssertThrow(std::isfinite(a) && a>0. && std::isfinite(b) && b>=0. && std::isfinite(dc) && dc>0.,
                        ExcMessage("Invalid local RSF coefficients for weakening-dependent slip resolution."));
            distance = dc;
            budget = b>0. && a<=b ? slip_fraction*(a/b) : 0.;
          }
        const bool check_slip = budget>0. && std::isfinite(distance);
        const double slip_limit = check_slip && rate != 0. ? budget*distance/std::abs(rate) :
                                  std::numeric_limits<double>::infinity();
        if (completed && check_slip && dt*std::abs(rate)/distance > max_slip)
          {
            max_slip=dt*std::abs(rate)/distance;
            max_slip_id=i;
          }
        if (slip_limit < next)
          {
            next=slip_limit;
            limiting=i;
            criterion="slip";
            limiting_budget=budget;
          }
        double parent_limit = slip_limit;
        bool reject = completed && dt > slip_limit;
        const char *cause = "slip";
        double change=0.;
        if (check_age)
          {
            const double age = ages[i];
            AssertThrow(std::isfinite(age) && age > 0., ExcMessage("Nonpositive or nonfinite accepted fault age."));
            const double endpoint = completed && dt > 0. ? rates.law.predict_age(rate, age, dt) : age;
            AssertThrow(std::isfinite(endpoint) && endpoint > 0., ExcMessage("Invalid predicted fault age."));
            // log1p preserves small increments; separate logs avoid ratio overflow for large ones.
            change = std::abs(endpoint-age) < .5*age ? std::log1p((endpoint-age)/age) :
                                  std::log(endpoint)-std::log(age);
            if (std::abs(change) > log_age_budget + 4.*std::numeric_limits<double>::epsilon())
              {
                reject = true;
                cause = parent_limit<dt ? "slip_and_age" : "age";
                parent_limit = std::min(parent_limit, rates.law.state_time_limit(rate, age, log_age_budget));
              }
            if (std::abs(change)>max_age)
              {
                max_age=std::abs(change);
                max_age_id=i;
              }
            const double age_limit=rates.law.state_time_limit(rate,endpoint,log_age_budget);
            if (age_limit < next)
              {
                next=age_limit;
                limiting=i;
                criterion="age";
                limiting_budget=log_age_budget;
              }
          }
        if (reject && .9*std::min(dt,parent_limit)<retry_duration)
          {
            retry_duration=.9*std::min(dt,parent_limit);
            retry_id=i;
            retry_reason=cause;
            retry_slip_budget=budget;
            retry_slip=dt*std::abs(rate)/distance;
            retry_age=std::abs(change);
          }
      }
    AssertThrow(next > 0. && retry_duration > 0., ExcMessage("Fault timestep is below floating point range."));
    const double raw_limit=next;
    if (next<std::numeric_limits<double>::max())
      next*=proposal_factor;
    if (report)
      {
        std::ostringstream row;
        row << std::setprecision(17) << "Fault timestep at " << this->get_time() << " s: dt=" << dt
            << " raw_limit=" << raw_limit << " proposal=" << next << " criterion=" << criterion << " id=" << limiting;
        if (!material.uses_growing_strips())
          {
            const auto &labels=material.labels();
            const auto station = [&](const unsigned int id)
            {
              double s=0.;
              for (unsigned int j=0; j<id; ++j)
                s+=labels[j+1].position.distance(labels[j].position);
              return s;
            };
            row << " station_m=" << station(limiting);
            if (retry_duration<dt)
              row << " retry_station_m=" << station(retry_id);

          }
        row << " limiting_budget=" << limiting_budget << " slip_fraction=" << max_slip << " abs_log_age=" << max_age;
        row << " slip_id=" << max_slip_id << " age_id=" << max_age_id;
        if (retry_duration<dt)
          row << " retry_id=" << retry_id << " retry_reason=" << retry_reason << " retry_slip_budget=" << retry_slip_budget
              << " retry_slip_fraction=" << retry_slip << " retry_abs_log_age=" << retry_age
              << " retry_age_budget=" << log_age_budget << " retry_dt=" << retry_duration;
        this->get_pcout() << row.str() << std::endl;
        material_proposal=next;
      }
    return next;
  }

  template <int dim>
  std::pair<Reaction,double> FaultRSF<dim>::determine_reaction(const TimeStepInfo &info)
  {
    if (report)
      {
        const auto &p=this->get_parameters();
        const double dt=this->get_timestep();
        const char *reason=(info.reduced_by_termination_plugin ? "termination" :
                            info.next_time_step_size==material_proposal ? "material" :
                            this->get_timestep_number()==0 && info.next_time_step_size==p.maximum_first_time_step ? "first" :
                            dt>0. && info.next_time_step_size==dt+dt*p.maximum_relative_increase_time_step ? "growth" :
                            info.next_time_step_size==p.maximum_time_step ? "maximum" : "other");
        std::ostringstream row;
        row << std::setprecision(17) << "Fault next dt=" << info.next_time_step_size << " limiter=" << reason;
        this->get_pcout() << row.str() << std::endl;
      }
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
    prm.declare_entry("Report timestep limits", "false", Patterns::Bool(),
                      "Print the limiting material station and slip and age increments for each attempt.");
    prm.declare_entry("Slip resolution policy", "uniform", Patterns::Selection("uniform|weakening dependent"),
                      "Uniform applies the slip fraction to every material identity. Weakening dependent "
                      "multiplies it by a/b where b is positive and a is at most b. Other IDs have no slip limit.");
    prm.declare_entry("Proposal safety factor", "1", Patterns::Double(0.,1.),
                      "Multiply the next finite fault timestep estimate by this positive factor. "
                      "Completed trials are checked against the original slip and age budgets.");
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
    weakening_policy = prm.get("Slip resolution policy") == "weakening dependent";
    proposal_factor = prm.get_double("Proposal safety factor");
    AssertThrow(proposal_factor>0.,ExcMessage("The fault proposal safety factor must be positive."));
    report = prm.get_bool("Report timestep limits");
    slip_fraction = prm.get_double("Maximum slip fraction");
    log_age_budget = prm.get_double("Maximum logarithmic state change");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  ASPECT_REGISTER_TIME_STEPPING_MODEL(FaultRSF, "fault RSF",
                                      "Predict and enforce independent slip and age resolution on material fault IDs.")
}
