/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system.h>
#include <aspect/simulator_signals.h>
#include <aspect/time_stepping/interface.h>
#include <fstream>
#include <iomanip>
namespace aspect {
namespace TimeStepping {
/** Accuracy controller only. Native nonlinear-failure cutback remains
 * separate; neither controller publishes any material history. */
template <int dim>
class CohesiveRSF : public Interface<dim>, public SimulatorAccess<dim> {
  double slip = .1, state = .1, speed = .2, damage = .02, safety = .8,
         growth = 1.25, minimum = 1e-10;
  double ratio = 0., proposed = 0.;
  bool reject = false, residual_resolved = false;
  double history_error = .0005;

public:
  void initialize() override {
    // The owner has been restored before this signal. Only shrink the
    // pending interval; the last accepted dt and all histories stay intact.
    this->get_signals().post_resume_time_step.connect(
        [this](const SimulatorAccess<dim> &, double &dt) {
          dt = std::min(
              {dt, execute(), this->get_parameters().maximum_time_step});
        });
  }
  static void declare_parameters(ParameterHandler &p) {
    p.enter_subsection("Time stepping");
    p.enter_subsection("Cohesive RSF");
    p.declare_entry("Use residual resolved estimator", "false", Patterns::Bool());
    p.declare_entry("Maximum continuous history error", ".0005", Patterns::Double(0.));
    p.declare_entry("Maximum slip increment in Dc", ".1", Patterns::Double(0.));
    p.declare_entry("Maximum logarithmic state change", ".1",
                    Patterns::Double(0.));
    p.declare_entry("Maximum regularized logarithmic speed change", ".2",
                    Patterns::Double(0.));
    p.declare_entry("Maximum damage increment", ".02", Patterns::Double(0.));
    p.declare_entry("Safety factor", ".8", Patterns::Double(0., 1.));
    p.declare_entry("Growth factor", "1.25", Patterns::Double(1.));
    p.declare_entry("Minimum timestep", "1e-10", Patterns::Double(0.));
    p.leave_subsection();
    p.leave_subsection();
  }
  void parse_parameters(ParameterHandler &p) override {
    p.enter_subsection("Time stepping");
    p.enter_subsection("Cohesive RSF");
    residual_resolved = p.get_bool("Use residual resolved estimator");
    history_error = p.get_double("Maximum continuous history error");
    slip = p.get_double("Maximum slip increment in Dc");
    state = p.get_double("Maximum logarithmic state change");
    speed = p.get_double("Maximum regularized logarithmic speed change");
    damage = p.get_double("Maximum damage increment");
    safety = p.get_double("Safety factor");
    growth = p.get_double("Growth factor");
    minimum = p.get_double("Minimum timestep");
    p.leave_subsection();
    p.leave_subsection();
    AssertThrow(history_error > 0. && slip > 0. && state > 0. && speed > 0. && damage > 0. &&
                    safety > 0. && safety < 1. && minimum > 0.,
                ExcMessage("Invalid cohesive adaptive-step controls."));
  }
  double execute() override {
    const auto *owner = this->get_cohesive_rsf_system();
    AssertThrow(
        owner,
        ExcMessage(
            "Cohesive RSF timestepping requires the native cohesive pilot."));
    const auto m = owner->step_metrics();
    const double increment_ratio = residual_resolved
        ? std::max({(m.slip_Dc+m.resolution_slip_Dc)/slip,
                    (m.log_state+m.resolution_slip_Dc)/state,
                    (m.damage_increment+m.resolution_damage)/damage})
        : std::max({m.slip_Dc/slip,m.log_state/state,m.log_speed/speed,m.damage_increment/damage});
    const double defect_ratio=residual_resolved?m.continuous_error/history_error:0.;
    ratio=std::max(increment_ratio,defect_ratio);
    reject = m.pending && ratio > 1.;
    if (m.dt == 0.)
      return this->get_parameters().maximum_first_time_step;
    proposed=m.dt*std::min({growth,increment_ratio>0.?safety/increment_ratio:growth,
                           defect_ratio>0.?std::sqrt(safety/defect_ratio):growth});
    if (m.pending &&
        Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0) {
      const auto name =
          this->get_output_directory() + "cohesive-adaptation.csv";
      const bool exists = std::ifstream(name).good();
      std::ofstream f(name, std::ios::app);
      if (!exists)
        f << "step,time_s,dt_s,speed_m_s,slip_Dc,log_state,log_speed,damage_"
             "increment,ratio,reject,next_dt_s\n";
      f << std::setprecision(17) << this->get_timestep_number() << ','
        << this->get_time() << ',' << m.dt << ',' << m.speed << ',' << m.slip_Dc
        << ',' << m.log_state << ',' << m.log_speed << ',' << m.damage_increment
        << ',' << ratio << ',' << reject << ',' << proposed << '\n';
      if (residual_resolved) {
        const std::string error_name=this->get_output_directory()+"cohesive-error-estimate.csv";
        const bool error_exists=std::ifstream(error_name).good();
        std::ofstream error(error_name,std::ios::app);
        if(!error_exists)error<<"step,time_s,dt_s,raw_log_speed,resolved_log_speed,displacement_resolution_m,continuous_error,numerical_resolution,increment_ratio,defect_ratio,reject,next_dt_s\n";
        error<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<m.dt
          <<','<<m.log_speed<<','<<m.resolved_log_speed<<','<<m.displacement_resolution
          <<','<<m.continuous_error<<','<<m.numerical_resolution<<','<<increment_ratio
          <<','<<defect_ratio<<','<<reject<<','<<proposed<<'\n';
      }
      if (proposed < minimum) {
        std::ofstream failure(this->get_output_directory()+"cohesive-rate-failure.csv");
        failure << "step,time_s,dt_s,ratio,next_dt_s,minimum_dt_s,rate_point,x_m,y_m,old_damage,new_damage,old_rate_m_s,new_rate_m_s,old_jump_m,new_jump_m\n"
          << std::setprecision(17) << this->get_timestep_number() << ',' << this->get_time()
          << ',' << m.dt << ',' << ratio << ',' << proposed << ',' << minimum << ',' << m.rate_point
          << ',' << m.rate_x << ',' << m.rate_y << ',' << m.old_damage << ',' << m.new_damage
          << ',' << m.old_rate << ',' << m.new_rate << ',' << m.old_jump << ',' << m.new_jump << '\n';
      }
    }
    // Preserve the failed, unpublished candidate's metrics before stopping.
    AssertThrow(proposed >= minimum,
                ExcMessage("Cohesive adaptive timestep reached its "
                           "computational lower limit before publication; "
                           "see cohesive-rate-failure.csv for the pending point and operands."));
    return proposed;
  }
  std::pair<Reaction, double>
  determine_reaction(const TimeStepInfo &) override {
    return {reject ? Reaction::repeat_step : Reaction::advance,
            reject ? proposed : std::numeric_limits<double>::max()};
  }
};
ASPECT_REGISTER_TIME_STEPPING_MODEL(
    CohesiveRSF, "cohesive RSF",
    "Default-off native cohesive accuracy control. Bound each Gauss slip/Dc, "
    "logarithmic age, "
    "log((abs(V)+V0)/(abs(Vold)+V0)), and damage increment. Retry from "
    "accepted state.")
} // namespace TimeStepping
} // namespace aspect
