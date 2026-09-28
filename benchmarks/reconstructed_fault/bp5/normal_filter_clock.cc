// Replay ACTUAL constitutive intervals, never differences of large timestamps.
#include <aspect/time_stepping/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/utilities.h>
#include "normal_stress_clock.h"
#include <algorithm>

namespace aspect::TimeStepping
{
  template <int dim> class BP5FilterClock : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &p)
      {
        p.enter_subsection("Time stepping");p.enter_subsection("BP5 filter clock");
        p.declare_entry("Actual intervals file","",Patterns::Anything(),"One positive actual dt in seconds per line.");
        p.declare_entry("Halve recorded intervals","false",Patterns::Bool(),
                        "Single bounded retry: halve each original R interval once without doubling the step count. All safety caps remain active.");
        p.leave_subsection();p.leave_subsection();
      }
      void parse_parameters(ParameterHandler &p) override
      {
        p.enter_subsection("Time stepping");p.enter_subsection("BP5 filter clock");
        path=p.get("Actual intervals file");halve=p.get_bool("Halve recorded intervals");
        p.leave_subsection();p.leave_subsection();
        p.enter_subsection("Postprocess");p.enter_subsection("BP5 normal diagnostic");
        origin=p.get_integer("Checkpoint accepted step");count=p.get_integer("New accepted steps");
        p.leave_subsection();p.leave_subsection();
      }
      void initialize() override
      {
        intervals=BP5NormalStress::read_filter_intervals(
          Utilities::read_and_distribute_file_content(path,this->get_mpi_communicator()),count,halve);
        this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim> &,double &dt)
          {dt=std::min(dt,intervals[0]);});
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
        {
          const auto k=this->get_timestep_number();
          AssertThrow(k>origin && k<=origin+count,ExcMessage("Filter replay exceeded its bound."));
          const double expected=intervals[k-origin-1];
          BP5NormalStress::check_filter_interval(k,expected,this->get_timestep());
        });
      }
      double execute() override
      {
        const auto k=this->get_timestep_number()-origin;
        return k<intervals.size()?intervals[k]:std::numeric_limits<double>::max();
      }
    private:
      std::string path;
      bool halve=false;
      unsigned int origin=0,count=0;
      std::vector<double> intervals;
  };
  ASPECT_REGISTER_TIME_STEPPING_MODEL(BP5FilterClock,"BP5 filter clock","Bounded actual-dt normal-filter restart comparison; all safety caps remain active.")
}
