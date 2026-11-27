/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_time_stepping_fault_rsf_h
#define _aspect_time_stepping_fault_rsf_h
#include <aspect/time_stepping/interface.h>

namespace aspect::TimeStepping
{
  /** Resolve slip and changes of RSF age across accepted steps. The manager
   * queries execute() after solving and before publishing the returned state.
   * Predict the next interval from that endpoint and independently test the
   * just-completed interval against its immutable parent. */
  template <int dim>
  class FaultRSF : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm) override;
      double execute() override;
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override;
    private:
      double slip_fraction, log_age_budget;
      double retry_duration = std::numeric_limits<double>::max();
  };
}
#endif
