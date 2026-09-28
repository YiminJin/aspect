// Tiny checkpoint test: no BP5 mechanics or production histories are advanced
// by the restart hook. Tests the actual core lifecycle on one and two ranks.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <array>
#include <fstream>
#include <iomanip>

namespace aspect::Postprocess
{
  template <int dim>
  class RestartClockTest : public Interface<dim>, public SimulatorAccess<dim>
  {
  public:
    static void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");prm.enter_subsection("Restart clock test");
      prm.declare_entry("Mode","keep",Patterns::Selection("keep|half|zero|increase|nonfinite|rank disagreement|external"));
      prm.leave_subsection();prm.leave_subsection();
    }
    void parse_parameters(ParameterHandler &prm) override
    {
      prm.enter_subsection("Postprocess");prm.enter_subsection("Restart clock test");mode=prm.get("Mode");
      prm.leave_subsection();prm.leave_subsection();
    }
    void initialize() override
    {
      if (!this->get_parameters().resume_computation) return;
      this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim> &,double &dt)
      {
        incoming_time=this->get_time();incoming_dt=this->get_timestep();old_dt=this->get_old_timestep();
        incoming_step=this->get_timestep_number();norms=history_norms();
        if (mode=="half") dt*=.5;
        if (mode=="zero") dt=0.;
        if (mode=="increase") dt*=2.;
        if (mode=="nonfinite") dt=std::numeric_limits<double>::infinity();
        if (mode=="rank disagreement") dt*=Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) ? .25:.5;
        expected_dt=mode=="external" ? incoming_dt/2.:dt;
      });
      this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
      {
        if (checked) return;
        AssertThrow(this->get_time()==incoming_time-incoming_dt+expected_dt && this->get_timestep()==expected_dt,
                    ExcMessage("Restart clock was not adjusted before the first solve."));
        AssertThrow(this->get_old_timestep()==old_dt && this->get_timestep_number()==incoming_step && norms==history_norms(),
                    ExcMessage("Restart reduction changed a committed history/vector or the step number."));
        this->get_pcout()<<"Restart clock test: incoming vectors, old dt and step unchanged; pending clock verified.\n";
        checked=true;
      });
    }
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      if (this->get_pcout().is_active())
        {
          std::ofstream out(this->get_output_directory()+"clock_test.csv",std::ios::app);
          out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<'\n';
        }
      return {"Restart clock test", "accepted"};
    }
  private:
    std::array<double,3> history_norms() const
    { return {{this->get_solution().l2_norm(),this->get_old_solution().l2_norm(),this->get_old_old_solution().l2_norm()}}; }
    std::string mode;
    double incoming_time=0.,incoming_dt=0.,old_dt=0.,expected_dt=0.;
    unsigned int incoming_step=0;
    bool checked=false;
    std::array<double,3> norms;
  };
  ASPECT_REGISTER_POSTPROCESSOR(RestartClockTest,"restart clock test","Test-only pending-clock reduction and history preservation.")
}
