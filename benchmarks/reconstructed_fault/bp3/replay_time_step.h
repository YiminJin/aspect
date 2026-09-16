#include <aspect/time_stepping/interface.h>

namespace aspect
{
  namespace TimeStepping
  {
    // A benchmark cap, not a replacement for CFL or fault-state restrictions.
    // Stop if the other models require smaller steps: unlike histories cannot
    // be presented as a matched-time spatial comparison.
    template <int dim>
    class BP3ReplayTimeStep : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double execute() override
        {
          if (times.empty())
            {
              const char *path=std::getenv("ASPECT_BP3_TIMESTEP_SEQUENCE");
              AssertThrow(path,ExcMessage("BP3 replay cap requires an accepted-step CSV."));
              std::ifstream in(path);
              AssertThrow(in,ExcMessage("Cannot read BP3 replay sequence."));
              std::string line;
              std::getline(in,line);
              while (std::getline(in,line))
                {
                  std::replace(line.begin(),line.end(),',',' ');
                  std::istringstream row(line);
                  unsigned int step;
                  double time,dt;
                  AssertThrow(row>>step>>time>>dt,ExcMessage("Invalid BP3 replay row."));
                  AssertThrow(step==times.size(),ExcMessage("Non-contiguous BP3 replay steps."));
                  times.push_back(time);
                  steps.push_back(dt);
                }
            }
          // Opt-in mechanism replay: retain comparison times, but allow the
          // production controllers to insert substeps. Do not index the saved
          // trajectory by the now different simulator timestep number.
          if (std::getenv("ASPECT_BP3_ADAPTIVE_REPLAY"))
            {
              const double time=this->get_time();
              const auto next=std::upper_bound(times.begin(),times.end(),
                time+1e-12*std::max(1.,std::abs(time)));
              return next==times.end() ? std::numeric_limits<double>::max() : *next-time;
            }
          const unsigned int k=this->get_timestep_number();
          AssertThrow(k<times.size() && std::abs(this->get_time()-times[k])
                      <=1e-12*std::max(1.,std::abs(times[k])),
                      ExcMessage("BP3 spatial replay left the matched physical times."));
          return k+1<steps.size() ? steps[k+1] : std::numeric_limits<double>::max();
        }

        std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override
        {
          if (std::getenv("ASPECT_BP3_ADAPTIVE_REPLAY"))
            {
              this->get_pcout()<<std::setprecision(17)<<"BP3 adaptive comparison clock: time="
                <<this->get_time()<<" selected="<<info.next_time_step_size<<std::endl;
              return {Reaction::advance,std::numeric_limits<double>::max()};
            }
          const unsigned int k=this->get_timestep_number();
          if (k+1<steps.size())
            {
              this->get_pcout()<<std::setprecision(17)<<"BP3 matched timestep: next="<<k+1
                <<" requested="<<steps[k+1]<<" selected="<<info.next_time_step_size<<std::endl;
              AssertThrow(std::abs(info.next_time_step_size-steps[k+1])<=1e-12*steps[k+1],
                          ExcMessage("BP3 controller requires a smaller timestep; stop the matched spatial comparison."));
            }
          return {Reaction::advance,std::numeric_limits<double>::max()};
        }

      private:
        std::vector<double> times,steps;
    };

    ASPECT_REGISTER_TIME_STEPPING_MODEL(BP3ReplayTimeStep,"BP3 replay cap",
      "Cap steps by saved accepted BP3 times, retaining all other controller restrictions.")
  }
}
