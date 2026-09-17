#ifndef ASPECT_BP3_OUTPUT_SCHEDULE_H
#define ASPECT_BP3_OUTPUT_SCHEDULE_H
#include <algorithm>
#include <cmath>
#include <vector>

namespace BP3
{
  // Scheduling observes committed signed slip. Only successful writers move
  // the reference; asking about a rejected candidate has no side effects.
  struct OutputSchedule
  {
    double slip_interval=0.1, time_interval=31557600.;
    double last_time=-1.;
    std::vector<double> reference;
    double increment(const std::vector<double> &slip) const
    {
      if (reference.empty()) return 0.;
      double maximum=0.;
      for (unsigned int i=0;i<slip.size();++i)
        maximum=std::max(maximum,std::abs(slip[i]-reference.at(i)));
      return maximum;
    }
    bool due(double time,const std::vector<double> &slip,bool force=false) const
    {
      return force || reference.empty() || increment(slip)>=slip_interval
             || (time_interval>0. && time-last_time>=time_interval);
    }
    void written(double time,const std::vector<double> &slip)
    { last_time=time; reference=slip; }
    template<class Archive> void serialize(Archive &ar,const unsigned int)
    { ar & slip_interval & time_interval & last_time & reference; }
  };
}
#endif
