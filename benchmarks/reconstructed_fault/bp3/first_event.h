#ifndef ASPECT_BP3_FIRST_EVENT_H
#define ASPECT_BP3_FIRST_EVENT_H

#include <limits>

namespace BP3
{
  // Accepted-state observer only. Equality breaks a sub-threshold streak.
  struct FirstEvent
  {
    bool started=false, complete=false;
    unsigned int below=0;
    double onset=-1., peak=0., peak_time=-1., peak_xd=0.;
    double down_crossing=-1., termination=-1.;
    static constexpr double threshold=1e-3;

    void observe(const double time, const double maximum, const double xd)
    {
      if (complete) return;
      if (!started && maximum>=threshold)
        { started=true; onset=time; }
      if (!started) return;
      if (maximum>peak)
        { peak=maximum; peak_time=time; peak_xd=xd; }
      if (maximum<threshold)
        {
          if (below==0) down_crossing=time;
          if (++below==5) { complete=true; termination=time; }
        }
      else
        { below=0; down_crossing=-1.; }
    }

    template<class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar & started & complete & below & onset & peak & peak_time & peak_xd
         & down_crossing & termination;
    }
  };
}
#endif
