// Accepted-state definition used by the historical BP3 run, in SI units.
#ifndef BP3_FIRST_EVENT_H
#define BP3_FIRST_EVENT_H

namespace BP3
{
  struct FirstEvent
  {
    double onset=-1., completion=-1.;
    unsigned int below=0;
    bool observe(double clock,double peak)
    {
      constexpr double threshold=1.e-3;
      if (completion>=0.)
        return false;
      if (onset<0.)
        {
          if (peak<threshold)
            return false;
          onset=clock;
          return true;
        }
      below=peak<threshold ? below+1 : 0;
      if (below!=5)
        return false;
      completion=clock;
      return true;
    }

    template<class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &onset &completion &below;
    }
  };
}
#endif
