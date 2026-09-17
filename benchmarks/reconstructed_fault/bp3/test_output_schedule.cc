#include "output_schedule.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <sstream>
#include <cassert>
#include <iostream>
int main()
{
  BP3::OutputSchedule s;
  assert(s.due(0.,{0.,0.})); s.written(0.,{1.,.95});
  // The maximum location changes; difference of spatial maxima would miss it.
  assert(s.due(1.,{1.01,1.06}));
  assert(!s.due(1.,{1.01,1.04}));
  assert(s.last_time==0. && s.reference[1]==.95); // rejected trial did not advance
  assert(s.due(2.,{1.,1.051}));
  s.written(2.,{1.,1.051});
  assert(!s.due(3.,{1.,1.052}));
  assert(s.due(3.,{1.,1.052},true)); // graceful final output
  assert(s.due(2.+31557600.,{1.,1.052}));
  s.time_interval=0.; assert(!s.due(1e10,{1.,1.052}));
  std::stringstream stream;
  { boost::archive::text_oarchive a(stream); a<<s; }
  BP3::OutputSchedule restored;
  { boost::archive::text_iarchive a(stream); a>>restored; }
  assert(restored.reference==s.reference && restored.last_time==s.last_time);
  assert(restored.due(4.,{1.101,1.052}));
  std::cout<<"Output schedule: moving maximum, threshold, rejected candidate, final/time trigger and save/load passed.\n";
}
