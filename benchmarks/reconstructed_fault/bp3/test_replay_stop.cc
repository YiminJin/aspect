#include "replay_stop.h"
#include <cassert>
#include <iostream>

int main()
{
  const double target=922804465.59751725;
  assert(BP3::replay_time_reached(922804465.59751701,target));
  assert(BP3::replay_time_reached(target,target));
  assert(BP3::replay_time_reached(std::nextafter(target,INFINITY),target));
  assert(!BP3::replay_time_reached(target-1e-4,target));
  assert(!BP3::replay_time_reached(483028744.22168565,target));
  assert(!BP3::replay_time_reached(-INFINITY,target));
  assert(!BP3::replay_time_reached(NAN,target));
  assert(BP3::replay_time_reached(0.,0.));
  std::cout<<"BP3 accepted-clock guard: captured two-ULP remainder, equality, overshoot, physical gap and unaccepted state pass.\n";
}
