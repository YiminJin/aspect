#include "first_event.h"
#include <cassert>
#include <iostream>

int main()
{
  BP3::FirstEvent event;
  event.observe(0.,1e-9,0.);
  for (unsigned int i=1;i<20;++i) event.observe(i,1e-8,100.);
  assert(!event.started && !event.complete);
  event.observe(20.,1e-3,2500.);
  assert(event.started && event.onset==20. && event.below==0);
  event.observe(21.,2e-3,5000.);
  event.observe(22.,1e-4,0.);
  event.observe(23.,1e-4,0.);
  event.observe(24.,1e-3,0.); // equality breaks consecutive sub-threshold states
  assert(event.below==0 && event.down_crossing==-1.);
  for (unsigned int i=25;i<29;++i) event.observe(i,1e-4,0.);
  assert(!event.complete && event.below==4);
  BP3::FirstEvent resumed=event;
  resumed.observe(29.,1e-4,0.);
  assert(resumed.complete && resumed.termination==29. && resumed.down_crossing==25.);
  assert(resumed.peak==2e-3 && resumed.peak_time==21. && resumed.peak_xd==5000.);
  resumed.observe(30.,1.,10000.);
  assert(resumed.termination==29. && resumed.peak==2e-3);
  std::cout<<"BP3 accepted-state event observer: onset, equality, re-entry, five-state termination and retained peak pass.\n";
}
