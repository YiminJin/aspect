// Pure model/observer equivalence across the runtime source extraction.
// The historical implementation remains an independent comparison here only.
#define BP3 HistoricalBP3
#define ASPECT_BP3_RESTORE_150X50
#include "bp3_model.h"
#include "first_event.h"
#include "output_schedule.h"
#undef ASPECT_BP3_RESTORE_150X50
#undef BP3
#undef ASPECT_BENCHMARK_BP3_MODEL_H
#undef ASPECT_BP3_FIRST_EVENT_H
#undef ASPECT_BP3_OUTPUT_SCHEDULE_H
#include "plugin/bp3_model.h"
#include "plugin/first_event.h"
#include "plugin/output_schedule.h"
#include <iostream>
#include <stdexcept>

namespace BP3 { double weakening_length=15000.; }

int main()
{
  unsigned int checks=0;
  const auto equal=[&](auto a,auto b)
    {
      if(a!=b) throw std::runtime_error("Restored runtime changed a model/observer result");
      ++checks;
    };
  for(double x:{-60000.,0.,10000.,28867.5134594813,90000.})
    for(double y:{0.,1000.,25000.,50000.})
      {
        equal(BP3::down_dip(x,y),HistoricalBP3::down_dip(x,y));
        equal(BP3::signed_normal(x,y),HistoricalBP3::signed_normal(x,y));
        equal(BP3::normal_distance(x,y),HistoricalBP3::normal_distance(x,y));
        equal(BP3::depth_fraction(y),HistoricalBP3::depth_fraction(y));
      }
  for(double s:{0.,14999.,15000.,16500.,18000.,40000.,57735.})
    equal(BP3::theta0(s,.008),HistoricalBP3::theta0(s,.008));
  for(double v:{1e-20,1e-12,1e-9,1e-3})
    for(double dt:{0.,75.,100.,4e6})
      equal(BP3::aging_state_reference(v,8000.,dt,.008),
            HistoricalBP3::aging_state_reference(v,8000.,dt,.008));
  BP3::FirstEvent event;
  HistoricalBP3::FirstEvent old_event;
  double t=0.;
  for(double v:{1e-9,1e-3,1e-2,1e-4,1e-4,1e-3,1e-4,1e-4,1e-4,1e-4,1e-4})
    {
      event.observe(++t,v,15000.);old_event.observe(t,v,15000.);
      equal(event.complete,old_event.complete);equal(event.peak,old_event.peak);
      equal(event.down_crossing,old_event.down_crossing);equal(event.below,old_event.below);
    }
  BP3::OutputSchedule output;
  HistoricalBP3::OutputSchedule old_output;
  for(double slip:{0.,.01,.1,.11,.21})
    {
      const std::vector<double> values={0.,slip};
      equal(output.due(t,values),old_output.due(t,values));
      equal(output.increment(values),old_output.increment(values));
      if(output.due(t,values)) {output.written(t,values);old_output.written(t,values);}
      ++t;
    }
  std::cout<<checks<<" exact restored model/event/output comparisons passed.\n";
}
