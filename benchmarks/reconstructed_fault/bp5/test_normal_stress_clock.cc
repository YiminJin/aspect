#include "normal_stress_clock.h"
#include <iostream>
#include <functional>

int main()
{
  using namespace aspect::BP5NormalStress;
  const double start=5310111071.5634108;
  double time=start;
  std::ostringstream csv;
  csv<<std::setprecision(17)<<"step,time_s,dt\n";
  const double dt[]={.0028219241006433027,.0028144282043784941,.0028069639581435046,.0027995311884659868};
  for (unsigned int i=0;i<4;++i) { time+=dt[i];csv<<5613+i<<','<<time<<','<<dt[i]<<'\n'; }
  const auto clock=read_half_step_clock(csv.str(),5612,start,4);
  const auto require=[](bool ok) { if (!ok) throw std::runtime_error("Clock regression failed"); };
  const auto rejects=[&](const std::function<void()> &f)
  { bool threw=false;try { f(); } catch (const std::exception &) { threw=true; }require(threw); };
  double previous=start;
  for (unsigned int i=0;i<8;++i)
    {
      const auto &c=clock.halves[i];require(previous+c.dt==c.time);
      check_clock(c,c.step,c.time,c.dt);
      if (i%2) require(c.time==clock.original[i/2].time);
      else require(c.dt==dt[i/2]/2.);
      previous=c.time;
    }
  require(previous==time);
  rejects([&] { read_half_step_clock(csv.str(),5611,start,4); });
  rejects([&] { read_half_step_clock(csv.str(),5612,start,3); });
  rejects([&] { read_half_step_clock(csv.str(),5612,start,5); });
  rejects([&] { read_half_step_clock("step,time_s,dt\n5613,nan,1\n",5612,start,1); });
  rejects([&] { read_half_step_clock("step,time_s,dt\n5613,1,0\n",5612,start,1); });
  rejects([&] { read_half_step_clock("step,time_s,dt\n5613,1,1junk\n",5612,0,1); });
  rejects([&] { const auto &c=clock.halves[1];check_clock(c,c.step,c.time,c.dt*.9); });
  std::cout<<"Clock parsing, eight endpoints, rounding adjustment and safety-reduction rejection passed.\n";

  std::ostringstream intervals;
  intervals<<std::setprecision(17);
  for (unsigned int i=0;i<10;++i) intervals<<dt[i%4]<<'\n';
  const auto original=read_filter_intervals(intervals.str(),10,false);
  const auto retry=read_filter_intervals(intervals.str(),10,true);
  double original_elapsed=0.,retry_elapsed=0.;
  require(retry.size()==10);
  for (unsigned int i=0;i<10;++i)
    {
      require(original[i]==dt[i%4] && retry[i]==original[i]*.5);
      original_elapsed+=original[i];retry_elapsed+=retry[i];
      check_filter_interval(5613+i,retry[i],retry[i]);
    }
  require(retry_elapsed==original_elapsed*.5);
  rejects([&] { read_filter_intervals(intervals.str(),9,true); });
  rejects([&] { read_filter_intervals(intervals.str(),11,true); });
  rejects([&] { read_filter_intervals(intervals.str()+"1e999",10,true); });
  rejects([&] { read_filter_intervals("",0,true); });
  for (const auto bad : {"0", "-1", "nan", "inf", "1junk"})
    rejects([&] { read_filter_intervals(bad,1,true); });
  rejects([&] { check_filter_interval(5614,.00281443,.00279062); });
  rejects([&] { check_filter_interval(5614,retry[1],retry[1]*.99); });
  rejects([&] { check_filter_interval(5614,retry[1],retry[1]*1.01); });
  try { check_filter_interval(5614,.00281443,.00279062); }
  catch (const std::exception &e)
    { require(std::string(e.what()).find("step=5614, expected_dt=")!=std::string::npos); }
  std::cout<<"Filter retry: ten half-sized intervals, unchanged raw clock, half duration, malformed-clock and mismatch rejection passed.\n";
}
