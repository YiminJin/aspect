// Benchmark-only clock parsing; independent of constitutive state and ASPECT.
#ifndef ASPECT_BP5_NORMAL_STRESS_CLOCK_H
#define ASPECT_BP5_NORMAL_STRESS_CLOCK_H

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aspect::BP5NormalStress
{
  // The filter retry halves each recorded interval once. It retains the count
  // and therefore covers half the elapsed interval, unlike the paired halves
  // below that preserve each original endpoint and double the step count.
  inline std::vector<double> read_filter_intervals(const std::string &text,
                                                  const unsigned int count,
                                                  const bool halve)
  {
    std::istringstream input(text);
    std::vector<double> intervals;
    std::string token;
    while (input>>token)
      {
        size_t used=0;
        double dt=std::stod(token,&used);
        if (used!=token.size())
          throw std::runtime_error("Malformed actual interval in the filter replay clock.");
        if (halve) dt*=0.5;
        if (!std::isfinite(dt) || !(dt>0.))
          throw std::runtime_error("Invalid actual interval in the filter replay clock.");
        intervals.push_back(dt);
      }
    if (!input.eof() || count==0 || count>10 || intervals.size()!=count)
      throw std::runtime_error("Filter replay needs the complete bounded actual-dt clock.");
    return intervals;
  }

  inline void check_filter_interval(const unsigned int step,
                                    const double expected, const double actual)
  {
    if (!(std::abs(actual-expected)<=16*std::numeric_limits<double>::epsilon()*expected))
      {
        std::ostringstream message;
        message<<std::setprecision(17)<<"Filter clock mismatch before mechanics: step="<<step
               <<", expected_dt="<<expected<<", actual_dt="<<actual
               <<", difference="<<actual-expected<<" s. Inspect timestep_selection.csv; "
               <<"preserve safety restrictions and stop without forcing acceptance.";
        throw std::runtime_error(message.str());
      }
  }

  struct ClockEntry
  {
    unsigned int step;
    double time, dt;
  };

  struct HalfStepClock
  {
    std::vector<ClockEntry> original, halves;
  };

  inline HalfStepClock read_half_step_clock(const std::string &csv,
                                            const unsigned int checkpoint_step,
                                            const double checkpoint_time,
                                            const unsigned int original_steps)
  {
    const auto columns=[](const std::string &line)
    {
      std::vector<std::string> result;
      std::istringstream row(line);
      std::string field;
      while (std::getline(row,field,',')) result.push_back(field);
      return result;
    };
    std::istringstream input(csv);
    std::string line;
    std::getline(input,line);
    const auto header=columns(line);
    const auto index=[&](const std::string &name)
    {
      const auto it=std::find(header.begin(),header.end(),name);
      if (it==header.end()) throw std::runtime_error("Reference trajectory is missing column "+name);
      return static_cast<unsigned int>(it-header.begin());
    };
    const auto step_column=index("step"),time_column=index("time_s"),dt_column=index("dt");
    const auto number=[](const std::string &field)
    {
      size_t used=0;
      const double value=std::stod(field,&used);
      if (!std::isfinite(value) || field.find_first_not_of(" \t\r",used)!=std::string::npos)
        throw std::runtime_error("Nonfinite or malformed reference clock value");
      return value;
    };
    HalfStepClock clock;
    double previous=checkpoint_time;
    while (std::getline(input,line))
      {
        if (line.find_first_not_of(" \t\r")==std::string::npos) continue;
        const auto row=columns(line);
        if (row.size()!=header.size()) throw std::runtime_error("Incomplete reference trajectory row");
        const unsigned int i=clock.original.size();
        const double step=number(row[step_column]),time=number(row[time_column]),dt=number(row[dt_column]);
        if (i>=original_steps || step!=checkpoint_step+i+1 || !(dt>0.) || previous+dt!=time)
          throw std::runtime_error("Reference trajectory is not the requested complete, contiguous accepted clock");
        const double first=dt/2.,middle=previous+first,second=time-middle;
        const double ulp=std::nextafter(time,std::numeric_limits<double>::infinity())-time;
        if (!(previous<middle && middle<time) || middle+second!=time || std::abs(second-first)>ulp)
          throw std::runtime_error("Half steps cannot preserve the represented reference endpoint");
        clock.original.push_back({checkpoint_step+i+1,time,dt});
        clock.halves.push_back({checkpoint_step+2*i+1,middle,first});
        clock.halves.push_back({checkpoint_step+2*i+2,time,second});
        previous=time;
      }
    if (clock.original.size()!=original_steps || original_steps==0)
      throw std::runtime_error("Reference trajectory has not completed the required accepted steps");
    return clock;
  }

  inline void check_clock(const ClockEntry &expected, const unsigned int step,
                          const double time, const double dt)
  {
    if (expected.step!=step || expected.time!=time || expected.dt!=dt)
      {
        std::ostringstream message;
        message<<std::setprecision(17)<<"Half-step clock changed before mechanics: actual (step,time,dt)="
               <<step<<','<<time<<','<<dt<<"; expected="<<expected.step<<','<<expected.time<<','<<expected.dt
               <<". Preserve the safety reduction; do not force a larger timestep or add solves.";
        throw std::runtime_error(message.str());
      }
  }
}
#endif
