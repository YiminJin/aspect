/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_reconstructed_fault_linear_performance_h
#define _aspect_reconstructed_fault_linear_performance_h

#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <aspect/global.h>

namespace aspect
{
  namespace internal
  {
    /** Opt-in, rank-local exclusive wall times for one coupled linearization.
     * Nested actions suspend their caller's clock; no MPI timing collectives. */
    struct FaultLinearTiming
    {
      enum Part { factor, inverse, A, B, G, preconditioner, krylov_vectors, other,
                  B_setup, G_setup, B_sparse, G_sparse, interface_setup, interface_apply, count };
      using Clock = std::chrono::steady_clock;
      bool active = false;
      Part current = other;
      Clock::time_point last;
      std::array<double,count> seconds{};
      std::array<unsigned long long,count> calls{};
      double B_relative_error=0., G_relative_error=0.;

      static FaultLinearTiming &get()
      {
        static thread_local FaultLinearTiming timing;
        return timing;
      }
      void charge()
      {
        const auto now = Clock::now();
        seconds[current] += std::chrono::duration<double>(now-last).count();
        last = now;
      }
    };

    class FaultLinearSection
    {
      public:
        explicit FaultLinearSection(const FaultLinearTiming::Part part, const bool selected=true)
          : timing(FaultLinearTiming::get()), enabled(timing.active && selected), parent(timing.current)
        {
          if (enabled)
            { timing.charge(); timing.current=part; ++timing.calls[part]; }
        }
        ~FaultLinearSection()
        {
          if (enabled) { timing.charge(); timing.current=parent; }
        }
      private:
        FaultLinearTiming &timing;
        const bool enabled;
        const FaultLinearTiming::Part parent;
    };

    class FaultLinearProfile
    {
      public:
        FaultLinearProfile(ConditionalOStream &output, const unsigned int step,
                           const unsigned int iteration)
          : output(output), step(step), iteration(iteration)
        {
          auto &t=FaultLinearTiming::get();
          t=FaultLinearTiming();
          t.active=std::getenv("ASPECT_FAULT_LINEAR_PERFORMANCE");
          t.last=FaultLinearTiming::Clock::now();
        }
        ~FaultLinearProfile() { report(); }
        void report()
        {
          auto &t=FaultLinearTiming::get();
          if (!t.active) return;
          t.charge(); t.active=false;
          const char *names[]={"factor","inverse","A","B","G","preconditioner","FGMRES_vectors","other",
                               "B_setup","G_setup","B_sparse","G_sparse","interface_setup","interface_apply"};
          std::ostringstream line;
          line << std::setprecision(17) << "Fault linear profile: step=" << step << ", newton=" << iteration;
          double total=0.;
          for (unsigned int i=0;i<FaultLinearTiming::count;++i)
            { total+=t.seconds[i]; line << ", " << names[i] << "_s=" << t.seconds[i]
                                      << ", " << names[i] << "_calls=" << t.calls[i]; }
          line << ", elapsed=" << total << ", B_relative_error=" << t.B_relative_error
               << ", G_relative_error=" << t.G_relative_error;
          output << line.str() << std::endl;
        }
      private:
        ConditionalOStream &output;
        const unsigned int step,iteration;
    };
  }
}
#endif
