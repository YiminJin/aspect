/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_reconstructed_fault_linear_performance_h
#define _aspect_reconstructed_fault_linear_performance_h

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <aspect/global.h>

namespace aspect
{
  namespace internal
  {
    /** Counts are rank-local, relaxed atomics; no search or sampling semantics
     * depend on them. They distinguish allocation/layout work from value work. */
    struct FaultSamplingCounters
    {
      enum Part { projection_queries, segment_tests, dg_fevalues, native_fevalues,
                  history_cache_misses, mass_inverse, roundoff_geometry,
                  lookup_builds, B_builds, G_builds, bulk_builds,
                  B_reference_checks,G_reference_checks,C_reference_checks,count };
      std::array<std::atomic<unsigned long long>,count> values{};
      static FaultSamplingCounters &get();
      static void add(Part part,unsigned long long n=1)
      { get().values[part].fetch_add(n,std::memory_order_relaxed); }
      std::array<std::atomic<double>,3> maximum_relative_error{};
      static void observe(unsigned int,double);
      static std::string report();
    };

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
                           const unsigned int iteration, const bool force=false)
          : output(output), step(step), iteration(iteration)
        {
          auto &t=FaultLinearTiming::get();
          t=FaultLinearTiming();
          t.active=force || std::getenv("ASPECT_FAULT_LINEAR_PERFORMANCE");
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
          line << ", elapsed=" << total;
          output << line.str() << std::endl;
        }
      private:
        ConditionalOStream &output;
        const unsigned int step,iteration;
    };
  }
}
#endif
