#ifndef BP3_REPLAY_STOP_H
#define BP3_REPLAY_STOP_H
#include <algorithm>
#include <cmath>
#include <limits>

namespace BP3
{
  // Only accept a roundoff-sized remainder, not a physical time tolerance.
  inline bool replay_time_reached(const double accepted, const double target)
  {
    return std::isfinite(accepted) && accepted >= target
      - 8*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(target));
  }
}
#endif
