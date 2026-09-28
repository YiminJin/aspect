/* Benchmark-only helpers retained for the standalone GMG reproduction probes.
 * No production vectors, coefficients, or solver controls are modified. */
#ifndef ASPECT_FAULT_GMG_DIAGNOSTICS_H
#define ASPECT_FAULT_GMG_DIAGNOSTICS_H

#include <deal.II/base/mpi.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>

namespace aspect::internal::FaultGMGDiagnostics
{
  // Inspect bits so finite-math compiler assumptions cannot turn the diagnostic
  // into an unconditional true. GMGNumberType is converted to IEEE double.
  inline bool finite(const double value)
  {
    static_assert(sizeof(double) == sizeof(std::uint64_t)
                  && std::numeric_limits<double>::is_iec559,
                  "GMG diagnostics require IEEE binary64.");
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  }

  // Each rank records its own first bad entry, including global numbering.
  // Reduce only the error flag before throwing, so empty-level ranks participate
  // and no rank unwinds while its peers enter the next diagnostic collective.
  template <typename Number>
  void vector_summary(const char *name,
                      const dealii::LinearAlgebra::distributed::Vector<Number> &v,
                      std::ostream &out,
                      const bool require_positive = false)
  {
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    unsigned int bad = 0;
    dealii::types::global_dof_index nonpositive = 0;
    for (unsigned int i = 0; i < v.locally_owned_size(); ++i)
      {
        const double value = v.local_element(i);
        if (!finite(value) || (require_positive && value <= 0.))
          {
            if (bad == 0)
              out << name << " first_bad_global="
                  << v.locally_owned_elements().nth_index_in_set(i)
                  << " value=" << value << '\n';
            bad = 1;
          }
        if (finite(value))
          {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            nonpositive += value <= 0.;
          }
      }
    out << name << " owned=" << v.locally_owned_size()
        << " nonpositive=" << nonpositive;
    if (v.locally_owned_size() != 0)
      out << " finite_min=" << minimum << " finite_max=" << maximum;
    out << std::endl;
    const unsigned int global_bad = dealii::Utilities::MPI::max(bad, v.get_mpi_communicator());
    AssertThrow(global_bad == 0,
                dealii::ExcMessage(std::string("Fault GMG diagnostic: invalid ") + name
                                   + "; inspect fault_gmg_rank*.log."));
  }
}
#endif
