/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_rheology_fault_resistance_h
#define _aspect_material_model_rheology_fault_resistance_h

#include <aspect/global.h>

namespace aspect::MaterialModel::Rheology
{
  /** Traction in Pa and its derivative with respect to positive slip speed
   * (Pa s/m), at fixed incoming constitutive state and compression. */
  struct FaultTraction
  {
    double traction, speed_derivative;
  };

  /** Material-owned surface law. Index i follows the material, independent of
   * mesh ownership. Evaluations are local and read only throughout an attempt.
   * The default law history is empty. Only acceptance advances physical state. */
  class FaultResistance
  {
    public:
      virtual ~FaultResistance() = default;
      /** Stable checkpoint identifier of the law. */
      virtual std::string name() const = 0;
      /** Evaluate at Q1 coordinate xi between vertices i and i+1. */
      virtual FaultTraction at_speed(unsigned int i, double xi, double speed, double compression) const = 0;
      /** Seed optional history once for the ordered vertices or strips. */
      virtual void initialize(unsigned int) {}
      /** Publish one material identity at the accepted signed rate in m/s. */
      virtual void accept_rate(unsigned int, double, double) {}
      /** Optional state, in material identity order, for checkpoints. */
      virtual const std::vector<double> &state() const
      {
        static const std::vector<double> empty;
        return empty;
      }
      virtual void restore_state(std::vector<double> values)
      {
        AssertThrow(values.empty(),ExcMessage("A state-free fault law cannot restore constitutive history."));
      }
  };
}
#endif
