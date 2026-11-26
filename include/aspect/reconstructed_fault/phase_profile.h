/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_reconstructed_fault_phase_profile_h
#define _aspect_reconstructed_fault_phase_profile_h

#include <aspect/reconstructed_fault/material_fault_trace.h>

namespace aspect
{
  /** Finite-strip bulk/section transfer on one oriented trace. ASPECT owns
   * the native DG phase. This operator owns its replicated Q1 normal
   * integrals in metres and uses the same inverse-length weight both ways. */
  template <int dim>
  class FaultPhaseProfile
  {
    public:
      explicit FaultPhaseProfile(const MaterialFaultTrace<dim> &trace, const int slip_sense = 1)
        : geometry(trace), slip_sense(slip_sense)
      {
        AssertThrow(slip_sense == 1 || slip_sense == -1, ExcMessage("Expected a slip sense of +1 or -1."));
      }

      /** The same association and inverse-length weight act in both directions. */
      struct Sample
      {
        typename MaterialFaultTrace<dim>::Projection foot;
        double chi; ///< Excess compliance divided by its projected normal integral (1/m).
        /** Interpolate the two material speeds (m/s) at the projected foot. */
        double rate(const std::vector<double> &rates) const;
        /** Add the two weighted Q1 test contributions to the supplied rows. */
        void add_load(double integrated_traction, std::vector<double> &rows) const;
      };

      /** Integrate h along both physical normal rays at three Gauss stations
       * per surface panel. Collectively check phase, quadrature convergence and the
       * discarded fraction before projecting with the consistent Q1 mass. */
      void initialize(const SimulatorAccess<dim> &sim,
                      unsigned int phase_field,
                      const LinearAlgebra::BlockVector &fields,
                      double degradation_magnitude,
                      unsigned int station_subdivisions = 1);
      double weight(double phase) const;
      Sample sample(const Point<dim> &x, double phase) const;
      /** Unit slip dyad for a segment: slip_sense times the outer product of
       * the tangent and geometric normal. Contract stress with this tensor to obtain
       * the shear traction that drives positive slip speed. */
      Tensor<2,dim> slip_dyad(unsigned int segment_index) const;
      Tensor<2,dim> slip_gradient(const Sample &sample, double rate) const;
      const MaterialFaultTrace<dim> &geometry;
      /** +1 if the positive-normal side slides along increasing material ID
       * relative to the negative-normal side, -1 for the opposite motion. */
      const int slip_sense;
      const std::vector<double> &normal_integrals() const
      {
        return integrals;
      }

    private:
      double magnitude;
      std::vector<double> integrals;
  };
}
#endif
