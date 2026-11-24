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
      explicit FaultPhaseProfile(const MaterialFaultTrace<dim> &trace) : geometry(trace) {}

      /** The same association and inverse-length weight act in both directions. */
      struct Sample
      {
        typename MaterialFaultTrace<dim>::Projection foot;
        double chi;
        double rate(const std::vector<double> &rates) const;
        void add_load(double integrated_traction, std::vector<double> &rows) const;
      };

      /** Integrate h along both physical normal rays at three Gauss stations
       * per surface panel. Collectively check phase, quadrature convergence and the
       * discarded fraction before projecting with the consistent Q1 mass. */
      void initialize(const SimulatorAccess<dim> &sim, unsigned int phase_field,
                      const LinearAlgebra::BlockVector &fields, double magnitude, unsigned int station_subdivisions = 1);
      double weight(double phase) const;
      Sample sample(const Point<dim> &x, double phase) const;
      Tensor<2,dim> slip_gradient(const Sample &sample, double rate) const;
      const MaterialFaultTrace<dim> &geometry;
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
