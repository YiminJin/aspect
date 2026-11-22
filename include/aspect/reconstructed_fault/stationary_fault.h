/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_reconstructed_fault_stationary_fault_h
#define _aspect_reconstructed_fault_stationary_fault_h

#include <aspect/simulator_access.h>
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <deal.II/base/symmetric_tensor.h>

namespace aspect
{

  /** A stationary, oriented Q1 trace joining two contacts of a Cartesian box.
   * Segment frames and arclengths are reconstructed from its vertices. Material
   * label identity is its index in the ordered list of vertices.
   * Geometry and accepted history are replicated on the simulator communicator.
   */
  template <int dim>
  class StationaryFault
  {
    public:
      /** Persistent SI state of one material vertex at the last accepted endpoint. */
      struct Label
      {
        Point<dim> position;
        double age, slip, rate; // s, m, m/s
      };

      /** Normal projection onto the segment joining labels left and left+1. */
      struct Projection
      {
        unsigned int left;
        double xi;
        bool active;
        double interpolate(double a, double b) const;
      };

      /** Frame on one segment, with arclength and length in metres. */
      struct Segment
      {
        double station, length;
        Tensor<1,dim> tangent, normal;
        Tensor<2,dim> dyad;
      };
      const Segment &segment(unsigned int j) const;
      /** Fit normal offsets to the initialized FE ridge once. Keep endpoints on
       * their seed contact faces, and retain the seed for constant phase. */
      void fit(const SimulatorAccess<dim> &sim, unsigned int phase_field,
               const LinearAlgebra::BlockVector &fields);
      /** Set the prescribed geometry and uniform initial state in physical units. */
      void initialize(const Point<dim> &origin, const Point<dim> &extent,
                      const std::vector<Point<dim>> &vertices, double age, double rate, double half_width = std::numeric_limits<double>::max());
      /** Find the two neighboring material labels and the local Q1 coordinate. */
      Projection project(const Point<dim> &point) const;

      /** Publish one endpoint per label using the same accepted age that
       * entered the Q1 friction law throughout this attempt. */
      void accept(const std::vector<double> &rates, double dt,
                  const MaterialModel::Rheology::RateAndStateFriction &law);
      const std::vector<Label> &labels() const;

    private:
      std::vector<Label> accepted;
      void rebuild_segments();
      std::vector<Segment> segments;
      std::array<unsigned int,2> contact_axes;
      double half_width;
      std::vector<double> arclength;
      Tensor<1,dim> tangent, normal;
  };
}
#endif
