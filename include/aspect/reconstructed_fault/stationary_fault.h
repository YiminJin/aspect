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

  /** An oriented Q1 material trace in a Cartesian box, fixed at two contacts
   * or advected between two interior material endpoints.
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
        double half_width, initial_shear, memory; // material normal radius (m), prestress (Pa), Maxwell slip memory (Pa m)
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
      /** Seed normal radius in metres. The moving strip interpolates current label widths. */
      double support_radius() const
      {
        return half_width;
      }
      const Segment &segment(unsigned int j) const;
      /** Collectively fit normal offsets to two-sided observations of the
       * initialized FE ridge. Constrain both crossings to the one-sided
       * phase maxima on their supplied faces. Constant phase retains the seed. */
      void fit(const SimulatorAccess<dim> &sim, unsigned int phase_field,
               const LinearAlgebra::BlockVector &fields);
      /** Set the prescribed geometry and uniform initial state in physical units. */
      void initialize(const Point<dim> &origin, const Point<dim> &extent,
                      const std::vector<Point<dim>> &vertices, double age, double rate, double half_width = std::numeric_limits<double>::max(),
                      bool interior = false);
      /** Carry each finite material vertex and its normal support with the
       * accepted FE velocity and deformation gradient. Collective on all ranks. */
      void advect(const SimulatorAccess<dim> &sim, double dt);
      /** Install ordered material histories and reconstruct the segment frames. */
      void set_labels(std::vector<Label> labels);
      /** Find the two neighboring material labels and the local Q1 coordinate. */
      Projection project(const Point<dim> &point) const;

      /** On a Cartesian cell cut by a finite segment or support edge, return
       * a unit-cell rule split at the association boundaries. An empty rule
       * selects the usual tensor-product quadrature on smooth cells. */
      Quadrature<dim> cut_quadrature(const Point<dim> &lower, const Point<dim> &upper) const;

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
