/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_reconstructed_fault_stationary_fault_h
#define _aspect_reconstructed_fault_stationary_fault_h

#include <aspect/global.h>
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <deal.II/base/symmetric_tensor.h>

namespace aspect
{

  /** A prescribed straight Q1 curve spanning a Cartesian box in its tangential
   * direction. Label identity is its index in the ordered list of vertices.
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
        double interpolate(double a, double b) const;
      };

      /** One owned bulk quadrature point in the section balance. The weight
       * is (h/I) JxW. Driving shear excludes the current slip increment. */
      struct BulkPoint
      {
        Projection foot;
        double weight, driving_shear, compression;
      };

      /** Set the prescribed geometry and uniform initial state in physical units. */
      void initialize(const Point<dim> &origin, const Point<dim> &extent,
                      const std::vector<Point<dim>> &vertices, double age, double rate, double slip = 0.);
      /** True if a cell spans only one Q1 segment in the tangential direction. */
      bool resolves_segments(const Point<dim> &lower, const Point<dim> &upper) const;
      /** Find the two neighboring material labels and the local Q1 coordinate. */
      Projection project(const Point<dim> &point) const;

      /** Full-normal h/I in 1/m for a spatially constant positive phase. */
      double localization() const;
      /** Slip dyad t (x) n with the positive slip sense along t. */
      const Tensor<2,dim> &slip_dyad() const;
      /** Unit left normal to the oriented label sequence. */
      const Tensor<1,dim> &normal_vector() const;

      /** Solve the consistent Q1 weak balance at a fixed Stokes iterate.
       * Only the tridiagonal rows are summed across MPI ranks. */
      std::vector<double> solve_rates(const std::vector<BulkPoint> &points,
                                      const MaterialModel::Rheology::RateAndStateFriction &law,
                                      double kappa, double minimum_speed, MPI_Comm comm) const;

      /** Publish one endpoint per label using the same accepted age that
       * entered the Q1 friction law throughout this attempt. */
      void accept(const std::vector<double> &rates, double dt,
                  const MaterialModel::Rheology::RateAndStateFriction &law);
      const std::vector<Label> &labels() const;
      void save(oarchive &archive) const;
      void load(iarchive &archive);

    private:
      std::vector<Label> accepted;
      std::vector<double> arclength;
      Tensor<1,dim> tangent, normal;
      Tensor<2,dim> slip_gradient;
      double width;
  };
}
#endif
