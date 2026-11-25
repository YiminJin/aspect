/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_reconstructed_fault_material_fault_trace_h
#define _aspect_reconstructed_fault_material_fault_trace_h

#include <aspect/simulator_access.h>
#include <deal.II/base/symmetric_tensor.h>

namespace aspect
{
  /**
   * One open, piecewise linear fault curve in a two-dimensional Cartesian
   * box. The index of a vertex in the ordered list is its material identity
   * throughout the run, including when it enters a different cell or rank.
   * Segment j points from vertex j to j+1. Its unit normal is the tangent
   * rotated counterclockwise by 90 degrees. A normal offset is the signed
   * distance along that normal from the segment to a bulk point. Normal
   * support is the interval of offsets admitted for the bulk/section transfer.
   *
   * A boundary contact is the endpoint where a boundary-spanning seed meets
   * a box face. At initialization its vertices can be fitted to the phase
   * ridge, the band of elevated values in the bulk finite-element phase field
   * that locates the diffuse fault. A prescribed interior seed has two
   * material endpoints inside the box. It can remain fixed or move with the
   * accepted bulk velocity.
   *
   * Every rank stores the complete ordered labels, their slip and speed,
   * and the derived segment frames and arclengths. PhaseFieldFault owns the
   * accepted trace and profile. For an interior seed it also owns a trial
   * copy, bound once per timestep attempt. A retry discards that copy.
   * Publication installs the converged trace and surface history together.
   * The Simulator owns the distributed bulk phase, velocity and stress.
   */
  template <int dim>
  class MaterialFaultTrace
  {
    public:
      /** Physical state attached to one material vertex in this copy of the trace. */
      struct Label
      {
        Point<dim> position;  ///< Spatial position in metres.
        double slip;          ///< Accumulated slip in metres.
        double rate;          ///< Section slip speed in m/s.
        double half_width;    ///< Material normal radius in metres, or zero for the full in-box ray.
      };

      /** Normal projection onto the segment starting at first_vertex. */
      struct Projection
      {
        unsigned int first_vertex;
        double xi;   ///< Q1 coordinate, zero at the first vertex and one at the next.
        bool active; ///< The point falls within the segment's admitted normal support.

        /** Interpolate the two vertex values at xi. Local, read-only. */
        double interpolate(double a, double b) const;
      };

      /** Current oriented segment frame and arclength from the first material vertex. */
      struct Segment
      {
        double station, length; ///< Arclength at the first vertex and segment length, in metres.
        Tensor<1,dim> tangent, normal;
        Tensor<2,dim> dyad;     ///< Tangent outer normal, used in the bulk slip gradient.
      };

      /**
       * Initialize a straight, ordered seed with uniform rate (m/s)
       * and zero accumulated slip. A finite support is the initial normal
       * radius in metres. The default admits the full in-box cross-section.
       * Interior vertices must all lie strictly inside the box. Otherwise the
       * first and last vertices specify the two boundary contacts. Local;
       * all ranks supply the same inputs before the first solve.
       */
      void initialize(const Point<dim> &origin, const Point<dim> &extent,
                      const std::vector<Point<dim>> &vertices, double rate,
                      double support = std::numeric_limits<double>::max(), bool interior = false);

      /**
       * Fit the boundary-spanning seed to the initialized FE phase ridge
       * before its first Stokes assembly. Set the contacts to the phase maxima
       * on their seed faces and displace the interior vertices along the seed
       * normal. Constant phase retains the seed. Collective on the simulator
       * communicator; the simulator and fields are borrowed for this call.
       */
      void fit_boundary_spanning_seed_to_phase_ridge(const SimulatorAccess<dim> &sim,
                                                     unsigned int phase_field,
                                                     const LinearAlgebra::BlockVector &fields);

      /**
       * Move the trial interior vertices over dt seconds using accepted FE
       * velocity and deform their finite normal radii with its gradient.
       * Rebuild the segment frames, preserving each vertex's history and
       * index. Called once per moving attempt before binding its phase
       * profile. Collective on the simulator communicator.
       */
      void advect_with_accepted_velocity(const SimulatorAccess<dim> &sim, double dt);

      /**
       * Replace the ordered vertex records at initialization, checkpoint
       * restore or acceptance and rebuild their frames. The caller supplies
       * one entry for each existing material identity in its original order.
       * Zero radii denote full normal rays. Finite material radii retain
       * their saved lengths. The support convention must match this trace.
       * Local; the caller gives identical records on every rank.
       */
      void replace_material_labels(std::vector<Label> labels);

      /**
       * Apply the converged rates to this trace's cumulative
       * slips at acceptance. At dt = 0 only the rates change. Local; the
       * section solver has already reduced the rates across ranks.
       */
      void accept_section_rates(const std::vector<double> &rates, double dt);

      /** Read this trace's ordered material records at its current time level. Local. */
      const std::vector<Label> &labels() const;

      /** Read the current frame on the segment starting at vertex j. Local. */
      const Segment &segment(unsigned int j) const;

      /**
       * Read the current normal support radius at coordinate xi on the
       * specified segment (m). Full in-box support returns infinity so the
       * integration endpoints are set by the physical mesh. Local.
       */
      double normal_support_radius(unsigned int segment_index, double xi) const;

      /**
       * Find the closest admitted segment and its bounded Q1 coordinate
       * on this trace. In a tie the earlier segment wins. If no segment
       * admits the point, return the closest with active set to false.
       * Read-only and local during assembly and output.
       */
      Projection project(const Point<dim> &point) const;

      /**
       * Build a reference-cell quadrature for a Cartesian cell split by a
       * segment or support edge, for the boundary-spanning stress projection
       * at acceptance. An empty rule selects ordinary smooth-cell quadrature.
       * Read-only and local to the calling cell owner.
       */
      Quadrature<dim> cut_quadrature(const Point<dim> &lower, const Point<dim> &upper) const;

    private:
      void rebuild_segments();
      std::vector<Label> material_labels;
      std::vector<Segment> segments;
      std::array<unsigned int,2> contact_face_axes; ///< Box face axes at the first and last contacts.
      double initial_half_width;
      std::vector<double> arclength;
      Tensor<1,dim> seed_tangent, seed_normal;
  };
}
#endif
