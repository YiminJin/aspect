/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

/* Internal implementation data for the restricted accepted FE-history path. */
#ifndef _aspect_material_model_phase_field_fault_fe_strain_internal_h
#define _aspect_material_model_phase_field_fault_fe_strain_internal_h

#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      /** Analytic normal-ray integral and derivatives with respect to its two endpoint phases. */
      struct FENormalIntegral
      {
        double v = 0.; ///< Integral (m).
        std::array<double, 2> d {}; ///< First endpoint derivatives (m).
        std::array<std::array<double, 2>, 2> h {}; ///< Endpoint Hessian (m).
      };
    }
    /** Scratch evaluation of I_h and optional dense/directional phase derivatives. */
    template <int dim>
    struct PhaseFieldFault<dim>::FENormalization
    {
      Vector<double> phase_values; ///< Replicated Q1 phase used by the fixed normal rays.
      std::vector<double> values; ///< Nodal I_h (m).
      std::vector<double> action; ///< Directional derivative of nodal I_h (m).
      FullMatrix<double> derivative; ///< Nodal-I derivative by phase DoF.
      FullMatrix<double> derivative_scale; ///< Absolute accumulation for roundoff bounds.
      std::vector<std::vector<internal::FENormalIntegral>> integrals; ///< Profile/ray jets.
      bool admissible = true; ///< False if physical phase/normalization checks fail.
    };

    /**
     * Material-side FE/profile owner. Normally accepted; during a moving attempt
     * a private candidate is bound for mechanics while DGMaterialTrial keeps the
     * accepted parent alive. BulkHistory holds constitutive data; ObjectiveAnchor
     * and Observations are rebuildable. Fixed-geometry integration data stay here
     * because normal rays, projection and cell maps are rebuilt together.
     *
     * Persistence is explicit in history.cc/dg_history.cc, not implicit member
     * serialization. Cache grouping must not change archive packet order.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::FEStrainState
    {
      /** One cell-clipped normal-ray interval for exact Q1 endpoint integration. */
      struct Ray
      {
        std::vector<types::global_dof_index> dofs;
        std::array<std::vector<double>, 2> shape;
        double length; ///< Interval length (m).
        double lower; ///< Signed starting normal coordinate (m).
        template<class Archive> void serialize(Archive &archive, const unsigned int)
        {
          archive &dofs &shape &length &lower;
        }
      };
      /** Normal section attached to one surface quadrature point. */
      struct Profile
      {
        unsigned int segment;
        double xi; ///< Segment interpolation coordinate.
        double weight; ///< Surface quadrature weight (m).
        std::vector<Ray> rays;
      };
      /** Bulk quadrature point and its fixed geometry/profile association. */
      struct BulkPoint
      {
        CellId cell;
        Point<dim> x;
        double weight; ///< Bulk integration measure (m^2 in 2-D).
        double xi; ///< Surface-segment coordinate.
        unsigned int segment;
        std::vector<types::global_dof_index> dofs;
        std::vector<double> shape;
      };
      /** Alternative bulk-history discretizations plus their shared FE profile. */
      struct BulkHistory
      {
        /**
         * Ghost-readable mixed FE vector. In native FE mode the velocity block
         * is the accepted displacement-like Maxwell memory d_R (m), and the
         * phase block is the physical profile/irreversibility obstacle. In DG
         * mode velocity entries remain zero and only the phase block is used;
         * tensor coefficients below are then authoritative. Never interpret the
         * entire mixed vector as a phase scalar or a physical stress.
         */
        LinearAlgebra::BlockVector native;
        /** Authoritative DG material history, or null in the native FE path. */
        std::shared_ptr<DGHistory> tensor;
      };
      /** Accepted bulk fields, or the privately bound geometry/profile candidate. */
      BulkHistory bulk_history;

      /** Derived objective anchor for one attempted interval, never checkpointed. */
      struct ObjectiveAnchor
      {
        /** Fixed transported/rotated parent used by every nonlinear candidate. */
        std::shared_ptr<DGHistory> tensor;
        /** Parent identity; weak ownership prevents extending an obsolete history. */
        std::weak_ptr<DGHistory> parent;
        /** Attempt interval length (s); a retry with another dt must rebuild. */
        double dt = -1.;
        /** Simulator step key; invalid means no prepared physical-step anchor. */
        unsigned int step = numbers::invalid_unsigned_int;
      };
      /** Scratch for this attempt; binding it does not accept the step. */
      ObjectiveAnchor objective;
      /** Mature auxiliary slip memory (Pa m); diagnostic only for direct tau.
       * In cohesive P2 this instead means I_h*tau_coh; never identify the laws. */
      std::vector<double> Z;
      /** Accepted/bound normal localization integral I_h at surface nodes (m). */
      std::vector<double> I;
      double shear_sense = 1.; ///< Signed slip orientation for the fixed fault.
      std::vector<Profile> profiles; ///< Fixed normal sections for normalization.
      std::vector<BulkPoint> bulk; ///< Bulk integration/phase pullback points.
      FullMatrix<double> projection; ///< Consistent surface mass projection.
      /** Locally owned native FE cells for collective assembly. */
      std::map<CellId, typename DoFHandler<dim>::active_cell_iterator> cells;
      /** Readable native FE cells, including ghosts for observation. */
      std::map<CellId, typename DoFHandler<dim>::active_cell_iterator> evaluation_cells;
      unsigned long long n_cells_global = 0; ///< Mesh identity guard.
      unsigned long long n_particles_global = 0; ///< Particle-layout identity guard.
      std::map<types::particle_index, Point<dim>> particles; ///< Reference particle positions (m).
      std::vector<Point<dim>> vertices; ///< Bound surface vertices (m).
      std::vector<Point<dim>> mesh_vertices; ///< Reference mesh coordinates (m).
      struct ParticleSamples;
      struct MechanicalSamples;

      /** Phase/localization observations independent of the stored stress value. */
      struct TensorPointSamples
      {
        /** One physical point with Q2 basis values and its fault/profile data. */
        struct Sample
        {
          /** Spatial coordinate (m) used as an exact cache key. */
          Point<dim> point;
          /** Nine scalar Q2 basis values for the supported 2-D tensor cell. */
          std::array<double, 9> shape;
          /** Dimensionless Q1 phase sampled at this point. */
          double phase;
          /** Derived fault segment/normal-profile association at this point. */
          ReconstructedFaultUtilities::NormalProfileProjection source;
        };
        /** Manager association generation; invalidated on material motion/transfer. */
        unsigned int geometry_generation = numbers::invalid_unsigned_int;
        /** Multiple point layouts per cell; adding diagnostics does not evict others. */
        std::map<CellId, std::vector<Sample>> cells;
      };

      /**
       * Rebuildable observations, never independent constitutive history.
       * Copying a state copies stress samples and shares the pointer caches, as
       * before this grouping. Material motion/publication invalidates them at
       * the existing call sites; transfer creates new FE data and clears tensor
       * points. Mechanical sampling serializes access with its existing mutex.
       */
      struct Observations
      {
        /** Physical stress samples of the immutable native anchor; not serialized. */
        mutable std::map<std::pair<CellId, std::array<double, dim>>, SymmetricTensor<2, dim>> stress;
        /** Particle lookup/basis data; reset after accepted-history changes. */
        mutable std::shared_ptr<ParticleSamples> particles;
        /** Mechanical quadrature layouts and stresses; protected by the sampling mutex. */
        mutable std::shared_ptr<MechanicalSamples> mechanics;
        /**
         * Shared by state clones, retaining the former shared-space cache lifetime.
         * Population is outside threaded mechanical assembly. Contains no stress
         * coefficients, and is invalidated explicitly on geometry changes.
         */
        std::shared_ptr<TensorPointSamples> tensor_points = std::make_shared<TensorPointSamples>();
      };
      /** Derived cache group; restart rebuilds it instead of reading archive data. */
      Observations observations;
      std::uint64_t geometry_version; ///< Association generation for fixed-geometry guards.
      types::global_dof_index n_dofs; ///< Mixed native FE vector size.
      double m = 0.; ///< Dimensionless compliance/localization parameter.
      double G = 0.; ///< Shear modulus (Pa).
      double eta = 0.; ///< Reference viscosity (Pa s).
      double initial_fracture = 0.; ///< Initial fracture-energy reference (J/m in 2-D).
      unsigned int committed_step = 0; ///< Last physical publication step.
      std::uint64_t accepted_publications = 0; ///< Physical endpoint publication count.
      /** Prescribed geometry commits, distinct from physical Maxwell publications. */
      std::uint64_t geometry_publications = 0;
    };

    /** Unpublished endpoint fields; publication swaps these into the bound owner. */
    template <int dim>
    struct PhaseFieldFault<dim>::FEStrainUpdate
    {
      /** Candidate bulk fields in the same representation as the accepted parent. */
      typename FEStrainState::BulkHistory bulk_history;
      std::vector<double> Z; ///< Candidate slip memory (Pa m), or cohesive I*q.
      std::vector<double> I; ///< Candidate normal integral (m).
      /** Preconditions for the retained closed-boundary prescribed Strang control. */
      struct PrescribedInterval
      {
        /** Accepted material parent; compared by identity before publication. */
        std::shared_ptr<const DGHistory> parent;
        /** Candidate cohesive traction (Pa) at surface labels. */
        std::vector<double> cohesive;
        /** Prescribed label slip rates (m/s), fixed during this interval. */
        std::vector<double> rates;
        /** Simulator step that created this candidate. */
        unsigned int step = 0;
        /** Start time (s), unlike MaterialHistoryInputs::end_time. */
        double start_time = 0.;
        /** Interval length (s); start_time+dt equals the simulator endpoint. */
        double dt = 0.;
      };
      /** Unused by the ordinary mature endpoint path; not serialized. */
      PrescribedInterval prescribed;
    };

    /** Retained prescribed geometry transaction, distinct from physical reaction. */
    template <int dim>
    struct DGGeometryUpdate
    {
      /** Accepted parent retained until the geometry candidate is validated. */
      std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainState> incoming;
      /** Prepared profile/normalization/history candidate; unpublished initially. */
      std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainState> next;
      /** Candidate manager; after swapping, retains the former geometry. */
      std::shared_ptr<ReconstructedFaultManager<dim>> manager;
      std::vector<std::vector<double>> normalization; ///< Candidate nodal I_h (m).
      std::vector<std::vector<double>> surface_temperatures; ///< Candidate temperatures (K).
      std::string parent_surface; ///< Serialized parent identity guard.
      bool published = false; ///< Prevents repeated geometry publication.
      bool identity = false; ///< Geometry move is an exact identity.
    };


  }
}
#endif
