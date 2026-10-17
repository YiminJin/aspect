/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef _aspect_material_model_phase_field_fault_dg_history_internal_h
#define _aspect_material_model_phase_field_fault_dg_history_internal_h

#include <aspect/material_model/phase_field_fault.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/sparse_matrix.h>
#include <set>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    struct PhaseFieldFault<dim>::DGHistory
    {
      struct Space
      {
        explicit Space(const Triangulation<dim> &triangulation)
          : element(2), dofs(triangulation)
        {
          dofs.distribute_dofs(element);
          relevant=DoFTools::extract_locally_relevant_dofs(dofs);
          for (const auto &cell : dofs.active_cell_iterators())
            if (!cell->is_artificial())
              cells.emplace(cell->id(),cell);
        }

        FE_DGQ<dim> element;
        DoFHandler<dim> dofs;
        IndexSet relevant;
        std::map<CellId,typename DoFHandler<dim>::active_cell_iterator> cells;
      };

      std::shared_ptr<Space> space;
      // In 2D these are xx, yy, xy in Pa. zz=-(xx+yy), never zero.
      std::array<LinearAlgebra::Vector,SymmetricTensor<2,dim>::n_independent_components> trial_stress;
      /** Scratch forward-error bounds for the native source in a trial only.
       * Not constitutive uncertainty, persistent history, or a limiter. */
      struct SourceRoundoff
      {
        SymmetricTensor<2,dim> projection;
        Tensor<2,dim> gradient_scale;
      };
      std::map<CellId,SourceRoundoff> source_roundoff;
    };


    /** Fixed serial spatial operator; no accepted state or caller velocity is
     * mutated. Quadrature loads already include the local inverse mass. */
    template <int dim> struct PhaseFieldFault<dim>::DGTransport
    {
      struct Cell
      {
        CellId id;
        std::vector<types::global_dof_index> indices;
        FullMatrix<double> inverse_mass;
      };
      struct Load
      {
        Point<dim> point;
        std::vector<types::global_dof_index> indices;
        Vector<double> weights;
      };
      std::shared_ptr<typename DGHistory::Space> space;
      std::vector<Cell> cells;
      SparsityPattern sparsity;
      SparseMatrix<double> matrix,divergence_correction;
      std::vector<Load> source,inflow;
      std::vector<Point<dim>> mesh_vertices;
      unsigned int quadrature_order;
      double inverse_advection_time=0.,relaxation_rate=0.;
    };

    /** Fixed affine Cartesian stencil for the optional nonpublishing experiment.
     * Neighbor polynomials are extrapolated to every target support point. */
    template <int dim> struct PhaseFieldFault<dim>::DGLimiter
    {
      enum class Detector
      {
        fu_shu, kxrcf, force, scaling
      };
      struct Candidate
      {
        std::vector<types::global_dof_index> indices;
        FullMatrix<double> extrapolate;
        // Neighbor-basis averages on every cell in the compact patch and
        // precomputed KKT corrections, indexed by the useful-neighbor mask.
        std::vector<Vector<double>> patch_means;
        std::vector<FullMatrix<double>> constrained_corrections;
      };
      struct FacePoint
      {
        Point<dim> point;
        double weight;
        Vector<double> inside,outside;
        std::vector<types::global_dof_index> neighbor;
      };
      struct Cell
      {
        std::vector<Candidate> candidates;
        std::vector<FacePoint> inflow;
        double h_power;
        double measure;
      };
      std::shared_ptr<typename DGHistory::Space> space;
      std::vector<Cell> cells;
      std::vector<Point<dim>> mesh_vertices;
      FullMatrix<double> values;
      FullMatrix<double> scaling_values;
      std::array<FullMatrix<double>,5> derivatives;
      Vector<double> mean;
      std::vector<double> quadrature_weights;
      Detector detector=Detector::fu_shu;
      double threshold=.1;
      // Diagnostic Q2 adaptation of constrained neighbor-cell fitting.
      bool constrained_candidates=false;
      // Diagnostic scalar envelope for the selected proportional tensor pulse.
      // No production stress bound: scalar = <A,reference>_3/|reference|_3^2.
      SymmetricTensor<2,dim> scaling_reference;
      double scaling_lower=-1.,scaling_upper=1.;
    };

    template <int dim> struct PhaseFieldFault<dim>::DGLimiterStats
    {
      struct Indicator
      {
        double mean=0.,polynomial=0.,kxrcf=0.,mean_error=0.;
        bool limited=false;
        std::vector<double> weights,beta;
        std::vector<unsigned int> useful_masks;
        std::vector<std::vector<SymmetricTensor<2,dim>>> candidate_values;
        SymmetricTensor<2,dim> physical_mean_before,physical_mean_after;
        double theta=1.;
      };
      struct NormChange
      {
        double time=0.,before=0.,after=0.,change=0.;
        unsigned int positive_cells=0,negative_cells=0;
        unsigned int means_outside_envelope=0;
        double maximum_mean_excess=0.;
        double maximum_physical_mean_change=0.;
        SymmetricTensor<2,dim> integral_mean_change;
      };
      std::vector<Indicator> last;
      unsigned long long visits=0,limited=0,mean_flagged=0,safeguard_only=0;
      double detector_seconds=0.,reconstruction_seconds=0.,maximum_mean_error=0.;
      // Optional diagnostic measurements; do not affect any reconstruction.
      bool measure_norm_changes=false,capture_first_call=false;
      double norm_measurement_seconds=0.;
      std::vector<NormChange> norm_changes;
      DGComponents first_incoming,first_outgoing;
      bool measure_physical_means=false;
      std::set<unsigned int> trace_cells;
      // Called after all cells are reconstructed, so neighbor flux changes
      // use the complete separate output rather than partially written data.
      std::function<void(const DGComponents &,const DGComponents &,double,
                         const std::vector<Indicator> &)> stage_observer;
    };
  }
}
#endif
