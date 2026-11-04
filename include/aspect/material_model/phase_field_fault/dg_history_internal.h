/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef _aspect_material_model_phase_field_fault_dg_history_internal_h
#define _aspect_material_model_phase_field_fault_dg_history_internal_h

#include <aspect/material_model/phase_field_fault.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/sparse_matrix.h>

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

        struct PointSample
        {
          Point<dim> point;
          std::array<double,9> shape;
          double phase;
          ReconstructedFaultUtilities::NormalProfileProjection source;
        };
        mutable unsigned int point_geometry_generation=numbers::invalid_unsigned_int;
        mutable std::map<CellId,std::vector<PointSample>> point_samples;
        struct SourceGeometry
        {
          FullMatrix<double> inverse;
          std::vector<long double> projection_factor;
        };
        // Exact JxW operands and affine cell widths identify reusable Gauss3
        // mass/inverse and inverse-defect analysis, not a stress/dt cache.
        mutable std::map<std::array<long double,9+dim>,SourceGeometry> source_geometry;
        FE_DGQ<dim> element;
        DoFHandler<dim> dofs;
        IndexSet relevant;
        std::map<CellId,typename DoFHandler<dim>::active_cell_iterator> cells;
      };

      enum class Representation { separated_trial, physical_stress };
      Representation representation = Representation::separated_trial;
      std::shared_ptr<Space> space;
      // In 2D these are xx, yy, xy in Pa. zz=-(xx+yy), never zero.
      std::array<LinearAlgebra::Vector,SymmetricTensor<2,dim>::n_independent_components> trial_stress;
      // Complete independent localized residual, present only for the opt-in
      // objective DG law. It shares the tensor space; no recursive history.
      std::shared_ptr<DGHistory> residual;
      /** Scratch forward-error bounds for the native source in a trial only.
       * Not constitutive uncertainty, persistent history, or a limiter. */
      struct SourceRoundoff
      {
        SymmetricTensor<2,dim> projection;
        Tensor<2,dim> gradient_scale;
      };
      std::map<CellId,SourceRoundoff> source_roundoff;
    };


    /** Scratch DG transport: retained serial diagnostic and owned-row variant.
     * No accepted state or caller velocity is mutated. Quadrature loads already
     * include the local inverse mass. */
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
      struct Block
      {
        std::vector<types::global_dof_index> rows,columns;
        FullMatrix<double> values;
      };
      // Locally owned rows and readable neighbor ghosts. Unlike the retained
      // serial diagnostic matrix this never allocates a global dense vector.
      bool distributed=false;
      std::vector<Block> blocks;
      SparsityPattern sparsity;
      SparseMatrix<double> matrix,divergence_correction;
      std::vector<Load> source,inflow;
      std::vector<Point<dim>> mesh_vertices;
      unsigned int quadrature_order;
      double inverse_advection_time=0.,relaxation_rate=0.;

    };

  }
}
#endif
