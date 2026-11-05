/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_TENSOR_HISTORY_H
#define ASPECT_MATERIAL_MODEL_TENSOR_HISTORY_H

#include <aspect/simulator_access.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <deal.II/lac/full_matrix.h>
#include <functional>
#include <array>
#include <map>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/sparse_matrix.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      template <int dim>
      struct TensorHistory
      {
        using Cells = std::map<CellId, typename DoFHandler<dim>::active_cell_iterator>;
        using MaterialField = std::function<SymmetricTensor<2, dim>(
                                const std::vector<types::global_dof_index> &, const Point<dim> &,
                                const Point<dim> &, double, const Tensor<2, dim> &)>;
        /** L2 projection of an explicit evaluator. Profile and gradient sampling
         * are supplied by the caller; this operation knows no material law. */
        static std::shared_ptr<TensorHistory> project(
          const SimulatorAccess<dim> &, const TensorHistory &, const Cells &,
          const LinearAlgebra::BlockVector &profile,
          const LinearAlgebra::BlockVector &accepted_velocity, unsigned int order,
          const MaterialField &);
        struct Space
        {
          explicit Space(const Triangulation<dim> &triangulation)
            : element(2), dofs(triangulation)
          {
            dofs.distribute_dofs(element);
            relevant = DoFTools::extract_locally_relevant_dofs(dofs);
            for (const auto &cell : dofs.active_cell_iterators())
              if (!cell->is_artificial())
                cells.emplace(cell->id(), cell);
          }

          struct PointSample
          {
            Point<dim> point;
            std::array<double, 9> shape;
            double phase;
            ReconstructedFaultUtilities::NormalProfileProjection source;
          };
          mutable unsigned int point_geometry_generation = numbers::invalid_unsigned_int;
          mutable std::map<CellId, std::vector<PointSample>> point_samples;
          struct SourceGeometry
          {
            FullMatrix<double> inverse;
            std::vector<long double> projection_factor;
          };
          // Exact JxW operands and affine cell widths identify reusable Gauss3
          // mass/inverse and inverse-defect analysis, not a stress/dt cache.
          mutable std::map < std::array < long double, 9 + dim >, SourceGeometry > source_geometry;
          FE_DGQ<dim> element;
          DoFHandler<dim> dofs;
          IndexSet relevant;
          std::map<CellId, typename DoFHandler<dim>::active_cell_iterator> cells;
        };

        enum class Representation
        {
          separated_trial, physical_stress
        };
        Representation representation = Representation::separated_trial;
        std::shared_ptr<Space> space;
        // The representation tag identifies physical tau or auxiliary T.
        // Checkpoint packets retain their existing ordering at serialization.
        // residual owns auxiliary R, never a cohesive elastic history. Mechanics
        // reads physical_history_stress instead of inspecting these containers.
        // In 2D these are xx, yy, xy in Pa. zz=-(xx+yy), never zero.
        std::array<LinearAlgebra::Vector, SymmetricTensor<2, dim>::n_independent_components> coefficients;
        /** Raw coefficient evaluation in the stored representation. Physical
         * reconstruction is a separate operation; never infer stress from T. */
        SymmetricTensor<2, dim> tensor_at(
          const std::vector<types::global_dof_index> &indices, const Point<dim> &unit) const
        {
          SymmetricTensor<2, dim> value;
          for (unsigned c = 0; c < coefficients.size(); ++c)
            for (unsigned i = 0; i < 9; ++i)
              value[SymmetricTensor<2, dim>::unrolled_to_component_indices(c)]
              += coefficients[c][indices[i]] * space->element.shape_value(i, unit);
          return value;
        }
        // Complete independent localized residual, present only for the opt-in
        // objective DG law. It shares the tensor space; no recursive history.
        std::shared_ptr<TensorHistory<dim>> residual;
        /** Scratch forward-error bounds for the native source in a trial only.
         * Not constitutive uncertainty, persistent history, or a limiter. */
        struct SourceRoundoff
        {
          SymmetricTensor<2, dim> projection;
          Tensor<2, dim> gradient_scale;
        };
        std::map<CellId, SourceRoundoff> source_roundoff;
      };


      /** Scratch DG transport: retained serial diagnostic and owned-row variant.
       * No accepted state or caller velocity is mutated. Quadrature loads already
       * include the local inverse mass. */
      template <int dim> struct TensorAdvection
      {
          using History = TensorHistory<dim>;
          using Cells = typename History::Cells;
          using Field = std::function<SymmetricTensor<2, dim>(const Point<dim> &, double)>;
          static std::shared_ptr<TensorAdvection> assemble(
            const SimulatorAccess<dim> &, const History &, const Cells &,
            const LinearAlgebra::BlockVector &, unsigned int, double relaxation_rate, bool distributed);
          static std::shared_ptr<History> project(
            const SimulatorAccess<dim> &, const TensorAdvection &, const Field &, double, unsigned int);
          static std::shared_ptr<History> advance(
            const SimulatorAccess<dim> &, const TensorAdvection &, const History &,
            const Field &inflow, const Field &source, double time, double dt);
        private:
          static std::shared_ptr<TensorAdvection> assemble_distributed(
            const SimulatorAccess<dim> &, const History &, const Cells &,
            const LinearAlgebra::BlockVector &, unsigned int, double);
          static std::shared_ptr<History> advance_distributed(
            const SimulatorAccess<dim> &, const TensorAdvection &, const History &,
            const Field &, const Field &, double, double);
        public:
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
          std::shared_ptr<typename TensorHistory<dim>::Space> space;
          std::vector<Cell> cells;
          struct Block
          {
            std::vector<types::global_dof_index> rows, columns;
            FullMatrix<double> values;
          };
          // Locally owned rows and readable neighbor ghosts. Unlike the retained
          // serial diagnostic matrix this never allocates a global dense vector.
          bool distributed = false;
          std::vector<Block> blocks;
          SparsityPattern sparsity;
          SparseMatrix<double> matrix, divergence_correction;
          std::vector<Load> source, inflow;
          std::vector<Point<dim>> mesh_vertices;
          unsigned int quadrature_order;
          double inverse_advection_time = 0., relaxation_rate = 0.;

      };

    }
  }
}
#endif
