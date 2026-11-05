/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_TENSOR_ADVECTION_H
#define ASPECT_MATERIAL_MODEL_TENSOR_ADVECTION_H

#include <aspect/material_model/history/tensor_field.h>
#include <deal.II/lac/sparse_matrix.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace HistoryDiscretization
    {
      /**
       * Fixed-velocity upwind DG operator for independent tensor components.
       * Implements the material equation dH/dt+v.grad(H)=source-rate*H;
       * divergence correction keeps this a material, not conservative, update.
       * No constitutive reconstruction, rotation, or history publication occurs.
       *
       * Supported geometry is fixed 2-D Cartesian Q2 without hanging/periodic
       * faces. The owned-row implementation uses readable neighbor ghosts; the
       * retained serial control uses a sparse matrix and global serial vectors.
       * Rebuild after mesh or velocity changes. Construction/advance are collective
       * on the simulator communicator and are called outside threaded assembly.
       */
      template <int dim>
      struct TensorAdvection
      {
          using History = TensorField<dim>;
          using Cells = typename History::Cells;
          /** Evaluate at spatial point (m), time (s). Inflow returns H; source H/s. */
          using Field = std::function<SymmetricTensor<2, dim>(const Point<dim> &, double)>;

          /** Assemble from ghost-readable velocity; relaxation_rate is in 1/s.
           * The supplied cells are locally owned native FE cells matching history. */
          static std::shared_ptr<TensorAdvection> assemble(
            const SimulatorAccess<dim> &context, const History &history,
            const Cells &cells, const LinearAlgebra::BlockVector &velocity,
            unsigned int quadrature_order, double relaxation_rate, bool distributed);

          /** Mass-project an explicit spatial/time field into a new numerical field.
           * Works with serial or distributed operators; refreshes coefficient ghosts. */
          static std::shared_ptr<History> project(
            const SimulatorAccess<dim> &context, const TensorAdvection &op,
            const Field &field, double time, unsigned int quadrature_order);

          /** Advance from time to time+dt with SSPRK3; never mutate history.
           * Callbacks are borrowed synchronously and evaluated at stage times.
           * Requires dt*(inverse_advection_time+relaxation_rate) <= 0.125; caller subcycles if needed.
           * Return owned coefficients with readable ghosts in the same space. */
          static std::shared_ptr<History> advance(
            const SimulatorAccess<dim> &context, const TensorAdvection &op,
            const History &history, const Field &inflow, const Field &source,
            double time, double dt);

        private:
          /** Owned-row assembly, with no replicated global dense field. */
          static std::shared_ptr<TensorAdvection> assemble_distributed(
            const SimulatorAccess<dim> &context, const History &history,
            const Cells &cells, const LinearAlgebra::BlockVector &velocity,
            unsigned int quadrature_order, double relaxation_rate);
          /** Owned-row SSPRK3; refresh ghost values at each stage. */
          static std::shared_ptr<History> advance_distributed(
            const SimulatorAccess<dim> &context, const TensorAdvection &op,
            const History &history, const Field &inflow, const Field &source,
            double time, double dt);

        public:
          /** Cell-local integration data, immutable after assembly. */
          struct Cell
          {
            CellId id; ///< Mesh identity for matching native velocity cells.
            std::vector<types::global_dof_index> indices; ///< Tensor DoFs.
            FullMatrix<double> inverse_mass; ///< Inverse scalar Q2 cell mass.
          };
          /** Quadrature forcing with inverse mass already applied to weights. */
          struct Load
          {
            Point<dim> point; ///< Spatial callback point (m).
            std::vector<types::global_dof_index> indices; ///< Target cell DoFs.
            Vector<double> weights; ///< Source volume or incoming face weights.
          };
          /** Shared numerical space; carries no fault associations. */
          std::shared_ptr<typename TensorField<dim>::Space> space;
          std::vector<Cell> cells; ///< Locally owned cells (all cells in serial).
          /** One self/neighbor contribution to locally owned rows. */
          struct Block
          {
            std::vector<types::global_dof_index> rows; ///< Owned target DoFs.
            std::vector<types::global_dof_index> columns; ///< Readable source DoFs.
            FullMatrix<double> values; ///< Mass-inverted transport contribution.
          };
          bool distributed = false; ///< Selects owned-row storage below.
          std::vector<Block> blocks; ///< Distributed operator contributions.
          SparsityPattern sparsity; ///< Serial operator graph.
          SparseMatrix<double> matrix; ///< Serial upwind operator.
          SparseMatrix<double> divergence_correction; ///< Serial material correction.
          std::vector<Load> source; ///< Volume source quadrature.
          std::vector<Load> inflow; ///< Incoming-boundary quadrature.
          std::vector<Point<dim>> mesh_vertices; ///< Fixed-mesh identity guard.
          unsigned int quadrature_order; ///< Volume/face transport Gauss order.
          double inverse_advection_time = 0.; ///< Global directional speed/h bound (1/s).
          double relaxation_rate = 0.; ///< Optional uniform decay rate (1/s).
      };
    }
  }
}
#endif
