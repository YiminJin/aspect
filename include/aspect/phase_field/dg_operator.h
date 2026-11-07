/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_DG_OPERATOR_H
#define ASPECT_PHASE_FIELD_DG_OPERATOR_H

#include <aspect/global.h>
#include <aspect/phase_field/dg_bounds.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <functional>

namespace aspect
{
  namespace PhaseField
  {
    /** Native scalar phase spatial operators. The DoFs belong to the simulator,
     * not to an auxiliary field. Assembly borrows explicit coefficients/velocity;
     * advance returns a candidate and never publishes accepted state.
     * Currently Cartesian 2-D cells, including nonmatching interior faces. */
    template <int dim>
    class DGOperator
    {
      public:
        using CellIterator = typename DoFHandler<dim>::active_cell_iterator;
        using Velocity = std::function<void(const CellIterator &, const std::vector<Point<dim>> &,
                                            std::vector<Tensor<1, dim>> &, std::vector<double> &)>;
        using Field = std::function<double(const Point<dim> &, double)>;
        using Coefficient = std::function<double(const Point<dim> &)>;

        DGOperator(const Mapping<dim> &, const DoFHandler<dim> &,
                   unsigned int component, types::global_dof_index offset,
                   const IndexSet &owned, const IndexSet &relevant, MPI_Comm);

        /** Assemble upwind material transport, including phi div(v). Uses
         * tensor Lobatto5 volume/face quadrature and a positive-mean CFL bound. */
        void assemble_transport(const Velocity &);

        struct TransportStatistics
        {
          unsigned int substeps = 0;
          unsigned long long limited_cells = 0;
          double minimum_alpha = 1.;
          double initial_integral = 0., final_integral = 0.;
          double integrated_source = 0., integrated_boundary_flux = 0., balance_error = 0.;
        };

        /** SSPRK3 with limiting at initialization and every stage. Inflow is
         * evaluated at stage time. With limit=false this is the smooth control.
         * The caller owns physical-step acceptance and the transported obstacle. */
        TransportStatistics transport(const LinearAlgebra::Vector &accepted,
                                      LinearAlgebra::Vector &candidate,
                                      const Field &inflow, double time, double dt,
                                      double lower, double upper, bool limit = true) const;

        /** Apply the whole-cell mean-preserving limiter to an explicit field. */
        void limit(LinearAlgebra::Vector &, double lower, double upper,
                   TransportStatistics &) const;
        double integral(const LinearAlgebra::Vector &) const;

        /** Assemble the SIPG gradient bilinear form with diffusivity k.
         * Energy is 1/2 phi^T K phi - phi^T rhs + boundary constant.
         * Default physical boundary is zero Neumann. Optional Dirichlet data
         * use symmetric Nitsche terms on all exterior faces, for operator tests.
         * Positive penalty factor multiplies (p+1)^2 max(k/h_n). */
        double assemble_sipg(LinearAlgebra::SparseMatrix &, LinearAlgebra::Vector &rhs,
                             const Coefficient &, double penalty,
                             const Coefficient &dirichlet = {}) const;
        void reinit_matrix(LinearAlgebra::SparseMatrix &) const;

        double maximum_mean_rate() const
        {
          return mean_rate;
        }
        std::size_t memory_consumption() const;

      private:
        struct Cell
        {
          CellIterator iterator;
          std::vector<types::global_dof_index> indices;
          FullMatrix<double> inverse_mass;
          Vector<double> mean_weights;
          double volume;
        };
        struct Block
        {
          std::vector<types::global_dof_index> rows, columns;
          FullMatrix<double> matrix;
        };
        struct BoundaryLoad
        {
          Point<dim> point;
          std::vector<types::global_dof_index> indices;
          Vector<double> values, trace;
          double flux_weight;
        };
        struct SourceSample
        {
          std::vector<types::global_dof_index> indices;
          Vector<double> shape;
          double weight;
        };
        std::vector<types::global_dof_index> indices(const CellIterator &) const;
        void evaluate(const LinearAlgebra::Vector &, LinearAlgebra::Vector &,
                      const Field &, double, double &, double &) const;
        const Mapping<dim> &mapping;
        const DoFHandler<dim> &dofs;
        const unsigned int component;
        const types::global_dof_index offset;
        const IndexSet owned, relevant;
        const MPI_Comm communicator;
        const FiniteElement<dim> &element;
        const Q2Bounds<dim> bounds;
        std::vector<unsigned int> local_indices;
        std::vector<Cell> cells;
        std::vector<Block> blocks;
        std::vector<BoundaryLoad> boundary;
        std::vector<SourceSample> sources;
        double mean_rate = 0.;
    };
  }
}
#endif
