/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_SIMULATOR_SOLVER_BULK_TIP_H
#define ASPECT_SIMULATOR_SOLVER_BULK_TIP_H

#include <aspect/simulator_access.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/phase_field/dg_bounds.h>
#include <deal.II/lac/affine_constraints.h>

namespace aspect
{
  namespace internal
  {
    /** Fixed-reference bulk-tip residual and exact condensed tangent.
     * The ordinary Simulator solver scheme owns nonlinear iteration and trial
     * acceptance. This operator neither advances time nor publishes history.
     * Section forces are eliminated by their monotone scalar law. Their exact
     * nonlocal tangent is an integration/adjoint action, not a dense FE matrix.
     * Unknowns use ordinary distributed ASPECT blocks: displacement increment,
     * p/G and cell-local Bernstein phase coefficients. Other blocks are inert. */
    template <int dim>
    class BulkTipCoupling : public SimulatorAccess<dim>
    {
      public:
        using Vector = LinearAlgebra::BlockVector;
        using Model = MaterialModel::PhaseFieldFault<dim>;
        using State = typename Model::BulkCohesiveState;
        using Point = MaterialModel::BulkCohesive::Tip::Response;
        using Law = typename MaterialModel::Rheology::FaultFriction<dim>::SectionEndpoint;

        /** Noncommitting response; all scalar energies use the physical 3-D
         * tensor metric and actual JxW. Only small section arrays are replicated. */
        struct Evaluation
        {
          Vector residual, raw;
          std::map<CellId, std::vector<Point>> points;
          std::vector<Law> laws;
          std::vector<double> rate, absolute_rate, positive_rate, negative_rate;
          double mechanical = 0., kkt = 0., section_defect = 0.;
          double bulk = 0., cohesive = 0., fracture = 0., functional = 0.;
          double storage_change = 0., internal_work = 0., external_work = 0.;
          double Db = 0., Dc = 0., Df = 0., friction = 0., radiation = 0.;
          double stress_loss = 0., traction_loss = 0., compliance_remainder = 0.;
          double fracture_loss = 0., phase_work = 0., ledger_error = 0., adjoint_error = 0.;
        };

        std::shared_ptr<Evaluation> accepted_evaluation;
        Vector accepted_coordinates;
        unsigned int newton_iterations=0, linear_iterations=0, stokes_solves=0, active_releases=0, search_shifts=0;

        BulkTipCoupling(Simulator<dim> &, Model &, const AffineConstraints<double> &);
        /** Build current boundary/accepted-history guess; no accepted mutation. */
        Vector initial_guess(const Vector &solution) const;
        Vector physical_field(const Vector &coordinates) const;
        Vector coordinates(const Vector &physical) const;
        std::shared_ptr<Evaluation> evaluate(const Vector &, bool tangent);
        /** Exact derivative at the most recently assembled tangent point. */
        void vmult(Vector &destination, const Vector &source) const;
        /** Positive phase search/preconditioner matrix, distinct from physics. */
        void phase_search_matrix(LinearAlgebra::SparseMatrix &, double shift) const;
        /** Read-only sparse coordinate and ownership information. */
        const LinearAlgebra::Vector &obstacle() const
        {
          return lower;
        }
        const LinearAlgebra::Vector &phase_mass() const
        {
          return mass;
        }
        const LinearAlgebra::SparseMatrix &fracture_hessian() const
        {
          return fracture_matrix;
        }
        const AffineConstraints<double> &constraints() const
        {
          return homogeneous;
        }
        const AffineConstraints<double> &boundary_constraints() const
        {
          return boundary;
        }
        const std::shared_ptr<const State> &accepted_parent() const
        {
          return parent;
        }
        unsigned int phase_block() const
        {
          return pb;
        }
        double shear_modulus() const
        {
          return G;
        }
        double relaxation() const
        {
          return beta;
        }
        double fracture_scale() const
        {
          return Kf;
        }
        double timestep() const
        {
          return dt;
        }
        double directional_work(const Evaluation &, const Vector &) const;
        std::uint64_t sparse_entries() const
        {
          return local_matrix.n_nonzero_elements();
        }
        /** Construct a candidate for the existing owner; caller publishes once. */
        std::shared_ptr<State> candidate(const Vector &, const Evaluation &) const;
        /** Zero-clock seed equilibration uses the existing SPD obstacle solver. */
        void equilibrate_seed(Vector &coordinates);

      private:
        Model &model;
        std::shared_ptr<const State> parent;
        unsigned int pb, pc;
        types::global_dof_index po, nv, pressure_begin, pressure_end;
        double dt, G, beta, Kf;
        AffineConstraints<double> boundary, homogeneous;
        PhaseField::Q2Bounds<dim> basis;
        std::map<types::global_dof_index,
            std::vector<std::pair<types::global_dof_index,double>>> phase_rows;
        LinearAlgebra::Vector lower, mass, fracture_linear;
        LinearAlgebra::SparseMatrix fracture_matrix;
        LinearAlgebra::BlockSparseMatrix local_matrix;
        /** Each section has rank-local sparse integration/adjoint coefficients. */
        std::vector<std::map<types::global_dof_index,double>> section_integrals, section_adjoints;
        std::vector<double> section_slopes;
        std::shared_ptr<Evaluation> linearization;
        Vector ghost(const Vector &) const;
        void transform_phase(Vector &, bool inverse) const;
    };
  }
}
#endif
