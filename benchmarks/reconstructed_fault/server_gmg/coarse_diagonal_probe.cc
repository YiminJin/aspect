/* Test-only isolation of the server's level-zero velocity diagonal failure.
 * No physical equations are solved and no production operator is replaced. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator/solver/matrix_free_operators.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include "fault_gmg_diagnostics.h"

#include <cmath>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace
  {
    using ProbeVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using ProbeEvaluation = FEEvaluation<2,2,3,2,double>;

    // Intentionally compiled in the plugin, separately from the production
    // ABlockOperator instantiation in ASPECT. This permits an optimization
    // comparison without replacing the production kernel.
    struct ProbeCellOperation
    {
      const Table<2,VectorizedArray<double>> &viscosity;

      void apply(ProbeEvaluation &velocity) const
      {
        velocity.evaluate(EvaluationFlags::gradients);
        const auto eta2 = 2. * viscosity(velocity.get_current_cell_index(), 0);
        for (const unsigned int q : velocity.quadrature_point_indices())
          {
            auto strain = velocity.get_symmetric_gradient(q);
            strain *= eta2;
            velocity.submit_symmetric_gradient(strain, q);
          }
        velocity.integrate(EvaluationFlags::gradients);
      }
    };

    bool agrees(const double value, const double reference)
    {
      return internal::FaultGMGDiagnostics::finite(value)
             && std::abs(value-reference) <= 1.e-11 * std::abs(reference);
    }

    bool run_coarse_probe(const MPI_Comm communicator, std::ostream &out,
                          const Mapping<2> &mapping,
                          const unsigned int refinements,
                          const char *case_name)
    {
      AssertThrow(Utilities::MPI::n_mpi_processes(communicator) == 1,
                  ExcMessage("The coarse diagonal isolation probe requires one MPI rank."));
      // Copied from the first coarse coefficient in both server logs, not a
      // new physical parameter or an alternative constitutive calculation.
      constexpr double eta = 24999999999.663513;
      out << std::setprecision(17)
          << "case=" << case_name << " refinements=" << refinements << '\n'
          << "eta=" << eta << " simd_width=" << VectorizedArray<double>::size()
          << " analytic_free_diagonal=" << (128./15.)*eta << std::endl;

      parallel::distributed::Triangulation<2> triangulation(
        communicator, Triangulation<2>::none,
        parallel::distributed::Triangulation<2>::construct_multigrid_hierarchy);
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(refinements);
      FESystem<2> fe_velocity(FE_Q<2>(2), 2);
      FE_Q<2> fe_pressure(1);
      DoFHandler<2> dofs_velocity(triangulation), dofs_pressure(triangulation);
      dofs_velocity.distribute_dofs(fe_velocity);
      dofs_pressure.distribute_dofs(fe_pressure);
      dofs_velocity.distribute_mg_dofs();
      dofs_pressure.distribute_mg_dofs();
      MGConstrainedDoFs mg_constraints;
      mg_constraints.initialize(dofs_velocity);
      mg_constraints.make_zero_boundary_constraints(dofs_velocity, {0});

      AffineConstraints<double> constraints_velocity, constraints_pressure;
      IndexSet relevant;
#if DEAL_II_VERSION_GTE(9,7,0)
      relevant = DoFTools::extract_locally_relevant_level_dofs(dofs_velocity, 0);
#else
      DoFTools::extract_locally_relevant_level_dofs(dofs_velocity, 0, relevant);
#endif
      constraints_velocity.reinit(dofs_velocity.locally_owned_mg_dofs(0), relevant);
      for (const auto index : mg_constraints.get_boundary_indices(0))
        constraints_velocity.constrain_dof_to_zero(index);
      constraints_velocity.close();
#if DEAL_II_VERSION_GTE(9,7,0)
      relevant = DoFTools::extract_locally_relevant_level_dofs(dofs_pressure, 0);
#else
      DoFTools::extract_locally_relevant_level_dofs(dofs_pressure, 0, relevant);
#endif
      constraints_pressure.reinit(dofs_pressure.locally_owned_mg_dofs(0), relevant);
      constraints_pressure.close();

      auto matrix_free = std::make_shared<MatrixFree<2,double>>();
      MatrixFree<2,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<2,double>::AdditionalData::none;
      additional_data.mapping_update_flags = update_gradients | update_JxW_values;
      additional_data.mg_level = 0;
      matrix_free->reinit(mapping,
                         std::vector<const DoFHandler<2> *>{&dofs_velocity, &dofs_pressure},
                         std::vector<const AffineConstraints<double> *>{&constraints_velocity,
                                                                       &constraints_pressure},
                         QGauss<1>(3), additional_data);

      MatrixFreeStokesOperators::OperatorCellData<2,double> cell_data{};
      cell_data.viscosity.reinit(matrix_free->n_cell_batches(), 1);
      for (unsigned int cell=0; cell<matrix_free->n_cell_batches(); ++cell)
        {
          cell_data.viscosity(cell, 0) = 0.;
          for (unsigned int lane=0; lane<matrix_free->n_active_entries_per_cell_batch(cell); ++lane)
            cell_data.viscosity(cell, 0)[lane] = eta;
        }

      MatrixFreeStokesOperators::ABlockOperator<2,2,double> production;
      production.initialize(matrix_free, mg_constraints, 0, {0});
      production.set_cell_data(cell_data);
      out << "production_compute_diagonal_begin" << std::endl;
      production.compute_diagonal();
      const auto &inverse = production.get_matrix_diagonal_inverse()->get_vector();
      out << "production_compute_diagonal_done\n"
          << "global_dof constrained production_inverse finite" << std::endl;
      for (types::global_dof_index i=0; i<dofs_velocity.n_dofs(0); ++i)
        out << i << ' ' << constraints_velocity.is_constrained(i) << ' '
            << inverse[i] << ' ' << internal::FaultGMGDiagnostics::finite(inverse[i]) << std::endl;

      // FEValues follows an independent quadrature path. Only the two free
      // central Q2 basis functions enter the comparison below.
      FEValues<2> fe_values(mapping, fe_velocity, QGauss<2>(3),
                            update_gradients | update_JxW_values);
      const auto coarse_cell = matrix_free->get_cell_iterator(0, 0);
      fe_values.reinit(coarse_cell);
      const FEValuesExtractors::Vector velocity(0);
      std::vector<types::global_dof_index> indices(fe_velocity.n_dofs_per_cell());
      coarse_cell->get_mg_dof_indices(indices);
      std::vector<double> reference(dofs_velocity.n_dofs(0), 0.);
      for (unsigned int i=0; i<indices.size(); ++i)
        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
          {
            const auto strain = fe_values[velocity].symmetric_gradient(i, q);
            reference[indices[i]] += 2.*eta*scalar_product(strain, strain)*fe_values.JxW(q);
          }

      ProbeVector raw_diagonal, basis, action;
      // Match the production caller: deal.II 9.6's helper expects allocated
      // vector storage and accumulates into it despite its initialization docstring.
      matrix_free->initialize_dof_vector(raw_diagonal);
      raw_diagonal = 0.;
      const ProbeCellOperation test_kernel{cell_data.viscosity};
      for (types::global_dof_index i=0; i<dofs_velocity.n_dofs(0); ++i)
        if (!constraints_velocity.is_constrained(i))
          out << "fevalues_free_dof=" << i << " diagonal=" << reference[i] << std::endl;
      out << "test_compiled_raw_diagonal_begin" << std::endl;
      MatrixFreeTools::compute_diagonal(*matrix_free, raw_diagonal,
                                        &ProbeCellOperation::apply, &test_kernel);
      out << "test_compiled_raw_diagonal_done" << std::endl;
      matrix_free->initialize_dof_vector(basis);
      matrix_free->initialize_dof_vector(action);
      bool passed = true;
      for (types::global_dof_index i=0; i<dofs_velocity.n_dofs(0); ++i)
        if (constraints_velocity.is_constrained(i))
          passed = agrees(inverse[i], 1.) && passed;
        else
          {
            out << "production_vmult_begin dof=" << i << std::endl;
            basis = 0.;
            basis[i] = 1.;
            production.vmult(action, basis);
            const bool entry_passed = agrees(reference[i], (128./15.)*eta)
                                      && agrees(inverse[i], 1./reference[i])
                                      && agrees(raw_diagonal[i], reference[i])
                                      && agrees(action[i], reference[i]);
            out << "free_dof=" << i
                << " fevalues=" << reference[i]
                << " production_inverse=" << inverse[i]
                << " test_raw_diagonal=" << raw_diagonal[i]
                << " production_vmult_diagonal=" << action[i]
                << " pass=" << entry_passed << std::endl;
            passed = entry_passed && passed;
          }
      out << "coarse_probe_pass=" << passed << std::endl;
      return passed;
    }
  }

  namespace Postprocess
  {
    template <int dim>
    class FaultGMGCoarseProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Fault GMG coarse probe");
          prm.declare_entry("Test mode", "baseline",
                            Patterns::Selection("baseline|context"),
                            "Select the baseline or the three mapping/parent-cell comparisons.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Fault GMG coarse probe");
          context_tests = prm.get("Test mode") == "context";
          prm.leave_subsection();
          prm.leave_subsection();
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim == 2, ExcMessage("The coarse diagonal probe is a 2D fixture."));
          std::ofstream out(this->get_output_directory() + "coarse_diagonal_probe.txt");
          AssertThrow(out, ExcMessage("Cannot open coarse diagonal probe output."));
          const MappingQ1<2> q1;
          const MappingCartesian<2> cartesian;
          bool passed;
          if (context_tests)
            {
              // Keep the tested coarse cell fixed; isolate mapping type and
              // the presence of six refined levels as in Stage I.
              passed = run_coarse_probe(this->get_mpi_communicator(), out, cartesian, 0, "cartesian_active");
              passed = run_coarse_probe(this->get_mpi_communicator(), out, q1, 6, "q1_parent") && passed;
              passed = run_coarse_probe(this->get_mpi_communicator(), out, cartesian, 6, "cartesian_parent") && passed;
            }
          else
            passed = run_coarse_probe(this->get_mpi_communicator(), out, q1, 0, "q1_active");
          AssertThrow(passed, ExcMessage("Coarse diagonal probe mismatch; preserve its output."));
          return {"GMG coarse diagonal probe:", "verified"};
        }

      private:
        bool context_tests = false;
    };

    ASPECT_REGISTER_POSTPROCESSOR(FaultGMGCoarseProbe,
                                 "fault GMG coarse probe",
                                 "Test-only one-cell comparison of production and reference velocity diagonals.")
  }
}
