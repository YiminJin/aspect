/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_bulk_history.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace GeometryModel
  {
    template <int dim> class CohesiveHistoryBlock : public Box<dim>
    {
    public:
      void
      create_coarse_mesh (parallel::distributed::Triangulation<dim> &grid) const override
      {
        Box<dim>::create_coarse_mesh (grid);
        if (std::getenv ("ASPECT_COHESIVE_HISTORY_AFFINE"))
          return;
        GridTools::transform (
            [] (const Point<dim> &x)
              {
                Point<dim> p = x;
                const double w = std::sin (numbers::PI * x[0]) * std::sin (numbers::PI * x[1]);
                p[0] += .035 * w;
                p[1] += .02 * w;
                return p;
              },
            grid);
      }
    };
    ASPECT_REGISTER_GEOMETRY_MODEL (
        CohesiveHistoryBlock, "cohesive history block",
        "Small fixed bilinear mesh for independent exact-jump Maxwell history checks.")
  }

  namespace Postprocess
  {
    template <int dim>
    class CohesiveHistoryChecks : public Interface<dim>, public SimulatorAccess<dim>
    {
      class Stress : public Function<dim>
      {
      public:
        Stress () : Function<dim> (3) {}
        double
        value (const Point<dim> &x, const unsigned int c = 0) const override
        {
          if (c == 0)
            return 170. + 90 * x[0] * x[0] * x[1] * x[1] + 35 * x[0] * x[1];
          if (c == 1)
            return -80. + 47 * x[0] * x[0] - 63 * x[1] * x[1] + 18 * x[0] * x[1] * x[1];
          return 260. + 33 * x[0] * x[0] * x[1] - 87 * x[0] * x[1] * x[1];
        }
      };
      class Increment : public Function<dim>
      {
      public:
        Increment (const unsigned int n, const double sign) : Function<dim> (n), sign (sign) {}
        double
        value (const Point<dim> &x, const unsigned int c = 0) const override
        {
          if (c == 0)
            return sign * 1e-5 * (x[0] * x[0] * x[1] + .2 * x[1] * x[1]);
          if (c == 1)
            return 1e-5 * (x[0] * x[1] * x[1] - .3 * x[0] * x[0]);
          return 0.;
        }

      private:
        double sign;
      };

    public:
      std::pair<std::string, std::string>
      execute (TableHandler &) override
      {
        if (this->get_timestep_number () != 0)
          return { "Cohesive native history", "no mutation" };
        if constexpr (dim == 2)
          {
            using History = CohesiveInterface::BulkHistory<dim>;
            const auto communicator = this->get_mpi_communicator ();
            const auto ranks = Utilities::MPI::n_mpi_processes (communicator);
            const auto &intro = this->introspection ();
            LinearAlgebra::BlockVector owned (intro.index_sets.system_partitioning, communicator),
                du;
            du.reinit (this->get_solution ());
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), 1.), owned);
            const bool unmasked_pressure_finite
                = std::isfinite (owned.block (intro.block_indices.pressure).l2_norm ());
            LinearAlgebra::BlockVector masked (intro.index_sets.system_partitioning, communicator);
            masked = 0.;
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), 1.), masked,
                                      intro.component_masks.velocities);
            const bool masked_pressure_zero
                = masked.block (intro.block_indices.pressure).l2_norm () == 0.;
            auto delta = masked.block (intro.block_indices.velocities);
            delta -= owned.block (intro.block_indices.velocities);
            AssertThrow (delta.l2_norm () == 0. && masked_pressure_zero,
                         ExcMessage ("Velocity-only interpolation changed its native field or "
                                     "pressure initialization."));
            du = owned;
            auto h = std::make_shared<History> (this->get_simulator (), 1e6, 1e8);
            h->initialize (Stress ());
            const auto before = h->pack ();
            const double E0 = h->storage (QGauss<dim> (8));
            auto rejected = h->advance (du, .2);
            const auto check = h->check_advance (*rejected, du, .2, QGauss<dim> (8));
            AssertThrow (h->pack () == before,
                         ExcMessage ("Rejected Maxwell history changed its accepted input."));
            auto retry = h->advance (du, .2);
            AssertThrow (rejected->pack () == retry->pack (),
                         ExcMessage ("Maxwell retry differs from the rejected trial."));
            const auto checkpoint = retry->pack ();
            History restored (this->get_simulator (), 1e6, 1e8);
            restored.restore (checkpoint);
            AssertThrow (restored.pack () == checkpoint,
                         ExcMessage ("Physical CellId Maxwell restart differs."));
            const double E1 = retry->storage (QGauss<dim> (8)),
                         Er = restored.storage (QGauss<dim> (8));
            AssertThrow (E1 == Er, ExcMessage ("Maxwell restored physical energy differs."));
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), -1.), owned);
            du = owned;
            auto next = retry->advance (du, .125), continued = restored.advance (du, .125);
            const auto second = retry->check_advance (*next, du, .125, QGauss<dim> (8));
            AssertThrow (next->pack () == continued->pack (),
                         ExcMessage ("Non-proportional Maxwell restart continuation differs."));
            double norm = 0., remainder = 0.;
            const FE_DGQ<dim> dg (2);
            const QGauss<dim> samples (8);
            FEValues<dim> fe (this->get_mapping (), this->get_fe (), QGauss<dim> (8),
                              update_gradients);
            std::vector<SymmetricTensor<2, dim>> strain (fe.n_quadrature_points);
            for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
              if (cell->is_locally_owned ())
                {
                  // The complete native tensor is retained at points other than
                  // the source rule; no claim that its DG projection is exact.
                  const auto stress = next->evaluate (cell, QGauss<dim> (8));
                  for (const auto &t : stress)
                    norm = std::max (norm, t.norm ());
                  const auto &coefficients = next->dg_coefficients ().at (cell->id ());
                  for (unsigned int q = 0; q < samples.size (); ++q)
                    {
                      SymmetricTensor<2, 3> projected;
                      for (unsigned int i = 0; i < 9; ++i)
                        {
                          projected[0][0]
                              += coefficients[i] * dg.shape_value (i, samples.point (q));
                          projected[1][1]
                              += coefficients[9 + i] * dg.shape_value (i, samples.point (q));
                          projected[0][1]
                              += coefficients[18 + i] * dg.shape_value (i, samples.point (q));
                        }
                      projected[2][2] = -projected[0][0] - projected[1][1];
                      remainder = std::max (remainder, (stress[q] - projected).norm ());
                    }
                }
            norm = Utilities::MPI::max (norm, communicator);
            remainder = Utilities::MPI::max (remainder, communicator);
            const auto records = next->pack ();
            if (Utilities::MPI::this_mpi_process (communicator) == 0)
              {
                std::ofstream f (this->get_output_directory () + "maxwell-history-checks.csv");
                f << std::setprecision (17)
                  << "ranks,cells,source_Pa,source_scaled,second_Pa,second_scaled,initial_J_m,"
                     "trial_J_m,restart_J_m,max_full_tensor_Pa,max_retained_non_DG_remainder_Pa,"
                     "retry_exact,restart_exact,unmasked_pressure_finite,masked_pressure_zero\n"
                  << ranks << ',' << before.size () << ',' << check.first << ',' << check.second
                  << ',' << second.first << ',' << second.second << ',' << E0 << ',' << E1 << ','
                  << Er << ',' << norm << ',' << remainder << ",1,1," << unmasked_pressure_finite
                  << ',' << masked_pressure_zero << '\n';
                std::ofstream rows (this->get_output_directory ()
                                    + "maxwell-history-cell-records.csv");
                rows << std::setprecision (17);
                for (const auto &entry : records)
                  {
                    rows << entry.first;
                    for (const double value : entry.second)
                      rows << ',' << value;
                    rows << '\n';
                  }
              }
            return { "Cohesive native history", "strict recurrence/retry/restart pass" };
          }
        return { "Cohesive native history", "2D only" };
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR (
        CohesiveHistoryChecks, "cohesive native history checks",
        "Independent general tensor/native remainder checks on native owned cells and ghosts.")
  }
}
