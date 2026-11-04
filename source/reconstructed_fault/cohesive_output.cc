/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>
namespace aspect
{
  namespace CohesiveInterface
  {
    std::vector<std::array<double, 6>>
    System::Implementation::traces (const State &s, const Field &solution) const
    {
      const auto points = s.geometry.surface_points ();
      std::vector<double> flat (6 * points.size ()), coverage (2 * points.size ());
      const auto pb = system.introspection ().extractors.pressure;
      // Match the retained sided-cell admission exactly. The offset selects a
      // cell only; fields are evaluated at the unshifted physical point.
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          for (unsigned q = 0; q < points.size (); ++q)
            for (int side : { -1, 1 })
              {
                const auto &point = points[q];
                dealii::Tensor<1, 2> normal;
                normal[0] = -point.tangent[1];
                normal[1] = point.tangent[0];
                const auto query = point.position + side * 1e-9 * support * normal;
                const auto lo = cell->vertex (0), hi = cell->vertex (3);
                if (!(query[0] >= lo[0] && query[0] < hi[0] && query[1] >= lo[1]
                      && query[1] < hi[1]))
                  continue;
                auto unit
                    = system.get_mapping ().transform_real_to_unit_cell (cell, point.position);
                for (unsigned a = 0; a < 2; ++a)
                  unit[a] = std::clamp (unit[a], 0., 1.);
                const Quadrature<2> rule (std::vector<dealii::Point<2>>{ unit });
                auto tau = s.bulk->evaluate_sided(cell,rule,side)[0];
                for (const auto &m : s.geometry.lifting (point.position, side))
                  tau += 2 * law.G * s.memory[m.index] * dev3 (m.gradient);
                FEValues<2> v (system.get_mapping (), system.get_fe (), rule, update_values);
                v.reinit (cell);
                std::vector<double> pressure (1);
                v[pb].get_function_values (solution, pressure);
                double nn = 0., tn = 0.;
                for (unsigned a = 0; a < 2; ++a)
                  for (unsigned b = 0; b < 2; ++b)
                    {
                      nn += normal[a] * tau[a][b] * normal[b];
                      tn += point.tangent[a] * tau[a][b] * normal[b];
                    }
                const unsigned k = side > 0;
                coverage[2 * q + k] = 1.;
                flat[6 * q + k] = law.p0 + pressure[0] - nn;
                flat[6 * q + 2 + k] = tn;
                flat[6 * q + 4 + k] = tau[0][1];
              }
      internal::endpoint_sum (flat, system.get_mpi_communicator ());
      internal::endpoint_sum (coverage, system.get_mpi_communicator ());
      std::vector<std::array<double, 6>> result (points.size ());
      for (unsigned q = 0; q < points.size (); ++q)
        {
          AssertThrow (coverage[2 * q] == 1. && coverage[2 * q + 1] == 1.,
                       ExcMessage ("Native sided trace has missing/duplicate MPI cell ownership."));
          for (unsigned k = 0; k < 6; ++k)
            result[q][k] = flat[6 * q + k];
        }
      return result;
    }

    void
    System::write_output () const
    {
      AssertThrow (data->accepted && !data->solving && data->active_trial.expired (),
                   ExcMessage ("Only accepted native cohesive state may be output."));
      data->write (*data->accepted);
    }

    void
    System::Implementation::write (const State &s) const
    {
      const auto communicator = system.get_mpi_communicator ();
      const unsigned rank = Utilities::MPI::this_mpi_process (communicator);
      const auto raw = traces (s, system.get_solution ());
      if (std::getenv ("ASPECT_COHESIVE_WRITE_HISTORY_RECORDS"))
        {
          const auto records = s.bulk->pack ();
          if (rank == 0)
            {
              std::ostringstream name;
              name << system.get_output_directory () << "cohesive-" << std::setw (6)
                   << std::setfill ('0') << s.step << "-physical-history.csv";
              std::ofstream record (name.str ());
              record << std::setprecision (17);
              for (const auto &cell : records)
                {
                  record << cell.first;
                  for (double x : cell.second)
                    record << ',' << x;
                  record << '\n';
                }
            }
        }
      struct rusage usage;
      getrusage (RUSAGE_SELF, &usage);
      const double rss
          = Utilities::MPI::max (static_cast<double> (usage.ru_maxrss) * 1024., communicator);
      std::ostringstream stem;
      stem << system.get_output_directory () << "cohesive-" << std::setw (6) << std::setfill ('0')
           << s.step;
      if (rank == 0)
        {
          std::ofstream d (stem.str () + "-diagnostics.csv");
          d << std::setprecision (17) << "quantity,value\n";
          for (const auto &entry : s.diagnostics)
            d << entry.first << ',' << entry.second << '\n';
          d << "accepted_time_s," << s.time << "\npublications," << s.publications
            << "\nregistry_publications," << s.registry_publications << "\npeak_rank_memory_bytes,"
            << rss << "\ncells," << system.get_triangulation ().n_global_active_cells ()
            << "\ndofs," << system.get_dof_handler ().n_dofs () << "\nranks,"
            << Utilities::MPI::n_mpi_processes (communicator) << '\n';
          const auto surface = s.geometry.surface_points ();
          const auto history = points (s, s.geometry);
          std::ofstream f (stem.str () + "-surface.csv");
          f << std::setprecision (17)
            << "q,cell,piece,id,x_m,y_m,weight_m,tangent_deg,formed,selected_s,admitted_s,"
               "preferred_deg,jump_m,damage,theta_normalized,reference_m,relaxed_jump_m,absolute_"
               "slip_m,velocity_m_s,q_Pa,resistance_Pa,pn_work_Pa,pn_minus_Pa,pn_plus_Pa,tau_t_"
               "minus_Pa,tau_t_plus_Pa,tau_xy_minus_Pa,tau_xy_plus_Pa\n";
          unsigned piece = 0, end = 3 * s.geometry.pieces ()[0].cells;
          for (unsigned q = 0; q < surface.size (); ++q)
            {
              while (q >= end)
                {
                  ++piece;
                  end += 3 * s.geometry.pieces ()[piece].cells;
                }
              const auto &p = surface[q];
              const auto &h = history[q];
              const auto &m = s.metadata[piece];
              std::array<double, 3> diag{};
              if (piece < s.formed)
                {
                  auto row = s.manager->get_fault (0).get_properties (q / 3);
                  for (unsigned k = 0; k < 3; ++k)
                    diag[k] = row[18 + 3 * (q % 3) + k];
                }
              double V = 0.;
              if (p.left > 0)
                V += (1 - p.xi) * s.rate[p.left - 1];
              if (p.right + 1 < s.geometry.vertices ().size ())
                V += p.xi * s.rate[p.right - 1];
              f << q << ',' << p.cell << ',' << piece << ','
                << (q < s.ids.size () ? s.ids[q] : "temporary") << ',' << p.position[0] << ','
                << p.position[1] << ',' << p.weight << ','
                << std::atan2 (p.tangent[1], p.tangent[0]) * 180 / numbers::PI << ','
                << (piece < s.formed) << ',' << m.selected_time << ',' << m.admitted_time << ','
                << m.preferred * 180 / numbers::PI << ',' << h.jump << ',' << h.damage << ','
                << h.theta << ',' << h.reference << ',' << h.relaxed_jump << ',' << h.absolute_slip
                << ',' << V;
              for (double x : diag)
                f << ',' << x;
              for (double x : raw[q])
                f << ',' << x;
              f << '\n';
            }
          std::ofstream geometry (stem.str () + "-geometry.csv");
          geometry << std::setprecision (17)
                   << "piece,x_m,y_m,end_m,a,b,c1,c2,c3,cells,hermite,formed\n";
          for (unsigned i = 0; i < s.geometry.pieces ().size (); ++i)
            {
              const auto &p = s.geometry.pieces ()[i];
              geometry << i << ',' << p.x << ',' << p.y << ',' << p.end << ',' << p.a << ',' << p.b
                       << ',' << p.c1 << ',' << p.c2 << ',' << p.c3 << ',' << p.cells << ','
                       << p.hermite << ',' << (i < s.formed) << '\n';
            }
          const std::string index = system.get_output_directory () + "cohesive-index.csv";
          const bool exists = std::ifstream (index).good ();
          std::ofstream ix (index, std::ios::app);
          if (!exists)
            ix << "step,time_s,publications,stem\n";
          ix << std::setprecision (17) << s.step << ',' << s.time << ',' << s.publications << ','
             << stem.str () << '\n';
        }
      std::ofstream field (stem.str () + "-bulk-r" + std::to_string (rank) + ".csv");
      field << std::setprecision (17)
            << "cell,x_m,y_m,area_m2,Txx_Pa,Tyy_Pa,Tzz_Pa,Txy_Pa,pprime_Pa,velocity_x_m_s,velocity_"
               "y_m_s\n";
      const Quadrature<2> centre (std::vector<dealii::Point<2>>{ dealii::Point<2> (.5, .5) });
      FEValues<2> values (system.get_mapping (), system.get_fe (), centre,
                          update_values | update_quadrature_points);
      const auto &intro = system.introspection ();
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            values.reinit (cell);
            const auto t = physical_stress (s, cell, centre)[0];
            const auto x = values.quadrature_point (0);
            std::vector<double> p (1);
            std::vector<dealii::Tensor<1, 2>> u (1);
            values[intro.extractors.pressure].get_function_values (system.get_solution (), p);
            values[intro.extractors.velocities].get_function_values (system.get_solution (), u);
            for (const auto &m : s.geometry.lifting (x))
              u[0] += s.rate[m.index] * m.value;
            field << cell->id ().to_string () << ',' << x[0] << ',' << x[1] << ','
                  << cell->measure () << ',' << t[0][0] << ',' << t[1][1] << ',' << t[2][2] << ','
                  << t[0][1] << ',' << p[0] << ',' << u[0][0] << ',' << u[0][1] << '\n';
          }
    }
  }
}
