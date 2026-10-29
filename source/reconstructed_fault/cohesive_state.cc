/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <deal.II/fe/fe_values.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <sstream>
#include <iomanip>

namespace aspect
{
  namespace CohesiveInterface
  {
    std::shared_ptr<ReconstructedFaultManager<2>>
    System::Implementation::manager (const State &s, const std::vector<PointState> &history,
                                     const std::vector<std::array<double, 3>> &diagnostics) const
    {
      const unsigned int cells = formed_cells (s);
      const auto surface = s.geometry.surface_points ();
      AssertThrow (history.size () >= 3 * cells && diagnostics.size () >= 3 * cells,
                   ExcMessage ("Incomplete formed Gauss histories."));
      std::shared_ptr<ReconstructedFaultManager<2>> result;
      unsigned int previous = 0;
      if (s.manager)
        {
          result = s.manager->clone_prescribed_geometry ();
          result->initialize_simulator (system.get_simulator ());
          previous = result->get_fault (0).n_vertices () - 1;
          AssertThrow (previous <= cells, ExcMessage ("Formed interface cannot be shortened."));
        }
      else
        {
          result = std::make_shared<ReconstructedFaultManager<2>> (system.get_simulator ());
          result->register_property ("native exact-graph Gauss3 physical history v1", 18);
          result->register_property ("native accepted Gauss3 q resistance normal diagnostics", 9);
        }
      std::vector<dealii::Point<2>> vertices;
      for (unsigned int i = s.manager ? previous + 1 : 0; i <= cells; ++i)
        {
          const double x = s.geometry.vertices ().at (i);
          vertices.emplace_back (x, s.geometry.sample (x).y);
        }
      if (!s.manager)
        {
          result->add_reconstructed_fault (vertices,
                                           std::vector<double> (vertices.size (), support));
          // This manager's legacy nonnegative nodal-rate channel is unused.
          // Initialize its required container, never put signed jump history
          // there: the native Q1 trace/Gauss rows remain authoritative.
          result->initialize_slip_rate (0, std::vector<double> (vertices.size (), 0.));
        }
      else if (!vertices.empty ())
        {
          std::vector<std::vector<double>> birth (vertices.size (), std::vector<double> (27, 0.));
          for (auto &row : birth)
            for (unsigned int q = 0; q < 3; ++q)
              row[6 * q + 2] = s.virgin_theta;
          result->append_reconstructed_fault (0, vertices,
                                              std::vector<double> (vertices.size (), support),
                                              birth, std::vector<double> (vertices.size (), 0.));
        }
      auto &fault = result->get_fault (0);
      for (unsigned int cell = 0; cell < cells; ++cell)
        {
          auto row = fault.get_properties (cell);
          for (unsigned int q = 0; q < 3; ++q)
            {
              const auto &h = history[3 * cell + q];
              const unsigned int at = 6 * q;
              row[at] = h.jump;
              row[at + 1] = h.damage;
              row[at + 2] = h.theta;
              row[at + 3] = h.reference;
              row[at + 4] = h.relaxed_jump;
              row[at + 5] = h.absolute_slip;
              for (unsigned int c = 0; c < 3; ++c)
                row[18 + 3 * q + c] = diagnostics[3 * cell + q][c];
            }
        }
      auto last = fault.get_properties (cells);
      for (unsigned int i = 0; i < 27; ++i)
        last[i] = 0.;
      for (unsigned int q = 0; q < 3; ++q)
        last[6 * q + 2] = s.virgin_theta;
      return result;
    }

    void
    System::Implementation::validate (const State &s) const
    {
      AssertThrow (s.bulk && s.manager && s.step == s.publications && s.formed > 0
                       && s.formed <= s.geometry.pieces ().size () && s.virgin_theta > 0,
                   ExcMessage ("Invalid native cohesive accepted-state ownership."));
      const unsigned int cells = formed_cells (s), modes = s.geometry.modes ();
      AssertThrow (s.manager->get_fault (0).n_vertices () == cells + 1 && s.ids.size () == 3 * cells
                       && s.metadata.size () == s.geometry.pieces ().size ()
                       && s.memory.size () == modes && s.jump.size () == modes
                       && s.rate.size () == modes
                       && s.offset.size () == s.geometry.surface_points ().size (),
                   ExcMessage ("Native cohesive registry/layout mismatch."));
      for (unsigned int i = cells - 1; i < modes; ++i)
        AssertThrow (s.memory[i] == 0. && s.jump[i] == 0. && s.rate[i] == 0.,
                     ExcMessage ("Temporary candidate carries nodal material history."));
      for (unsigned int q = 3 * cells; q < s.offset.size (); ++q)
        AssertThrow (s.offset[q] == 0.,
                     ExcMessage ("Temporary candidate carries physical jump offset."));
      const auto history = points (s, s.geometry);
      const auto surface = s.geometry.surface_points ();
      for (unsigned int v = 0; v <= cells; ++v)
        {
          const double x = s.geometry.vertices ()[v];
          AssertThrow (
              s.manager->get_fault (0).vertex (v) == dealii::Point<2> (x, s.geometry.sample (x).y),
              ExcMessage ("Native registry endpoints differ from the exact graph geometry."));
        }
      for (unsigned int q = 0; q < 3 * cells; ++q)
        {
          const auto &h = history[q];
          AssertThrow (std::isfinite (h.jump) && std::isfinite (h.damage) && std::isfinite (h.theta)
                           && std::isfinite (h.reference) && std::isfinite (h.relaxed_jump)
                           && std::isfinite (h.absolute_slip),
                       ExcMessage ("Nonfinite native cohesive physical history."));
          AssertThrow (h.damage >= 0 && h.damage <= 1 && h.theta > 0 && h.absolute_slip >= 0,
                       ExcMessage ("Inadmissible native cohesive history."));
          const auto &point = surface[q];
          double jump = s.offset[q], memory = 0.;
          if (point.left > 0)
            {
              jump += (1 - point.xi) * s.jump[point.left - 1];
              memory += (1 - point.xi) * s.memory[point.left - 1];
            }
          if (point.right + 1 < s.geometry.vertices ().size ())
            {
              jump += point.xi * s.jump[point.right - 1];
              memory += point.xi * s.memory[point.right - 1];
            }
          AssertThrow (
              std::abs (jump - h.jump) < 1e-13 && std::abs (memory - h.relaxed_jump) < 1e-13,
              ExcMessage ("Physical Gauss jump/Maxwell memory differs from its native trace."));
        }
    }

    std::pair<System::Implementation::Tensor, double>
    System::Implementation::sample (const State &s) const
    {
      const auto &piece = s.geometry.pieces ().at (s.formed - 1);
      const auto end = s.geometry.sample (piece.end);
      dealii::Point<2> centre (piece.end, end.y);
      centre += R * end.tangent;
      const QGauss<2> rule (selection_order);
      FEValues<2> values (system.get_mapping (), system.get_fe (), rule,
                          update_quadrature_points | update_JxW_values);
      Tensor tensor;
      double mass = 0.;
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            values.reinit (cell);
            const auto stress = physical_stress (s, cell, rule);
            for (unsigned int q = 0; q < rule.size (); ++q)
              {
                const double r = values.quadrature_point (q).distance_square (centre) / (R * R);
                if (r < 1.)
                  {
                    const double w = values.JxW (q) * (1 - r) * (1 - r);
                    mass += w;
                    tensor += w * stress[q];
                  }
              }
          }
      mass = Utilities::MPI::sum (mass, system.get_mpi_communicator ());
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = i; j < 3; ++j)
          tensor[i][j] = Utilities::MPI::sum (tensor[i][j], system.get_mpi_communicator ());
      AssertThrow (mass > 0, ExcMessage ("Empty native path-sampling disk."));
      tensor /= mass;
      return { tensor, Geometry::preferred_direction (tensor, std::atan (piece.b)) };
    }

    void
    System::publish (const std::shared_ptr<Trial> &trial)
    {
      AssertThrow (trial && trial->parent == data->accepted
                       && trial->state->step == data->accepted->step + 1
                       && trial->state->publications == data->accepted->publications + 1,
                   ExcMessage ("Stale or duplicate native cohesive publication."));
      // All MPI communication, allocation and checks completed in solve().
      data->accepted = trial->state;
      data->active_trial.reset ();
      data->clock_attempts = 0;
    }

    const System::Field &
    System::solution (const Trial &trial) const
    {
      return *trial.native_solution;
    }

    std::string
    System::checkpoint () const
    {
      const auto &d = *data;
      AssertThrow (d.accepted, ExcMessage ("Missing accepted native cohesive checkpoint state."));
      AssertThrow (
          !d.solving && d.active_trial.expired (),
          ExcMessage ("A native cohesive trial cannot be checkpointed before the outer decision."));
      d.validate (*d.accepted);
      AssertThrow (
          d.accepted->step + 1 == get_timestep_number (),
          ExcMessage ("Native cohesive checkpoint must follow outer acceptance/advance_time."));
      auto records = d.accepted->bulk->pack ();
      const auto &s = *d.accepted;
      std::ostringstream out;
      {
        aspect::oarchive ar (out);
        auto configuration = d.configuration ();
        ar & configuration & s.geometry & s.formed & s.step & s.registry_publications
            & s.publications & s.identity_clock & s.time & s.virgin_theta & s.candidate_epoch
            & s.candidate_anchor & s.metadata & s.ids & s.jump & s.memory & s.rate & s.offset
            & s.diagnostics;
        ar &*s.manager & records;
      }
      return out.str ();
    }

    void
    System::load_checkpoint (const std::string &record)
    {
      AssertThrow (!data->accepted && data->pending_checkpoint.empty (),
                   ExcMessage ("Native cohesive checkpoint loaded twice."));
      data->pending_checkpoint = record;
    }

    void
    System::restore ()
    {
      auto &d = *data;
      AssertThrow (!d.pending_checkpoint.empty () && !d.accepted,
                   ExcMessage ("Missing native cohesive restart record."));
      auto s = std::make_shared<Implementation::State> ();
      s->manager = std::make_shared<ReconstructedFaultManager<2>> (get_simulator ());
      BulkHistory<2>::CellRecords records;
      std::string configuration;
      std::istringstream input (d.pending_checkpoint);
      {
        aspect::iarchive ar (input);
        ar & configuration;
        AssertThrow (configuration == d.configuration (),
                     ExcMessage ("Native cohesive checkpoint version/material/geometry settings "
                                 "changed; conversion is unsupported."));
        ar & s->geometry & s->formed & s->step & s->registry_publications & s->publications
            & s->identity_clock & s->time & s->virgin_theta & s->candidate_epoch
            & s->candidate_anchor & s->metadata & s->ids & s->jump & s->memory & s->rate & s->offset
            & s->diagnostics;
        ar & * s->manager & records;
      }
      s->bulk = std::make_shared<BulkHistory<2>> (get_simulator (), d.law.G, d.law.eta);
      s->bulk->restore (records);
      AssertThrow (
          s->bulk->pack () == records,
          ExcMessage (
              "Native checkpoint restore changed physical CellId tensor/native-memory records."));
      d.validate (*s);
      auto owned_solution = d.owned ();
      *owned_solution = get_solution ();
      AssertThrow (std::isfinite (owned_solution->l2_norm ()),
                   ExcMessage ("Native checkpoint contains a nonfinite mixed physical solution."));
      AssertThrow (
          s->step + 1 == get_timestep_number (),
          ExcMessage ("Native cohesive restored clock/publication disagrees with Simulator."));
      d.accepted = std::move (s);
      d.pending_checkpoint.clear ();
      d.cached_columns.clear ();
      d.cached_normals.clear ();
      d.matrix_ready = false;
      d.cached_prepared.reset ();
      d.prepared_signature.clear ();
      get_pcout () << "Restored native cohesive geometry, Gauss histories and exact physical "
                      "CellId records: accepted step "
                   << d.accepted->step << ", time " << std::setprecision (17) << d.accepted->time
                   << ", pending dt " << get_timestep () << ".\n";
      write_output ();
    }
  }
}
