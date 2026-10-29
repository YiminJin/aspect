/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <aspect/utilities.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <zlib.h>

namespace aspect
{
  namespace CohesiveInterface
  {
    namespace
    {
      std::vector<double>
      json_values (const boost::property_tree::ptree &tree, const std::string &name)
      {
        std::vector<double> values;
        for (const auto &entry : tree.get_child (name))
          values.push_back (entry.second.get_value<double> ());
        return values;
      }
      std::vector<std::vector<double>>
      numeric_csv (const std::string &path, bool header = false)
      {
        std::ifstream in (path);
        AssertThrow (in, ExcMessage ("Cannot read " + path));
        std::vector<std::vector<double>> rows;
        std::string line;
        if (header)
          std::getline (in, line);
        while (std::getline (in, line))
          {
            std::replace (line.begin (), line.end (), ',', ' ');
            std::istringstream row (line);
            std::vector<double> values;
            double x;
            while (row >> x)
              values.push_back (x);
            if (!values.empty ())
              rows.push_back (std::move (values));
          }
        return rows;
      }
      class AffineMemory : public Function<2>
      {
      public:
        AffineMemory (unsigned int n, const std::array<double, 2> &c) : Function<2> (n), c (c) {}
        double
        value (const dealii::Point<2> &x, unsigned int component = 0) const override
        {
          constexpr double axial = .41954981558863996;
          if (component == 0)
            return c[0] * (-axial * x[0] + .5 * x[1]) + c[1] * (-.5 * x[0] - axial * x[1]);
          if (component == 1)
            return c[0] * (.5 * x[0] + axial * x[1]) + c[1] * (-axial * x[0] + .5 * x[1]);
          return 0.;
        }

      private:
        std::array<double, 2> c;
      };
      class SeedTensor : public Function<2>
      {
      public:
        SeedTensor (const Function<2> &extra, double G) : Function<2> (3), extra (extra), G (G) {}
        double
        value (const dealii::Point<2> &x, unsigned int c = 0) const override
        {
          constexpr double gamma = .0003, axial = .41954981558863996;
          return extra.value (x, c)
                 + (c == 0   ? -2 * G * gamma * axial
                    : c == 1 ? 2 * G * gamma * axial
                             : G * gamma);
        }

      private:
        const Function<2> &extra;
        double G;
      };
    }

    std::shared_ptr<System::Field>
    System::Implementation::elastic (const Geometry &geometry, const Field &particular,
                                     const std::vector<double> &jump,
                                     const BulkSolve &inverse) const
    {
      AssertDimension (jump.size (), geometry.modes ());
      const auto &fe = system.get_fe ();
      const auto &intro = system.introspection ();
      std::vector<unsigned int> local;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell (); ++i)
        if (intro.is_stokes_component (fe.system_to_component_index (i).first))
          local.push_back (i);
      std::vector<types::global_dof_index> indices (fe.n_dofs_per_cell ()), ids (local.size ());
      auto rhs = owned ();
      *rhs = 0.;
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto rule = geometry.quadrature (cell->vertex (0), cell->vertex (3), bulk_order);
            FEValues<2> values (system.get_mapping (), fe, rule,
                                update_values | update_gradients | update_quadrature_points
                                    | update_JxW_values);
            values.reinit (cell);
            cell->get_dof_indices (indices);
            for (unsigned int i = 0; i < local.size (); ++i)
              ids[i] = indices[local[i]];
            std::vector<dealii::Tensor<2, 2>> gradient (rule.size ());
            values[intro.extractors.velocities].get_function_gradients (particular, gradient);
            Vector<double> load (local.size ());
            for (unsigned int q = 0; q < rule.size (); ++q)
              {
                auto total = gradient[q];
                for (const auto &mode : geometry.lifting (values.quadrature_point (q)))
                  total += jump[mode.index] * mode.gradient;
                const Tensor stress = 2 * law.G * dev3 (total);
                for (unsigned int i = 0; i < local.size (); ++i)
                  load[i] += values.JxW (q)
                             * (-stress
                                    * dev3 (values[intro.extractors.velocities].symmetric_gradient (
                                        local[i], q))
                                + pressure_scaling * trace (total)
                                      * values[intro.extractors.pressure].value (local[i], q));
              }
            system.get_current_constraints ().distribute_local_to_global (load, ids, *rhs);
          }
      rhs->compress (VectorOperation::add);
      auto correction = inverse (*rhs), result = owned ();
      *result = 0.;
      for (unsigned int b = 0; b < result->n_blocks (); ++b)
        for (const auto i : result->block (b).locally_owned_elements ())
          result->block (b)[i] = particular.block (b)[i] + correction->block (b)[i];
      result->compress (VectorOperation::insert);
      auto field = ghosted ();
      *field = *result;
      return field;
    }

    std::shared_ptr<System::Implementation::State>
    System::Implementation::initialize_seed ()
    {
      auto s = std::make_shared<State> ();
      Geometry::Piece seed;
      seed.x = 3.;
      seed.end = 6.;
      seed.cells = 32;
      s->geometry = Geometry ({ seed }, support);
      s->formed = 1;
      s->time = system.get_time ();
      AssertThrow (s->time == 0.,
                   ExcMessage ("Fresh physical seed initializes at t=0; use an explicit checked "
                               "history snapshot for a loaded start."));
      s->metadata.resize (1);
      s->metadata[0].admitted_time = 0.;
      const unsigned int n = s->geometry.modes ();
      const auto qp = s->geometry.surface_points ();
      s->jump.assign (n, 0.);
      s->memory.assign (n, 0.);
      s->rate.assign (n, 0.);
      s->offset.resize (qp.size ());
      std::vector<PointState> history (qp.size ());
      std::vector<std::array<double, 3>> diagnostics (qp.size ());
      const double C0 = 2 * law.Gc / (law.peak * law.peak), q = law.G * .0003;
      for (unsigned int i = 0; i < qp.size (); ++i)
        {
          const double z = (qp[i].position[0] - 3.) / 3., shape = 4 * z * (1 - z);
          auto &h = history[i];
          h.damage = .75 + .05 * shape;
          h.jump = .0004 * shape;
          h.theta = 1.2 + .1 * shape;
          h.reference = h.jump - C0 * h.damage / (1 - h.damage) * q;
          h.absolute_slip = std::abs (h.jump);
          s->offset[i] = h.jump;
          s->ids.push_back ("seed:" + std::to_string (i));
          diagnostics[i] = { q, q, law.p0 - 2 * law.G * .0003 * .41954981558863996 };
        }
      s->manager = manager (*s, history, diagnostics);
      s->bulk = std::make_shared<BulkHistory<2>> (system.get_simulator (), law.G, law.eta);
      s->bulk->initialize (SeedTensor (independent_tensor, law.G));
      initial_pressure = ghosted ();
      validate (*s);
      return s;
    }

    std::shared_ptr<System::Implementation::State>
    System::Implementation::import (const BulkSolve &inverse)
    {
      AssertThrow (!initial_file.empty () && law.G == 1e6 && law.eta == 1e8 && law.Gc == 1.
                       && law.peak == 3000.,
                   ExcMessage ("Checked snapshot import requires the retained physical law and an "
                               "explicit input file."));
      boost::property_tree::ptree root;
      boost::property_tree::read_json (initial_file, root);
      const auto &tree = root.get_child_optional ("state") ? root.get_child ("state") : root;
      auto s = std::make_shared<State> ();
      std::vector<Geometry::Piece> pieces;
      for (const auto &entry : tree.get_child ("geometry"))
        {
          const auto &a = entry.second;
          Geometry::Piece p;
          p.x = a.get<double> ("x");
          p.y = a.get<double> ("y");
          p.end = a.get<double> ("end");
          p.a = a.get<double> ("a");
          p.b = a.get<double> ("b");
          p.cells = a.get<unsigned int> ("cells");
          p.hermite = bool (a.get_optional<double> ("c1"));
          if (p.hermite)
            {
              p.c1 = a.get<double> ("c1");
              p.c2 = a.get<double> ("c2");
              p.c3 = a.get<double> ("c3");
            }
          pieces.push_back (p);
        }
      s->geometry = Geometry (pieces, support);
      s->formed = tree.get<unsigned int> ("formed_arcs");
      s->time = tree.get<double> ("clock_s");
      s->virgin_theta = tree.get<double> ("virgin_theta");
      s->identity_clock = tree.get<std::uint64_t> ("physical_publications");
      s->registry_publications = tree.get<unsigned int> ("publications");
      s->candidate_epoch = tree.get<int> ("candidate_epoch");
      s->candidate_anchor = tree.get<unsigned int> ("candidate_anchor");
      AssertThrow (
          s->time == system.get_time (),
          ExcMessage ("ASPECT start clock differs from the explicit imported physical state."));
      s->jump = json_values (tree, "delta");
      s->memory = json_values (tree, "memory");
      s->rate = json_values (tree, "nodal_rate");
      s->offset = json_values (tree, "jump_offset");
      for (const auto &entry : tree.get_child ("record_ids"))
        s->ids.push_back (entry.second.get_value<std::string> ());
      for (const auto &entry : tree.get_child ("arc_metadata"))
        {
          Metadata m;
          const auto number = [&] (const std::string &name, double fallback)
            {
              const auto v = entry.second.get_optional<std::string> (name);
              return !v || *v == "null" ? fallback : std::stod (*v);
            };
          m.selected_time = number ("selected_from_time_s", -1.);
          m.admitted_time = number ("committed_time_s", -1.);
          m.preferred = number ("preferred_deg", 0.) * numbers::PI / 180.;
          s->metadata.push_back (m);
        }
      const auto jump = json_values (tree, "histories.jump"),
                 damage = json_values (tree, "histories.damage"),
                 theta = json_values (tree, "histories.theta");
      const auto reference = json_values (tree, "histories.reference"),
                 relaxed = json_values (tree, "histories.relaxed_jump"),
                 absolute = json_values (tree, "histories.absolute_slip");
      const auto resistance = json_values (tree, "last_resistance");
      std::vector<PointState> history (jump.size ());
      std::vector<std::array<double, 3>> diagnostics (jump.size ());
      const auto affine = json_values (tree, "affine_memory");
      const std::string path = tree.get<std::string> ("kernel_path");
      const auto N = numeric_csv (path + "/normal-work.csv");
      std::vector<double> c = affine;
      c.insert (c.end (), s->memory.begin (), s->memory.end ());
      const double C0 = 2 * law.Gc / (law.peak * law.peak);
      for (unsigned int i = 0; i < history.size (); ++i)
        {
          history[i] = { jump[i], damage[i], theta[i], reference[i], relaxed[i], absolute[i] };
          double normal = law.p0;
          for (unsigned int j = 0; j < c.size (); ++j)
            normal += N.at (i).at (j) * c[j];
          const double q
              = damage[i] == 0 ? 0. : (1 - damage[i]) * (jump[i] - reference[i]) / (C0 * damage[i]);
          diagnostics[i] = { q, resistance[i], normal };
        }
      s->manager = manager (*s, history, diagnostics);
      auto reference_owned = owned ();
      *reference_owned = 0.;
      VectorTools::interpolate (
          system.get_mapping (), system.get_dof_handler (),
          AffineMemory (system.get_fe ().n_components (), { affine.at (0), affine.at (1) }),
          *reference_owned, system.introspection ().component_masks.velocities);
      auto particular = ghosted ();
      *particular = *reference_owned;
      const auto field = elastic (s->geometry, *particular, s->memory, inverse);
      s->bulk = std::make_shared<BulkHistory<2>> (system.get_simulator (), law.G, law.eta);
      s->bulk->initialize_native (*field, independent_tensor);
      initial_pressure = ghosted ();
      const auto pb = system.introspection ().block_indices.pressure;
      auto p = owned ();
      *p = 0.;
      p->block (pb) = field->block (pb);
      *initial_pressure = *p;
      validate (*s);
      conversion_check (*s, path);
      // Compare the actual native condensed operators before any new path
      // decision. These are diagnostics, not a response-span history backend.
      auto prepared = prepare (*s, s->geometry, inverse);
      const auto old_K = numeric_csv (path + "/stiffness.csv");
      double dk = 0., kn = 0., dn = 0., nn = 0.;
      for (unsigned i = 0; i < s->geometry.modes (); ++i)
        for (unsigned j = 0; j < s->geometry.modes (); ++j)
          {
            const double a = old_K[i + 2][j + 2], b = prepared->surface.K (i + 1, j + 1);
            dk += (a - b) * (a - b);
            kn += a * a;
          }
      for (unsigned q = 0; q < N.size (); ++q)
        for (unsigned j = 0; j < s->geometry.modes (); ++j)
          {
            const double a = N[q][j + 2], b = prepared->surface.Nm (q, j + 1);
            dn += (a - b) * (a - b);
            nn += a * a;
          }
      const double rk = std::sqrt (dk / std::max (1., kn)), rn = std::sqrt (dn / std::max (1., nn));
      s->diagnostics["initial_K_relative"] = rk;
      s->diagnostics["initial_N_relative"] = rn;
      AssertThrow (rk < 1e-10 && rn < 1e-10,
                   ExcMessage ("Native/reference jump or current-pressure operator mismatch: K="
                               + std::to_string (rk) + ", N=" + std::to_string (rn)));
      return s;
    }

    void
    System::Implementation::conversion_check (State &s, const std::string &path) const
    {
      const auto cells = numeric_csv (path + "/cells.csv", true),
                 K = numeric_csv (path + "/stiffness.csv");
      AssertThrow (cells.size () == system.get_triangulation ().n_global_active_cells (),
                   ExcMessage ("Snapshot conversion requires the matching native bulk mesh; "
                               "remeshing is unsupported."));
      boost::property_tree::ptree root;
      boost::property_tree::read_json (initial_file, root);
      const auto &tree = root.get_child_optional ("state") ? root.get_child ("state") : root;
      auto coefficients = json_values (tree, "affine_memory");
      coefficients.insert (coefficients.end (), s.memory.begin (), s.memory.end ());
      std::vector<double> raw (cells.size () * coefficients.size () * 5);
      gzFile file = gzopen ((path + "/field-center.bin.gz").c_str (), "rb");
      AssertThrow (
          file,
          ExcMessage ("Missing physical snapshot field for explicit stress conversion check."));
      const auto bytes = gzread (file, raw.data (), raw.size () * sizeof (double));
      gzclose (file);
      AssertThrow (bytes == static_cast<int> (raw.size () * sizeof (double)),
                   ExcMessage ("Incomplete physical snapshot field."));
      std::map<std::pair<double, double>, unsigned int> locations;
      for (unsigned int i = 0; i < cells.size (); ++i)
        locations[{ cells[i][1], cells[i][2] }] = i;
      double error = 0., pressure_error = 0., energy = 0., independent = 0.;
      const Quadrature<2> centre (std::vector<dealii::Point<2>>{ dealii::Point<2> (.5, .5) });
      FEValues<2> centre_values (system.get_mapping (), system.get_fe (), centre, update_values);
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto x
                = system.get_mapping ().transform_unit_to_real_cell (cell, centre.point (0));
            const auto i = locations.at ({ x[0], x[1] });
            Tensor expected;
            double value[5] = {};
            for (unsigned int j = 0; j < coefficients.size (); ++j)
              for (unsigned int k = 0; k < 5; ++k)
                value[k] += raw[5 * (i * coefficients.size () + j) + k] * coefficients[j];
            expected[0][0] = value[0];
            expected[1][1] = value[1];
            expected[2][2] = value[2];
            expected[0][1] = value[3];
            error = std::max (error, (physical_stress (s, cell, centre)[0] - expected).norm ());
            centre_values.reinit (cell);
            std::vector<double> p (1);
            centre_values[system.introspection ().extractors.pressure].get_function_values (
                *initial_pressure, p);
            AssertThrow (std::isfinite (p[0]),
                         ExcMessage ("Nonfinite imported physical pressure."));
            pressure_error = std::max (pressure_error, std::abs (p[0] - value[4]));
            for (unsigned int c = 0; c < 3; ++c)
              independent = std::max (independent, std::abs (independent_tensor.value (x, c)));
            const auto rule
                = s.geometry.quadrature (cell->vertex (0), cell->vertex (3), bulk_order);
            FEValues<2> values (system.get_mapping (), system.get_fe (), rule, update_JxW_values);
            values.reinit (cell);
            const auto stress = physical_stress (s, cell, rule);
            for (unsigned int q = 0; q < rule.size (); ++q)
              energy += values.JxW (q) * (stress[q] * stress[q]) / (4 * law.G);
          }
      error = Utilities::MPI::max (error, system.get_mpi_communicator ());
      pressure_error = Utilities::MPI::max (pressure_error, system.get_mpi_communicator ());
      energy = Utilities::MPI::sum (energy, system.get_mpi_communicator ());
      independent = Utilities::MPI::max (independent, system.get_mpi_communicator ());
      double reference_energy = 0.;
      for (unsigned int i = 0; i < coefficients.size (); ++i)
        for (unsigned int j = 0; j < coefficients.size (); ++j)
          reference_energy += .5 * coefficients[i] * K.at (i).at (j) * coefficients[j];
      AssertThrow (independent == 0.,
                   ExcMessage ("A checked snapshot cannot add new initial stress; use a fresh "
                               "independent-history control."));
      std::ostringstream message;
      message << std::setprecision (17)
              << "Native representation conversion: stress error=" << error
              << " Pa (bound 1e-7), energy error=" << energy - reference_energy
              << " J/m, bound=" << 1e-10 * std::max (1., std::abs (reference_energy));
      AssertThrow (error < 1e-7 && pressure_error < 1e-7
                       && std::abs (energy - reference_energy)
                              < 1e-10 * std::max (1., std::abs (reference_energy)),
                   ExcMessage (message.str ()));
      s.diagnostics["conversion_stress_Pa"] = error;
      s.diagnostics["conversion_pressure_Pa"] = pressure_error;
      s.diagnostics["conversion_energy_J_m"] = energy - reference_energy;
      s.diagnostics["bulk_J_m"] = energy;
    }

    std::shared_ptr<System::Field>
    System::initialize (const Field &particular_velocity, const BulkSolve &inverse)
    {
      AssertThrow (!data->accepted, ExcMessage ("Native cohesive state initializes once."));
      auto state = data->initial_mode == "checked prototype snapshot" ? data->import (inverse)
                                                                      : data->initialize_seed ();
      auto &s = *state;
      auto field = data->elastic (s.geometry, particular_velocity, s.rate, inverse);
      auto owned = data->owned ();
      *owned = *field;
      owned->block (introspection ().block_indices.pressure)
          = data->initial_pressure->block (introspection ().block_indices.pressure);
      auto result = data->ghosted ();
      *result = *owned;
      AssertThrow (
          std::isfinite (owned->l2_norm ()),
          ExcMessage ("Native cohesive initialization produced a nonfinite mixed solution."));
      if (!s.diagnostics.count ("bulk_J_m"))
        {
          double energy = 0.;
          for (const auto &cell : get_dof_handler ().active_cell_iterators ())
            if (cell->is_locally_owned ())
              {
                const auto rule
                    = s.geometry.quadrature (cell->vertex (0), cell->vertex (3), data->bulk_order);
                FEValues<2> values (get_mapping (), get_fe (), rule, update_JxW_values);
                values.reinit (cell);
                const auto stress = data->physical_stress (s, cell, rule);
                for (unsigned q = 0; q < rule.size (); ++q)
                  energy += values.JxW (q) * (stress[q] * stress[q]) / (4 * data->law.G);
              }
          s.diagnostics["bulk_J_m"] = Utilities::MPI::sum (energy, get_mpi_communicator ());
        }
      s.diagnostics["clock_outer_rejections"] = 0.;
      data->validate (s);
      data->accepted = std::move (state);
      return result;
    }
  }
}
