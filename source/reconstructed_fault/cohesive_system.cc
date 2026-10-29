/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <sstream>
#include <iomanip>

namespace aspect
{
  namespace CohesiveInterface
  {
    System::System (const Simulator<2> &sim)
    {
      initialize_simulator (sim);
      data = std::make_unique<Implementation> (*this);
    }
    System::~System () = default;
    System::Implementation::Implementation (System &s) : system (s), independent_tensor (3) {}

    System::Implementation::Tensor
    System::Implementation::dev3 (const SymmetricTensor<2, 2> &strain)
    {
      Tensor result;
      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = i; j < 2; ++j)
          result[i][j] = strain[i][j];
      const double tr = trace (result) / 3.;
      for (unsigned int i = 0; i < 3; ++i)
        result[i][i] -= tr;
      return result;
    }
    System::Implementation::Tensor
    System::Implementation::dev3 (const dealii::Tensor<2, 2> &gradient)
    {
      return dev3 (symmetrize (gradient));
    }

    void
    System::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Cohesive RSF pilot");
      {
        prm.declare_entry ("Initial state", "physical seed",
                           Patterns::Selection ("physical seed|checked prototype snapshot"));
        prm.declare_entry (
            "Initial state file", "", Patterns::Anything (),
            "Explicit checked initialization, never a trajectory or native restart.");
        prm.declare_entry ("Shear modulus", "1e6", Patterns::Double (0));
        prm.declare_entry ("Maxwell viscosity", "1e8", Patterns::Double (0));
        prm.declare_entry ("Fracture energy", "1", Patterns::Double (0));
        prm.declare_entry ("Intact strength", "3000", Patterns::Double (0));
        prm.declare_entry ("Background confinement", "10000", Patterns::Double (0));
        prm.declare_entry ("Reference friction coefficient", "0.1", Patterns::Double (0));
        prm.declare_entry ("RSF direct coefficient", "0.005", Patterns::Double (0));
        prm.declare_entry ("RSF evolution coefficient", "0.01", Patterns::Double (0));
        prm.declare_entry ("Reference velocity", "5e-6", Patterns::Double (0));
        prm.declare_entry ("State slip distance", "0.001", Patterns::Double (0));
        prm.declare_entry ("Radiation impedance", "25836.021365527624", Patterns::Double (0));
        prm.declare_entry ("Physical persistence length", "0.75", Patterns::Double (0));
        prm.declare_entry ("Numerical lifting half width", "0.75", Patterns::Double (0));
        prm.declare_entry ("Future surface spacing", "0.09375", Patterns::Double (0));
        prm.declare_entry ("Candidate extent", "0.1875", Patterns::Double (0));
        prm.declare_entry ("Candidate refresh interval", "0.0025", Patterns::Double (0));
        prm.declare_entry ("Maximum formed length", "7.5", Patterns::Double (0));
        prm.enter_subsection ("Initial independent tensor");
        Functions::ParsedFunction<2>::declare_parameters (prm, 3);
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    void
    System::parse_parameters (ParameterHandler &prm)
    {
      auto &d = *data;
      d.end_time = prm.get_double ("End time");
      prm.enter_subsection ("Cohesive RSF pilot");
      {
        d.initial_mode = prm.get ("Initial state");
        d.initial_file = prm.get ("Initial state file");
        d.law.G = prm.get_double ("Shear modulus");
        d.law.eta = prm.get_double ("Maxwell viscosity");
        d.law.Gc = prm.get_double ("Fracture energy");
        d.law.peak = prm.get_double ("Intact strength");
        d.law.p0 = prm.get_double ("Background confinement");
        d.law.mu0 = prm.get_double ("Reference friction coefficient");
        d.law.direct = prm.get_double ("RSF direct coefficient");
        d.law.evolution = prm.get_double ("RSF evolution coefficient");
        d.law.V0 = prm.get_double ("Reference velocity");
        d.law.Dc = prm.get_double ("State slip distance");
        d.law.radiation = prm.get_double ("Radiation impedance");
        d.R = prm.get_double ("Physical persistence length");
        d.support = prm.get_double ("Numerical lifting half width");
        d.spacing = prm.get_double ("Future surface spacing");
        d.candidate_extent = prm.get_double ("Candidate extent");
        d.cadence = prm.get_double ("Candidate refresh interval");
        d.max_length = prm.get_double ("Maximum formed length");
        prm.enter_subsection ("Initial independent tensor");
        d.initial_expression = prm.get ("Function expression");
        d.independent_tensor.parse_parameters (prm);
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      const auto &p = get_parameters ();
      AssertThrow (
          !p.enable_phase_field && !p.reconstruct_faults && !p.enable_elasticity
              && !p.include_melt_transport && !p.mesh_deformation_enabled
              && p.initial_adaptive_refinement == 0 && p.adaptive_refinement_interval == 0
              && p.stokes_velocity_degree == 2 && p.use_locally_conservative_discretization
              && p.stokes_solver_type == aspect::Parameters<2>::StokesSolverType::block_amg
              && p.pressure_normalization == "volume" && !p.convert_to_years
              && p.nonlinear_solver == aspect::Parameters<2>::NonlinearSolver::single_Advection_iterated_Newton_Stokes
              && !p.run_postprocessors_on_nonlinear_iterations,
          ExcMessage (
              "Cohesive RSF pilot requires fixed 2D Q2/P1-discontinuous native AMG mechanics, "
              "mean-zero perturbation pressure, seconds, and no other "
              "fracture/elasticity/melt/AMR/mesh motion."));
      AssertThrow (get_geometry_model ().get_periodic_boundary_pairs ().empty (),
                   ExcMessage ("Cohesive RSF pilot is nonperiodic."));
      AssertThrow (d.law.G > 0 && d.law.eta > 0 && d.law.Gc > 0 && d.law.peak > 0 && d.law.p0 > 0
                       && d.law.direct > 0 && d.law.Dc > 0 && d.law.V0 > 0 && d.R > 0
                       && d.support > 0 && d.spacing > 0 && d.cadence > 0
                       && d.candidate_extent >= d.spacing,
                   ExcMessage ("Invalid cohesive RSF physical/numerical coefficients."));
    }

    std::string
    System::Implementation::configuration () const
    {
      std::ostringstream out;
      out << std::setprecision (17) << "native exact jump cohesive RSF v2;";
      for (const double x :
           { law.G, law.eta, law.Gc, law.peak, law.p0, law.direct, law.evolution, law.mu0, law.V0,
             law.Dc, law.radiation, R, support, spacing, candidate_extent, cadence, max_length })
        out << x << ';';
      return out.str ();
    }

    std::shared_ptr<System::Field>
    System::Implementation::owned () const
    {
      return std::make_shared<Field> (system.introspection ().index_sets.system_partitioning,
                                      system.get_mpi_communicator ());
    }
    std::shared_ptr<System::Field>
    System::Implementation::ghosted () const
    {
      auto field = std::make_shared<Field> ();
      field->reinit (system.get_solution ());
      *field = 0.;
      return field;
    }
    std::shared_ptr<System::Field>
    System::Implementation::scaled (const Field &field) const
    {
      auto result = owned ();
      *result = field;
      result->block (system.introspection ().block_indices.pressure) /= pressure_scaling;
      return result;
    }

    bool
    System::bulk_matrix_ready () const
    {
      return data->matrix_ready;
    }
    bool
    System::initialized () const
    {
      return bool (data->accepted);
    }
    bool
    System::has_restart_record () const
    {
      return !data->pending_checkpoint.empty ();
    }
    bool
    System::at_end_time () const
    {
      if (!data->accepted)
        return false;
      const double t = data->accepted->time;
      const double roundoff = data->clock_bound (*data->accepted);
      return t >= data->end_time || data->end_time - t <= roundoff;
    }

    double
    System::Implementation::clock_bound (const State &s) const
    {
      const auto found = s.diagnostics.find ("clock_outer_rejections");
      const double retries = found == s.diagnostics.end () ? 0. : found->second;
      // One addition per accepted advance_time; each outer retry uses a
      // subtraction and addition. Three extra operations cover the two input
      // roundings and the final endpoint subtraction. All clock operands in
      // this positive-time interval are bounded by max(t,end). Use the standard
      // gamma_n bound, rounded outward, rather than a fixed number of ulps.
      const double u = .5 * std::numeric_limits<double>::epsilon ();
      const double nu = (s.step + 2 * retries + 3) * u;
      AssertThrow (nu < .5,
                   ExcMessage ("Native clock operation count exceeded its rounding bound."));
      const double gamma = nu / (1 - nu);
      return std::nextafter (gamma / (1 - gamma)
                                 * std::max (std::abs (s.time), std::abs (end_time)),
                             std::numeric_limits<double>::infinity ());
    }

    void
    System::assemble_bulk_matrix (LinearAlgebra::BlockSparseMatrix &matrix,
                                  const double pressure_scale)
    {
      auto &d = *data;
      const auto &boundary = get_boundary_velocity_manager ();
      for (const auto id : get_geometry_model ().get_used_boundary_indicators ())
        {
          const bool zero = boundary.get_zero_boundary_velocity_indicators ().count (id);
          const bool prescribed
              = boundary.get_prescribed_boundary_velocity_indicators ().count (id);
          AssertThrow (
              zero || prescribed,
              ExcMessage ("Native cohesive pilot requires fully prescribed boundary velocity."));
          if (prescribed)
            {
              const auto mask = boundary.get_component_mask (id);
              for (const auto c : introspection ().component_indices.velocities)
                AssertThrow (mask[c], ExcMessage ("Native cohesive pilot does not admit partial "
                                                  "boundary velocity constraints."));
            }
        }
      bool cartesian = true;
      for (const auto &cell : get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto lo = cell->vertex (0), hi = cell->vertex (3);
            cartesian &= cell->level () == 0;
            for (unsigned v = 0; v < 4; ++v)
              cartesian &= cell->vertex (v)
                           == dealii::Point<2> (v % 2 ? hi[0] : lo[0], v / 2 ? hi[1] : lo[1]);
          }
      AssertThrow (
          Utilities::MPI::min (static_cast<unsigned> (cartesian), get_mpi_communicator ()),
          ExcMessage ("Native cohesive pilot currently requires the fixed affine Cartesian coarse "
                      "mesh; general/hanging mesh transfer is unsupported."));
      AssertThrow (
          !d.matrix_ready,
          ExcMessage ("Native cohesive reference matrix assembled twice without invalidation."));
      AssertThrow (std::isfinite (pressure_scale) && pressure_scale > 0.,
                   ExcMessage ("Native pressure scaling was not initialized."));
      d.pressure_scaling = pressure_scale;
      matrix = 0.;
      const auto &fe = get_fe ();
      const auto &intro = introspection ();
      std::vector<unsigned int> indices;
      for (unsigned int i = 0; i < fe.n_dofs_per_cell (); ++i)
        if (intro.is_stokes_component (fe.system_to_component_index (i).first))
          indices.push_back (i);
      const QGauss<2> rule (6);
      FEValues<2> values (get_mapping (), fe, rule,
                          update_gradients | update_values | update_JxW_values);
      std::vector<types::global_dof_index> ids (fe.n_dofs_per_cell ()), stokes (indices.size ());
      for (const auto &cell : get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            values.reinit (cell);
            cell->get_dof_indices (ids);
            for (unsigned int i = 0; i < indices.size (); ++i)
              stokes[i] = ids[indices[i]];
            FullMatrix<double> local (indices.size (), indices.size ());
            std::vector<Implementation::Tensor> E (indices.size ());
            std::vector<double> div (indices.size ()), p (indices.size ());
            for (unsigned int q = 0; q < rule.size (); ++q)
              {
                for (unsigned int i = 0; i < indices.size (); ++i)
                  {
                    E[i] = Implementation::dev3 (
                        values[intro.extractors.velocities].symmetric_gradient (indices[i], q));
                    div[i] = values[intro.extractors.velocities].divergence (indices[i], q);
                    p[i] = values[intro.extractors.pressure].value (indices[i], q);
                  }
                for (unsigned int i = 0; i < indices.size (); ++i)
                  for (unsigned int j = 0; j < indices.size (); ++j)
                    local (i, j) += values.JxW (q)
                                    * (2 * d.law.G * (E[i] * E[j])
                                       - pressure_scale * (div[i] * p[j] + p[i] * div[j]));
              }
            get_current_constraints ().distribute_local_to_global (local, stokes, matrix);
          }
      matrix.compress (VectorOperation::add);
      d.matrix_ready = true;
    }

    unsigned int
    System::Implementation::formed_cells (const State &s) const
    {
      unsigned int cells = 0;
      for (unsigned int i = 0; i < s.formed; ++i)
        cells += s.geometry.pieces ()[i].cells;
      return cells;
    }

    std::vector<PointState>
    System::Implementation::points (const State &s, const Geometry &geometry) const
    {
      std::vector<PointState> result (geometry.surface_points ().size ());
      for (auto &p : result)
        p.theta = s.virgin_theta;
      const auto &fault = s.manager->get_fault (0);
      for (unsigned int cell = 0; cell < formed_cells (s); ++cell)
        {
          const auto row = fault.get_properties (cell);
          for (unsigned int q = 0; q < 3; ++q)
            {
              const unsigned int at = history_position + 6 * q;
              result[3 * cell + q]
                  = { row[at], row[at + 1], row[at + 2], row[at + 3], row[at + 4], row[at + 5] };
            }
        }
      return result;
    }

    std::vector<System::Implementation::Tensor>
    System::Implementation::physical_stress (const State &s, const BulkHistory<2>::Cell &cell,
                                             const Quadrature<2> &rule) const
    {
      auto result = s.bulk->evaluate (cell, rule);
      for (unsigned int q = 0; q < rule.size (); ++q)
        {
          const auto p = system.get_mapping ().transform_unit_to_real_cell (cell, rule.point (q));
          for (const auto &mode : s.geometry.lifting (p))
            result[q] += 2 * law.G * s.memory.at (mode.index) * dev3 (mode.gradient);
        }
      return result;
    }
  }
}
