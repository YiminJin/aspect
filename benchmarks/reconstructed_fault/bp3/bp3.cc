#include "bp3_model.h"
#include "first_event.h"
#include "output_schedule.h"

#include <aspect/initial_composition/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/termination_criteria/interface.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include "mature_fault.h"
#include "work_replay.h"
#include "matched_resolution.h"
#if defined(ASPECT_BP5_WEAK_INITIALIZATION) && defined(ASPECT_BP5_STEADY_INITIALIZATION)
#error Select only one BP5 initialization procedure
#endif
#ifdef ASPECT_BP5_WEAK_INITIALIZATION
#include "../bp5/weak_initialization.h"
#endif
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
#include "../bp5/steady_initialization.h"
#endif
#ifdef ASPECT_BP5_NORMAL_CONTROL
#include "../bp5/normal_control_checks.h"
#endif
#ifdef ASPECT_BP3_DISTURBANCE_TEST
#include "disturbance_diagnostic.h"
#endif

namespace aspect
{
  namespace BP3Benchmark
  {
    bool converged = false;
    bool long_run_stop = false;
    bool restored_history = false;
    std::string bottom_normalization_completion_file;
    unsigned int newton_updates = 0, krylov_iterations = 0;
    double minimum_alpha = 1.;
    double accepted_nonlinear_residual = std::numeric_limits<double>::infinity ();
    std::vector<std::vector<bool>> final_active;

    template <int dim>
    void
    verify_velocity_constraints (const SimulatorAccess<dim> &sim)
    {
      // Inspect the realized physical lift, including hanging constraints.
      // This private diagnostic does not publish or modify the solver iterate.
      LinearAlgebra::BlockVector owned (sim.introspection ().index_sets.system_partitioning,
                                        sim.get_mpi_communicator ());
      owned = sim.get_solution ();
      sim.get_current_constraints ().distribute (owned);
      LinearAlgebra::BlockVector lifted (sim.introspection ().index_sets.system_partitioning,
                                         sim.introspection ().index_sets.system_relevant_partitioning,
                                         sim.get_mpi_communicator ());
      lifted = owned;
      const auto &fe = sim.get_fe ();
      std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
      double error[2] = { 0, 0 };
      unsigned int count[2] = { 0, 0 };
      const auto &box
          = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>> (sim.get_geometry_model ());
      const double left = box.get_origin ()[0], right = left + box.get_extents ()[0];
      for (const auto &cell : sim.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned () && cell->at_boundary ())
          {
            cell->get_dof_indices (dofs);
            for (unsigned int j = 0; j < dofs.size (); ++j)
              for (unsigned int d = 0; d < 2; ++d)
                if (fe.system_to_component_index (j).first
                    == sim.introspection ().component_indices.velocities[d])
                  {
                    const auto p = sim.get_mapping ().transform_unit_to_real_cell (
                        cell, fe.get_unit_support_points ()[j]);
                    if (p[0] != left && p[0] != right)
                      continue;
                    const unsigned int side = p[0] == left ? 0 : 1;
                    const double expected
                        = (side == 0 ? 1 : -1) * .5 * BP3::Vp * (d == 0 ? BP3::cosine : BP3::sine);
                    error[side] = std::max (error[side], std::abs (lifted[dofs[j]] - expected));
                    ++count[side];
                  }
          }
      std::ofstream out;
      if (sim.get_pcout ().is_active ())
        {
          out.open (sim.get_output_directory () + "velocity_constraints.csv");
          out << std::setprecision (17)
              << "side,expected_ux,expected_uy,expected_speed,max_actual_error,samples\n";
        }
      for (unsigned int side = 0; side < 2; ++side)
        {
          const auto n = Utilities::MPI::sum (count[side], sim.get_mpi_communicator ());
          const double maximum = Utilities::MPI::max (error[side], sim.get_mpi_communicator ());
          AssertThrow (n > 0 && maximum < 1e-22,
                       ExcMessage ("BP3 realized lateral velocity constraints are incorrect."));
          const double sign = side == 0 ? 1 : -1;
          if (out)
            out << (side == 0 ? "left" : "right") << ',' << sign * .5 * BP3::Vp * BP3::cosine << ','
                << sign * .5 * BP3::Vp * BP3::sine << ',' << .5 * BP3::Vp << ',' << maximum << ',' << n
                << '\n';
        }
    }

    template <int dim>
    void
    prescribe_phase (const SimulatorAccess<dim> &sim, AffineConstraints<double> &constraints)
    {
      AssertThrow (dim == 2, ExcMessage ("BP3 currently supports two dimensions."));
      const auto profiles = sim.get_phase_field_handler ().get_phase_field_profiles (BP3::core_phi);
      const auto &fe = sim.get_fe ();
      const auto phi = sim.introspection ().variable ("phase_field").first_component_index;
      std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
      for (const auto &cell : sim.get_dof_handler ().active_cell_iterators ())
        if (!cell->is_artificial ())
          {
            cell->get_dof_indices (dofs);
            for (unsigned int j = 0; j < dofs.size (); ++j)
              if (fe.system_to_component_index (j).first == phi && constraints.can_store_line (dofs[j])
                  && !constraints.is_constrained (dofs[j]))
                {
                  const auto p = sim.get_mapping ().transform_unit_to_real_cell (
                      cell, fe.get_unit_support_points ()[j]);
                  constraints.add_line (dofs[j]);
                  constraints.set_inhomogeneity (dofs[j],
                                                 profiles[0]->value (BP3::normal_distance (p[0], p[1])));
                }
          }
    }

    template <int dim>
    void
    initial_history (const SimulatorAccess<dim> &sim)
    {
      // Extend the straight stationary distance field through the box boundaries;
      // use the current handler's H law and configured activation, not old BP3 H.
      auto &pm = sim.get_phase_field_handler ().get_associated_particle_manager ();
      const auto H
          = pm.get_property_manager ().get_data_info ().get_position_by_field_name ("crack_driving_force");
      const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
          sim.get_material_model ());
      const auto profiles = sim.get_phase_field_handler ().get_phase_field_profiles (BP3::core_phi);
      for (auto &particle : pm.get_particle_handler ())
        {
          const auto p = particle.get_location ();
          const double phi = profiles[0]->value (BP3::normal_distance (p[0], p[1]));
          const double f = BP3::depth_fraction (p[1]);
          if (phi > model.get_phase_field_activation_threshold ())
            particle.get_properties ()[H] = sim.get_phase_field_handler ().stationary_crack_driving_force (
                { 1 - f, f }, phi, BP3::core_phi);
        }
    }

    template <int dim>
    void
    prepare (const SimulatorAccess<dim> &sim, bool temperature, unsigned int, const SolverControl &)
    {
      if (!temperature)
        return;
      auto &manager = sim.get_reconstructed_fault_manager ();
      AssertThrow (dim == 2 && manager.get_faults ().size () == 1,
                   ExcMessage ("Modified BP3 requires one fixed two-dimensional fault."));
      const auto &fault = manager.get_fault (0);
      for (unsigned int v = 0; v < fault.n_vertices (); ++v)
        AssertThrow (BP3::normal_distance (fault.vertex (v)[0], fault.vertex (v)[1]) < 1e-8,
                     ExcMessage ("BP3 reconstructed dip changed."));
      manager.set_prescribed_slip_rates (std::vector<std::map<unsigned int, double>> (1));

      // Reattach runtime selectors on every entry, including restart. The
      // manager owns the checkpointed coefficients, not a new initial solve.
      auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &> (
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (sim.get_material_model ()));
      AssertThrow (
          model.is_mature_frictional_fault () && !bottom_normalization_completion_file.empty (),
          ExcMessage ("Modified BP3 requires frozen mature mechanics and paired completion inputs."));
      model.set_boundary_normalization_completion_file (bottom_normalization_completion_file);
      const auto &box
          = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>> (sim.get_geometry_model ());
      manager.enable_bottom_source_continuation (0, box.get_origin (),
                                                 box.get_origin () + box.get_extents ());
      manager.enable_top_source_continuation ();
      sim.get_reconstructed_fault_surface_system ().enable_bulk_work_measure ();
      if (sim.get_timestep_number () != 0 || restored_history)
        model.set_reconstructed_fault_background_traction_property (
            manager.get_property_index ("background tractions"),
            manager.get_property_index ("BP3 fixed shear correction"));
      if (sim.get_timestep_number () != 0 || restored_history)
        return;
      verify_paired_mesh (sim);
      verify_velocity_constraints (sim);
      manager.initialize_slip_rate (0, std::vector<double> (fault.n_vertices (), BP3::Vinit));
      // Record the realized grid before projection can fail; center-sampled
      // refinement functions need not refine every cell crossed by the fault.
      std::ofstream mesh (sim.get_output_directory () + "initial_mesh_"
                          + std::to_string (Utilities::MPI::this_mpi_process (sim.get_mpi_communicator ()))
                          + ".csv");
      mesh << std::setprecision (17) << "cell,level,x,y,h,distance\n";
      for (const auto &cell : sim.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto p = cell->center ();
            mesh << cell->id ().to_string () << ',' << cell->level () << ',' << p[0] << ',' << p[1] << ','
                 << cell->diameter () / std::sqrt (2.) << ',' << BP3::normal_distance (p[0], p[1]) << '\n';
          }
      mesh.close ();
      // The simulator owns a mutable material object. This initialization-only
      // callback precedes the particle-to-FE transfer; no assembly loop casts.
      model.prepare_reconstructed_fault_mechanical_solve ();

#ifdef ASPECT_BP5_STEADY_INITIALIZATION
      initialize_steady_prestress (sim, model);
#else
      // Set the selected initial state only after surface material preparation.
      // The ordinary particle property supplies the same initial function.
      const auto state
          = manager.get_property_information ()[manager.get_property_index ("phase field fault state")]
                .position;
      for (unsigned int v = 0; v < fault.n_vertices (); ++v)
        manager.get_fault (0).get_properties (v)[state]
            = BP3::configured_initial_state (
                BP3::down_dip (fault.vertex (v)[0], fault.vertex (v)[1]), model.get_fault_friction ());

      initialize_mature_prestress (sim, model);
#ifdef ASPECT_BP5_WEAK_INITIALIZATION
      initialize_weak_state (sim, model);
#endif
#endif
    }
  }

  template <int dim>
  void
  connect_bp3 (SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect (&BP3Benchmark::prescribe_phase<dim>);
    // Install after manager/particle initialization slots have been registered.
    signals.post_simulator_initialization.connect (
        [] (const SimulatorAccess<dim> &sim)
          {
            sim.get_reconstructed_fault_manager ().register_property ("background tractions", 2);
            sim.get_reconstructed_fault_manager ().register_property ("cumulative_signed_slip_m", 1);
            sim.get_reconstructed_fault_manager ().register_property ("BP3 fixed shear correction", 3);
            sim.get_signals ().post_set_initial_state.connect (&BP3Benchmark::initial_history<dim>);
          });
    signals.post_advection_solver.connect (&BP3Benchmark::prepare<dim>);
    signals.start_timestep.connect ([] (const SimulatorAccess<dim> &) { BP3Benchmark::converged = false; });
#ifdef ASPECT_BP3_DISTURBANCE_TEST
    signals.start_timestep.connect (&BP3Disturbance::begin<dim>);
    signals.post_advection_solver.connect (&BP3Disturbance::enable_normal_control<dim>);
#endif
    signals.post_nonlinear_solver.connect (
        [] (const SolverControl &c)
          {
            BP3Benchmark::converged
                = c.last_check () == SolverControl::success && c.last_value () < c.tolerance ();
            BP3Benchmark::accepted_nonlinear_residual = c.last_value ();
          });
    signals.post_reconstructed_fault_solver.connect (
        [] (unsigned int n, unsigned int k, double alpha, const std::vector<std::vector<bool>> &active)
          {
            BP3Benchmark::newton_updates = n;
            BP3Benchmark::krylov_iterations = k;
            BP3Benchmark::minimum_alpha = alpha;
            BP3Benchmark::final_active = active;
          });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR (connect_bp3<2>, connect_bp3<3>)

  namespace InitialComposition
  {
    template <int dim> class BP3Initial : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      void initialize () override
      {
        friction = &Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
                 this->get_material_model ()).get_fault_friction ();
      }

      double
      initial_composition (const Point<dim> &p, const unsigned int field) const override
      {
        // Extend the official 15--18 km down-dip transition horizontally
        // into the bulk, just as for a. The sharp-fault values are unchanged.
        const double xd = (BP3::box_size - p[1]) / BP3::sine;
        const auto &name = this->introspection ().name_for_compositional_index (field);
        if (name == "theta_initial")
          return BP3::configured_initial_state (xd, *friction);
        if (name == "strengthening")
          return BP3::depth_fraction (p[1]);
        AssertThrow (name == "tau_xx" || name == "tau_yy" || name == "tau_xy",
                     ExcMessage ("Unexpected BP3 composition."));
        return 0.; // Maxwell stores Delta tau, not the official prestress.
      }

    private:
      const MaterialModel::Rheology::FaultFriction<dim> *friction = nullptr;
    };
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL (BP3Initial, "reconstructed fault BP3",
                                               "Official BP3 state and zero initial bulk stress change.")
  }

  namespace BoundaryVelocity
  {
    template <int dim> class BP3Velocity : public Interface<dim>
    {
    public:
      Tensor<1, dim>
      boundary_velocity (const types::boundary_id, const Point<dim> &p) const override
      {
        Tensor<1, dim> v;
        const double signed_normal = (BP3::trace_x - p[0]) * BP3::sine - (BP3::box_size - p[1]) * BP3::cosine;
        const double sign = signed_normal >= 0 ? 1 : -1;
        v[0] = sign * .5 * BP3::Vp * BP3::cosine;
        v[1] = sign * .5 * BP3::Vp * BP3::sine;
        return v;
      }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL (
        BP3Velocity, "reconstructed fault BP3",
        "BP3 far-field rigid translation in the documented rotated chart.")
  }

  namespace BoundaryTraction
  {
    template <int dim> class BP3Traction : public Interface<dim>
    {
    public:
      Tensor<1, dim>
      boundary_traction (const types::boundary_id, const Point<dim> &, const Tensor<1, dim> &) const override
      {
        return {}; // Zero perturbation traction; no Airy load enters Stokes.
      }
    };
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL (
        BP3Traction, "reconstructed fault BP3",
        "Zero bottom stress-change traction for the finite-domain BP3 pilot.")
  }

  namespace Postprocess
  {
    template <int dim> class BP3Output : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void
      declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection ("Postprocess");
        prm.enter_subsection ("BP3");
        prm.declare_entry ("Weakening region length", "15000", Patterns::Double (0),
                           "Down-dip extent in metres of the uniform weakening region; followed by a 3 km transition.");
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
        prm.declare_entry ("Weakening initial state ratio", "1", Patterns::Double (0, 1),
                           "Initial Vinit*Theta/Dc in the weakening material. Must be positive. "
                           "The projected strengthening mixture blends this ratio geometrically to one.");
#endif
        prm.declare_entry ("Heavy output slip interval", "0.1", Patterns::Double (0),
                           "Maximum nodal slip change in metres since the last heavy output.");
        prm.declare_entry ("Heavy output time interval", "31557600", Patterns::Double (0),
                           "Maximum physical seconds between heavy outputs; zero disables.");
        prm.declare_entry ("Profile slip interval", "0.01", Patterns::Double (0),
                           "Maximum nodal slip change in metres between lightweight profiles.");
        prm.declare_entry ("Profile time interval", "3155760", Patterns::Double (0),
                           "Maximum physical seconds between profiles; zero disables.");
        prm.declare_entry ("Graceful wall seconds", "86400", Patterns::Double (1),
                           "Stop at the next accepted state after this process wall budget.");
        prm.declare_entry ("Last accepted step", "2147483647", Patterns::Integer (0),
                           "Optional bounded verification stop; not a timestep controller.");
        prm.declare_entry (
            "Stop after first event", "false", Patterns::Bool (),
            "First-event pilot only; false permits recurrence research without claiming qualification.");
        prm.declare_entry (
            "Audit full state every step", "false", Patterns::Bool (),
            "Export bulk DoFs and stable-ID particle histories for the short restart regression.");
        prm.declare_entry ("Mature prestress file", "", Patterns::Anything (),
                           "Fresh-only initial fixed prestress coefficients x,y,shear,normal,a,b,d; "
                           "checkpointed thereafter.");
        prm.declare_entry (
            "Bottom normalization completion file", "", Patterns::Anything (),
            "Paired top/bottom outside-profile integrals (count, then id x y integral). "
            "The table must match the fixed mesh, phase profile and fault geometry. "
            "Both in-box source continuations use the same work measure, without changing connectivity.");
        prm.leave_subsection ();
        prm.leave_subsection ();
      }

      void
      parse_parameters (ParameterHandler &prm) override
      {
        prm.enter_subsection ("Postprocess");
        prm.enter_subsection ("BP3");
        BP3::weakening_length = prm.get_double ("Weakening region length");
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
        BP3::weakening_initial_state_ratio = prm.get_double ("Weakening initial state ratio");
        AssertThrow (BP3::weakening_initial_state_ratio > 0.,
                     ExcMessage ("Weakening initial state ratio must be positive."));
#endif
        heavy.slip_interval = prm.get_double ("Heavy output slip interval");
        heavy.time_interval = prm.get_double ("Heavy output time interval");
        profiles.slip_interval = prm.get_double ("Profile slip interval");
        profiles.time_interval = prm.get_double ("Profile time interval");
        AssertThrow (heavy.slip_interval > 0. && profiles.slip_interval > 0.,
                     ExcMessage ("Output slip intervals must be positive."));
        wall_seconds = prm.get_double ("Graceful wall seconds");
        last_requested_step = prm.get_integer ("Last accepted step");
        stop_after_event = prm.get_bool ("Stop after first event");
        audit_states = prm.get_bool ("Audit full state every step");
        BP3Benchmark::mature_prestress_file = prm.get ("Mature prestress file");
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
        AssertThrow (BP3Benchmark::mature_prestress_file.empty (),
                     ExcMessage ("Steady BP5 constructs its native weak background; remove the captured Mature prestress file."));
#endif
        BP3Benchmark::bottom_normalization_completion_file = prm.get ("Bottom normalization completion file");
        prm.leave_subsection ();
        prm.leave_subsection ();
      }

      void
      initialize () override
      {
        wall_start = std::chrono::steady_clock::now ();
        this->get_signals ().allow_native_output.connect (
            [this] (const std::string &)
              {
                AssertThrow (last_step == this->get_timestep_number (),
                             ExcMessage ("BP3 output decision must precede native writers."));
                return heavy_pending;
              });
        this->get_signals ().post_checkpoint.connect (
            [this] (const std::string &path)
              {
                // This bounded diagnostic needs the accepted-step-2 checkpoint
                // and termination checkpoint only; no changes to its clock.
                if (std::getenv ("ASPECT_BP5_SHORT_TEST") && last_step == 2)
                  const_cast<Parameters<dim> &>(this->get_parameters ()).checkpoint_steps = 0;
                if (this->get_pcout ().is_active ())
                  {
                    std::ofstream out (path + "/bp3_accepted_state.txt");
                    out << std::setprecision (17) << last_step << ' ' << last_accepted_time << '\n';
                    AssertThrow (out, ExcMessage ("Cannot label BP3 checkpoint accepted state."));
                    // Snapshot only small time-series metadata, never copy a
                    // whole event state. A restart branch restores this prefix.
                    const auto destination = path + "/bp3_output_metadata";
                    std::filesystem::create_directories (destination);
                    for (const auto &entry :
                         std::filesystem::directory_iterator (this->get_output_directory ()))
                      if (entry.is_regular_file ())
                        {
                          const auto name = entry.path ().filename ().string ();
                          const auto extension = entry.path ().extension ().string ();
                          if (extension == ".pvd" || extension == ".visit" || extension == ".xdmf"
                              || name == "profiles.csv" || name == "heavy_outputs.csv"
                              || name == "stations.csv" || name == "accepted_steps.csv"
                              || name == "first_event.csv" || name == "statistics")
                            std::filesystem::copy_file (entry.path (), destination + "/" + name,
                                                        std::filesystem::copy_options::overwrite_existing);
                        }
                  }
              });
      }

      void
      save (std::map<std::string, std::string> &status) const override
      {
        std::ostringstream stream;
        {
          aspect::oarchive archive (stream);
          const unsigned int version = 5;
          const bool selected = true;
          // Keep the qualified v5 wire layout. These retired custom-writer
          // fields have no role in native output; they are not runtime state.
          const double unused_time = -std::numeric_limits<double>::max ();
          const std::vector<std::string> unused_labels;
          archive << version << selected << slip << previous_theta << last_step << event << unused_time
                  << unused_time << unused_labels << selected << selected << BP3Benchmark::work_initial_H
                  << BP3Benchmark::work_initial_geometry << BP3Benchmark::work_initial_I << selected << heavy
                  << profiles << last_accepted_time;
        }
        status["BP3 accepted history"] = stream.str ();
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
        status["BP5 initial condition"] = BP3::initial_condition_identity ();
#endif
      }

      void
      load (const std::map<std::string, std::string> &status) override
      {
        const auto initialization = status.find ("BP5 initial condition");
#ifdef ASPECT_BP5_STEADY_INITIALIZATION
        AssertThrow (initialization != status.end ()
                       && initialization->second == BP3::initial_condition_identity (),
                     ExcMessage ("BP5 checkpoint initial-condition identity/ratio differs; histories cannot be converted on restart."));
#else
        AssertThrow (initialization == status.end (),
                     ExcMessage ("This checkpoint requires the steady BP5 initialization plugin."));
#endif
        const auto entry = status.find ("BP3 accepted history");
        AssertThrow (entry != status.end (), ExcMessage ("Checkpoint lacks BP3 history."));
        std::istringstream stream (entry->second);
        aspect::iarchive archive (stream);
        unsigned int version;
        bool mature, frictional, work, native;
        double unused_profile_time, unused_bulk_time;
        std::vector<std::string> unused_labels;
        archive >> version;
        AssertThrow (version == 5,
                     ExcMessage ("Modified BP3 requires its version-5 particle/output checkpoint."));
        archive >> mature >> slip >> previous_theta >> last_step >> event >> unused_profile_time
            >> unused_bulk_time >> unused_labels >> frictional >> work >> BP3Benchmark::work_initial_H
            >> BP3Benchmark::work_initial_geometry >> BP3Benchmark::work_initial_I >> native >> heavy
            >> profiles >> last_accepted_time;
        AssertThrow (mature && frictional && work && native,
                     ExcMessage ("Cannot convert a different BP3 formulation on restart."));
        AssertThrow (!slip.empty () && slip.size () == previous_theta.size ()
                         && !BP3Benchmark::work_initial_H.empty ()
                         && BP3Benchmark::work_initial_geometry.size () == slip.size ()
                         && BP3Benchmark::work_initial_I.size () == slip.size (),
                     ExcMessage ("Incomplete BP3 checkpoint histories."));
        BP3Benchmark::restored_history = true;
#ifdef ASPECT_BP3_DISTURBANCE_TEST
        BP3Disturbance::perturb_audit(previous_theta, BP3Benchmark::work_initial_geometry);
#endif
      }

      std::pair<std::string, std::string>
      execute (TableHandler &) override
      {
        AssertThrow (BP3Benchmark::converged, ExcMessage ("BP3 requires genuine bulk/surface convergence."));
#ifdef ASPECT_BP5_NORMAL_CONTROL
        BP3Benchmark::check_normal_control(*this);
#endif
        const double Dc = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
            this->get_material_model ()).characteristic_fault_slip_distance ();
#ifdef ASPECT_BP3_DISTURBANCE_TEST
        BP3Disturbance::accepted(*this);
#endif
        const auto &manager = this->get_reconstructed_fault_manager ();
        const auto &fault = manager.get_faults ()[0];
        const auto &weak = this->get_reconstructed_fault_surface_system ().get_linearization_residual ();
        const auto position = [&] (const std::string &name)
          { return manager.get_property_information ()[manager.get_property_index (name)].position; };
        const auto state = position ("phase field fault state"),
                   C = position ("phase field fault cohesive traction");
        const unsigned int n = fault.n_vertices ();
        AssertThrow (last_step == numbers::invalid_unsigned_int
                         || this->get_timestep_number () == last_step + 1,
                     ExcMessage ("BP3 output must visit each accepted state exactly once."));
        if (slip.empty ())
          slip.assign (n, 0.);
        const auto &V = manager.get_timestep_committed_slip_rate (0);
        const auto prescribed = manager.prescribed_slip_rate_mask ();
        if (this->get_timestep_number () > 0)
          for (unsigned int j = 0; j < n; ++j)
            slip[j] += this->get_timestep () * V[j];
        const auto qfield = ReconstructedFaultUtilities::solve_tridiagonal_system (
            weak.mass_diagonal[0], weak.mass_off_diagonal[0], weak.shear_traction[0]);
        const auto Cfield = ReconstructedFaultUtilities::solve_tridiagonal_system (
            weak.mass_diagonal[0], weak.mass_off_diagonal[0], weak.cohesive_traction[0]);
        const auto normal_field = ReconstructedFaultUtilities::solve_tridiagonal_system (
            weak.mass_diagonal[0], weak.mass_off_diagonal[0], weak.normal_traction[0]);
        // The selected length-scale diagnostic exports current mechanical
        // work/source observations without dumping every bulk/particle field.
        const bool work_files = audit_states || std::getenv ("ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC");
#ifdef ASPECT_BP3_DISTURBANCE_TEST
        // The normal-feedback control has a prescribed friction traction but
        // still has its own physical bulk normal stress. Keep the independent
        // working-stress observer checking that physical quantity.
        BP3Benchmark::export_work_replay (*this, BP3Disturbance::physical_weak(weak), work_files);
#else
        BP3Benchmark::export_work_replay (*this, weak, work_files);
#endif

        // A Q1 rate attains its physical maximum at a vertex. Prescribed deep
        // nodes participate in max(V), but not in free/lower-contact counts.
        const unsigned int imax = std::max_element (V.begin (), V.end ()) - V.begin ();
        const double maximum = V[imax];
        const double maximum_xd = BP3::down_dip (fault.vertex (imax)[0], fault.vertex (imax)[1]);
        double minimum_free = std::numeric_limits<double>::infinity (), maximum_free = 0.;
        unsigned int free_count = 0, lower_count = 0;
        for (unsigned int j = 0; j < n; ++j)
          if (!prescribed[0][j])
            {
              if (BP3Benchmark::final_active[0][j])
                ++lower_count;
              else
                {
                  ++free_count;
                  minimum_free = std::min (minimum_free, V[j]);
                  maximum_free = std::max (maximum_free, V[j]);
                }
            }
        const bool started_before = event.started;
        const unsigned int previous_below = event.below;
        const bool previous_complete = event.complete;
        event.observe (this->get_time (), maximum, maximum_xd);
        const char *audit_from = std::getenv ("ASPECT_BP3_LENGTH_FULL_AUDIT_FROM");
        const bool full_audit = audit_states || (audit_from && this->get_timestep_number () >= std::stoul (audit_from));
        if (full_audit)
          write_audit_state ();
        if (this->get_pcout ().is_active ())
          {
            write_stations (V, qfield, normal_field, state);
            {
              std::ofstream summary (this->get_output_directory () + "first_event.csv");
              summary
                  << std::setprecision (17)
                  << "started,complete,onset,peak_V,peak_xd,peak_time,down_crossing,termination,below_count\n"
                  << event.started << ',' << event.complete << ',' << event.onset << ',' << event.peak << ','
                  << event.peak_xd << ',' << event.peak_time << ',' << event.down_crossing << ','
                  << event.termination << ',' << event.below << '\n';
            }
          }
        // Audit the split aging-law update independently in extended precision.
        // No Maxwell response is evaluated here: the particle array is already
        // committed, whereas weak traction above belongs to the accepted solve.
        double theta_error = 0.;
        std::string theta_failure;
        std::ofstream state_audit;
        if (std::getenv ("ASPECT_BP5_SHORT_TEST") && this->get_pcout ().is_active ())
          {
            state_audit.open (this->get_output_directory () + "state_work_"
                              + std::to_string (this->get_timestep_number ()) + ".csv");
            state_audit << std::setprecision (17)
                        << "node,xd,time,dt,V,Theta_in,Theta_out,slip,weak_q,weak_sigma,weak_friction,weak_damping,weak_residual\n";
          }
        for (unsigned int j = 0; j < n; ++j)
          {
            const double actual = fault.get_properties (j)[state];
            const double xd = BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1]);
            const double old_theta = this->get_timestep_number () > 0 ? previous_theta[j]
#ifdef ASPECT_BP5_WEAK_INITIALIZATION
              : BP3Benchmark::weak_initial_state.at(j);
#elif defined(ASPECT_BP5_STEADY_INITIALIZATION)
              : (BP3Benchmark::restored_history ? previous_theta.at(j)
                                               : BP3Benchmark::prepared_initial_state.at(j));
#else
              : BP3::configured_initial_state (xd,
                  Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
                    this->get_material_model ()).get_fault_friction ());
#endif
            const long double expected
                = this->get_timestep_number () > 0
                      ? BP3::aging_state_reference (
#ifdef ASPECT_BP3_DISTURBANCE_TEST
                            BP3Disturbance::aging_rate(j, V[j]),
#else
                            V[j],
#endif
                            old_theta, this->get_timestep (), Dc)
                      : old_theta;
            if (state_audit)
              state_audit << j << ',' << xd << ',' << this->get_time () << ',' << this->get_timestep ()
                          << ',' << V[j] << ',' << old_theta << ',' << actual << ',' << slip[j]
                          << ',' << weak.shear_traction[0][j] << ',' << weak.normal_traction[0][j]
                          << ',' << weak.friction_traction[0][j] << ',' << weak.damping_traction[0][j]
                          << ',' << weak.values[0][j] << '\n';
            const double relative_error = std::abs (actual / static_cast<double> (expected) - 1.);
            if (!std::isfinite (relative_error) || relative_error > theta_error)
              {
                theta_error = std::isfinite (relative_error) ? relative_error
                                                             : std::numeric_limits<double>::infinity ();
                if (!(theta_error < 1e-12))
                  {
                    std::ostringstream message;
                    message << std::setprecision (std::numeric_limits<long double>::max_digits10)
                            << "BP3 Theta audit: step=" << this->get_timestep_number ()
                            << ", fault=0, node=" << j << ", xd=" << xd << ", accepted V=" << V[j]
                            << ", old Theta=" << old_theta << ", committed Theta=" << actual
                            << ", reference=" << expected
                            << ", absolute error=" << std::abs (actual - expected)
                            << ", relative error=" << relative_error;
                    theta_failure = message.str ();
                  }
              }
          }
        // Report one worst node collectively before any rank throws. The
        // observer must not lose the offending values in an early MPI abort.
        const double maximum_theta_error = Utilities::MPI::max (theta_error, this->get_mpi_communicator ());
        if (!(maximum_theta_error < 1e-12))
          {
            const unsigned int rank = Utilities::MPI::this_mpi_process (this->get_mpi_communicator ());
            const unsigned int reporting_rank = Utilities::MPI::min (
                theta_error == maximum_theta_error ? rank : numbers::invalid_unsigned_int,
                this->get_mpi_communicator ());
            theta_failure
                = Utilities::MPI::broadcast (this->get_mpi_communicator (), theta_failure, reporting_rank);
            this->get_pcout () << theta_failure << std::endl;
            AssertThrow (false,
                         ExcMessage ("BP3 retained/updated Theta does not follow the split history cycle.\n"
                                     + theta_failure));
          }
        previous_theta.resize (n);
        for (unsigned int j = 0; j < n; ++j)
          previous_theta[j] = fault.get_properties (j)[state];
        last_step = this->get_timestep_number ();
        const auto &particles = this->get_phase_field_handler ().get_associated_particle_manager ();
        const auto stress_position
            = particles.get_property_manager ().get_data_info ().get_position_by_field_name (
                "maxwell stress");
        double maximum_stress = 0.;
        for (const auto &particle : particles.get_particle_handler ())
          for (unsigned int c = 0; c < 3; ++c)
            maximum_stress
                = std::max (maximum_stress, std::abs (particle.get_properties ()[stress_position + c]));
        maximum_stress = Utilities::MPI::max (maximum_stress, this->get_mpi_communicator ());
        if (this->get_timestep_number () == 0)
          AssertThrow (maximum_stress == 0.,
                       ExcMessage ("BP3 initial Maxwell perturbation history was changed."));
        {
          for (unsigned int j = 0; j < n; ++j)
            AssertThrow (fault.get_properties (j)[C] == 0. && Cfield[j] == 0.,
                         ExcMessage ("Mature BP3 published cohesive resistance."));
          // H remains initialized profile data. Stable IDs let the offline
          // audit verify retention despite particle movement and exchange.
          const auto H = particles.get_property_manager ().get_data_info ().get_position_by_field_name (
              "crack_driving_force");
          if (full_audit)
            {
              std::ofstream out (
                  this->get_output_directory () + "mature_history_"
                  + std::to_string (this->get_timestep_number ()) + "_rank"
                  + std::to_string (Utilities::MPI::this_mpi_process (this->get_mpi_communicator ()))
                  + ".csv");
              out << std::setprecision (17) << "id,H_inert,tau_xx,tau_yy,tau_xy\n";
              for (const auto &p : particles.get_particle_handler ())
                out << p.get_id () << ',' << p.get_properties ()[H] << ','
                    << p.get_properties ()[stress_position] << ',' << p.get_properties ()[stress_position + 1]
                    << ',' << p.get_properties ()[stress_position + 2] << '\n';
            }
        }
        // The summary is published only after the accepted history checks.
        const double surface_rms = this->get_reconstructed_fault_surface_system ().surface_residual_rms (
            weak, BP3Benchmark::final_active);
        if (this->get_pcout ().is_active ())
          {
            const auto path = this->get_output_directory () + "accepted_steps.csv";
            const bool header = !std::filesystem::exists (path);
            std::ofstream out (path, std::ios::app);
            if (header)
              out << "step,time,dt,max_V,xd_at_max_V,min_sigma_n,max_sigma_n,min_free_V,max_free_V,free,"
                     "lower_active,newton_updates,krylov_iterations,min_alpha,normalized_nonlinear_residual,"
                     "surface_RMS_Pa,Theta_relative_error,max_committed_stress_Pa,max_step_slip_over_Dc,"
                     "fresh_linear_checks_passed\n";
            out << std::setprecision (17) << this->get_timestep_number () << ',' << this->get_time () << ','
                << this->get_timestep () << ',' << maximum << ',' << maximum_xd << ','
                << weak.minimum_normal_traction << ',' << weak.maximum_normal_traction << ',' << minimum_free
                << ',' << maximum_free << ',' << free_count << ',' << lower_count << ','
                << BP3Benchmark::newton_updates << ',' << BP3Benchmark::krylov_iterations << ','
                << BP3Benchmark::minimum_alpha << ',' << BP3Benchmark::accepted_nonlinear_residual << ','
                << surface_rms << ',' << theta_error << ',' << maximum_stress << ','
                << (this->get_timestep_number () ? maximum * this->get_timestep () / Dc : 0.) << ",1\n";
            AssertThrow (out, ExcMessage ("Cannot append BP3 accepted-step diagnostics."));
          }
        last_accepted_time = this->get_time ();
        // One append-only table records every accepted vertex state, independently
        // of visualization throttling. The artificial initialization interval adds no slip.
        if (this->get_pcout ().is_active ())
          {
            const auto path = this->get_output_directory () + "cumulative_slip.csv";
            const bool header = !std::filesystem::exists (path);
            std::ofstream out (path, std::ios::app);
            if (header)
              out << "step,time_s,fault,node,s_m,xd_m,slip_m\n";
            out << std::setprecision (17);
            double s = 0.;
            for (unsigned int j = 0; j < n; ++j)
              {
                if (j > 0)
                  s += fault.vertex (j).distance (fault.vertex (j - 1));
                out << last_step << ',' << last_accepted_time << ",0," << j << ',' << s << ','
                    << BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1]) << ',' << slip[j] << '\n';
              }
            out.close ();
            AssertThrow (out, ExcMessage ("Cannot append BP3 cumulative slip history."));
          }
        {
          const double elapsed
              = std::chrono::duration<double> (std::chrono::steady_clock::now () - wall_start).count ();
          BP3Benchmark::long_run_stop = this->get_time () >= this->get_parameters ().end_time
                                        || this->get_timestep_number () >= last_requested_step
                                        || elapsed >= wall_seconds || (stop_after_event && event.complete);
          const bool milestone_now = (!started_before && event.started)
                                     || (event.below == 1 && previous_below == 0)
                                     || (!previous_complete && event.complete);
          heavy_pending = heavy.due (this->get_time (), slip, BP3Benchmark::long_run_stop);
          const double increment
              = Utilities::MPI::max (heavy.increment (slip), this->get_mpi_communicator ());
          heavy_pending |= increment >= heavy.slip_interval;
          auto &mutable_manager = this->get_reconstructed_fault_manager ();
          const auto slip_position = mutable_manager
                                         .get_property_information ()[mutable_manager.get_property_index (
                                             "cumulative_signed_slip_m")]
                                         .position;
          for (unsigned int j = 0; j < n; ++j)
            mutable_manager.get_fault (0).get_properties (j)[slip_position] = slip[j];
          if (heavy_pending
              || profiles.due (this->get_time (), slip, milestone_now || BP3Benchmark::long_run_stop))
            write_long_profile (V, qfield, normal_field, state);
          if (BP3Benchmark::long_run_stop)
            this->get_pcout () << "BP3 LONG RUN GRACEFUL STOP at accepted step " << last_step
                               << ", time=" << last_accepted_time << " s." << std::endl;
        }
        return { "BP3 accepted state", std::to_string (this->get_timestep_number ()) };
      }

      void
      finish_native_output ()
      {
        if (!heavy_pending)
          return;
        if (this->get_pcout ().is_active ())
          {
            const auto path = this->get_output_directory () + "heavy_outputs.csv";
            const bool header = !std::filesystem::exists (path);
            std::ofstream out (path, std::ios::app);
            if (header)
              out << "step,time_s,max_slip_change_m\n";
            out << std::setprecision (17) << last_step << ',' << last_accepted_time << ','
                << heavy.increment (slip) << '\n';
            AssertThrow (out, ExcMessage ("Cannot record successful coordinated native output."));
          }
        heavy.written (last_accepted_time, slip);
        heavy_pending = false;
      }

    private:
      void
      write_long_profile (const std::vector<double> &V, const std::vector<double> &q,
                          const std::vector<double> &normal, unsigned int state)
      {
        if (this->get_pcout ().is_active ())
          {
            std::filesystem::create_directories (this->get_output_directory () + "profiles");
            const std::string file = "profiles/fault_" + std::to_string (last_step) + ".csv";
            std::ofstream out (this->get_output_directory () + file);
            out << "fault,node,step,time_s,xd_m,x_m,y_m,slip_m,V_m_per_s,Theta_s,q_weak_Pa,sigma_n_weak_Pa\n"
                << std::setprecision (17);
            const auto &fault = this->get_reconstructed_fault_manager ().get_fault (0);
            for (unsigned int j = 0; j < V.size (); ++j)
              out << 0 << ',' << j << ',' << last_step << ',' << last_accepted_time << ','
                  << BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1]) << ',' << fault.vertex (j)[0]
                  << ',' << fault.vertex (j)[1] << ',' << slip[j] << ',' << V[j] << ','
                  << fault.get_properties (j)[state] << ',' << q[j] << ',' << normal[j] << '\n';
            out.close ();
            AssertThrow (out, ExcMessage ("Cannot write BP3 slip profile."));
            const auto index = this->get_output_directory () + "profiles.csv";
            const bool header = !std::filesystem::exists (index);
            std::ofstream list (index, std::ios::app);
            if (header)
              list << "step,time_s,file,max_slip_change_m\n";
            list << std::setprecision (17) << last_step << ',' << last_accepted_time << ',' << file << ','
                 << profiles.increment (slip) << '\n';
            AssertThrow (list, ExcMessage ("Cannot append BP3 profile index."));
          }
        profiles.written (last_accepted_time, slip);
      }
      void
      write_stations (const std::vector<double> &V, const std::vector<double> &shear,
                      const std::vector<double> &normal, const unsigned int state) const
      {
        const auto &fault = this->get_reconstructed_fault_manager ().get_faults ()[0];
        const double stations[]
            = { 0., 2500., 5000., 7500., 10000., 12500., 15000., 17500., 20000., 25000., 30000., 35000. };
        const auto path = this->get_output_directory () + "stations.csv";
        const bool header = !std::filesystem::exists (path);
        std::ofstream out (path, std::ios::app);
        if (header)
          out << "step,time,dt,xd,V,Theta,slip,tau_total,sigma_n_total\n";
        out << std::setprecision (17);
        for (const double xd : stations)
          {
            unsigned int segment = 0;
            double xi = 0.;
            bool found = false;
            for (; segment < fault.n_cells (); ++segment)
              {
                const auto a = fault.vertex (segment), b = fault.vertex (segment + 1);
                const double s0 = BP3::down_dip (a[0], a[1]), s1 = BP3::down_dip (b[0], b[1]);
                if (xd >= std::min (s0, s1) - 1e-7 && xd <= std::max (s0, s1) + 1e-7)
                  {
                    xi = std::clamp ((xd - s0) / (s1 - s0), 0., 1.);
                    found = true;
                    break;
                  }
              }
            AssertThrow (found, ExcMessage ("Official BP3 station is outside the represented fault."));
            const auto interpolate
                = [&] (const std::vector<double> &v) { return (1. - xi) * v[segment] + xi * v[segment + 1]; };
            const double theta = (1. - xi) * fault.get_properties (segment)[state]
                                 + xi * fault.get_properties (segment + 1)[state];
            out << this->get_timestep_number () << ',' << this->get_time () << ',' << this->get_timestep ()
                << ',' << xd << ',' << interpolate (V) << ',' << theta << ',' << interpolate (slip) << ','
                << interpolate (shear) << ',' << interpolate (normal) << '\n';
          }
        AssertThrow (out, ExcMessage ("Cannot append BP3 station histories."));
      }

      void
      write_audit_state () const
      {
        const auto tag = std::to_string (this->get_timestep_number ()) + "_rank"
                         + std::to_string (Utilities::MPI::this_mpi_process (this->get_mpi_communicator ()))
                         + ".csv";
        std::ofstream bulk (this->get_output_directory () + "audit_bulk_" + tag);
        const auto &owned = this->get_dof_handler ().locally_owned_dofs ();
        std::vector<unsigned int> components (owned.n_elements ());
        std::vector<types::global_dof_index> indices (this->get_fe ().n_dofs_per_cell ());
        for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
          if (!cell->is_artificial ())
            {
              cell->get_dof_indices (indices);
              for (unsigned int j = 0; j < indices.size (); ++j)
                if (owned.is_element (indices[j]))
                  components[owned.index_within_set (indices[j])]
                      = this->get_fe ().system_to_component_index (j).first;
            }
        bulk << std::setprecision (17) << "dof,component,value\n";
        for (const auto i : owned)
          bulk << i << ',' << components[owned.index_within_set (i)] << ',' << this->get_solution ()[i]
               << '\n';
        const auto &particles
            = this->get_phase_field_handler ().get_associated_particle_manager ().get_particle_handler ();
        std::ofstream out (this->get_output_directory () + "audit_particles_" + tag);
        out << std::setprecision (17) << "id,x,y,properties\n";
        for (const auto &particle : particles)
          {
            out << particle.get_id () << ',' << particle.get_location ()[0] << ','
                << particle.get_location ()[1];
            for (const auto v : particle.get_properties ())
              out << ',' << v;
            out << '\n';
          }
      }

      bool audit_states = false;
      BP3::FirstEvent event;
      std::vector<double> slip;
      BP3::OutputSchedule heavy, profiles;
      bool heavy_pending = false, stop_after_event = false;
      double wall_seconds = 86400., last_accepted_time = -1.;
      unsigned int last_requested_step = 2147483647;
      std::chrono::steady_clock::time_point wall_start;
      std::vector<double> previous_theta;
      unsigned int last_step = numbers::invalid_unsigned_int;
    };
    ASPECT_REGISTER_POSTPROCESSOR (
        BP3Output, "reconstructed fault BP3",
        "Accepted BP3 fault profiles and actual weak traction, without reevaluating committed stress.")

    template <int dim> class BP3OutputComplete : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      std::list<std::string>
      required_other_postprocessors () const override
      {
        return { "reconstructed fault BP3", "visualization", "particles", "reconstructed faults" };
      }
      std::pair<std::string, std::string>
      execute (TableHandler &) override
      {
        auto &output = const_cast<BP3Output<dim> &> (
            this->get_postprocess_manager ().template get_matching_active_plugin<BP3Output<dim>> ());
        output.finish_native_output ();
        return {};
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR (
        BP3OutputComplete, "BP3 output complete",
        "Commit the shared slip-output reference only after all native writers succeed.")
  }

  namespace TerminationCriteria
  {
    template <int dim> class BP3LongRunComplete : public Interface<dim>
    {
    public:
      bool
      execute () override
      {
        return BP3Benchmark::long_run_stop;
      }
    };
    ASPECT_REGISTER_TERMINATION_CRITERION (
        BP3LongRunComplete, "BP3 long run complete",
        "Accepted-state physical, wall, step or first-event stop; never changes a timestep.")
  }
}
