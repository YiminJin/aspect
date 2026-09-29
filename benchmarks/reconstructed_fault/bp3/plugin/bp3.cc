#include "runtime.h"
#include "bp3_model.h"

#include <aspect/initial_composition/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator_signals.h>

#include <fstream>
#include <iomanip>


namespace BP3 { double weakening_length = 15000.; }
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
namespace BP3 { double local_state_disturbance = 0.; }
#endif

// BP3 initialization of fixed prestress data, not a runtime cohesive law.
namespace aspect
{
  namespace BP3Benchmark
  {

    template <int dim>
    void
    initialize_mature_prestress (const SimulatorAccess<dim> &sim, MaterialModel::PhaseFieldFault<dim> &model)
    {
      auto &manager = sim.get_reconstructed_fault_manager ();
      auto &fault = manager.get_fault (0);
      const auto background = manager.get_property_index ("background tractions");
      const auto correction = manager.get_property_index ("BP3 fixed shear correction");
      const auto p = manager.get_property_information ()[background].position;
      const auto c = manager.get_property_information ()[correction].position;
      AssertThrow(mature_prestress_file.empty(),
                  ExcMessage("Restored BP3 uses uniform nominal background, not captured prestress."));
      for (unsigned int v=0;v<fault.n_vertices();++v)
        {
          auto data=fault.get_properties(v);
          data[p]=BP3::tau0; data[p+1]=BP3::sigma0;
          data[c]=data[c+1]=0.; data[c+2]=1.;
        }
      model.set_reconstructed_fault_background_traction_property(background,correction);

    }

  }
}

namespace aspect
{
  namespace BP3Benchmark
  {
    bool converged = false;
    bool long_run_stop = false;
    bool restored_history = false;
    bool detailed_diagnostics = false;
    std::string bottom_normalization_completion_file;
    std::string mature_prestress_file;
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
      constexpr unsigned int n_sides=3;
      double error[n_sides] = {};
      unsigned int count[n_sides] = {};
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
                    if (p[0] != left && p[0] != right && !(n_sides==3 && p[1]==box.get_origin()[1]))
                      continue;
                    const unsigned int side = p[0] == left ? 0 : (p[0]==right ? 1 : 2);
                    if (side==2 && BP3Restore::bottom_velocity_constraint=="fault parallel")
                      {
                        if (d!=0) continue;
                        unsigned int k=0;
                        for (;k<dofs.size();++k)
                          if (fe.system_to_component_index(k).first==sim.introspection().component_indices.velocities[1]
                              && fe.get_unit_support_points()[k]==fe.get_unit_support_points()[j]) break;
                        AssertThrow(k<dofs.size(),ExcMessage("Unpaired bottom velocity support point."));
                        const auto t=BP3Restore::bottom_tangent;
                        const auto prescribed=BP3Restore::loading(sim,p);
                        error[side]=std::max(error[side],std::abs(t[0]*(lifted[dofs[j]]-prescribed[0])
                                                                +t[1]*(lifted[dofs[k]]-prescribed[1])));
                        ++count[side];
                        continue;
                      }
                    const double expected=BP3Restore::loading(sim,p)[d];
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
      for (unsigned int side = 0; side < n_sides; ++side)
        {
          const auto n = Utilities::MPI::sum (count[side], sim.get_mpi_communicator ());
          const double maximum = Utilities::MPI::max (error[side], sim.get_mpi_communicator ());
          AssertThrow (n > 0 && maximum < 1e-22,
                       ExcMessage ("BP3 realized lateral velocity constraints are incorrect."));
          const double sign = side == 0 ? 1 : -1;
          if (out)
            out << (side == 0 ? "left" : (side==1 ? "right" :
                    (BP3Restore::bottom_velocity_constraint=="full" ? "bottom_profile":"bottom_parallel"))) << ',' << (side==2?0.:sign * .5 * BP3::Vp * BP3::cosine) << ','
                << (side==2?0.:sign * .5 * BP3::Vp * BP3::horizontal_sign*BP3::sine) << ',' << .5 * BP3::Vp << ',' << maximum << ',' << n
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
      manager.set_shear_sense(0,-1);
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
      // Mesh verification above is mandatory; its per-cell dump is diagnostic.
      if (detailed_diagnostics)
      {
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
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(bool(mesh)),sim.get_mpi_communicator()),
                  ExcMessage("Cannot write initial mesh diagnostics."));
      }
      // The simulator owns a mutable material object. This initialization-only
      // callback precedes the particle-to-FE transfer; no assembly loop casts.
      model.prepare_reconstructed_fault_mechanical_solve ();

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
    }
  }

  template <int dim>
  void
  connect_bp3 (SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect (&BP3Benchmark::prescribe_phase<dim>);
    signals.post_constraints_creation.connect (&BP3Restore::constrain_bottom<dim>);
    // Register preparation before the monitor attaches its incoming-state
    // observer during postprocessor initialization. Preparation restores
    // selectors, but never reinitializes histories after a restart.
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
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
        const double xd = BP3::down_dip(p[0],p[1]);
#else
        const double xd = (BP3::box_size - p[1]) / BP3::sine;
#endif
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
    template <int dim> class BP3Velocity : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      Tensor<1, dim>
      boundary_velocity (const types::boundary_id, const Point<dim> &p) const override
      {
        return BP3Restore::loading(*this,p);
      }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL (
        BP3Velocity, "reconstructed fault BP3",
        "BP3 far-field rigid translation in the documented rotated chart.")
  }

}
