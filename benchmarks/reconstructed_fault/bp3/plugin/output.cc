#include "runtime.h"
#include "bp3_model.h"
#include "first_event.h"
#include "output_schedule.h"
#include "execution_environment.h"
#include "output_files.h"

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


namespace aspect
{
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
        prm.declare_entry ("Heavy output slip interval", "0.1", Patterns::Double (0),
                           "Maximum nodal slip change in metres since the last heavy output.");
        prm.declare_entry ("Heavy output time interval", "31557600", Patterns::Double (0),
                           "Maximum physical seconds between heavy outputs; zero disables.");
        prm.declare_entry ("Profile slip interval", "0.1", Patterns::Double (0),
                           "Maximum nodal slip change in metres between lightweight profiles.");
        prm.declare_entry ("Profile time interval", "31557600", Patterns::Double (0),
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
                           "Deprecated compatibility entry; must be empty for restored BP3.");
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
        AssertThrow (BP3Benchmark::mature_prestress_file.empty (),
                     ExcMessage ("Mature prestress file is unsupported in restored BP3; leave it empty."));
        BP3Benchmark::bottom_normalization_completion_file = prm.get ("Bottom normalization completion file");
        prm.leave_subsection ();
        prm.leave_subsection ();
      }

      void
      initialize () override
      {
        const char *unexpected = BP3::unexpected_execution_switch ();
        AssertThrow (!unexpected,
                     ExcMessage (std::string ("Ordinary BP3 rejects inherited diagnostic selector: ")
                                 + (unexpected ? unexpected : "")
                                 + ". Unset it or use the dedicated diagnostic plugin."));
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
                BP3::collective_root_write(this->get_mpi_communicator(), [&]()
                  {
                    std::ofstream out (path + "/bp3_accepted_state.txt");
                    out.exceptions(std::ios::failbit|std::ios::badbit);
                    out << std::setprecision (17) << last_step << ' ' << last_accepted_time << '\n';
                    out.close();
                    AssertThrow (out, ExcMessage ("Cannot label BP3 checkpoint accepted state."));
                    // Snapshot only small time-series metadata, never copy a
                    // whole event state. A restart branch restores this prefix.
                    const auto destination = path + "/bp3_output_metadata";
                    std::filesystem::create_directories (destination);
                    std::ofstream origin(destination+"/source_output_directory.txt");
                    origin.exceptions(std::ios::failbit|std::ios::badbit);
                    origin<<std::filesystem::absolute(this->get_output_directory()).string()<<'\n';
                    origin.close();
                    for (const auto &entry :
                         std::filesystem::directory_iterator (this->get_output_directory ()))
                      if (entry.is_regular_file ())
                        {
                          const auto name = entry.path ().filename ().string ();
                          const auto extension = entry.path ().extension ().string ();
                          if (extension == ".pvd" || extension == ".visit" || extension == ".xdmf"
                              || name == "profiles.csv" || name == "heavy_outputs.csv"
                              || name == "stations.csv" || name == "accepted_steps.csv"
                              || name == "first_event.csv" || name == "restored_growth.csv" || name == "statistics")
                            std::filesystem::copy_file (entry.path (), destination + "/" + name,
                                                        std::filesystem::copy_options::overwrite_existing);
                        }
                  });
              });
      }

      void
      save (std::map<std::string, std::string> &status) const override
      {
        std::ostringstream stream;
        {
          aspect::oarchive archive (stream);
          const unsigned int version = 6;
          const bool selected = true;
          // Version 6 stores current owned H baselines, gathered before saving.
          // These retired custom-writer
          // fields have no role in native output; they are not runtime state.
          const double unused_time = -std::numeric_limits<double>::max ();
          const std::vector<std::string> unused_labels;
          archive << version << selected << slip << previous_theta << last_step << event << unused_time
                  << unused_time << unused_labels << selected << selected << BP3Benchmark::checkpoint_particle_H
                  << BP3Benchmark::work_initial_geometry << BP3Benchmark::work_initial_I << selected << heavy
                  << profiles << last_accepted_time
                  << this->get_phase_field_handler().get_associated_particle_manager()
                     .get_particle_handler().get_next_free_particle_index();
        }
        status["BP3 accepted history"] = stream.str ();
      }

      void
      load (const std::map<std::string, std::string> &status) override
      {
        const auto initialization = status.find ("BP5 initial condition");
        AssertThrow (initialization == status.end (),
                     ExcMessage ("This checkpoint requires the steady BP5 initialization plugin."));
        const auto entry = status.find ("BP3 accepted history");
        AssertThrow (entry != status.end (), ExcMessage ("Checkpoint lacks BP3 history."));
        std::istringstream stream (entry->second);
        // Version 5 stores intervals as well as references. Output cadence may
        // change on restart; preserve the loaded reference but use parsed PRM intervals.
        const auto requested_heavy=heavy, requested_profiles=profiles;
        aspect::iarchive archive (stream);
        unsigned int version;
        bool mature, frictional, work, native;
        double unused_profile_time, unused_bulk_time;
        std::vector<std::string> unused_labels;
        archive >> version;
        AssertThrow (version == 5 || version == 6,
                     ExcMessage ("Modified BP3 requires a version-5 or version-6 particle/output checkpoint."));
        archive >> mature >> slip >> previous_theta >> last_step >> event >> unused_profile_time
            >> unused_bulk_time >> unused_labels >> frictional >> work >> BP3Benchmark::work_initial_H
            >> BP3Benchmark::work_initial_geometry >> BP3Benchmark::work_initial_I >> native >> heavy
            >> profiles >> last_accepted_time;
        types::particle_index next_id = BP3Benchmark::work_initial_H.empty()
                                        ? 0 : BP3Benchmark::work_initial_H.rbegin()->first+1;
        if (version >= 6) archive >> next_id;
        BP3Benchmark::restore_particle_audit(next_id);
        heavy.slip_interval=requested_heavy.slip_interval;
        heavy.time_interval=requested_heavy.time_interval;
        profiles.slip_interval=requested_profiles.slip_interval;
        profiles.time_interval=requested_profiles.time_interval;
        AssertThrow (mature && frictional && work && native,
                     ExcMessage ("Cannot convert a different BP3 formulation on restart."));
        AssertThrow (!slip.empty () && slip.size () == previous_theta.size ()
                         && BP3Benchmark::work_initial_geometry.size () == slip.size ()
                         && BP3Benchmark::work_initial_I.size () == slip.size (),
                     ExcMessage ("Incomplete BP3 checkpoint histories."));
        BP3Benchmark::restored_history = true;
        BP3::collective_root_write(this->get_mpi_communicator(), [&]()
          { BP3::check_restart_output(this->get_output_directory(),last_step,last_accepted_time); });
      }

      std::pair<std::string, std::string>
      execute (TableHandler &) override
      {
        if (last_step==numbers::invalid_unsigned_int)
          BP3::collective_root_write(this->get_mpi_communicator(),[&]()
            {
              for (const auto *name : {"profiles.csv","accepted_steps.csv","stations.csv",
                                       "restored_growth.csv","heavy_outputs.csv"})
                AssertThrow(BP3::needs_header(this->get_output_directory()+name),
                            ExcMessage("Fresh BP3 run requires an empty output history: "+std::string(name)));
            });
        AssertThrow (BP3Benchmark::converged, ExcMessage ("BP3 requires genuine bulk/surface convergence."));
        const double Dc = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
            this->get_material_model ()).characteristic_fault_slip_distance ();
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
        BP3Benchmark::verify_accepted_work (*this, weak,
                                          audit_states || BP3Benchmark::detailed_diagnostics);

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
        const bool full_audit = audit_states;
        if (full_audit)
          write_audit_state ();
        BP3::collective_root_write(this->get_mpi_communicator(),[&]()
          {
            write_stations (V, qfield, normal_field, state);
            {
              std::ofstream summary (this->get_output_directory () + "first_event.csv");
              summary.exceptions(std::ios::failbit|std::ios::badbit);
              summary
                  << std::setprecision (17)
                  << "started,complete,onset,peak_V,peak_xd,peak_time,down_crossing,termination,below_count\n"
                  << event.started << ',' << event.complete << ',' << event.onset << ',' << event.peak << ','
                  << event.peak_xd << ',' << event.peak_time << ',' << event.down_crossing << ','
                  << event.termination << ',' << event.below << '\n';
              summary.close();
            }
          });
        // Audit the split aging-law update independently in extended precision.
        // No Maxwell response is evaluated here: the particle array is already
        // committed, whereas weak traction above belongs to the accepted solve.
        double theta_error = 0.;
        std::string theta_failure;
        for (unsigned int j = 0; j < n; ++j)
          {
            const double actual = fault.get_properties (j)[state];
            const double xd = BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1]);
            const double old_theta = this->get_timestep_number () > 0 ? previous_theta[j]
              : BP3::configured_initial_state (xd,
                  Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
                    this->get_material_model ()).get_fault_friction ());
            const long double expected
                = this->get_timestep_number () > 0
                      ? BP3::aging_state_reference (
                            V[j],
                            old_theta, this->get_timestep (), Dc)
                      : old_theta;
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
              out.close();
              AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(bool(out)),this->get_mpi_communicator()),
                          ExcMessage("Cannot write mature particle audit."));
            }
        }
        // The summary is published only after the accepted history checks.
        const double surface_rms = this->get_reconstructed_fault_surface_system ().surface_residual_rms (
            weak, BP3Benchmark::final_active);
        BP3::collective_root_write(this->get_mpi_communicator(),[&]()
          {
            const auto path = this->get_output_directory () + "accepted_steps.csv";
            const bool header = BP3::needs_header(path);
            std::ofstream out (path, std::ios::app);
            out.exceptions(std::ios::failbit|std::ios::badbit);
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
            out.close();
          });
        last_accepted_time = this->get_time ();
        // Slip is integrated/published/checkpointed every accepted step above
        // and below. Scheduled full-precision profiles are its only CSV history.
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
        BP3::collective_root_write(this->get_mpi_communicator(), [&]()
          {
            const auto path = this->get_output_directory () + "heavy_outputs.csv";
            const bool header = BP3::needs_header(path);
            std::ofstream out (path, std::ios::app);
            out.exceptions(std::ios::failbit|std::ios::badbit);
            if (header)
              out << "step,time_s,max_slip_change_m\n";
            out << std::setprecision (17) << last_step << ',' << last_accepted_time << ','
                << heavy.increment (slip) << '\n';
            out.close();
          });
        heavy.written (last_accepted_time, slip);
        heavy_pending = false;
      }

    private:
      void
      write_long_profile (const std::vector<double> &V, const std::vector<double> &q,
                          const std::vector<double> &normal, unsigned int state)
      {
        BP3::collective_root_write(this->get_mpi_communicator(), [&]()
          {
            std::filesystem::create_directories (this->get_output_directory () + "profiles");
            const std::string file = "profiles/fault_" + std::to_string (last_step) + ".csv";
            AssertThrow(!std::filesystem::exists(this->get_output_directory()+file),
                        ExcMessage("Refusing to overwrite an existing BP3 profile: "+file));
            std::ofstream out (this->get_output_directory () + file);
            out.exceptions(std::ios::failbit|std::ios::badbit);
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
            const bool header = BP3::needs_header(index);
            std::ofstream list (index, std::ios::app);
            list.exceptions(std::ios::failbit|std::ios::badbit);
            if (header)
              list << "step,time_s,file,max_slip_change_m\n";
            list << std::setprecision (17) << last_step << ',' << last_accepted_time << ',' << file << ','
                 << profiles.increment (slip) << '\n';
            list.close();
          });
        profiles.written (last_accepted_time, slip);
      }
      void
      write_stations (const std::vector<double> &V, const std::vector<double> &shear,
                      const std::vector<double> &normal, const unsigned int state) const
      {
        const auto &fault = this->get_reconstructed_fault_manager ().get_faults ()[0];
        const double stations[]
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
            = {0.,200.,500.,1000.,BP3::box_size/BP3::sine};
#else
            = { 0., 2500., 5000., 7500., 10000., 12500., 15000., 17500., 20000., 25000., 30000., 35000. };
#endif
        const auto path = this->get_output_directory () + "stations.csv";
        const bool header = BP3::needs_header(path);
        std::ofstream out (path, std::ios::app);
        out.exceptions(std::ios::failbit|std::ios::badbit);
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
        out.close();
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
        bulk.close(); out.close();
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(bool(bulk) && bool(out)),this->get_mpi_communicator()),
                    ExcMessage("Cannot write full-state rank audit."));
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
