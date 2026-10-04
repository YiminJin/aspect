// Bounded section-5 observations. No solution or particle history is modified.
#include "../bp3/plugin/bp3_model.h"
#include "../bp3/plugin/runtime.h"
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
namespace Postprocess
{
template <int dim>
class BP3LocalProbes : public Interface<dim>, public SimulatorAccess<dim>
{
  // Replicated accepted diagnostics let surviving particles retain their path through
  // MPI migration. Births start at zero; deleted IDs are discarded.
  // Checkpointed solely to continue these observations, never to restore
  // physical history.
  std::map<types::particle_index, std::vector<double>> previous;
  std::vector<double> incoming_theta;
  double previous_dt = 0;

public:
  void
  initialize () override
  {
    this->get_signals ().post_simulator_initialization.connect (
        [this] (const SimulatorAccess<dim> &)
          {
            this->get_reconstructed_fault_surface_system ()
                .set_normal_traction_diagnostic (
                    [] (const Point<dim> &p)
                      {
                        const double s = BP3::down_dip (p[0], p[1]);
                        return std::abs (s - 100 * std::round (s / 100)) < 2.;
                      });
          });
    this->get_signals ().post_advection_solver.connect (
        [this] (const SimulatorAccess<dim> &, bool temperature, unsigned int,
                const SolverControl &)
          {
            if (!temperature)
              return;
            const auto &m = this->get_reconstructed_fault_manager ();
            const auto &f = m.get_fault (0);
            const auto pos
                = m.get_property_information ()[m.get_property_index (
                                                    "phase field fault state")]
                      .position;
            incoming_theta.resize (f.n_vertices ());
            for (unsigned int i = 0; i < f.n_vertices (); ++i)
              incoming_theta[i] = f.get_properties (i)[pos];
          });
  }
  void
  save (std::map<std::string, std::string> &status) const override
  {
    std::ostringstream s;
    boost::archive::text_oarchive a (s);
    a << previous << previous_dt;
    status["BP3 local observations"] = s.str ();
  }
  void
  load (const std::map<std::string, std::string> &status) override
  {
    std::istringstream s (status.at ("BP3 local observations"));
    boost::archive::text_iarchive a (s);
    a >> previous >> previous_dt;
  }
  std::pair<std::string, std::string>
  execute (TableHandler &) override
  {
    const auto step = this->get_timestep_number ();
    const double time = this->get_time (), dt = this->get_timestep ();
    const auto comm = this->get_mpi_communicator ();
    const auto rank = Utilities::MPI::this_mpi_process (comm);
    const std::string prefix = this->get_output_directory ();
    const std::string tag
        = std::to_string (step) + "_rank" + std::to_string (rank) + ".csv";
    auto &pm
        = this->get_phase_field_handler ().get_associated_particle_manager ();
    auto &ph = pm.get_particle_handler ();
    const auto &info = pm.get_property_manager ().get_data_info ();
    const auto stress = info.get_position_by_field_name ("maxwell stress");
    const auto H = info.get_position_by_field_name ("crack_driving_force");
    std::map<types::particle_index, std::vector<double>> current;
    unsigned int pop_min = numbers::invalid_unsigned_int, pop_max = 0,
                 crossed = 0;
    double cp = 0, ap = 0, stress_increment = 0;
    const bool full = step == 10 || step == 12 || step == 18 || step == 20;
    std::ofstream particles;
    if (full)
      {
        particles.open (prefix + "local_particles_" + tag);
        particles << std::setprecision (17) << "id,x,y,H,xx,yy,xy,h,Ap\n";
      }
    for (const auto &cell :
         this->get_triangulation ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        {
          const auto n = ph.n_particles_in_cell (cell);
          pop_min = std::min (pop_min, n);
          pop_max = std::max (pop_max, n);
          for (const auto &p : ph.particles_in_cell (cell))
            {
              const auto x = p.get_location ();
              const auto v = p.get_properties ();
              const double h = cell->diameter () / std::sqrt (double (dim));
              double path = 0;
              const auto old = previous.find (p.get_id ());
              // A native birth can reuse a retired ID. Start its path at zero
              // even when that ID is present in the previous accepted map.
              if (old != previous.end () && !BP3Benchmark::attempt_births.count (p.get_id ()))
                {
                  // Incoming-cell edge is the denominator, before this
                  // accepted move.
                  const auto &o = old->second;
                  const double c
                      = std::hypot (x[0] - o[0], x[1] - o[1]) / o[2];
                  path = o[3] + c;
                  cp = std::max (cp, c);
                  crossed += h != o[2];
                  for (unsigned int k = 0; k < 3; ++k)
                    stress_increment = std::max (
                        stress_increment, std::abs (v[stress + k] - o[4 + k]));
                }
              ap = std::max (ap, path);
              current[p.get_id ()]
                  = { x[0],          x[1],         h, path, v[stress],
                      v[stress + 1], v[stress + 2] };
              if (full)
                particles << p.get_id () << ',' << x[0] << ',' << x[1] << ','
                          << v[H] << ',' << v[stress] << ',' << v[stress + 1]
                          << ',' << v[stress + 2] << ',' << h << ',' << path
                          << '\n';
            }
        }
    previous.clear ();
    for (const auto &part : Utilities::MPI::all_gather (comm, current))
      previous.insert (part.begin (), part.end ());
    cp = Utilities::MPI::max (cp, comm);
    ap = Utilities::MPI::max (ap, comm);
    crossed = Utilities::MPI::sum (crossed, comm);
    stress_increment = Utilities::MPI::max (stress_increment, comm);
    pop_min = Utilities::MPI::min (pop_min, comm);
    pop_max = Utilities::MPI::max (pop_max, comm);
    std::ofstream probes (prefix + "velocity_" + tag);
    probes << std::setprecision (17)
           << "x,y,ux,uy,dt_ux,dt_uy,p,mapped_xx,mapped_yy,mapped_xy,h\n";
    std::ofstream mesh;
    if (step == 0)
      {
        mesh.open (prefix + "mesh_rank" + std::to_string (rank) + ".csv");
        mesh << "x,y,h\n" << std::setprecision (17);
      }
    const auto &intro = this->introspection ();
    double umax = 0;
    for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        {
          const auto bounds = cell->bounding_box ().get_boundary_points ();
          const auto lo = bounds.first, hi = bounds.second;
          const double h = hi[0] - lo[0];
          if (step == 0)
            mesh << cell->center ()[0] << ',' << cell->center ()[1] << ',' << h
                 << '\n';
          std::vector<Point<dim>> units;
          for (double y : { 250., 500., 750. })
            if (y >= lo[1] && y < hi[1])
              for (int i = std::max (0, int (std::ceil ((lo[0] + 900) / 2)));
                   i <= 900 && -900 + 2 * i < hi[0]; ++i)
                {
                  Point<dim> p;
                  p[0] = (-900 + 2 * i - lo[0]) / h;
                  p[1] = (y - lo[1]) / h;
                  units.push_back (p);
                }
          if (units.empty ())
            continue;
          FEValues<dim> fe (this->get_mapping (), this->get_fe (),
                            Quadrature<dim> (units),
                            update_values | update_quadrature_points);
          fe.reinit (cell);
          std::vector<Vector<double>> values (
              units.size (), Vector<double> (this->get_fe ().n_components ()));
          fe.get_function_values (this->get_solution (), values);
          for (unsigned int i = 0; i < units.size (); ++i)
            {
              const auto &p = fe.quadrature_point (i);
              const auto &v = values[i];
              const double ux = v[intro.component_indices.velocities[0]],
                           uy = v[intro.component_indices.velocities[1]];
              umax = std::max (umax, std::hypot (ux, uy));
              probes << p[0] << ',' << p[1] << ',' << ux << ',' << uy << ','
                     << dt * ux << ',' << dt * uy << ','
                     << v[intro.component_indices.pressure];
              for (unsigned int k = 0; k < 3; ++k)
                probes << ','
                       << v[intro.component_indices.compositional_fields[k]];
              probes << ',' << h << '\n';
            }
        }
    umax = Utilities::MPI::max (umax, comm);
    if (full)
      {
        std::ofstream bulk (prefix + "local_bulk_" + tag);
        bulk << std::setprecision (17) << "dof,value\n";
        for (const auto i : this->get_dof_handler ().locally_owned_dofs ())
          bulk << i << ',' << this->get_solution ()[i] << '\n';
      }
    const auto &m = this->get_reconstructed_fault_manager ();
    const auto &f = m.get_fault (0);
    const auto &V = m.get_timestep_committed_slip_rate (0);
    const auto &w = this->get_reconstructed_fault_surface_system ()
                        .get_linearization_residual ();
    const auto solve = [&] (const auto &rhs)
      {
        return ReconstructedFaultUtilities::solve_tridiagonal_system (
            w.mass_diagonal[0], w.mass_off_diagonal[0], rhs);
      };
    const auto raw = solve (w.raw_normal_traction[0]),
               normal = solve (w.normal_traction[0]);
    const auto state
        = m.get_property_information ()[m.get_property_index (
                                            "phase field fault state")]
              .position;
    const auto &law = Plugins::get_plugin_as_type<
                          const MaterialModel::PhaseFieldFault<dim>> (
                          this->get_material_model ())
                          .get_fault_friction ();
    double dlog = 0;
    if (rank == 0)
      {
        std::ofstream out (prefix + "fault_" + std::to_string (step) + ".csv");
        out << std::setprecision (17)
            << "s,V,Theta_in,Theta,Omega,raw_normal_Q1,friction_normal_Q1,mu,"
               "mu_V,mu_Theta\n";
        for (unsigned int i = 0; i < V.size (); ++i)
          {
            const double theta = f.get_properties (i)[state];
            dlog = std::max (dlog,
                             std::abs (std::log (theta / incoming_theta[i])));
            out << BP3::down_dip (f.vertex (i)[0], f.vertex (i)[1]) << ','
                << V[i] << ',' << incoming_theta[i] << ',' << theta << ','
                << V[i] * theta / law.get_characteristic_slip_distance ()
                << ',' << raw[i] << ',' << normal[i] << ','
                << law.friction_coefficient ({ 0., 1. }, V[i],
                                             incoming_theta[i])
                << ','
                << law.friction_coefficient_derivative_wrt_slip_rate (
                       { 0., 1. }, V[i], incoming_theta[i])
                << ','
                << law.friction_coefficient_derivative_wrt_state (
                       { 0., 1. }, V[i], incoming_theta[i])
                << '\n';
          }
        const auto path = prefix + "local_summary.csv";
        const bool header = !std::ifstream (path).good ();
        std::ofstream summary (path, std::ios::app);
        summary << std::setprecision (17);
        if (header)
          summary << "step,time,dt,dt_ratio,pop_min,pop_max,Cp_max,Ap_max,"
                     "level_crossings,max_delta_stress,max_probe_u,max_delta_"
                     "log_Theta\n";
        summary << step << ',' << time << ',' << dt << ','
                << (previous_dt > 0 ? dt / previous_dt : 0) << ',' << pop_min
                << ',' << pop_max << ',' << cp << ',' << ap << ',' << crossed
                << ',' << stress_increment << ',' << umax << ',' << dlog
                << '\n';
      }
    const auto &diag = this->get_reconstructed_fault_surface_system ()
                           .get_normal_traction_diagnostic ();
    std::ofstream actual (prefix + "friction_samples_" + tag);
    actual << std::setprecision (17)
           << "x,y,s,V,raw_normal,actual_normal,mu,eta_ve,beta,incoming_xx,"
              "incoming_yy,incoming_xy\n";
    for (const auto &q : diag.samples)
      actual << q.position[0] << ',' << q.position[1] << ','
             << BP3::down_dip (q.surface_position[0], q.surface_position[1])
             << ',' << m.interpolate_slip_rate (q.fault, q.segment, q.xi)
             << ',' << q.total << ',' << q.friction_normal << ','
             << q.friction_coefficient << ',' << q.eta_ve << ',' << q.beta
             << ',' << q.incoming_stress[0][0] << ','
             << q.incoming_stress[1][1] << ',' << q.incoming_stress[0][1]
             << '\n';
    previous_dt = dt;
    return { "BP3 local observations:", "accepted" };
  }
};
ASPECT_REGISTER_POSTPROCESSOR (
    BP3LocalProbes, "BP3 local probes",
    "Bounded read-only velocity, history and particle motion observations.")
}
}
