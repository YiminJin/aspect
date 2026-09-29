#include "../plugin/bp3_model.h"
#include "../plugin/runtime.h"
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <chrono>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <set>

namespace aspect
{
namespace Postprocess
{
template <int dim>
class RotatedBottomAudit : public Interface<dim>, public SimulatorAccess<dim>
{
  std::vector<double> initial_raw, previous_raw, old_theta;
  std::chrono::steady_clock::time_point start
      = std::chrono::steady_clock::now ();

public:
  void
  initialize () override
  {
    this->get_signals ().post_simulator_initialization.connect (
        [this] (const SimulatorAccess<dim> &)
          {
            this->get_reconstructed_fault_surface_system ()
                .set_normal_traction_diagnostic ([] (const Point<dim> &)
                                                   { return true; });
          });
  }
  std::list<std::string>
  required_other_postprocessors () const override
  {
    return { "reconstructed fault BP3", "BP3 restored monitor" };
  }
  std::pair<std::string, std::string>
  execute (TableHandler &) override
  {
    const auto mpi = this->get_mpi_communicator ();
    const auto &fe = this->get_fe ();
    const auto &intro = this->introspection ();
    const auto &manager = this->get_reconstructed_fault_manager ();
    const auto &fault = manager.get_fault (0);
    const auto &surface = this->get_reconstructed_fault_surface_system ();
    const auto &w = surface.get_linearization_residual ();
    const auto &diag = surface.get_normal_traction_diagnostic ();
    auto project = [&] (const std::vector<double> &load)
      {
        return ReconstructedFaultUtilities::solve_tridiagonal_system (
            w.mass_diagonal[0], w.mass_off_diagonal[0], load);
      };
    auto raw = project (w.raw_normal_traction[0]),
         used = project (w.normal_traction[0]),
         shear = project (w.shear_traction[0]);
    auto pressure = project (diag.pressure_load[0]),
         deviatoric = project (diag.deviatoric_load[0]);
    const auto state
        = manager
              .get_property_information ()[manager.get_property_index (
                  "phase field fault state")]
              .position;
    const auto &V = manager.get_timestep_committed_slip_rate (0);
    if (initial_raw.empty ())
      {
        initial_raw = previous_raw = raw;
        old_theta.resize (V.size ());
        for (unsigned j = 0; j < V.size (); ++j)
          old_theta[j] = fault.get_properties (j)[state];
      }
    double norm[3] = {}, rate_norm[3] = {}, weight[3] = {}, maximum[3] = {},
           location[3] = {}, vmin[3] = { 1e100, 1e100, 1e100 }, vmax[3] = {},
           theta_change = 0.;
    for (unsigned j = 0; j < V.size (); ++j)
      {
        const double xd
            = BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1]);
        const double length = BP3::box_size / BP3::sine;
        const unsigned region = xd > length - 200 ? 0 : (xd < 200 ? 2 : 1);
        const double mass
            = w.mass_diagonal[0][j] + (j ? w.mass_off_diagonal[0][j - 1] : 0.)
              + (j + 1 < V.size () ? w.mass_off_diagonal[0][j] : 0.);
        const double delta = raw[j] - initial_raw[j],
                     rate
                     = this->get_timestep_number ()
                           ? (raw[j] - previous_raw[j]) / this->get_timestep ()
                           : 0.;
        norm[region] += mass * delta * delta;
        rate_norm[region] += mass * rate * rate;
        weight[region] += mass;
        if (std::abs (delta) > maximum[region])
          {
            maximum[region] = std::abs (delta);
            location[region] = xd;
          }
        vmin[region] = std::min (vmin[region], V[j] / BP3::Vp);
        vmax[region] = std::max (vmax[region], V[j] / BP3::Vp);
        theta_change = std::max (
            theta_change, std::abs (std::log (fault.get_properties (j)[state]
                                              / old_theta[j])));
        old_theta[j] = fault.get_properties (j)[state];
      }
    LinearAlgebra::BlockVector lift (intro.index_sets.system_partitioning,
                                     mpi),
        variation (lift);
    lift = 0.;
    variation = 0.;
    this->get_current_constraints ().distribute (lift);
    for (auto i :
         intro.index_sets.system_partitioning[intro.block_indices.velocities])
      variation[i] = BP3::Vp;
    AffineConstraints<double> homogeneous;
    homogeneous.copy_from (this->get_current_constraints ());
    for (const auto &line : homogeneous.get_lines ())
      homogeneous.set_inhomogeneity (line.index, 0.);
    homogeneous.distribute (variation);
    LinearAlgebra::BlockVector lift_g (
        intro.index_sets.system_partitioning,
        intro.index_sets.system_relevant_partitioning, mpi),
        var_g (lift_g);
    lift_g = lift;
    var_g = variation;
    FEFaceValues<dim> face (this->get_mapping (), fe, QGauss<dim - 1> (4),
                            update_values | update_quadrature_points
                                | update_normal_vectors | update_JxW_values);
    std::vector<Tensor<1, dim>> u (face.n_quadrature_points), l (u), v (u);
    double flux[4] = {}, strong = 0., continuous = 0., hom_error = 0.,
           free_variation = 0., normal_speed = 0., weak_max = 0.;
    std::set<types::global_dof_index> visited;
    std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
    // Per-rank compact boundary profiles keep MPI ownership explicit.
    std::ofstream local_boundary (
        this->get_output_directory () + "bottom_"
        + std::to_string (this->get_timestep_number ()) + "_rank"
        + std::to_string (Utilities::MPI::this_mpi_process (mpi)) + ".csv");
    local_boundary << "x,y,ut,un,ut_reference\n" << std::setprecision (17);
    for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        for (unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face (f)->at_boundary ())
            {
              const auto id = cell->face (f)->boundary_id ();
              face.reinit (cell, f);
              face[intro.extractors.velocities].get_function_values (
                  this->get_solution (), u);
              face[intro.extractors.velocities].get_function_values (lift_g,
                                                                     l);
              face[intro.extractors.velocities].get_function_values (var_g, v);
              for (unsigned q = 0; q < u.size (); ++q)
                {
                  flux[id] += u[q] * face.normal_vector (q) * face.JxW (q);
                  if (id != 2)
                    continue;
                  auto t = [] (const auto &a)
                    { return .5 * a[0] - BP3::sine * a[1]; };
                  auto n = [] (const auto &a)
                    { return BP3::sine * a[0] + .5 * a[1]; };
                  const auto p = face.quadrature_point (q);
                  const double target = t (BP3Restore::loading (*this, p));
                  strong = std::max (strong, std::abs (t (u[q]) - t (l[q])));
                  continuous
                      = std::max (continuous, std::abs (t (u[q]) - target));
                  hom_error = std::max (hom_error, std::abs (t (v[q])));
                  free_variation
                      = std::max (free_variation, std::abs (n (v[q])));
                  normal_speed = std::max (normal_speed, std::abs (n (u[q])));
                  local_boundary << p[0] << ',' << p[1] << ',' << t (u[q])
                                 << ',' << n (u[q]) << ',' << target << '\n';
                }
              if (id == 2)
                {
                  cell->get_dof_indices (dofs);
                  for (unsigned j = 0; j < dofs.size (); ++j)
                    if (fe.system_to_component_index (j).first
                            == intro.component_indices.velocities[0]
                        && fe.has_support_on_face (j, f)
                        && this->get_dof_handler ()
                               .locally_owned_dofs ()
                               .is_element (dofs[j])
                        && !this->get_current_constraints ().is_constrained (
                            dofs[j])
                        && visited.insert (dofs[j]).second)
                      weak_max = std::max (
                          weak_max,
                          std::abs (BP3::sine
                                    * this->get_system_rhs ()[dofs[j]]));
                }
            }
    for (auto &f : flux)
      f = Utilities::MPI::sum (f, mpi);
    strong = Utilities::MPI::max (strong, mpi);
    continuous = Utilities::MPI::max (continuous, mpi);
    hom_error = Utilities::MPI::max (hom_error, mpi);
    free_variation = Utilities::MPI::max (free_variation, mpi);
    normal_speed = Utilities::MPI::max (normal_speed, mpi);
    weak_max = Utilities::MPI::max (weak_max, mpi);
    // Keep the box corners separate from fault endpoint diagnostics. Pressure
    // is sampled from the accepted FE solution; deviatoric stress is the
    // accepted particle history (zero at step zero by the BP3 convention).
    const std::array<Point<dim>, 4> corners
        = { { Point<dim> (-1000, 0), Point<dim> (1000, 0),
              Point<dim> (-1000, 1000), Point<dim> (1000, 1000) } };
    double corner_p[4] = {}, corner_stress[4] = {};
    FEValues<dim> volume (this->get_mapping (), fe, QGauss<dim> (3),
                          update_values | update_quadrature_points);
    std::vector<double> pressures (volume.n_quadrature_points);
    for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        {
          volume.reinit (cell);
          volume[intro.extractors.pressure].get_function_values (
              this->get_solution (), pressures);
          for (unsigned q = 0; q < pressures.size (); ++q)
            for (unsigned k = 0; k < 4; ++k)
              if (volume.quadrature_point (q).distance (corners[k]) < 200.)
                corner_p[k] = std::max (corner_p[k], std::abs (pressures[q]));
        }
    const auto &pm
        = this->get_phase_field_handler ().get_associated_particle_manager ();
    const auto sp = pm.get_property_manager ()
                        .get_data_info ()
                        .get_position_by_field_name ("maxwell stress");
    for (const auto &particle : pm.get_particle_handler ())
      for (unsigned k = 0; k < 4; ++k)
        if (particle.get_location ().distance (corners[k]) < 200.)
          {
            SymmetricTensor<2, dim> stress;
            for (unsigned c = 0; c < 3; ++c)
              stress[SymmetricTensor<2, dim>::unrolled_to_component_indices (
                  c)] = particle.get_properties ()[sp + c];
            corner_stress[k] = std::max (corner_stress[k], stress.norm ());
          }
    for (unsigned k = 0; k < 4; ++k)
      {
        corner_p[k] = Utilities::MPI::max (corner_p[k], mpi);
        corner_stress[k] = Utilities::MPI::max (corner_stress[k], mpi);
      }
    AssertThrow (
        strong < 1e-20 && hom_error < 1e-20,
        ExcMessage ("Rotated physical or homogeneous trace audit failed."));
    AssertThrow (
        BP3Restore::bottom_velocity_constraint == "full"
            || free_variation > 1e-10,
        ExcMessage ("Bottom normal variation was inadvertently eliminated."));
    if (this->get_pcout ().is_active ())
      {
        const auto step = this->get_timestep_number ();
        std::ofstream corner_out (this->get_output_directory ()
                                      + "corner_metrics.csv",
                                  std::ios::app);
        if (step == 0)
          corner_out << "step,time,corner,max_abs_pressure,max_committed_"
                        "deviatoric_norm\n";
        for (unsigned k = 0; k < 4; ++k)
          corner_out << std::setprecision (17) << step << ','
                     << this->get_time () << ',' << k << ',' << corner_p[k]
                     << ',' << corner_stress[k] << '\n';
        std::ofstream profile (this->get_output_directory () + "local_fault_"
                               + std::to_string (step) + ".csv");
        profile << std::setprecision (17)
                << "xd,y,V,Theta,shear,raw_normal,used_normal,pressure,"
                   "deviatoric,normal_change,normal_rate\n";
        for (unsigned j = 0; j < V.size (); ++j)
          profile << BP3::down_dip (fault.vertex (j)[0], fault.vertex (j)[1])
                  << ',' << fault.vertex (j)[1] << ',' << V[j] << ','
                  << fault.get_properties (j)[state] << ',' << shear[j] << ','
                  << raw[j] << ',' << used[j] << ',' << pressure[j] << ','
                  << deviatoric[j] << ',' << raw[j] - initial_raw[j] << ','
                  << (step ? (raw[j] - previous_raw[j]) / this->get_timestep ()
                           : 0.)
                  << '\n';
        std::ofstream summary (this->get_output_directory ()
                                   + "local_metrics.csv",
                               std::ios::app);
        if (step == 0)
          summary << "step,time,dt,cells,dofs,tangent_error,continuous_error,"
                     "homogeneous_error,free_normal_probe,max_bottom_normal,"
                     "weak_bottom_residual,flux_left,flux_right,flux_bottom,"
                     "flux_top,net_flux,deep_rms,deep_rate_rms,deep_max,deep_"
                     "location,deep_Vmin,deep_Vmax,interior_rms,interior_rate_"
                     "rms,interior_max,interior_location,interior_Vmin,"
                     "interior_Vmax,top_rms,top_rate_rms,top_max,top_location,"
                     "top_Vmin,top_Vmax,max_log_theta_change,wall_seconds\n";
        summary << std::setprecision (17) << step << ',' << this->get_time ()
                << ',' << this->get_timestep () << ','
                << this->get_triangulation ().n_global_active_cells () << ','
                << this->get_dof_handler ().n_dofs () << ',' << strong << ','
                << continuous << ',' << hom_error << ',' << free_variation
                << ',' << normal_speed << ',' << weak_max;
        double net = 0.;
        for (auto f : flux)
          {
            summary << ',' << f;
            net += f;
          }
        summary << ',' << net;
        for (unsigned k = 0; k < 3; ++k)
          summary << ',' << std::sqrt (norm[k] / weight[k]) << ','
                  << std::sqrt (rate_norm[k] / weight[k]) << ',' << maximum[k]
                  << ',' << location[k] << ',' << vmin[k] << ',' << vmax[k];
        summary << ',' << theta_change << ','
                << std::chrono::duration<double> (
                       std::chrono::steady_clock::now () - start)
                       .count ()
                << '\n';
      }
    previous_raw = raw;
    return {
      "Local bottom audit",
      "physical/homogeneous constraints, flux and coupled endpoint response"
    };
  }
};
ASPECT_REGISTER_POSTPROCESSOR (RotatedBottomAudit, "rotated bottom audit",
                               "Small coupled A/B boundary experiment.")
}
}
