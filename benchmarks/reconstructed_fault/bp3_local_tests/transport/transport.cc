// Curved-history transport from particle_replenishment, with exact-position
// shadow components routed through the native particle -> continuous Q2 path.
#include "../../bp3/plugin/bp3_model.h"
#include "../../bp3/plugin/particle_initialization.h"
#include "../../bp3/plugin/runtime.h"
#include <aspect/initial_composition/interface.h>
#include <aspect/particle/interpolator/linear_least_squares.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
namespace aspect
{
namespace LocalTransport
{
template <int dim>
double
exact (const Point<dim> &p, double t)
{
  return 1e8
         * (1
            + .1 * std::sin (numbers::PI * (p[0] - t) / 64)
                  * std::cos (numbers::PI * (p[1] - .6 * t) / 64));
}
}
namespace InitialComposition
{
template <int dim>
class CurvedHistory : public Interface<dim>, public SimulatorAccess<dim>
{
public:
  double
  initial_composition (const Point<dim> &p, unsigned int c) const override
  {
    if (c == 6)
      return 8.;
    if (c == 7)
      return 1.;
    return (c % 3 == 0   ? 1.
            : c % 3 == 1 ? -1.
                         : .5)
           * LocalTransport::exact (p, this->get_time ());
  }
};
ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL (
    CurvedHistory, "local curved history",
    "Exact transported tensor and a separate spatially refreshed shadow copy.")
}
namespace Particle
{
namespace Interpolator
{
template <int dim> class TransportLLS : public LinearLeastSquares<dim>
{
public:
  void
  parse_parameters (ParameterHandler &prm) override
  {
    const auto &info
        = this->get_particle_manager (this->get_particle_manager_index ())
              .get_property_manager ()
              .get_data_info ();
    std::string mask;
    const auto count = info.n_components ()
                       - info.get_components_by_field_name (
                           "internal: integrator properties");
    for (unsigned int k = 0; k < count; ++k)
      {
        bool selected
            = k == info.get_position_by_field_name ("crack_driving_force");
        for (const std::string name :
             { "maxwell stress", "shadow_xx", "shadow_yy", "shadow_xy" })
          selected
              = selected
                || (k >= info.get_position_by_field_name (name)
                    && k < info.get_position_by_field_name (name)
                               + info.get_components_by_field_name (name));
        if (k)
          mask += ",";
        mask += selected ? "true" : "false";
      }
    prm.enter_subsection ("Interpolator");
    prm.enter_subsection ("Linear least squares");
    prm.set ("Use linear least squares limiter", mask);
    prm.set ("Use boundary extrapolation", "false");
    prm.leave_subsection ();
    prm.leave_subsection ();
    LinearLeastSquares<dim>::parse_parameters (prm);
  }
};
ASPECT_REGISTER_PARTICLE_INTERPOLATOR (
    TransportLLS, "local transport LLS",
    "Native LLS with the same H/stress limiter also applied to exact shadow "
    "tensor components.")
}
}
namespace Postprocess
{
template <int dim>
class TransportProbe : public Interface<dim>, public SimulatorAccess<dim>
{
  std::map<types::particle_index, std::vector<double>> previous;

public:
  static void
  declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Postprocess");
    prm.enter_subsection ("BP3");
    prm.declare_entry ("Weakening region length", "15000",
                       Patterns::Double (0));
    prm.declare_entry ("Allow truncated transition", "true",
                       Patterns::Bool ());
    prm.leave_subsection ();
    prm.leave_subsection ();
  }
  std::pair<std::string, std::string>
  execute (TableHandler &) override
  {
    const auto step = this->get_timestep_number ();
    const auto time = this->get_time ();
    if (step == 0)
      BP3Benchmark::verify_paired_mesh (*this);
    auto &pm = this->get_particle_manager (0);
    const auto &ph = pm.get_particle_handler ();
    const auto &info = pm.get_property_manager ().get_data_info ();
    const auto stress = info.get_position_by_field_name ("maxwell stress"),
               shadow = info.get_position_by_field_name ("shadow_xx"),
               H = info.get_position_by_field_name ("crack_driving_force");
    const auto rank
        = Utilities::MPI::this_mpi_process (this->get_mpi_communicator ());
    const auto tag
        = std::to_string (step) + "_rank" + std::to_string (rank) + ".csv";
    std::ofstream out (this->get_output_directory () + "transport_" + tag);
    out << std::setprecision (17)
        << "x,y,h,interface,boundary,n,Cp_max,Ap_max,crossings,born,reused_id_"
           "births,newborn_H_max_error,survivor_H_max_change,newborn_stress_"
           "max_error,retained_stress_max_error,tensor_constraint_max,stored_"
           "particle_max_error,shadow_particle_max_error,H_min,LLS_stored_rms,"
           "LLS_shadow_rms,Q2_stored_rms,Q2_shadow_rms,Q2_shadow_max\n";
    std::ofstream reused (this->get_output_directory () + "reused_ids_" + tag);
    reused << std::setprecision (17) << "id,old_x,old_y,new_x,new_y\n";
    const auto profiles
        = this->get_phase_field_handler ().get_phase_field_profiles (
            BP3::geometry ().peak_phase);
    std::map<types::particle_index, std::vector<double>> now;
    QGauss<dim> quadrature (3);
    FEValues<dim> fe (this->get_mapping (), this->get_fe (), quadrature,
                      update_values | update_quadrature_points);
    for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
      if (cell->is_locally_owned ())
        {
          const double h = cell->diameter () / std::sqrt (double (dim));
          bool interface = false;
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (!cell->face (f)->at_boundary ())
              interface = interface || cell->neighbor_is_coarser (f)
                          || cell->neighbor (f)->has_children ();
          double cp = 0, ap = 0, stored_error = 0, shadow_error = 0,
                 hmin = std::numeric_limits<double>::infinity ();
          unsigned int crossed = 0, born = 0, reused_count = 0;
          double newborn_H_error = 0, survivor_H_change = 0,
                 newborn_stress_error = 0, retained_stress_error = 0,
                 tensor_error = 0;
          for (const auto &p : ph.particles_in_cell (cell))
            {
              const auto x = p.get_location ();
              double path = 0;
              const auto old = previous.find (p.get_id ());
              // Native IDs can be reused after the previous maximum exits.
              // Match a trajectory only when it also agrees with this exact
              // constant flow; an old ID at a new birth position starts a new
              // path, never a teleport.
              const bool retained
                  = old != previous.end ()
                    && std::hypot (
                           x[0] - old->second[0] - this->get_timestep (),
                           x[1] - old->second[1] - .6 * this->get_timestep ())
                           < 256 * std::numeric_limits<double>::epsilon ()
                                 * std::max (
                                     { 1., std::abs (x[0]), std::abs (x[1]) });
              if (retained)
                {
                  const auto &o = old->second;
                  const double c
                      = std::hypot (x[0] - o[0], x[1] - o[1]) / o[2];
                  path = o[3] + c;
                  cp = std::max (cp, c);
                  crossed += h != o[2];
                }
              else if (step)
                {
                  ++born;
                  newborn_H_error = std::max (
                      newborn_H_error,
                      std::abs (p.get_properties ()[H]
                                - BP3Benchmark::stationary_particle_H (
                                    *this, x, *profiles[0])));
                  if (old != previous.end ())
                    {
                      ++reused_count;
                      reused << p.get_id () << ',' << old->second[0] << ','
                             << old->second[1] << ',' << x[0] << ',' << x[1]
                             << '\n';
                    }
                }
              ap = std::max (ap, path);
              now[p.get_id ()]
                  = { x[0], x[1], h, path, p.get_properties ()[H] };
              const double q = LocalTransport::exact (x, time);
              const auto v = p.get_properties ();
              if (retained)
                {
                  survivor_H_change = std::max (
                      survivor_H_change, std::abs (v[H] - old->second[4]));
                  retained_stress_error = std::max (retained_stress_error,
                                                    std::abs (v[stress] - q));
                }
              else if (step)
                newborn_stress_error = std::max (newborn_stress_error,
                                                 std::abs (v[stress] - q));
              for (const auto slot : { stress, shadow })
                tensor_error = std::max (
                    { tensor_error, std::abs (v[slot] + v[slot + 1]),
                      std::abs (v[slot + 2] - .5 * v[slot]) });
              stored_error = std::max (stored_error, std::abs (v[stress] - q));
              shadow_error = std::max (shadow_error, std::abs (v[shadow] - q));
              hmin = std::min (hmin, v[H]);
            }
          fe.reinit (cell);
          const auto values = pm.get_interpolator ().properties_at_points (
              ph, fe.get_quadrature_points (),
              ComponentMask (info.n_components (), true), cell);
          std::vector<Vector<double>> field (
              quadrature.size (),
              Vector<double> (this->get_fe ().n_components ()));
          fe.get_function_values (this->get_solution (), field);
          double errors[4] = {}, maximum = 0;
          for (unsigned int i = 0; i < quadrature.size (); ++i)
            {
              const double ref
                  = LocalTransport::exact (fe.quadrature_point (i), time);
              const double a[] = {
                values[i][stress], values[i][shadow],
                field[i][this->introspection ()
                             .component_indices.compositional_fields[0]],
                field[i][this->introspection ()
                             .component_indices.compositional_fields[3]]
              };
              for (unsigned int j = 0; j < 4; ++j)
                errors[j] += quadrature.weight (i) * std::pow (a[j] - ref, 2);
              maximum = std::max (maximum, std::abs (a[3] - ref));
            }
          out << cell->center ()[0] << ',' << cell->center ()[1] << ',' << h
              << ',' << interface << ',' << cell->at_boundary () << ','
              << ph.n_particles_in_cell (cell) << ',' << cp << ',' << ap << ','
              << crossed << ',' << born << ',' << reused_count << ','
              << newborn_H_error << ',' << survivor_H_change << ','
              << newborn_stress_error << ',' << retained_stress_error << ','
              << tensor_error << ',' << stored_error << ',' << shadow_error
              << ',' << hmin;
          for (const double e : errors)
            out << ',' << std::sqrt (e);
          out << ',' << maximum << '\n';
        }
    previous.clear ();
    for (const auto &part :
         Utilities::MPI::all_gather (this->get_mpi_communicator (), now))
      previous.insert (part.begin (), part.end ());
    AssertThrow (
        previous.size () == ph.n_global_particles (),
        ExcMessage (
            "Duplicate active particle IDs in transport observation."));
    return { "Curved transport:", "native stored/shadow Q2 sampled" };
  }
};
ASPECT_REGISTER_POSTPROCESSOR (
    TransportProbe, "local transport probes",
    "Observe stored and exact shadow tensor reconstruction without a coupled "
    "history update.")
}
}
