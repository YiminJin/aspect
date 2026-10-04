#include "../../bp3/plugin/runtime.h"
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>
namespace aspect
{
namespace Postprocess
{
template <int dim>
class LocalRestartClock : public Interface<dim>, public SimulatorAccess<dim>
{
  unsigned int stop_step = 0;
  void
  snapshot (const std::string &label) const
  {
    const auto rank
        = Utilities::MPI::this_mpi_process (this->get_mpi_communicator ());
    const auto prefix = this->get_output_directory () + label + "_rank"
                        + std::to_string (rank);
    std::ofstream bulk (prefix + "_bulk.csv");
    bulk << std::setprecision (17) << "dof,value\n";
    for (const auto i : this->get_dof_handler ().locally_owned_dofs ())
      bulk << i << ',' << this->get_solution ()[i] << '\n';
    const auto &pm
        = this->get_phase_field_handler ().get_associated_particle_manager ();
    std::ofstream particles (prefix + "_particles.csv");
    particles << std::setprecision (17);
    for (const auto &p : pm.get_particle_handler ())
      {
        particles << p.get_id () << ',' << p.get_location ()[0] << ','
                  << p.get_location ()[1];
        for (double v : p.get_properties ())
          particles << ',' << v;
        particles << '\n';
      }
    std::ofstream rng (prefix + "_rng.txt");
    rng << pm.get_random_number_state ();
  }

public:
  void
  parse_parameters (ParameterHandler &prm) override
  {
    prm.enter_subsection ("Postprocess");
    prm.enter_subsection ("BP3");
    stop_step = prm.get_integer ("Last accepted step");
    prm.leave_subsection ();
    prm.leave_subsection ();
  }
  void
  initialize () override
  {
    // Reuse the qualified bounded restart hook: only the pending interval is
    // reduced; the simulator preserves accepted time, old dt and all
    // histories.
    this->get_signals ().post_resume_time_step.connect (
        [this] (const auto &, double &dt)
          {
            snapshot ("common_restart");
            dt = std::min (dt, this->get_parameters ().maximum_time_step);
          });
  }
  std::pair<std::string, std::string>
  execute (TableHandler &) override
  {
    if (this->get_timestep_number () == stop_step)
      snapshot ("common_end");
    return { "Local restart clock:", "bounded" };
  }
};
ASPECT_REGISTER_POSTPROCESSOR (
    LocalRestartClock, "local restart clock",
    "Qualified pending-interval reduction and common-state snapshots for a "
    "bounded diagnostic.")
}
}
