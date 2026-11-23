/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/simulator_signals.h>
#include <aspect/simulator_access.h>

#include <fstream>

namespace
{
  template <int dim>
  void read_model(const aspect::SimulatorAccess<dim> &simulator,
                  aspect::Assemblers::Manager<dim> &)
  {
    const auto &model = simulator.get_parameters().surface_energy_model;
    model.validate();
    const auto field = simulator.introspection().compositional_index_for_name(model.composition);
    if (dealii::Utilities::MPI::this_mpi_process(simulator.get_mpi_communicator()) == 0)
      {
        std::ofstream out(simulator.get_output_directory() + "configured-model");
        out << model.composition << ' ' << field << ' '
            << model.fracture_energy << ' ' << model.length_scale << '\n'
            << model.geometry.value(.25) << ' ' << model.geometry.first_derivative(.25)
            << ' ' << model.geometry.second_derivative(.25) << '\n';
      }
  }

  template <int dim>
  void connect(aspect::SimulatorSignals<dim> &signals)
  {
    signals.set_assemblers.connect(&read_model<dim>);
  }
}

ASPECT_REGISTER_SIGNALS_CONNECTOR(connect<2>, connect<3>)
