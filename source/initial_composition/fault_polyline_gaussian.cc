/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/initial_composition/fault_polyline_gaussian.h>
#include <aspect/reconstructed_fault/fault_vertex_table.h>
#include <aspect/utilities.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect::InitialComposition
{
  template <int dim>
  double FaultPolylineGaussian<dim>::initial_composition(const Point<dim> &position,
                                                         const unsigned int compositional_index) const
  {
    if (compositional_index != phase_index)
      return 0.;

    if constexpr (dim == 2)
      {
        double distance = std::numeric_limits<double>::max();
        for (unsigned int i=1; i<vertices.size(); ++i)
          distance = std::min(distance, Utilities::distance_to_line({{vertices[i-1], vertices[i]}}, position));

        return background + (peak-background) * std::exp(-Utilities::fixed_power<2>(distance/width));
      }

    AssertThrow(false, ExcNotImplemented());
  }


  template <int dim>
  void FaultPolylineGaussian<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Initial composition model");
    prm.enter_subsection("Fault polyline Gaussian");
    {
      prm.declare_entry("Background phase", "0.03", Patterns::Double(0.,1.),
                        "Dimensionless phase far from the prescribed polyline.");
      prm.declare_entry("Peak phase", "0.35", Patterns::Double(0.,1.),
                        "Dimensionless phase on the prescribed polyline.");
      prm.declare_entry("Gaussian width", "0.15", Patterns::Double(0.),
                        "Euclidean distance in metres at which the phase excess above the background "
                        "falls to 1/e of its peak. This width describes only the initial phase.");
    }
    prm.leave_subsection();
    prm.leave_subsection();
  }


  template <int dim>
  void FaultPolylineGaussian<dim>::parse_parameters(ParameterHandler &prm)
  {
    AssertThrow(dim == 2, ExcMessage("The fault polyline Gaussian initial composition requires a 2-D run."));
    const auto &phase_name = this->get_parameters().surface_energy_model.composition;
    phase_index = this->introspection().compositional_index_for_name(phase_name);

    prm.enter_subsection("Initial composition model");
    prm.enter_subsection("Fault polyline Gaussian");
    {
      background = prm.get_double("Background phase");
      peak = prm.get_double("Peak phase");
      width = prm.get_double("Gaussian width");
      AssertThrow(peak >= background && width > 0. && std::isfinite(width),
                  ExcMessage("The Gaussian peak must be at least the background phase, and the width "
                             "must be finite and positive in metres."));
    }
    prm.leave_subsection();
    prm.leave_subsection();

    const std::string filename = prm.get({"Material model", "Phase field fault"}, "Fault vertex file");
    AssertThrow(!filename.empty(),
                ExcMessage("The fault polyline Gaussian initial composition requires the ordered table "
                           "in Material model / Phase field fault / Fault vertex file."));
    vertices = FaultVertexTable<dim>::read(filename, this->get_mpi_communicator()).positions;
  }

  ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(FaultPolylineGaussian,
                                            "fault polyline Gaussian",
                                            "Initialize the configured phase composition from the shortest Cartesian "
                                            "distance to the finite segments in Material model / Phase field fault / "
                                            "Fault vertex file. The closest point may be a vertex. Return zero for "
                                            "all other fields. This model is available in two dimensions.")
}
