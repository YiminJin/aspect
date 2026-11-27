/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/fault_slip_output.h>
#include <aspect/material_model/phase_field_fault.h>

namespace aspect::Postprocess
{
  template <int dim>
  bool FaultSlipOutput<dim>::requests_graphical_output() const
  {
    const auto &fault = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
    const auto &labels = fault.labels();
    if (slip_at_frame.empty()) return true;
    AssertThrow(labels.size() == slip_at_frame.size(), ExcMessage("The graphical fault reference has different material IDs."));
    for (unsigned int i=0; i<labels.size(); ++i)
      if (std::abs(labels[i].slip-slip_at_frame[i]) >= slip_interval)
        return true;
    return false;
  }

  template <int dim>
  void FaultSlipOutput<dim>::record_graphical_output()
  {
    const auto &fault = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
    slip_at_frame.clear();
    for (const auto &label : fault.labels())
      slip_at_frame.push_back(label.slip);
  }

  template <int dim>
  void FaultSlipOutput<dim>::save(std::map<std::string,std::string> &status) const
  {
    std::ostringstream stream;
    {
      aspect::oarchive archive(stream);
      archive << slip_at_frame;
    }
    status["Mature slip at graphical frame v1"] = stream.str();
  }

  template <int dim>
  void FaultSlipOutput<dim>::load(const std::map<std::string,std::string> &status)
  {
    // A checkpoint predating this observer has no graphical reference. Start
    // its new series at the first accepted state following restoration.
    const auto record = status.find("Mature slip at graphical frame v1");
    if (record != status.end())
      {
        std::istringstream stream(record->second);
        aspect::iarchive archive(stream);
        archive >> slip_at_frame;
      }
  }

  template <int dim>
  void FaultSlipOutput<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Postprocess");
    prm.enter_subsection("Visualization");
    prm.declare_entry("Fault slip interval", "0.1", Patterns::Double(0.),
                      "Cumulative slip difference (m) at any mature material ID that requests a native "
                      "frame when the fault slip output postprocessor is selected.");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void FaultSlipOutput<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Postprocess");
    prm.enter_subsection("Visualization");
    slip_interval = prm.get_double("Fault slip interval");
    prm.leave_subsection();
    prm.leave_subsection();
    AssertThrow(std::isfinite(slip_interval) && slip_interval > 0., ExcMessage("Fault slip interval must be finite and positive."));
    const auto &fault = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
    AssertThrow(!fault.uses_growing_strips(), ExcMessage("Fault slip output requires persistent mature trace vertices."));
  }

  ASPECT_REGISTER_POSTPROCESSOR(FaultSlipOutput, "fault slip output",
                                "Schedule native visualization at accepted mature fault slip intervals.")
}
