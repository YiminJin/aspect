/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_postprocess_fault_slip_output_h
#define _aspect_postprocess_fault_slip_output_h
#include <aspect/postprocess/interface.h>
#include <aspect/postprocess/visualization_trigger.h>

namespace aspect::Postprocess
{
  /** Request a graphical frame after a material vertex has slipped far
   * enough since the last frame. Accepted mature vertices and their stable
   * material IDs are replicated, including after repartitioning. */
  template <int dim>
  class FaultSlipOutput : public Interface<dim>, public SimulatorAccess<dim>, public VisualizationTrigger
  {
    public:
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm) override;
      bool requests_graphical_output() const override;
      void record_graphical_output() override;
      std::list<std::string> required_other_postprocessors() const override
      {
        return {"visualization"};
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        return {};
      }
      void save(std::map<std::string,std::string> &status) const override;
      void load(const std::map<std::string,std::string> &status) override;
    private:
      double slip_interval;
      std::vector<double> slip_at_frame;  ///< Metres, indexed by persistent material ID.
  };
}
#endif
