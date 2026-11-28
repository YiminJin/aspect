/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/mesh_refinement/growing_fault.h>

namespace aspect::TimeStepping
{
  template <int dim>
  class GrowingFaultRefinement : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      double execute() override
      {
        return std::numeric_limits<double>::max();
      }

      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        const auto &marker = this->get_mesh_refinement_manager()
                             .template get_matching_active_plugin<MeshRefinement::GrowingFault<dim>>();
        const auto reaction = marker.assess_endpoint();
        return {reaction, reaction == Reaction::refine_and_repeat_step ?
                this->get_timestep() : std::numeric_limits<double>::max()
               };
      }
  };

  ASPECT_REGISTER_TIME_STEPPING_MODEL(GrowingFaultRefinement, "growing fault refinement",
                                      "Check the growing fault refinement target before accepting a midpoint step. "
                                      "A missed extension refines the parent and repeats at the same duration "
                                      "unless another timestep model also requests a cutback.")
}
