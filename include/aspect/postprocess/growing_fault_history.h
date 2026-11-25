/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_postprocess_growing_fault_history_h
#define _aspect_postprocess_growing_fault_history_h
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect::Postprocess
{
  /** Export accepted material-point stress (tension-positive deviator, Pa),
   * strain and strip slip on their native fixed quadrature after acceptance. */
  template <int dim>
  class GrowingFaultHistory : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void initialize() override;
      std::pair<std::string,std::string> execute(TableHandler &) override;
  };
}
#endif
