/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/fixed_fault.h>
#include <aspect/postprocess/interface.h>

namespace aspect
{
  namespace Postprocess
  {
    /** Endpoint values on the homogeneous mature fault. */
    template <int dim>
    class FixedFaultStatistics : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &statistics) override
        {
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::FixedFault<dim>>(this->get_material_model());
          const auto &section = model.accepted_section();
          const std::map<std::string,double> columns = {{"Fault age (s)",section.age},
            {"Cumulative fault slip (m)",section.slip},
            {"Fault slip rate (m/s)",section.rate}
          };
          for (const auto &column : columns)
            {
              statistics.add_value(column.first,column.second);
              statistics.set_precision(column.first,16);
              statistics.set_scientific(column.first,true);
            }
          return {"Fault slip rate", Utilities::to_string(section.rate) + " m/s"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(FixedFaultStatistics,"fixed fault statistics",
                                  "Report the accepted age, slip and slip rate on the homogeneous fixed fault.")
  }
}
