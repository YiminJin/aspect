/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    /** Accepted label profiles and arclength averages on the stationary fault. */
    template <int dim>
    class FixedFaultStatistics : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &statistics) override
        {
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &labels = model.labels();
          const double length = labels.front().position.distance(labels.back().position);
          double age = 0., slip = 0., rate = 0.;
          for (unsigned int j=1; j<labels.size(); ++j)
            {
              const auto &a = labels[j-1];
              const auto &b = labels[j];
              const double weight = .5*a.position.distance(b.position)/length;
              age += weight*(a.age+b.age);
              slip += weight*(a.slip+b.slip);
              rate += weight*(a.rate+b.rate);
            }
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              const std::string path = this->get_output_directory()+"fault-labels-"+Utilities::int_to_string(this->get_timestep_number(),5)+".txt";
              std::ofstream out(path);
              out << "# label x(m) y(m) age(s) slip(m) rate(m/s)\n" << std::setprecision(17);
              for (unsigned int j=0; j<labels.size(); ++j)
                out << j << ' ' << labels[j].position << ' ' << labels[j].age << ' ' << labels[j].slip << ' ' << labels[j].rate << '\n';
              out.close();
              AssertThrow(out,ExcIO());
            }
          const std::map<std::string,double> columns = {{"Fault age (s)",age},
            {"Cumulative fault slip (m)",slip},{"Fault slip rate (m/s)",rate}
          };
          for (const auto &column : columns)
            {
              statistics.add_value(column.first,column.second);
              statistics.set_precision(column.first,16);
              statistics.set_scientific(column.first,true);
            }
          return {"Fault slip rate", Utilities::to_string(rate) + " m/s"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(FixedFaultStatistics,"fixed fault statistics",
                                  "Write accepted label coordinates, age, slip and rate and their arclength averages.")
  }
}
