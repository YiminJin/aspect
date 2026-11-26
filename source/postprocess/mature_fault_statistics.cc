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
    /** Accepted label profiles and arclength averages on the current mature material trace. */
    template <int dim>
    class MatureFaultStatistics : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &statistics) override
        {
          AssertThrow(dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(&this->get_material_model()),
                      ExcMessage("Mature fault statistics require a mature material trace."));
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto &labels = model.labels();
          const auto &ages = model.label_ages();
          const auto &last = model.phase_profile().geometry.segment(labels.size()-2);
          const double length = last.station+last.length;
          double age = 0., slip = 0., rate = 0.;
          for (unsigned int j=1; j<labels.size(); ++j)
            {
              const auto &a = labels[j-1];
              const auto &b = labels[j];
              const double weight = .5*a.position.distance(b.position)/length;
              if (!ages.empty()) age += weight*(ages[j-1]+ages[j]);
              slip += weight*(a.slip+b.slip);
              rate += weight*(a.rate+b.rate);
            }
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            {
              const std::string path = this->get_output_directory()+"fault-labels-"+Utilities::int_to_string(this->get_timestep_number(),5)+".txt";
              std::ofstream out(path);
              out << "# label x(m) y(m) " << (ages.empty() ? "" : "age(s) ") << "slip(m) rate(m/s)\n" << std::setprecision(17);
              for (unsigned int j=0; j<labels.size(); ++j)
                {
                  out << j << ' ' << labels[j].position << ' ';
                  if (!ages.empty()) out << ages[j] << ' ';
                  out << labels[j].slip << ' ' << labels[j].rate << '\n';
                }
              out.close();
              AssertThrow(out,ExcIO());
            }
          std::map<std::string,double> columns =
          {
            {"Cumulative fault slip (m)",slip},{"Fault slip rate (m/s)",rate}
          };
          if (!ages.empty()) columns.emplace("Fault age (s)",age);
          for (const auto &column : columns)
            {
              statistics.add_value(column.first,column.second);
              statistics.set_precision(column.first,16);
              statistics.set_scientific(column.first,true);
            }
          return {"Fault slip rate", Utilities::to_string(rate) + " m/s"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(MatureFaultStatistics,"mature fault statistics",
                                  "Write accepted material vertex positions, optional age, accumulated slip distance, "
                                  "positive slip speed and arclength averages.")
    // Retain the parameter-file spelling used by the stationary controls.
    namespace
    {
      aspect::internal::Plugins::RegisterHelper<Interface<2>,MatureFaultStatistics<2>>
      legacy_statistics_2(&Manager<2>::register_postprocessor,"fixed fault statistics","Compatibility name for mature fault statistics.");
      aspect::internal::Plugins::RegisterHelper<Interface<3>,MatureFaultStatistics<3>>
      legacy_statistics_3(&Manager<3>::register_postprocessor,"fixed fault statistics","Compatibility name for mature fault statistics.");
    }
  }
}
