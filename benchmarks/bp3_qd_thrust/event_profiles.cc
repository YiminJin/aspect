/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include "first_event.h"
#include <aspect/postprocess/interface.h>
#include <aspect/postprocess/visualization_trigger.h>
#include <aspect/termination_criteria/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <fstream>

namespace aspect::Postprocess
{
  template <int dim>
  class HistoricalProfiles : public Interface<dim>, public SimulatorAccess<dim>, public VisualizationTrigger
  {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        prm.enter_subsection("BP3 profiles");
        prm.declare_entry("Time interval", "31557600", Patterns::Double(0.),
                          "Maximum accepted time in seconds between full material profiles.");
        prm.declare_entry("Slip interval", "0.1", Patterns::Double(0.),
                          "Request a full profile when the slip of any material ID changes by this many metres.");
        prm.leave_subsection();
        prm.leave_subsection();
      }

      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Postprocess");
        prm.enter_subsection("BP3 profiles");
        interval=prm.get_double("Time interval");
        slip_interval=prm.get_double("Slip interval");
        prm.leave_subsection();
        prm.leave_subsection();
      }

      // Visualization can run before or after this observer. Look ahead on a
      // copy until execute() records the accepted peak. Only execute changes the streak.
      bool requests_graphical_output() const override
      {
        const double clock=this->get_time();
        if (clock==last_observed)
          return event.onset==clock || event.completion==clock;
        const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        double peak=0.;
        for (const auto &label:model.labels())
          peak=std::max(peak,label.rate);
        auto prediction=event;
        return prediction.observe(clock,peak);
      }

      void record_graphical_output() override {}

      bool complete() const
      {
        return event.completion>=0.;
      }

      std::pair<std::string,std::string> execute(TableHandler &statistics) override
      {
        const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        const auto &labels=model.labels();
        const auto &age=model.label_ages();
        const double clock=this->get_time();
        double peak=0., increment=0.;
        AssertThrow(slip.empty() || slip.size()==labels.size(),ExcMessage("Restored profile vertex count differs."));
        for (unsigned int i=0; i<labels.size(); ++i)
          {
            peak=std::max(peak,labels[i].rate);
            if (!slip.empty()) increment=std::max(increment,std::abs(labels[i].slip-slip[i]));
          }
        const bool milestone=event.observe(clock,peak);
        if (milestone && event.onset==clock)
          onset_parent=last_observed;
        if (milestone && event.completion==clock)
          completion_parent=last_observed;
        last_observed=clock;
        statistics.add_value("BP3 accepted time before onset (s)",onset_parent);
        statistics.add_value("BP3 accepted time before completion (s)",completion_parent);
        statistics.add_value("BP3 event onset (s)",event.onset);
        statistics.add_value("BP3 event below-threshold streak",event.below);
        statistics.add_value("BP3 event completion (s)",event.completion);
        for (const auto *column:{"BP3 accepted time before onset (s)", "BP3 accepted time before completion (s)",
                                 "BP3 event onset (s)", "BP3 event completion (s)"})
          {
            statistics.set_precision(column,17);
            statistics.set_scientific(column,true);
          }
        if (!slip.empty() && !milestone && increment<slip_interval && clock-last_clock<interval)
          return {};
        const auto &tractions=model.section_tractions();
        AssertThrow(tractions[0].size()==labels.size(),ExcMessage("BP3 profiles require Record section tractions."));
        const std::string filename="fault-profile-"+Utilities::int_to_string(this->get_timestep_number(),7)+".txt";
        bool success=true;
        if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
          {
            const auto path=this->get_output_directory()+filename;
            std::ofstream out(path+".tmp");
            out << std::setprecision(17) << "# accepted_clock_s " << clock
                << "\n# shear is positive up dip and includes radiation damping; compression positive in closure"
                << "\nid s_m x_m y_m rate_m_s age_s slip_m shear_Pa raw_compression_Pa friction_compression_Pa normal_integral_m\n";
            for (unsigned int i=0; i<labels.size(); ++i)
              out << i << ' ' << labels[i].position.distance(labels.front().position) << ' '
                  << labels[i].position << ' ' << labels[i].rate << ' ' << age.at(i) << ' ' << labels[i].slip << ' '
                  << tractions[2][i] << ' ' << tractions[0][i] << ' ' << tractions[1][i] << ' '
                  << model.phase_profile().normal_integrals()[i] << '\n';
            out.close();
            success=out.good() && std::rename((path+".tmp").c_str(),path.c_str())==0;
          }
        AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(success),this->get_mpi_communicator()),
                    ExcMessage("Failed to write accepted BP3 fault profile."));
        slip.resize(labels.size());
        for (unsigned int i=0; i<labels.size(); ++i) slip[i]=labels[i].slip;
        last_clock=clock;
        return {"BP3 accepted profile",filename};
      }

      void save(std::map<std::string,std::string> &status) const override
      {
        std::ostringstream bytes;
        {
          aspect::oarchive archive(bytes);
          archive << event << last_clock << slip;
        }
        status["Historical BP3 first event v1"]=bytes.str();
        std::ostringstream clocks;
        clocks << std::setprecision(17) << last_observed << ' ' << onset_parent << ' ' << completion_parent;
        status["Historical BP3 accepted event clocks"]=clocks.str();
      }

      void load(const std::map<std::string,std::string> &status) override
      {
        std::istringstream bytes(status.at("Historical BP3 first event v1"));
        aspect::iarchive archive(bytes);
        archive >> event >> last_clock >> slip;
        const auto clocks=status.find("Historical BP3 accepted event clocks");
        if (clocks!=status.end())
          {
            std::istringstream input(clocks->second);
            AssertThrow(input >> last_observed >> onset_parent >> completion_parent,
                        ExcMessage("Invalid checkpointed BP3 event clocks."));
          }
      }

    private:
      BP3::FirstEvent event;
      double last_clock=-1., interval, slip_interval;
      double last_observed=-1., onset_parent=-1., completion_parent=-1.;
      std::vector<double> slip;
  };

  ASPECT_REGISTER_POSTPROCESSOR(HistoricalProfiles,"BP3 first event profiles",
                                "Accepted material profiles at 0.1 m slip, one year and first-event milestones.")
}

namespace aspect::TerminationCriteria
{
  template <int dim>
  class FirstEventComplete : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      bool execute() override
      {
        return this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::HistoricalProfiles<dim>>().complete();
      }
  };
  ASPECT_REGISTER_TERMINATION_CRITERION(FirstEventComplete,"BP3 first event complete",
                                        "Stop after the fifth consecutive accepted sub-threshold peak following onset.")
}
