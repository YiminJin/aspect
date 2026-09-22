// Opt-in startup accuracy heuristic. No history is advanced by this predictor.
#include <aspect/time_stepping/interface.h>
#include <aspect/time_stepping/convection_time_step.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/plugins.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace TimeStepping
  {
    template <int dim>
    class BP5StateStartup : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Time stepping");
        prm.enter_subsection("BP5 state startup");
        prm.declare_entry("Maximum logarithmic state change", "0.1", Patterns::Double(0.),
                          "Positive bound on max (b/a)*abs(log(Theta_pred/Theta)). "
                          "An accuracy heuristic, not a nonlinear stopping tolerance.");
        prm.declare_entry("Record timestep selection", "false", Patterns::Bool(),
                          "Write controller proposals and the actual selected step; no full-state export.");
        prm.leave_subsection();prm.leave_subsection();
      }

      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Time stepping");prm.enter_subsection("BP5 state startup");
        limit=prm.get_double("Maximum logarithmic state change");
        record_selection=prm.get_bool("Record timestep selection") || std::getenv("ASPECT_BP5_TIMESTEP_AUDIT");
        AssertThrow(limit>0.,ExcMessage("The startup state-change bound must be positive."));
        prm.leave_subsection();prm.leave_subsection();
      }

      double execute() override
      {
        const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
          this->get_material_model());
        const auto &law=model.get_fault_friction();
        const auto &manager=this->get_reconstructed_fault_manager();
        AssertThrow(law.has_state_variable() && manager.slip_rates_are_initialized(),
                    ExcMessage("BP5 startup requires initialized stateful fault friction."));
        const auto state=manager.get_property_information()[manager.get_property_index(
          "phase field fault state")].position;
        std::vector<unsigned int> composition;
        for (const auto &name:this->introspection().chemical_composition_field_names())
          composition.push_back(manager.get_property_information()[manager.get_property_index(
            "phase field fault chemical composition "+name)].position);

        struct Node { double V,theta,ratio;std::vector<double> mixture; };
        std::vector<Node> nodes;
        for (unsigned int f=0;f<manager.get_faults().size();++f)
          for (unsigned int i=0;i<manager.get_fault(f).n_vertices();++i)
            {
              std::vector<double> chemical;
              for (const auto p:composition)chemical.push_back(manager.get_fault(f).get_properties(i)[p]);
              const auto mixture=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
              const double V=manager.get_timestep_committed_slip_rate(f)[i];
              const double theta=manager.get_fault(f).get_properties(i)[state];
              // The regularization multiplier cancels: theta*mu_theta/(V*mu_V)=b/a.
              // Use the configured law instead of duplicating material parameters.
              const double ratio=theta*law.friction_coefficient_derivative_wrt_state(mixture,V,theta)
                                 /(V*law.friction_coefficient_derivative_wrt_slip_rate(mixture,V,theta));
              AssertThrow(std::isfinite(ratio) && ratio>=0.,ExcMessage("Invalid startup b/a ratio."));
              nodes.push_back({V,theta,ratio,mixture});
            }
        const auto measure=[&](const double dt)
        {
          double result=0.;
          for (const auto &n:nodes)
            result=std::max(result,n.ratio*std::abs(std::log(
              law.update_state(n.mixture,n.V,n.theta,dt)/n.theta)));
          return result;
        };

        // For each frozen V, Theta moves monotonically toward Dc/V, so the
        // maximum absolute log-change is monotone. Retain the safe bracket end.
        const double ceiling=this->get_parameters().maximum_time_step;
        double dt=ceiling;
        if (measure(dt)>limit)
          {
            double lower=0.,upper=dt;
            for (unsigned int i=0;i<60;++i)
              {
                const double middle=lower+.5*(upper-lower);
                if (measure(middle)<=limit)lower=middle;else upper=middle;
              }
            dt=lower;
          }
        AssertThrow(dt>0. && std::isfinite(dt),ExcMessage("Invalid BP5 startup time restriction."));
        // Fault data are replicated; MPI min protects the global selection.
        dt=Utilities::MPI::min(dt,this->get_mpi_communicator());
        predictor_proposal=dt;
        if (record_selection)
          {
            // Re-evaluate the read-only production controllers, not approximations
            // from exported samples. The manager still makes the actual selection.
            ConvectionTimeStep<dim> convection;
            convection.initialize_simulator(this->get_simulator());
            convection_proposal=convection.execute();
            fault_proposal=model.compute_reconstructed_fault_time_step(this->get_parameters().CFL_number);
          }
        if (this->get_pcout().is_active())
          {
            const auto path=this->get_output_directory()+"state_startup_predictor.csv";
            const bool first=this->get_timestep_number()==0;
            std::ofstream out(path,first ? std::ios::out : std::ios::app);
            if (first)out<<"accepted_step,time,maximum_dt,proposed_dt,measure,limit\n";
            out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','
               <<ceiling<<','<<dt<<','<<measure(dt)<<','<<limit<<'\n';
          }
        return dt;
      }

      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override
      {
        if (record_selection && this->get_pcout().is_active())
          {
            const auto path=this->get_output_directory()+"timestep_selection.csv";
            const bool header=!std::ifstream(path).good();
            std::ofstream out(path,std::ios::app);
            if (header)out<<"accepted_step,time,convection,fault,state_predictor,ceiling,first_cap,growth_cap,termination_reduced,selected\n";
            const auto &p=this->get_parameters();
            const double growth=this->get_timestep()==0 ? std::numeric_limits<double>::max()
              : this->get_timestep()*(1+p.maximum_relative_increase_time_step);
            out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','
               <<convection_proposal<<','<<fault_proposal<<','<<predictor_proposal<<','
               <<p.maximum_time_step<<','
               <<(this->get_timestep_number()==0 ? p.maximum_first_time_step : std::numeric_limits<double>::max())
               <<','<<growth<<','<<info.reduced_by_termination_plugin<<','<<info.next_time_step_size<<'\n';
          }
        return Interface<dim>::determine_reaction(info);
      }
    private:
      double limit=0.1;
      bool record_selection=false;
      double predictor_proposal=0.,convection_proposal=0.,fault_proposal=0.;
    };

    ASPECT_REGISTER_TIME_STEPPING_MODEL(BP5StateStartup,"BP5 state startup",
      "Opt-in benchmark restriction on predicted logarithmic state change, "
      "using committed state/rate and the configured constant-rate aging law.")
  }
}
