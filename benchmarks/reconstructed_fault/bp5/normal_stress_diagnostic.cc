// Opt-in accepted-state observer. It never reevaluates committed Maxwell history.
#include <aspect/postprocess/interface.h>
#include <aspect/termination_criteria/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/plugins.h>
#include <aspect/particle/manager.h>
#include <aspect/utilities.h>
#include <aspect/time_stepping/interface.h>
#include "normal_stress_clock.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace aspect
{
  namespace BP5NormalStress
  {
    // Diagnostic termination state belongs only to this disposable process.
    bool finished = false;
    constexpr double sine = .86602540378443864676;
    template <int dim> double xd(const Point<dim> &p) { return (100000.-p[1])/sine; }

    std::vector<double> multiply(const std::vector<double> &diagonal,
                                 const std::vector<double> &off,
                                 const std::vector<double> &x)
    {
      std::vector<double> y(x.size());
      for (unsigned int i=0;i<x.size();++i)
        {
          y[i]=diagonal[i]*x[i];
          if (i) y[i]+=off[i-1]*x[i-1];
          if (i+1<x.size()) y[i]+=off[i]*x[i+1];
        }
      return y;
    }
  }

  namespace TimeStepping
  {
    // A benchmark-only MIN cap. Ordinary convection/fault/state controllers
    // remain active; the observer rejects a shortened schedule before mechanics.
    template <int dim>
    class BP5RecordedHalfSteps : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Time stepping");
        prm.enter_subsection("BP5 recorded half steps");
        prm.declare_entry("Reference trajectory file","",Patterns::Anything(),
                          "Completed four-step A normal_summary.csv. Read directly; no generated schedule or edited checkpoint.");
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Time stepping");prm.enter_subsection("BP5 recorded half steps");
        path=prm.get("Reference trajectory file");
        prm.leave_subsection();prm.leave_subsection();
        prm.enter_subsection("Postprocess");prm.enter_subsection("BP5 normal diagnostic");
        checkpoint_step=prm.get_integer("Checkpoint accepted step");
        checkpoint_time=prm.get_double("Checkpoint physical time");
        AssertThrow(prm.get_integer("New accepted steps")==8 && prm.get("Expected clock file").empty(),
                    ExcMessage("Recorded half steps require eight steps and no separate Expected clock file."));
        prm.leave_subsection();prm.leave_subsection();
      }
      void initialize() override
      {
        AssertThrow(this->get_parameters().resume_computation && !this->convert_output_to_years() && !path.empty(),
                    ExcMessage("Recorded half steps require a restart, seconds, and A's trajectory file."));
        clock=BP5NormalStress::read_half_step_clock(
          Utilities::read_and_distribute_file_content(path,this->get_mpi_communicator()),checkpoint_step,checkpoint_time,4);
        this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim> &,double &dt)
        {
          // Validate the original pending interval before requesting its half.
          // The core changes only time and dt, after all histories are restored.
          BP5NormalStress::check_clock(clock.original.front(),this->get_timestep_number(),this->get_time(),this->get_timestep());
          AssertThrow(this->get_time()-this->get_timestep()==checkpoint_time,
                      ExcMessage("The restored accepted time differs from the reference trajectory origin."));
          dt=std::min(dt,clock.halves.front().dt);
          if (this->get_pcout().is_active())
            {
              std::ofstream out(this->get_output_directory()+"normal_half_step_clock.csv");
              out<<std::setprecision(17)<<"step,time_s,dt,second_half_rounding_adjustment_s\n";
              for (unsigned int i=0;i<clock.halves.size();++i)
                {
                  const auto &c=clock.halves[i];
                  out<<c.step<<','<<c.time<<','<<c.dt<<','
                     <<(i%2 ? c.dt-clock.original[i/2].dt/2. : 0.)<<'\n';
                }
            }
        });
        // Independent of postprocessor ordering; no need to duplicate a
        // schedule file or to wait for the first accepted output to check it.
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
        {
          const auto step=this->get_timestep_number();
          AssertThrow(step>checkpoint_step && step<=checkpoint_step+clock.halves.size(),
                      ExcMessage("Recorded half-step experiment exceeded its eight-step bound."));
          BP5NormalStress::check_clock(clock.halves[step-checkpoint_step-1],step,this->get_time(),this->get_timestep());
        });
      }
      double execute() override
      {
        const auto step=this->get_timestep_number();
        AssertThrow(step>checkpoint_step && step<=checkpoint_step+clock.halves.size(),
                    ExcMessage("Recorded half-step cap requested outside its accepted sequence."));
        if (step==checkpoint_step+clock.halves.size()) return std::numeric_limits<double>::max();
        return clock.halves[step-checkpoint_step].dt;
      }
    private:
      std::string path;
      unsigned int checkpoint_step=0;
      double checkpoint_time=0.;
      BP5NormalStress::HalfStepClock clock;
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BP5RecordedHalfSteps,"BP5 recorded half steps",
                                       "Bounded replay of four recorded normal intervals as eight safe half steps.")
  }

  namespace Postprocess
  {
    template <int dim>
    class BP5NormalDiagnostic : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess"); prm.enter_subsection("BP5 normal diagnostic");
        prm.declare_entry("Checkpoint accepted step","5612",Patterns::Integer(0));
        prm.declare_entry("Checkpoint physical time","5310111071.5634108",Patterns::Double(0));
        prm.declare_entry("New accepted steps","5",Patterns::Integer(1));
        prm.declare_entry("Wall seconds","600",Patterns::Double(1));
        prm.declare_entry("Capture stress split","true",Patterns::Bool(),
                          "Disable for the single-step unchanged-mechanics control.");
        prm.declare_entry("Small windows only","false",Patterns::Bool());
        prm.declare_entry("Raw every step","false",Patterns::Bool());
        prm.declare_entry("Filter experiment diagnostics","false",Patterns::Bool(),"Capture the initial resumed base and compact filter moments.");
        prm.declare_entry("Friction normal input","raw",Patterns::Selection("raw|projected|helmholtz"),
                          "Experimental full-fault normal-input filter. Raw bypasses it completely.");
        prm.declare_entry("Normal filter length","0",Patterns::Double(0),"Physical arc-length scale in metres.");
        prm.declare_entry("Native centerline","false",Patterns::Bool(),"Evaluate native frozen fields on r=0, retaining both cell traces.");
        prm.declare_entry("Expected clock file","",Patterns::Anything(),
                          "Optional step/time/dt schedule. Stop before mechanics if a safety controller shortened a scheduled step.");
        prm.leave_subsection(); prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Postprocess"); prm.enter_subsection("BP5 normal diagnostic");
        checkpoint_step=prm.get_integer("Checkpoint accepted step");
        checkpoint_time=prm.get_double("Checkpoint physical time");
        count=prm.get_integer("New accepted steps"); wall_seconds=prm.get_double("Wall seconds");
        capture=prm.get_bool("Capture stress split");
        small_windows=prm.get_bool("Small windows only"); raw_every_step=prm.get_bool("Raw every step");
        filter_audit=prm.get_bool("Filter experiment diagnostics");
        filter_mode=prm.get("Friction normal input");filter_length=prm.get_double("Normal filter length");
        native_line=prm.get_bool("Native centerline");
        clock_file=prm.get("Expected clock file");
        prm.leave_subsection(); prm.leave_subsection();
      }
      std::list<std::string> required_other_postprocessors() const override
      { return {"reconstructed fault BP3", "BP3 output complete"}; }

      void initialize() override
      {
        AssertThrow(dim==2 && this->get_parameters().resume_computation,
                    ExcMessage("BP5 normal diagnostic requires a 2-D restart."));
        // Postprocessors initialize before the simulator constructs coupling.
        this->get_signals().post_simulator_initialization.connect([this](const SimulatorAccess<dim> &)
        {
        this->get_reconstructed_fault_surface_system().set_normal_stress_filter(filter_mode,filter_length);
        if (filter_audit)
        this->get_reconstructed_fault_surface_system().normal_diagnostic_observer=
          [this](const ReconstructedFaultSurfaceResidual &weak,
                 const typename ReconstructedFaultSurfaceSystem<dim>::NormalTractionDiagnostic &d)
          {
            if (captured_initial) return;
            captured_initial=true;
            // This is the FIRST BASE of the resumed solve, with checkpoint
            // histories and pending dt, not a reevaluation of accepted step 5612.
            if (this->get_pcout().is_active())
              {
                std::ofstream out(this->get_output_directory()+"normal_initial_operator.csv");
                out<<std::setprecision(17)<<"vertex,xd,pending_step,actual_dt,M_diag,M_right,K_diag,K_right,Mmu_diag,Mmu_right,raw_normal_load,friction_load,pressure_load,deviatoric_load,background_load,V,incoming_Theta\n";
                const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
                const auto &diag=weak.mass_diagonal[0],&off=weak.mass_off_diagonal[0];
                const auto &raw=weak.raw_normal_traction.empty()?weak.normal_traction[0]:weak.raw_normal_traction[0];
                for (unsigned int i=0;i<diag.size();++i)
                  out<<i<<','<<BP5NormalStress::xd(fault.vertex(i))<<','<<d.step<<','<<this->get_timestep()
                    <<','<<diag[i]<<','<<(i<off.size()?off[i]:0.)<<','<<d.filter_stiffness_diagonal[0][i]
                    <<','<<(i<off.size()?d.filter_stiffness_off_diagonal[0][i]:0.)
                    <<','<<d.friction_mass_diagonal[0][i]<<','<<(i<off.size()?d.friction_mass_off_diagonal[0][i]:0.)
                    <<','<<raw[i]<<','<<weak.friction_traction[0][i]<<','<<d.pressure_load[0][i]
                    <<','<<d.deviatoric_load[0][i]<<','<<d.background_load[0][i]
                    <<','<<d.rates[0][i]<<','<<incoming_theta[i]<<'\n';
              }
            write_samples(d,0,0.);
            const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            // Preserve the base separately; accepted-step output uses the same
            // step number but a different lifecycle stage.
            for (const auto prefix:{"normal_qp_","normal_friction_qp_","normal_incoming_particles_"})
              {
                const auto path=this->get_output_directory()+prefix+std::to_string(d.step)+"_rank"+std::to_string(rank)+".csv";
                const auto target=this->get_output_directory()+"initial_"+prefix+"rank"+std::to_string(rank)+".csv";
                AssertThrow(std::rename(path.c_str(),target.c_str())==0,ExcMessage("Cannot preserve initial filter diagnostic."));
              }
          };
        });
        if (!clock_file.empty())
          {
            std::istringstream input(Utilities::read_and_distribute_file_content(clock_file,this->get_mpi_communicator()));
            unsigned int step;double time,dt;
            while (input>>step>>time>>dt) expected_clock.push_back({step,time,dt});
            AssertThrow(input.eof() && expected_clock.size()==count,
                        ExcMessage("Expected clock must contain exactly the requested accepted steps."));
          }
        // Both BP3 bookkeeping and native writers remain present. Their output
        // is vetoed here; lightweight accepted-step/slip records still run.
        this->get_signals().allow_native_output.connect([](const std::string &) { return false; });
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &) { begin_step(); });
        this->get_signals().post_nonlinear_solver.connect([this](const SolverControl &control)
          { converged=control.last_check()==SolverControl::success; });
      }

      void begin_step()
      {
        converged=false;
        const auto step=this->get_timestep_number();
        if (!expected_clock.empty())
          {
            AssertThrow(step>checkpoint_step && step-checkpoint_step<=expected_clock.size(),
                        ExcMessage("Unexpected step in the half-step experiment."));
            const auto &expected=expected_clock[step-checkpoint_step-1];
            AssertThrow(step==expected.step && this->get_time()==expected.time && this->get_timestep()==expected.dt,
                        ExcMessage("Scheduled half-step was changed by a controller or checkpoint clock. "
                                   "Stop; do not force a larger step or add extra solves."));
          }
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        const auto state=position("phase field fault state");
        const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        const auto &law=model.get_fault_friction();
        const auto chemical=position("phase field fault chemical composition strengthening");
        incoming_theta.resize(fault.n_vertices()); predicted.resize(fault.n_vertices()); ratios.resize(fault.n_vertices());
        if (!started)
          {
            AssertThrow(step==checkpoint_step+1 && this->get_time()>checkpoint_time,
                        ExcMessage("Restart does not match the staged accepted-step metadata."));
            started=true; wall_start=std::chrono::steady_clock::now();
            restored_properties.resize(fault.n_vertices());
            for (unsigned int i=0;i<fault.n_vertices();++i)
              {
                const auto data=fault.get_properties(i);
                restored_properties[i].assign(data.begin(),data.end());
                restored_geometry.push_back(fault.vertex(i));
              }
            // This is deliberately a restored-history inventory, not traction.
            if (this->get_pcout().is_active())
              {
                std::ofstream schema(this->get_output_directory()+"normal_property_schema.csv");
                schema<<"position,components,name\n";
                for (const auto &property:manager.get_property_information())
                  schema<<property.position<<','<<property.n_components<<','<<property.name<<'\n';
                std::ofstream out(this->get_output_directory()+"normal_restored_fault.csv");
                out << std::setprecision(17) << "checkpoint_step,checkpoint_time,vertex,x,y,V";
                for (unsigned int c=0;c<restored_properties[0].size();++c) out << ",property_" << c;
                out << '\n';
                for (unsigned int i=0;i<fault.n_vertices();++i)
                  {
                    out<<checkpoint_step<<','<<checkpoint_time<<','<<i<<','<<fault.vertex(i)[0]<<','<<fault.vertex(i)[1]
                       <<','<<manager.get_timestep_committed_slip_rate(0)[i];
                    for (const auto v:restored_properties[i]) out<<','<<v;
                    out<<'\n';
                  }
              }
            // Particle summaries use real local owners, not ghosts. No update.
            const auto &pm=this->get_particle_manager(0);
            const auto stress=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
            const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
            double sums[5]={}, squares[5]={}; unsigned long long particles=0;
            for (const auto &p:pm.get_particle_handler())
              {
                ++particles;
                const double values[]={double(p.get_id()),p.get_properties()[H],p.get_properties()[stress],
                                       p.get_properties()[stress+1],p.get_properties()[stress+2]};
                for (unsigned int i=0;i<5;++i) { sums[i]+=values[i]; squares[i]+=values[i]*values[i]; }
              }
            std::ofstream inventory(this->get_output_directory()+"normal_restored_particles_rank"
              +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
            inventory<<std::setprecision(17)<<"count,id_sum,H_sum,xx_sum,yy_sum,xy_sum,id_squared,H_squared,xx_squared,yy_squared,xy_squared\n"<<particles;
            for (double v:sums) inventory<<','<<v;
            for (double v:squares) inventory<<','<<v;
            inventory<<'\n';
          }
        AssertThrow(step<=checkpoint_step+count,ExcMessage("Diagnostic exceeded its accepted-step bound."));
        for (unsigned int i=0;i<fault.n_vertices();++i)
          {
            incoming_theta[i]=fault.get_properties(i)[state];
            const auto fractions=MaterialModel::MaterialUtilities::compute_composition_fractions({fault.get_properties(i)[chemical]});
            const double V=manager.get_timestep_committed_slip_rate(0)[i],theta=incoming_theta[i];
            predicted[i]=law.update_state(fractions,V,theta,this->get_timestep());
            ratios[i]=theta*law.friction_coefficient_derivative_wrt_state(fractions,V,theta)
                      /(V*law.friction_coefficient_derivative_wrt_slip_rate(fractions,V,theta));
          }
        if (capture)
          {
            std::vector<std::pair<Point<dim>,Point<dim>>> lines;
            if (native_line)
              {
                const auto point=[&](double s)
                {
                  for (unsigned int j=0;j<fault.n_cells();++j)
                    {
                      const double a=BP5NormalStress::xd(fault.vertex(j)),b=BP5NormalStress::xd(fault.vertex(j+1));
                      const double xi=(s-a)/(b-a);
                      if (xi>=0. && xi<=1.) return (1.-xi)*fault.vertex(j)+xi*fault.vertex(j+1);
                    }
                  AssertThrow(false,ExcMessage("Centerline window is outside the fault."));
                  return Point<dim>();
                };
                lines.emplace_back(point(70000.),point(71000.));lines.emplace_back(point(79000.),point(80000.));
              }
            const double max_s=std::max(BP5NormalStress::xd(fault.vertex(0)),BP5NormalStress::xd(fault.vertex(fault.n_vertices()-1)));
            this->get_reconstructed_fault_surface_system().set_normal_traction_diagnostic([this,max_s](const Point<dim> &p)
            { const double s=BP5NormalStress::xd(p);
              if (filter_audit) return (s>=22000. && s<=35000.) || (s>=70000. && s<=71000.)
                                      || (s>=79000. && s<=80000.) || s<=2000. || s>=max_s-2000.;
              return small_windows ? ((s>=70000. && s<=71000.) || (s>=79000. && s<=80000.))
                                   : ((s>=22000. && s<=35000.) || (s>=60000. && s<=90000.)); },std::move(lines));
          }
      }

      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        using namespace BP5NormalStress;
        AssertThrow(started && converged,ExcMessage("Normal diagnostics require a newly accepted converged solve."));
        const unsigned int step=this->get_timestep_number(),relative=step-checkpoint_step;
        elapsed_actual+=this->get_timestep();
        const double t=this->get_time(), elapsed=elapsed_actual;
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        auto &surface=this->get_reconstructed_fault_surface_system();
        const auto &weak=surface.get_linearization_residual();
        const auto &diag=weak.mass_diagonal[0], &off=weak.mass_off_diagonal[0];
        const auto project=[&](const std::vector<double> &f)
        { return ReconstructedFaultUtilities::solve_tridiagonal_system(diag,off,f); };
        const auto &raw_load=weak.raw_normal_traction.empty() ? weak.normal_traction[0] : weak.raw_normal_traction[0];
        const auto normal=project(raw_load);
        const auto actual_normal=project(weak.normal_traction[0]);
        const auto mass=multiply(diag,off,std::vector<double>(diag.size(),1.));
        const auto &V=manager.get_timestep_committed_slip_rate(0);
        const unsigned int state=position("phase field fault state"),I=position("phase field fault previous I h"),slip=position("cumulative_signed_slip_m");
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count();
        finished=relative>=count || wall>=wall_seconds;
        std::vector<double> p(V.size()),d(V.size()),bg(V.size());
        std::array<std::vector<double>,3> components;
        double maximum=0.,sum2=0.,mass_sum=0.,projection_residual=0.;
        double max_predicted=0.,max_realized=0.,geometry_error=0.,I_change=0.;
        unsigned int limiting_node=0;
        const typename ReconstructedFaultSurfaceSystem<dim>::NormalTractionDiagnostic *data=nullptr;
        if (capture)
          {
            data=&surface.get_normal_traction_diagnostic();
            AssertThrow(data->step==step && data->time==t && data->rates[0]==V,
                        ExcMessage("Normal capture is not the accepted absolute mechanical state."));
            p=project(data->pressure_load[0]); d=project(data->deviatoric_load[0]); bg=project(data->background_load[0]);
            for (unsigned int c=0;c<3;++c) components[c]=project(data->deviatoric_component_loads[c][0]);
            const std::vector<double> fields[]={p,d,bg,normal};
            const std::vector<double> *loads[]={&data->pressure_load[0],&data->deviatoric_load[0],&data->background_load[0],&raw_load};
            for (unsigned int c=0;c<4;++c)
              {
                const auto product=multiply(diag,off,fields[c]);
                for (unsigned int i=0;i<V.size();++i)
                  projection_residual=std::max(projection_residual,std::abs(product[i]-(*loads[c])[i])/mass[i]);
              }
          }
        std::ofstream out;
        if (this->get_pcout().is_active())
          {
            out.open(this->get_output_directory()+"normal_profile_"+std::to_string(step)+".csv");
            out<<std::setprecision(17)<<"time_s,elapsed_s,accepted_step,diagnostic_step,data_stage,fault_id,vertex_id,down_dip_s_m,x_m,y_m,weak_pressure_Pa,weak_minus_n_tau_n_Pa,weak_reference_normal_Pa,weak_normal_sum_Pa,existing_production_weak_normal_Pa,closure_error_Pa,pressure_load,deviatoric_normal_load,reference_load,normal_row_mass,pressure_load_over_row_mass,deviatoric_load_over_row_mass,I_h,V_m_per_s,incoming_Theta_s,committed_Theta_s,cumulative_slip_m,predicted_weighted_log_change,realized_weighted_log_change,prescribed,mass_diagonal,mass_left,mass_right,weak_d_history_Pa,weak_d_strain_Pa,weak_d_slip_Pa,d_history_load,d_strain_load,d_slip_load,d_components_closure_Pa\n";
          }
        for (unsigned int i=0;i<V.size();++i)
          {
            AssertThrow(mass[i]>0.,ExcMessage("Diagnostic physical projection has a zero-mass row."));
            const auto properties=fault.get_properties(i);
            const double error=capture ? p[i]+d[i]+bg[i]-normal[i] : 0.;
            maximum=std::max(maximum,std::abs(error)); sum2+=mass[i]*error*error; mass_sum+=mass[i];
            const double predicted_change=ratios[i]*std::abs(std::log(predicted[i]/incoming_theta[i]));
            const double actual_change=ratios[i]*std::abs(std::log(properties[state]/incoming_theta[i]));
            if (predicted_change>max_predicted) { max_predicted=predicted_change; limiting_node=i; }
            max_realized=std::max(max_realized,actual_change);
            geometry_error=std::max(geometry_error,fault.vertex(i).distance(restored_geometry[i]));
            I_change=std::max(I_change,std::abs(properties[I]/restored_properties[i][I]-1.));
            if (out)
              {
                out<<t<<','<<elapsed<<','<<step<<','<<relative<<",accepted_mechanics_before_history/committed_state,0,"<<i<<','<<xd(fault.vertex(i))<<','<<fault.vertex(i)[0]<<','<<fault.vertex(i)[1]<<',';
                if (capture)
                  out<<p[i]<<','<<d[i]<<','<<bg[i]<<','<<p[i]+d[i]+bg[i]<<','<<actual_normal[i]<<','<<error<<','<<data->pressure_load[0][i]<<','<<data->deviatoric_load[0][i]<<','<<data->background_load[0][i]<<','<<mass[i]<<','<<data->pressure_load[0][i]/mass[i]<<','<<data->deviatoric_load[0][i]/mass[i];
                else out<<"nan,nan,nan,nan,"<<normal[i]<<",nan,nan,nan,nan,"<<mass[i]<<",nan,nan";
                out<<','<<properties[I]<<','<<V[i]<<','<<incoming_theta[i]<<','<<properties[state]<<','<<properties[slip]<<','<<predicted_change<<','<<actual_change<<','<<manager.prescribed_slip_rate_mask()[0][i]<<','<<diag[i]<<','<<(i ? off[i-1]:0.)<<','<<(i+1<V.size() ? off[i]:0.);
                if (capture)
                  {
                    for (const auto &c:components) out<<','<<c[i];
                    for (const auto &c:data->deviatoric_component_loads) out<<','<<c[0][i];
                    out<<','<<components[0][i]+components[1][i]+components[2][i]-d[i];
                  }
                else out<<",nan,nan,nan,nan,nan,nan,nan";
                out<<'\n';
              }
          }
        const auto comm=this->get_mpi_communicator();
        if (filter_audit && data && this->get_pcout().is_active())
          {
            std::ofstream extrema(this->get_output_directory()+"normal_filter_extrema_"+std::to_string(step)+".csv");
            extrema<<std::setprecision(17)<<"raw_QP_min_Pa,raw_QP_max_Pa,friction_QP_min_Pa,friction_QP_max_Pa\n"
              <<(weak.raw_normal_traction.empty()?weak.minimum_normal_traction:weak.minimum_raw_normal_traction)<<','
              <<(weak.raw_normal_traction.empty()?weak.maximum_normal_traction:weak.maximum_raw_normal_traction)<<','
              <<weak.minimum_normal_traction<<','<<weak.maximum_normal_traction<<'\n';
            std::ofstream filter(this->get_output_directory()+"normal_filter_"+std::to_string(step)+".csv");
            filter<<std::setprecision(17)<<"vertex,xd,mode,L,actual_dt,elapsed,M_diag,M_right,K_diag,K_right,Mmu_diag,Mmu_right,raw_normal_load,actual_normal_load,friction_load,shear_load,damping_load,residual,normal_coefficient\n";
            for (unsigned int i=0;i<V.size();++i)
              filter<<i<<','<<xd(fault.vertex(i))<<','<<filter_mode<<','<<filter_length<<','<<this->get_timestep()<<','<<elapsed
                <<','<<diag[i]<<','<<(i<off.size()?off[i]:0.)
                <<','<<data->filter_stiffness_diagonal[0][i]<<','<<(i<off.size()?data->filter_stiffness_off_diagonal[0][i]:0.)
                <<','<<data->friction_mass_diagonal[0][i]<<','<<(i<off.size()?data->friction_mass_off_diagonal[0][i]:0.)
                <<','<<raw_load[i]<<','<<weak.normal_traction[0][i]<<','<<weak.friction_traction[0][i]
                <<','<<weak.shear_traction[0][i]<<','<<weak.damping_traction[0][i]<<','<<weak.values[0][i]
                <<','<<(weak.normal_filter_coefficients.empty()?normal[i]:weak.normal_filter_coefficients[0][i])<<'\n';
          }
        const unsigned int sample_count=data ? Utilities::MPI::sum(static_cast<unsigned int>(data->samples.size()),comm):0;
        const unsigned int unassociated=data ? Utilities::MPI::sum(data->unassociated_phase_points,comm):0;
        if (capture && (raw_every_step || relative==1 || finished)) write_samples(*data,relative,elapsed);
        // Phase and accepted particle histories are fingerprinted separately
        // from the captured incoming-history stress; never recompute an update.
        const auto &pm=this->get_particle_manager(0);
        const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
        const auto hp=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
        std::array<double,8> particle_sums={};
        for (const auto &particle:pm.get_particle_handler())
          for (unsigned int c=0;c<4;++c)
            {
              const double v=particle.get_properties()[c==0 ? hp : sp+c-1];
              particle_sums[c]+=v; particle_sums[c+4]+=v*v;
            }
        for (auto &v:particle_sums) v=Utilities::MPI::sum(v,comm);
        const auto phase_block=this->introspection().variable("phase_field").block_index;
        // The published solution is ghosted. A global norm must count each
        // owned coefficient once, rather than call Epetra Norm2 on overlap maps.
        double phase_squared=0.;
        for (const auto i:this->introspection().index_sets.system_partitioning[phase_block])
          {
            const double value=this->get_solution().block(phase_block)[i];
            phase_squared+=value*value;
          }
        const double phase_norm=std::sqrt(Utilities::MPI::sum(phase_squared,comm));
        if (this->get_pcout().is_active())
          {
            std::ofstream checks(this->get_output_directory()+"normal_checks_"+std::to_string(step)+".csv");
            checks<<std::setprecision(17)<<"phase_l2,H_sum,xx_sum,yy_sum,xy_sum,H_squared,xx_squared,yy_squared,xy_squared\n"<<phase_norm;
            for (double v:particle_sums) checks<<','<<v;
            checks<<'\n';
            if (capture)
              {
                std::ofstream loads(this->get_output_directory()+"normal_totals_"+std::to_string(step)+".csv");
                loads<<std::setprecision(17)<<"component,owned_QP_integral_Pa_m,assembled_row_sum_Pa_m\n";
                const std::vector<double> *terms[]={&data->pressure_load[0],&data->deviatoric_load[0],&data->background_load[0],
                  &data->deviatoric_component_loads[0][0],&data->deviatoric_component_loads[1][0],&data->deviatoric_component_loads[2][0]};
                for (unsigned int c=0;c<6;++c)
                  loads<<c<<','<<data->integrated_loads[c]<<','<<std::accumulate(terms[c]->begin(),terms[c]->end(),0.)<<'\n';
              }
            std::ofstream summary(this->get_output_directory()+"normal_summary.csv",std::ios::app);
            if (relative==1) summary<<"step,time_s,elapsed_s,dt,closure_max_Pa,closure_RMS_Pa,projection_residual_over_mass_Pa,V_max,predicted_weighted_log_change,realized_weighted_log_change,predictor_node,predictor_xd,geometry_change_m,I_relative_change_from_restore,raw_window_samples,unassociated_positive_phase_samples,wall_s\n";
            summary<<std::setprecision(17)<<step<<','<<t<<','<<elapsed<<','<<this->get_timestep()<<','<<maximum<<','<<std::sqrt(sum2/mass_sum)<<','<<projection_residual<<','<<*std::max_element(V.begin(),V.end())<<','<<max_predicted<<','<<max_realized<<','<<limiting_node<<','<<xd(fault.vertex(limiting_node))<<','<<geometry_error<<','<<I_change<<','<<sample_count<<','<<unassociated<<','<<wall<<'\n';
            if (filter_audit)
              {
                // Record accepted constitutive intervals directly: the next
                // branch can replay them without subtracting large timestamps
                // or running an external schedule-generation script.
                std::ofstream clock(this->get_output_directory()+"normal_actual_intervals.txt",
                                    relative==1 ? std::ios::out : std::ios::app);
                clock<<std::setprecision(17)<<this->get_timestep()<<'\n';
              }
          }
        return {"BP5 normal diagnostic",std::to_string(relative)+" accepted steps"};
      }

    private:
      unsigned int position(const std::string &name) const
      {
        const auto &m=this->get_reconstructed_fault_manager();
        return m.get_property_information()[m.get_property_index(name)].position;
      }
      void write_samples(const typename ReconstructedFaultSurfaceSystem<dim>::NormalTractionDiagnostic &d,
                         unsigned int relative,double elapsed) const
      {
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        std::ofstream friction(this->get_output_directory()+"normal_friction_qp_"+std::to_string(d.step)+"_rank"+std::to_string(rank)+".csv");
        friction<<std::setprecision(17)<<"cell,qp,fault,segment,xi,x,y,xd,weight,mu,raw_normal,friction_normal\n";
        for (const auto &s:d.samples)
          friction<<s.cell<<','<<s.qp<<','<<s.fault<<','<<s.segment<<','<<s.xi<<','<<s.position[0]<<','<<s.position[1]
            <<','<<BP5NormalStress::xd(s.surface_position)<<','<<s.weight<<','<<s.friction_coefficient<<','<<s.total<<','<<s.friction_normal<<'\n';
        for (unsigned int kind=0;kind<2;++kind)
          {
        const auto &samples=kind ? d.line_samples:d.samples;
        if (kind && !native_line) continue;
        std::ofstream out(this->get_output_directory()+(kind ? "normal_line_":"normal_qp_")+std::to_string(d.step)+"_rank"+std::to_string(rank)+".csv");
        out<<std::setprecision(17)<<"time_s,elapsed_s,accepted_step,diagnostic_step,data_stage,rank,cell,qp,fault,segment,xi,s_m,down_dip_s_m,r_m,x,y,level,cell_diameter,n_x,n_y,p,tau_xx,tau_yy,tau_xy,minus_n_tau_n,perturbation_normal,reference_normal,total_normal,phase,I_h,chi,JxW,work_weight,incoming_FE_xx,incoming_FE_yy,incoming_FE_xy,history_xx,history_yy,history_xy,strain_xx,strain_yy,strain_xy,slip_xx,slip_yy,slip_xy,d_history,d_strain,d_slip,d_components_closure,ref_x,ref_y,particle_interp_xx,particle_interp_yy,particle_interp_xy,update_xx,update_yy,update_xy,d_update,sample_kind,stress_dt,beta,kappa,grad_xx,grad_xy,grad_yx,grad_yy\n";
        const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
        for (const auto &s:samples)
          {
            out<<d.time<<','<<elapsed<<','<<d.step<<','<<relative<<(relative ? ",accepted_mechanics_before_history," : ",first_resumed_base_checkpoint_histories_pending_dt,")<<rank<<','<<s.cell<<','<<s.qp<<','<<s.fault<<','<<s.segment<<','<<s.xi<<','<<s.surface_position.distance(fault.vertex(0))<<','<<BP5NormalStress::xd(s.surface_position)<<','<<(s.position-s.surface_position)*s.normal<<','<<s.position[0]<<','<<s.position[1]<<','<<s.level<<','<<s.cell_size<<','<<s.normal[0]<<','<<s.normal[1]<<','<<s.pressure<<','<<s.stress[0][0]<<','<<s.stress[1][1]<<','<<s.stress[0][1]<<','<<s.deviatoric<<','<<s.pressure+s.deviatoric<<','<<s.background<<','<<s.total<<','<<s.phase<<','<<s.I_h<<','<<s.chi<<','<<s.JxW<<','<<s.weight;
            out<<','<<s.incoming_stress[0][0]<<','<<s.incoming_stress[1][1]<<','<<s.incoming_stress[0][1];
            for (const auto &t:s.stress_components) out<<','<<t[0][0]<<','<<t[1][1]<<','<<t[0][1];
            const auto N=symmetrize(outer_product(s.normal,s.normal));
            double total=0.;
            for (const auto &t:s.stress_components) { const double value=-(t*N);out<<','<<value;total+=value; }
            const auto update=s.stress_components[1]+s.stress_components[2];
            out<<','<<total-s.deviatoric<<','<<s.reference_position[0]<<','<<s.reference_position[1]
               <<','<<s.particle_interpolated_stress[0][0]<<','<<s.particle_interpolated_stress[1][1]<<','<<s.particle_interpolated_stress[0][1]
               <<','<<update[0][0]<<','<<update[1][1]<<','<<update[0][1]<<','<<-(update*N)
               <<','<<(kind ? "native_line_trace":"production_QP")<<','<<s.stress_time_step<<','<<s.beta<<','<<s.kappa
               <<','<<s.velocity_gradient[0][0]<<','<<s.velocity_gradient[0][1]
               <<','<<s.velocity_gradient[1][0]<<','<<s.velocity_gradient[1][1]<<'\n';
          }
          }
        std::ofstream particles(this->get_output_directory()+"normal_incoming_particles_"+std::to_string(d.step)+"_rank"+std::to_string(rank)+".csv");
        particles<<std::setprecision(17)<<"time_s,accepted_step,rank,cell,particle_id,owner_rank,is_ghost,x,y,tau_xx,tau_yy,tau_xy,data_stage\n";
        for (const auto &p:d.particles)
          particles<<d.time<<','<<d.step<<','<<rank<<','<<p.cell<<','<<p.id<<','<<p.owner_rank<<','<<(p.owner_rank!=rank)<<','<<p.position[0]<<','<<p.position[1]
                   <<','<<p.stress[0][0]<<','<<p.stress[1][1]<<','<<p.stress[0][1]<<",incoming_precommit\n";
      }
      unsigned int checkpoint_step=5612,count=5;
      double checkpoint_time=5310111071.5634108,wall_seconds=600.;
      bool capture=true,started=false,converged=false,captured_initial=false;
      bool small_windows=false,raw_every_step=false,native_line=false,filter_audit=false;
      std::string filter_mode="raw";
      double filter_length=0.,elapsed_actual=0.;
      struct ClockEntry { unsigned int step; double time,dt; };
      std::string clock_file;
      std::vector<ClockEntry> expected_clock;
      std::chrono::steady_clock::time_point wall_start;
      std::vector<double> incoming_theta,predicted,ratios;
      std::vector<std::vector<double>> restored_properties;
      std::vector<Point<dim>> restored_geometry;
    };
    ASPECT_REGISTER_POSTPROCESSOR(BP5NormalDiagnostic,"BP5 normal diagnostic",
      "Opt-in normal-traction split captured before history commit and published after acceptance.")
  }
  namespace TerminationCriteria
  {
    template <int dim> class BP5NormalDiagnosticStop : public Interface<dim>
    { public: bool execute() override { return BP5NormalStress::finished; } };
    ASPECT_REGISTER_TERMINATION_CRITERION(BP5NormalDiagnosticStop,"BP5 normal diagnostic complete",
      "Stop after the requested newly accepted diagnostic states or accepted-state wall limit.")
  }
}
