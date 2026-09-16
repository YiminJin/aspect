#include "bp3_model.h"
#include "first_event.h"
#include "replay_stop.h"

#include <aspect/initial_composition/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/termination_criteria/interface.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iomanip>
#include <filesystem>

#include "junction_diagnostic.h"
#include "history_load_diagnostic.h"
#include "history_mechanics_diagnostic.h"
#include "theta_exact_diagnostic.h"
#include "theta_history_diagnostic.h"
#include "cohesion_diagnostic.h"
#include "mature_fault.h"
#include "uniform_sliding.h"
#include "work_measure_diagnostic.h"
#include "work_replay.h"
#include "within_step_diagnostic.h"
#include "replay_time_step.h"
#include "matched_resolution.h"

namespace aspect
{
  namespace BP3Benchmark
  {
    bool converged = false;
    double audited_accepted_time = -std::numeric_limits<double>::infinity();
    unsigned int audited_accepted_step = numbers::invalid_unsigned_int;
    bool first_event_run = false;
    bool first_event_started = false;
    bool first_event_complete = false;
    std::string bottom_normalization_completion_file;
    unsigned int newton_updates=0, krylov_iterations=0;
    double minimum_alpha=1.;
    std::vector<std::vector<bool>> final_active;

    template <int dim>
    void verify_velocity_constraints(const SimulatorAccess<dim> &sim)
    {
      // Inspect the realized physical lift, including hanging constraints.
      // This private diagnostic does not publish or modify the solver iterate.
      LinearAlgebra::BlockVector owned(sim.introspection().index_sets.system_partitioning,
                                       sim.get_mpi_communicator());
      owned=sim.get_solution();
      sim.get_current_constraints().distribute(owned);
      LinearAlgebra::BlockVector lifted(sim.introspection().index_sets.system_partitioning,
                                        sim.introspection().index_sets.system_relevant_partitioning,
                                        sim.get_mpi_communicator());
      lifted=owned;
      const auto &fe=sim.get_fe();
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      double error[2]={0,0};
      unsigned int count[2]={0,0};
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            cell->get_dof_indices(dofs);
            for (unsigned int j=0; j<dofs.size(); ++j)
              for (unsigned int d=0; d<2; ++d)
                if (fe.system_to_component_index(j).first==sim.introspection().component_indices.velocities[d])
                  {
                    const auto p=sim.get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j]);
                    if (p[0]!=0 && p[0]!=BP3::box_size) continue;
                    const unsigned int side=p[0]==0 ? 0 : 1;
                    const double expected=(side==0 ? 1 : -1)*.5*BP3::Vp*(d==0 ? BP3::cosine : BP3::sine);
                    error[side]=std::max(error[side],std::abs(lifted[dofs[j]]-expected));
                    ++count[side];
                  }
          }
      std::ofstream out;
      if (sim.get_pcout().is_active())
        {
          out.open(sim.get_output_directory()+"velocity_constraints.csv");
          out<<std::setprecision(17)<<"side,expected_ux,expected_uy,expected_speed,max_actual_error,samples\n";
        }
      for (unsigned int side=0; side<2; ++side)
        {
          const auto n=Utilities::MPI::sum(count[side],sim.get_mpi_communicator());
          const double maximum=Utilities::MPI::max(error[side],sim.get_mpi_communicator());
          AssertThrow(n>0 && maximum<1e-22,ExcMessage("BP3 realized lateral velocity constraints are incorrect."));
          const double sign=side==0 ? 1 : -1;
          if (out) out<<(side==0 ? "left" : "right")<<','<<sign*.5*BP3::Vp*BP3::cosine<<','
                     <<sign*.5*BP3::Vp*BP3::sine<<','<<.5*BP3::Vp<<','<<maximum<<','<<n<<'\n';
        }
    }

    template <int dim>
    void prescribe_phase(const SimulatorAccess<dim> &sim, AffineConstraints<double> &constraints)
    {
      AssertThrow(dim==2,ExcMessage("BP3 currently supports two dimensions."));
      const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(BP3::core_phi);
      const auto &fe=sim.get_fe();
      const auto phi=sim.introspection().variable("phase_field").first_component_index;
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            cell->get_dof_indices(dofs);
            for (unsigned int j=0;j<dofs.size();++j)
              if (fe.system_to_component_index(j).first==phi
                  && constraints.can_store_line(dofs[j]) && !constraints.is_constrained(dofs[j]))
                {
                  const auto p=sim.get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j]);
                  constraints.add_line(dofs[j]);
                  constraints.set_inhomogeneity(dofs[j],profiles[0]->value(BP3::normal_distance(p[0],p[1])));
                }
          }
    }

    template <int dim>
    void initial_history(const SimulatorAccess<dim> &sim)
    {
      // Extend the straight stationary distance field through the box boundaries;
      // use the current handler's H law and configured activation, not old BP3 H.
      auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
      const auto H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(BP3::core_phi);
      for (auto &particle:pm.get_particle_handler())
        {
          const auto p=particle.get_location();
          const double phi=profiles[0]->value(BP3::normal_distance(p[0],p[1]));
          const double f=BP3::depth_fraction(p[1]);
          if (phi>model.get_phase_field_activation_threshold())
            particle.get_properties()[H]=sim.get_phase_field_handler().stationary_crack_driving_force(
              {1-f,f},phi,BP3::core_phi);
        }
    }

    template <int dim>
    void prepare(const SimulatorAccess<dim> &sim, bool temperature, unsigned int, const SolverControl &)
    {
      if (!temperature) return;
      if (std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC")
          || std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY")) within_step_ready=true;
      if ((std::getenv("ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC")
           || !mature_prestress_file.empty()) && sim.get_timestep_number()==1
          && !uniform_sliding_test())
        cohesion_derivative_ready=true;
      if (std::getenv("ASPECT_BP3_THETA_EXACT_DIAGNOSTIC") && sim.get_timestep_number()==13)
        theta_exact_ready=true;
      if (std::getenv("ASPECT_BP3_SHARED_CLOCK"))paired_history_ready=true;
      if ((std::getenv("ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC") || std::getenv("ASPECT_BP3_FROZEN_SURFACE")) && sim.get_timestep_number()==12)
        history_load_ready=true;
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &faults=manager.get_faults();
      AssertThrow(faults.size()==1,ExcMessage("BP3 needs one fixed through-going fault."));
      const auto &fault=faults[0];
      const bool junction_diagnostic=std::getenv("ASPECT_BP3_JUNCTION_DIAGNOSTIC");
      if (junction_diagnostic)
        AssertThrow(sim.get_timestep_number()==12
                    && std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC"),
                    ExcMessage("The BP3 junction diagnostic requires a noncommitting step-12 replay."));
      const double prescribed_start=junction_diagnostic ? 35000. : BP3::Wf;
      std::vector<std::map<unsigned int,double>> prescribed(1);
      for (unsigned int v=0;v<fault.n_vertices();++v)
        {
          const auto p=fault.vertex(v);
          AssertThrow(BP3::normal_distance(p[0],p[1])<1e-8,ExcMessage("BP3 reconstructed dip changed."));
          if (uniform_sliding_test() || BP3::down_dip(p[0],p[1])>=prescribed_start-1e-7)
            prescribed[0][v]=BP3::Vp;
        }
      manager.set_prescribed_slip_rates(prescribed);
      // The registry/property data are checkpointed by the manager; the
      // material selector is reconstructible and must be reattached on resume.
      auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(
        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
      if (!bottom_normalization_completion_file.empty())
        model.set_boundary_normalization_completion_file(bottom_normalization_completion_file);
      if (!bottom_normalization_completion_file.empty()
          || std::getenv("ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION"))
        {
          AssertThrow(model.is_mature_frictional_fault()
                      && prescribed[0].count(0) && prescribed[0].at(0)==BP3::Vp,
                      ExcMessage("BP3 bottom continuation requires a mature fault and prescribed endpoint Vp."));
          if (bottom_normalization_completion_file.empty())
            AssertThrow(uniform_sliding_test() && std::getenv("ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC"),
                        ExcMessage("The diagnostic source switch requires completed uniform sliding."));
          Point<dim> lower, upper;
          upper[0]=BP3::box_size; upper[1]=BP3::box_size;
          manager.enable_bottom_source_continuation(0,lower,upper);
          if (std::getenv("ASPECT_BP3_TOP_SOURCE_EXPERIMENT") || work_measure_replay)
            {
              AssertThrow(uniform_sliding_test() || std::getenv("ASPECT_BP3_WORK_MEASURE") || work_measure_replay,
                          ExcMessage("Top continuation is restricted to prescribed uniform sliding until the surface coupling is qualified."));
              manager.enable_top_source_continuation();
            }
        }
      if (work_measure_replay)
        {
          AssertThrow(!uniform_sliding_test() && (!sim.get_parameters().resume_computation
                      || std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY"))
                      && !bottom_normalization_completion_file.empty()
                      && !std::getenv("ASPECT_BP3_WORK_MEASURE")
                      && !std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC"),
                      ExcMessage("Committing work replay requires a fresh mature run with paired completion, not a probe or restart."));
          if (const char *substeps=std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY"))
            {
              const unsigned int count=std::stoi(substeps);
              AssertThrow((count==2 || count==4) && sim.get_parameters().resume_computation
                          && std::getenv("ASPECT_FAULT_WITHIN_STEP_STATE")
                          && sim.get_timestep_number()>=10 && sim.get_timestep_number()<10+count,
                          ExcMessage("Coupled-state replay is limited to two/four substeps from accepted step 9."));
            }
          sim.get_reconstructed_fault_surface_system().enable_bulk_work_measure();
        }
      if (std::getenv("ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC"))
        {
          AssertThrow(sim.get_parameters().resume_computation && sim.get_timestep_number()==10
                      && std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC")
                      && !work_measure_replay && !bottom_normalization_completion_file.empty(),
                      ExcMessage("Within-step comparison requires the revised step-9 checkpoint and a disposable solve."));
          manager.enable_top_source_continuation();
          sim.get_reconstructed_fault_surface_system().enable_bulk_work_measure();
        }
      if (sim.get_timestep_number()!=0)
        model.set_reconstructed_fault_background_traction_property(
          manager.get_property_index("background tractions"),
          model.is_mature_frictional_fault() ? manager.get_property_index("BP3 fixed shear correction")
                                           : numbers::invalid_unsigned_int);
      if (sim.get_timestep_number()!=0) return;
      verify_paired_mesh(sim);
      verify_velocity_constraints(sim);
      manager.initialize_slip_rate(0,std::vector<double>(fault.n_vertices(),BP3::Vinit));
      // Record the realized grid before projection can fail; center-sampled
      // refinement functions need not refine every cell crossed by the fault.
      std::ofstream mesh(sim.get_output_directory()+"initial_mesh_"+
                         std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".csv");
      mesh << std::setprecision(17) << "cell,level,x,y,h,distance\n";
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto p=cell->center();
            mesh << cell->id().to_string()<<','<<cell->level()<<','<<p[0]<<','<<p[1]<<','
                 <<cell->diameter()/std::sqrt(2.)<<','<<BP3::normal_distance(p[0],p[1])<<'\n';
          }
      mesh.close();
      // The simulator owns a mutable material object. This initialization-only
      // callback precedes the particle-to-FE transfer; no assembly loop casts.
      model.prepare_reconstructed_fault_mechanical_solve();

      // Preserve official nodal Theta. The ordinary particle property supplies
      // the same initial function; subsequent states use the production update.
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      for (unsigned int v=0;v<fault.n_vertices();++v)
        manager.get_fault(0).get_properties(v)[state]=BP3::theta0(BP3::down_dip(fault.vertex(v)[0],fault.vertex(v)[1]));
      initialize_theta_function(sim,state);

      if (model.is_mature_frictional_fault())
        {
          initialize_mature_prestress(sim,model);
          if (std::getenv("ASPECT_BP3_WORK_MEASURE"))
            {
              AssertThrow(!uniform_sliding_test() && std::getenv("ASPECT_BP3_TOP_SOURCE_EXPERIMENT")
                          && std::getenv("ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC")
                          && !sim.get_parameters().resume_computation,
                          ExcMessage("Work-measure qualification requires a fresh, noncommitting free-top case."));
              sim.get_reconstructed_fault_surface_system().enable_bulk_work_measure();
              prepare_work_measure_test(sim);
            }
          return;
        }

      const unsigned int background=manager.get_property_index("background tractions");
      const auto position=manager.get_property_information()[background].position;
      for (unsigned int j=0; j<fault.n_vertices(); ++j)
        {
          auto data=manager.get_fault(0).get_properties(j);
          data[position]=0.;
          data[position+1]=BP3::sigma0;
        }
      model.set_reconstructed_fault_background_traction_property(background);

      // R = tau_bg + Delta_tau:S - C_eval - mu*sigma_total - damping*V.
      // Use the production domain quadrature to balance the represented initial
      // resistance at zero stress perturbation, not a nodal sample of C_eval.
      // C_eval = kappa_Gamma*Vinit/Ih0 + beta_Gamma*C0; C0 and Theta0 are retained.
      LinearAlgebra::BlockVector owned(sim.introspection().index_sets.system_partitioning,
                                       sim.get_mpi_communicator());
      owned=sim.get_solution();
      owned.block(sim.introspection().block_indices.velocities)=0.;
      owned.block(sim.introspection().block_indices.pressure)=0.;
      LinearAlgebra::BlockVector probe(sim.introspection().index_sets.system_partitioning,
                                       sim.introspection().index_sets.system_relevant_partitioning,
                                       sim.get_mpi_communicator());
      probe=owned;
      const auto resistance=sim.get_reconstructed_fault_surface_system().evaluate_surface_residual(
        probe,{std::vector<double>(fault.n_vertices(),BP3::Vinit)});
      std::vector<double> load(fault.n_vertices());
      for (unsigned int j=0; j<fault.n_vertices(); ++j)
        load[j]=resistance.cohesive_traction[0][j]+resistance.friction_traction[0][j]
                +resistance.damping_traction[0][j];
      // The zero-rate probe has slip-induced shear, which must NOT be included
      // in the background. Its normal contraction vanishes (S:N=0), so the
      // resistance terms still use exactly sigma0. This is not a bulk solution.
      const auto shear_background=ReconstructedFaultUtilities::solve_tridiagonal_system(
        resistance.mass_diagonal[0],resistance.mass_off_diagonal[0],load);
      const auto evaluated_C=ReconstructedFaultUtilities::solve_tridiagonal_system(
        resistance.mass_diagonal[0],resistance.mass_off_diagonal[0],resistance.cohesive_traction[0]);
      double maximum_balance_error=0.;
      for (unsigned int j=0; j<fault.n_vertices(); ++j)
        {
          manager.get_fault(0).get_properties(j)[position]=shear_background[j];
          double represented=resistance.mass_diagonal[0][j]*shear_background[j];
          if (j>0) represented+=resistance.mass_off_diagonal[0][j-1]*shear_background[j-1];
          if (j+1<fault.n_vertices()) represented+=resistance.mass_off_diagonal[0][j]*shear_background[j+1];
          maximum_balance_error=std::max(maximum_balance_error,std::abs(represented-load[j])/load[j]);
        }
      AssertThrow(maximum_balance_error<1e-12,
                  ExcMessage("BP3 frozen background does not balance the discrete initial resistance."));
      // These offsets are never recomputed at real timesteps: later changes of
      // cohesion and friction are mechanical driving, not new background loads.
      if (sim.get_pcout().is_active())
        {
          std::ofstream out(sim.get_output_directory()+"initial_traction_target.csv");
          out << std::setprecision(17) << "xd,q_target,sigma_target,Theta,a,C_eval,Q1_friction_correction,weak_relative_error\n";
          for (unsigned int v=fault.n_vertices(); v>0; --v)
            {
              const auto p=fault.vertex(v-1);
              const double xd=std::max(0.,BP3::down_dip(p[0],p[1]));
              out<<xd<<','<<shear_background[v-1]<<','<<BP3::sigma0<<','
                 <<BP3::theta0(xd)<<','<<BP3::direct_effect(xd)<<','<<evaluated_C[v-1]<<','
                 <<shear_background[v-1]-evaluated_C[v-1]-BP3::tau0<<','<<maximum_balance_error<<'\n';
            }
        }
    }
  }

  template <int dim> void connect_bp3(SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect(&BP3Benchmark::prescribe_phase<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::capture_junction_history<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::audit_history_load<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::compare_frozen_surface<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::exact_theta_comparison<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::within_step_check<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::check_frozen_cohesion<dim>);
    signals.post_constraints_creation.connect(&BP3Benchmark::paired_common_history<dim>);
    signals.post_nonlinear_solver.connect(&BP3Benchmark::verify_junction_rollback<dim>);
    // Install after manager/particle initialization slots have been registered.
    signals.post_simulator_initialization.connect([](const SimulatorAccess<dim> &sim)
    {
      sim.get_reconstructed_fault_manager().register_property("background tractions",2);
      if (Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            sim.get_material_model()).is_mature_frictional_fault())
        sim.get_reconstructed_fault_manager().register_property("BP3 fixed shear correction",3);
      sim.get_signals().post_set_initial_state.connect(&BP3Benchmark::initial_history<dim>);
    });
    signals.post_advection_solver.connect(&BP3Benchmark::prepare<dim>);
    signals.start_timestep.connect([](const SimulatorAccess<dim> &sim)
    {
      BP3Benchmark::converged=false; BP3Benchmark::paired_history_ready=false;
      // Postprocessor resume slots precede particle deserialization. At this
      // boundary all incoming histories are loaded and advection has not run.
      if (std::getenv("ASPECT_BP3_COUPLED_STATE_REPLAY") && sim.get_timestep_number()==10)
        BP3Benchmark::capture_work_invariants(sim);
    });
    signals.post_nonlinear_solver.connect([](const SolverControl &c)
    { BP3Benchmark::converged=c.last_check()==SolverControl::success && c.last_value()<c.tolerance(); });
    signals.post_reconstructed_fault_solver.connect([](unsigned int n, unsigned int k, double alpha,
                                                       const std::vector<std::vector<bool>> &active)
    {
      BP3Benchmark::newton_updates=n;
      BP3Benchmark::krylov_iterations=k;
      BP3Benchmark::minimum_alpha=alpha;
      BP3Benchmark::final_active=active;
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_bp3<2>,connect_bp3<3>)

  namespace InitialComposition
  {
    template<int dim> class BP3Initial : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double initial_composition(const Point<dim> &p, const unsigned int field) const override
        {
          // Extend the official 15--18 km down-dip transition horizontally
          // into the bulk, just as for a. The sharp-fault values are unchanged.
          const double xd=(BP3::box_size-p[1])/BP3::sine;
          const auto &name=this->introspection().name_for_compositional_index(field);
          if (name=="theta_initial") return BP3::theta0(xd);
          if (name=="strengthening") return BP3::depth_fraction(p[1]);
          AssertThrow(name=="tau_xx" || name=="tau_yy" || name=="tau_xy",
                      ExcMessage("Unexpected BP3 composition."));
          return 0.; // Maxwell stores Delta tau, not the official prestress.
        }
    };
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(BP3Initial,"reconstructed fault BP3","Official BP3 state and zero initial bulk stress change.")
  }

  namespace Particle
  {
    namespace Property
    {
      template<int dim> class BP3Strengthening : public Interface<dim>
      {
        public:
          void initialize_one_particle_property(const Point<dim> &p, std::vector<double> &data) const override
          { data.push_back(BP3::depth_fraction(p[1])); }
          void update_particle_properties(const ParticleUpdateInputs<dim> &,
            typename ParticleHandler<dim>::particle_iterator_range &particles) const override
          {
            // a is a prescribed spatial function, not an advected material marker.
            for (auto &particle:particles)
              { const auto p=particle.get_location();
                particle.get_properties()[this->data_position]=BP3::depth_fraction(p[1]); }
          }
          UpdateTimeFlags need_update() const override { return update_time_step; }
          UpdateFlags get_update_flags(unsigned int) const override { return update_default; }
          std::vector<std::pair<std::string,unsigned int>> get_property_information() const override
          { return {{"bp3 strengthening",1}}; }
      };
      ASPECT_REGISTER_PARTICLE_PROPERTY(BP3Strengthening,"bp3 strengthening","Official spatial VW/VS transition, refreshed after particle advection.")
    }
  }

  namespace BoundaryVelocity
  {
    template<int dim> class BP3Velocity : public Interface<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id, const Point<dim> &p) const override
        {
          Tensor<1,dim> v;
          const double signed_normal=(BP3::trace_x-p[0])*BP3::sine-(BP3::box_size-p[1])*BP3::cosine;
          const double sign=signed_normal>=0 ? 1 : -1;
          v[0]=sign*.5*BP3::Vp*BP3::cosine;v[1]=sign*.5*BP3::Vp*BP3::sine;
          return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(BP3Velocity,"reconstructed fault BP3","BP3 far-field rigid translation in the documented rotated chart.")
  }

  namespace BoundaryTraction
  {
    template<int dim> class BP3Traction : public Interface<dim>
    {
      public:
        Tensor<1,dim> boundary_traction(const types::boundary_id, const Point<dim> &,
                                      const Tensor<1,dim> &) const override
        {
          return {}; // Zero perturbation traction; no Airy load enters Stokes.
        }
    };
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(BP3Traction,"reconstructed fault BP3","Zero bottom stress-change traction for the finite-domain BP3 pilot.")
  }

  namespace Postprocess
  {
    template<int dim> class BP3Output : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("BP3");
          prm.declare_entry("First event run","false",Patterns::Bool(),
                            "Observe the first event and adapt benchmark output, without changing timesteps.");
          prm.declare_entry("Audit full state every step","false",Patterns::Bool(),
                            "Export bulk DoFs and stable-ID particle histories for the short restart regression.");
          prm.declare_entry("Mature prestress file","",Patterns::Anything(),
                            "Fresh-only initial fixed prestress coefficients x,y,shear,normal,a,b,d; checkpointed thereafter.");
          prm.declare_entry("Committing work-measure replay","false",Patterns::Bool(),
                            "Select the fresh straight/frozen/mature BP3 replay with bulk-QP mechanical work measure, "
                            "both paired boundary continuations and ordinary accepted history publication. "
                            "Requires a paired completion table; not a restart conversion or noncommitting probe.");
          prm.declare_entry("Bottom normalization completion file","",Patterns::Anything(),
                            "Opt-in fixed-profile mature BP3 treatment: outside-bottom profile integrals (count, then id x y integral), "
                            "projected with physical I_h, plus in-box straight endpoint source continuation. "
                            "The table must match this mesh, phase profile and fault geometry and remain immutable, including on restart. "
                            "Top continuation additionally requires the selected work replay or dedicated diagnostic. No surface connectivity change.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("BP3");
          BP3Benchmark::first_event_run=prm.get_bool("First event run");
          audit_states=prm.get_bool("Audit full state every step");
          BP3Benchmark::mature_prestress_file=prm.get("Mature prestress file");
          BP3Benchmark::work_measure_replay=prm.get_bool("Committing work-measure replay");
          BP3Benchmark::bottom_normalization_completion_file=prm.get("Bottom normalization completion file");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void initialize() override
        {
          this->get_signals().post_checkpoint.connect([this](const std::string &path)
          { archive_checkpoint(path); });
          this->get_signals().post_resume_load_user_data.connect([this](auto &)
          {
            BP3Benchmark::export_checkpoint_bulk<dim>(*this);
            // A checkpoint can be complete even if a killed job never archived
            // its event milestone. Finish that output from the restored state.
            unsigned int id=0;
            if (this->get_pcout().is_active())
              { std::ifstream marker(this->get_output_directory()+"restart/last_good_checkpoint.txt"); marker>>id; }
            id=Utilities::MPI::broadcast(this->get_mpi_communicator(),id,0);
            AssertThrow(id>0,ExcMessage("BP3 restart lacks a last-good checkpoint marker."));
            archive_checkpoint(this->get_output_directory()+"restart/"+Utilities::int_to_string(id,2));
          });
        }

        void save(std::map<std::string,std::string> &status) const override
        {
          std::ostringstream stream;
          {
            aspect::oarchive archive(stream);
            const unsigned int version=3;
            const bool mature=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
              this->get_material_model()).is_mature_frictional_fault();
            archive << version << mature << slip << previous_theta << last_step << event
                    << last_profile_time << last_bulk_time << checkpoint_labels;
          }
          status["BP3 accepted history"]=stream.str();
        }

        void load(const std::map<std::string,std::string> &status) override
        {
          AssertThrow(!std::getenv("ASPECT_FAULT_THETA_HISTORY_DIAGNOSTIC"),
                      ExcMessage("Functional Theta experiment is fresh-start only; its external accepted-rate history is not checkpointed."));
          AssertThrow(!std::getenv("ASPECT_FAULT_FROZEN_COHESION_DIAGNOSTIC"),
                      ExcMessage("Frozen cohesion experiment is fresh-start only."));
          const auto entry=status.find("BP3 accepted history");
          AssertThrow(entry!=status.end(),ExcMessage("Checkpoint lacks restartable BP3 benchmark history."));
          std::istringstream stream(entry->second);
          aspect::iarchive archive(stream);
          unsigned int version;
          archive >> version;
          AssertThrow(version==2 || version==3,ExcMessage("Unsupported BP3 benchmark checkpoint version."));
          bool stored_mature=false;
          if (version==3) archive >> stored_mature;
          AssertThrow(stored_mature==Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
                        this->get_material_model()).is_mature_frictional_fault(),
                      ExcMessage("Cannot convert cohesive/mature BP3 histories on restart."));
          archive >> slip >> previous_theta >> last_step >> event
                  >> last_profile_time >> last_bulk_time >> checkpoint_labels;
          // Pending labels are archived from this checkpoint by the post-resume
          // callback, never from the subsequent accepted physical state.
          BP3Benchmark::first_event_complete=event.complete;
          BP3Benchmark::first_event_started=event.started;
          AssertThrow(!slip.empty() && slip.size()==previous_theta.size(),
                      ExcMessage("Invalid BP3 slip/state checkpoint layout."));
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(BP3Benchmark::converged,ExcMessage("BP3 requires genuine bulk/surface convergence."));
          const auto &manager=this->get_reconstructed_fault_manager();const auto &fault=manager.get_faults()[0];
          const auto &weak=this->get_reconstructed_fault_surface_system().get_linearization_residual();
          const auto position=[&](const std::string &name){return manager.get_property_information()[manager.get_property_index(name)].position;};
          const auto state=position("phase field fault state"), C=position("phase field fault cohesive traction"), I=position("phase field fault previous I h"), background=position("background tractions");
          const unsigned int n=fault.n_vertices();
          AssertThrow(last_step==numbers::invalid_unsigned_int
                      || this->get_timestep_number()==last_step+1,
                      ExcMessage("BP3 output must visit each accepted state exactly once."));
          if (slip.empty()) slip.assign(n,0.);
          const auto &V=manager.get_timestep_committed_slip_rate(0);
          const auto prescribed=manager.prescribed_slip_rate_mask();
          BP3Benchmark::export_uniform_sliding(*this);
          if (this->get_timestep_number()>0)
            for (unsigned int j=0;j<n;++j) slip[j]+=this->get_timestep()*V[j];
          const auto qfield=ReconstructedFaultUtilities::solve_tridiagonal_system(
            weak.mass_diagonal[0],weak.mass_off_diagonal[0],weak.shear_traction[0]);
          const auto Ffield=ReconstructedFaultUtilities::solve_tridiagonal_system(
            weak.mass_diagonal[0],weak.mass_off_diagonal[0],weak.values[0]);
          const auto Cfield=ReconstructedFaultUtilities::solve_tridiagonal_system(
            weak.mass_diagonal[0],weak.mass_off_diagonal[0],weak.cohesive_traction[0]);
          const auto friction_field=ReconstructedFaultUtilities::solve_tridiagonal_system(
            weak.mass_diagonal[0],weak.mass_off_diagonal[0],weak.friction_traction[0]);
          const auto normal_field=ReconstructedFaultUtilities::solve_tridiagonal_system(
            weak.mass_diagonal[0],weak.mass_off_diagonal[0],weak.normal_traction[0]);
          const bool mature=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model()).is_mature_frictional_fault();
          std::vector<double> background_field(n);
          if (BP3Benchmark::work_measure_replay)
            background_field=BP3Benchmark::export_work_replay(*this,weak);
          else if (mature)
            background_field=BP3Benchmark::projected_mature_background(*this,weak);
          else
            for (unsigned int j=0;j<n;++j) background_field[j]=fault.get_properties(j)[background];
          if (std::getenv("ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC") && this->get_pcout().is_active())
            {
              // Accepted weak stress was frozen before publication. Export it
              // without another Maxwell evaluation using newly committed history.
              std::ofstream out(this->get_output_directory()+"stress_projected_"
                                +std::to_string(this->get_timestep_number())+".csv");
              out.exceptions(std::ios::failbit | std::ios::badbit);
              out<<"step,time,node,xd,x,y,V,slip,prescribed,lower_active,sigma_n,q,Theta_committed,mass_diagonal,mass_upper\n";
              for (unsigned int j=0; j<n; ++j)
                {
                  const auto p=fault.vertex(j);
                  const double xd=BP3::down_dip(p[0],p[1]);
                  out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<j<<','
                     <<xd<<','<<p[0]<<','<<p[1]<<','<<V[j]<<','<<slip[j]<<','<<prescribed[0][j]<<','
                     <<(!prescribed[0][j] && BP3Benchmark::final_active[0][j])<<','
                     <<normal_field[j]<<','<<qfield[j]<<','<<fault.get_properties(j)[state]<<','
                     <<weak.mass_diagonal[0][j]<<','<<(j+1<n ? weak.mass_off_diagonal[0][j] : 0.)<<'\n';
                }
            }

          // A Q1 rate attains its physical maximum at a vertex. Prescribed deep
          // nodes participate in max(V), but not in free/lower-contact counts.
          const unsigned int imax=std::max_element(V.begin(),V.end())-V.begin();
          const double maximum=V[imax];
          const double maximum_xd=BP3::down_dip(fault.vertex(imax)[0],fault.vertex(imax)[1]);
          double minimum_free=std::numeric_limits<double>::infinity(), maximum_free=0.;
          unsigned int free_count=0, lower_count=0;
          for (unsigned int j=0;j<n;++j)
            if (!prescribed[0][j])
              {
                if (BP3Benchmark::final_active[0][j]) ++lower_count;
                else
                  { ++free_count; minimum_free=std::min(minimum_free,V[j]);
                    maximum_free=std::max(maximum_free,V[j]); }
              }
          const bool started_before=event.started;
          const double previous_peak=event.peak;
          const unsigned int previous_below=event.below;
          if (BP3Benchmark::first_event_run)
            event.observe(this->get_time(),maximum,maximum_xd);
          BP3Benchmark::first_event_complete=event.complete;
          BP3Benchmark::first_event_started=event.started;
          if (!started_before && event.started)
            {
              // The latest completed checkpoint is the immediately preceding
              // accepted state, since this fixture checkpoints every step.
              bool failed=false;
              if (this->get_pcout().is_active())
                try
                  {
                    std::ifstream marker(this->get_output_directory()+"restart/last_good_checkpoint.txt");
                    unsigned int id=0;
                    marker >> id;
                    AssertThrow(marker && id>0,ExcMessage("No pre-onset BP3 checkpoint exists."));
                    const auto destination=this->get_output_directory()+"event_states/before_onset";
                    std::filesystem::create_directories(destination);
                    std::filesystem::copy(this->get_output_directory()+"restart/"+Utilities::int_to_string(id,2),
                      destination,std::filesystem::copy_options::recursive
                                  |std::filesystem::copy_options::overwrite_existing);
                  }
                catch (const std::exception &) { failed=true; }
              AssertThrow(!Utilities::MPI::max(static_cast<unsigned int>(failed),this->get_mpi_communicator()),
                          ExcMessage("BP3 pre-onset checkpoint archival failed."));
              checkpoint_labels.push_back("onset");
              this->get_pcout()<<"BP3 FIRST EVENT ONSET: time="<<event.onset<<" s"<<std::endl;
            }
          if (event.peak>previous_peak) checkpoint_labels.push_back("peak");
          if (event.below==1 && previous_below==0) checkpoint_labels.push_back("down_crossing");
          if (event.complete) checkpoint_labels.push_back("termination");
          const bool milestone=!checkpoint_labels.empty();
          constexpr double year=31557600.;
          // Output never constrains dt. Pre-event profiles become every-step
          // at 1e-5 m/s; event profiles and bulk fields are every accepted state.
          const bool profile_due=!BP3Benchmark::first_event_run || audit_states || milestone || std::getenv("ASPECT_BP3_SHARED_CLOCK") || std::getenv("ASPECT_BP3_TIMESTEP_SEQUENCE")
            || maximum>=1e-5 || this->get_time()-last_profile_time>=.1*year || last_step==numbers::invalid_unsigned_int;
          const bool bulk_due=BP3Benchmark::first_event_run && (audit_states || milestone
            || maximum>=BP3::FirstEvent::threshold || this->get_time()-last_bulk_time>=year
            || last_step==numbers::invalid_unsigned_int);
          if (profile_due) last_profile_time=this->get_time();
          if (bulk_due) { write_bulk_state(); last_bulk_time=this->get_time(); }
          if (audit_states) write_audit_state();
          if (this->get_pcout().is_active())
            {
              const auto path=this->get_output_directory()+"accepted_steps.csv";
              const bool header=!std::filesystem::exists(path);
              std::ofstream out(path,std::ios::app);
              if (header) out<<"step,time,dt,max_V,xd_at_max_V,min_sigma_n,max_sigma_n,min_free_V,max_free_V,free,lower_active,newton_updates,krylov_iterations,min_alpha\n";
              out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','
                 <<maximum<<','<<maximum_xd<<','<<weak.minimum_normal_traction<<','<<weak.maximum_normal_traction<<','
                 <<minimum_free<<','<<maximum_free<<','<<free_count<<','<<lower_count<<','
                 <<BP3Benchmark::newton_updates<<','<<BP3Benchmark::krylov_iterations<<','<<BP3Benchmark::minimum_alpha<<'\n';
              AssertThrow(out,ExcMessage("Cannot append BP3 accepted-step diagnostics."));
              write_stations(V,qfield,normal_field,state);
              if (BP3Benchmark::first_event_run)
                {
                  std::ofstream summary(this->get_output_directory()+"first_event.csv");
                  summary<<std::setprecision(17)<<"started,complete,onset,peak_V,peak_xd,peak_time,down_crossing,termination,below_count\n"
                         <<event.started<<','<<event.complete<<','<<event.onset<<','<<event.peak<<','<<event.peak_xd<<','
                         <<event.peak_time<<','<<event.down_crossing<<','<<event.termination<<','<<event.below<<'\n';
                }
            }
          // Audit the split aging-law update independently in extended precision.
          // No Maxwell response is evaluated here: the particle array is already
          // committed, whereas weak traction above belongs to the accepted solve.
          double theta_error=0.;
          std::string theta_failure;
          for (unsigned int j=0;j<n;++j)
            {
              const double actual=fault.get_properties(j)[state];
              const double xd=BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]);
              const double old_theta=this->get_timestep_number()>0
                                     ? previous_theta[j] : BP3::theta0(xd);
              const long double expected=this->get_timestep_number()>0
                ? BP3::aging_state_reference(V[j],old_theta,this->get_timestep())
                : old_theta;
              const double relative_error=std::abs(actual/static_cast<double>(expected)-1.);
              if (!std::isfinite(relative_error) || relative_error>theta_error)
                {
                  theta_error=std::isfinite(relative_error) ? relative_error
                              : std::numeric_limits<double>::infinity();
                  if (!(theta_error<1e-12))
                    {
                      std::ostringstream message;
                      message<<std::setprecision(std::numeric_limits<long double>::max_digits10)
                             <<"BP3 Theta audit: step="<<this->get_timestep_number()
                             <<", fault=0, node="<<j<<", xd="<<xd
                             <<", accepted V="<<V[j]<<", old Theta="<<old_theta
                             <<", committed Theta="<<actual<<", reference="<<expected
                             <<", absolute error="<<std::abs(actual-expected)
                             <<", relative error="<<relative_error;
                      theta_failure=message.str();
                    }
                }
            }
          // Report one worst node collectively before any rank throws. The
          // observer must not lose the offending values in an early MPI abort.
          const double maximum_theta_error=Utilities::MPI::max(theta_error,this->get_mpi_communicator());
          if (!(maximum_theta_error<1e-12))
            {
              const unsigned int rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
              const unsigned int reporting_rank=Utilities::MPI::min(
                theta_error==maximum_theta_error ? rank : numbers::invalid_unsigned_int,
                this->get_mpi_communicator());
              theta_failure=Utilities::MPI::broadcast(this->get_mpi_communicator(),theta_failure,reporting_rank);
              this->get_pcout()<<theta_failure<<std::endl;
              AssertThrow(false,ExcMessage("BP3 retained/updated Theta does not follow the split history cycle.\n"
                                           +theta_failure));
            }
          previous_theta.resize(n);
          for (unsigned int j=0;j<n;++j) previous_theta[j]=fault.get_properties(j)[state];
          BP3Benchmark::verify_coupled_state_commit<dim>(*this);
          last_step=this->get_timestep_number();
          const auto &particles=this->get_phase_field_handler().get_associated_particle_manager();
          const auto stress_position=particles.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          double maximum_stress=0.;
          for (const auto &particle:particles.get_particle_handler())
            for (unsigned int c=0;c<3;++c)
              maximum_stress=std::max(maximum_stress,std::abs(particle.get_properties()[stress_position+c]));
          maximum_stress=Utilities::MPI::max(maximum_stress,this->get_mpi_communicator());
          if (this->get_timestep_number()==0)
            AssertThrow(maximum_stress==0.,ExcMessage("BP3 initial Maxwell perturbation history was changed."));
          if (mature)
            {
              for (unsigned int j=0;j<n;++j)
                AssertThrow(fault.get_properties(j)[C]==0. && Cfield[j]==0.,
                            ExcMessage("Mature BP3 published cohesive resistance."));
              // H remains initialized profile data. Stable IDs let the offline
              // audit verify retention despite particle movement and exchange.
              const auto H=particles.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
              std::ofstream out(this->get_output_directory()+"mature_history_"+
                std::to_string(this->get_timestep_number())+"_rank"+
                std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
              out<<std::setprecision(17)<<"id,H_inert,tau_xx,tau_yy,tau_xy\n";
              for (const auto &p:particles.get_particle_handler())
                out<<p.get_id()<<','<<p.get_properties()[H]<<','<<p.get_properties()[stress_position]<<','
                   <<p.get_properties()[stress_position+1]<<','<<p.get_properties()[stress_position+2]<<'\n';
              this->get_pcout()<<"Mature history verified: C=0, W_coh=0; H is inert profile metadata."<<std::endl;
            }
          if (this->get_pcout().is_active() && profile_due)
            {
              std::ofstream history(this->get_output_directory()+"history_"+std::to_string(this->get_timestep_number())+".csv");
              history<<std::setprecision(17)<<"step,time,Theta_reference_relative_error,committed_particle_stress_max_Pa\n"
                     <<this->get_timestep_number()<<','<<this->get_time()<<','<<theta_error<<','<<maximum_stress<<'\n';
              if (profile_due)
              {
              std::ofstream out(this->get_output_directory()+"fault_"+std::to_string(this->get_timestep_number())+".csv");
              out<<std::setprecision(17)<<"xd,x,y,time,dt,V,Theta,C,Ih,slip,q,F,prescribed,mass_diagonal,mass_upper,tau_bg,delta_tau,tau_total,sigma_n_bg,C_evaluated,friction_traction,weak_residual\n";
              for (unsigned int j=0;j<n;++j)
                {const auto p=fault.vertex(j);const auto data=fault.get_properties(j);const double xd=BP3::down_dip(p[0],p[1]);
                 out<<xd<<','<<p[0]<<','<<p[1]<<','<<this->get_time()<<','<<this->get_timestep()<<','<<V[j]<<','<<data[state]<<','<<data[C]<<','<<data[I]<<','<<slip[j]<<','<<qfield[j]<<','<<Ffield[j]<<','<<prescribed[0][j]<<','<<weak.mass_diagonal[0][j]<<','<<(j+1<n ? weak.mass_off_diagonal[0][j] : 0.)<<','<<background_field[j]<<','<<qfield[j]-background_field[j]<<','<<qfield[j]<<','<<data[background+1]<<','<<Cfield[j]<<','<<friction_field[j]<<','<<weak.values[0][j]<<'\n';}
              }
            }
          if (!BP3Benchmark::work_measure_replay
              && (std::getenv("ASPECT_BP3_SHARED_CLOCK") || std::getenv("ASPECT_BP3_TIMESTEP_SEQUENCE")))
            {
              const auto rank=std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
              std::filesystem::copy_file(this->get_output_directory()+"history_surface_rank"+rank+".csv",
                this->get_output_directory()+"history_surface_step"+std::to_string(this->get_timestep_number())+"_rank"+rank+".csv");
            }
          BP3Benchmark::append_theta_function(*this,V);
          BP3Benchmark::snapshot_initial_cohesion(*this,V);
          if (BP3Benchmark::work_measure_replay && this->get_timestep_number()==1)
            this->get_pcout()<<"BP3 WORK REPLAY FIRST UPDATE PASSED: exact inert H/geometry, Maxwell update, retained FE input, "
                           <<"C=0 and independent Theta reference; continuing with unchanged controllers."<<std::endl;
          BP3Benchmark::audited_accepted_time=this->get_time();
          BP3Benchmark::audited_accepted_step=this->get_timestep_number();
          return {"BP3 accepted state",std::to_string(this->get_timestep_number())};
        }
      private:
        void archive_checkpoint(const std::string &path)
        {
          bool failed=false;
          if (this->get_pcout().is_active())
            try
              {
                for (const auto &label : checkpoint_labels)
                  {
                    const auto destination=this->get_output_directory()+"event_states/"+label;
                    std::filesystem::create_directories(destination);
                    std::filesystem::copy(path,destination,
                      std::filesystem::copy_options::recursive
                      |std::filesystem::copy_options::overwrite_existing);
                  }
              }
            catch (const std::exception &) { failed=true; }
          AssertThrow(!Utilities::MPI::max(static_cast<unsigned int>(failed),this->get_mpi_communicator()),
                      ExcMessage("BP3 completed-checkpoint event archival failed."));
          checkpoint_labels.clear();
        }

        void write_stations(const std::vector<double> &V, const std::vector<double> &shear,
                            const std::vector<double> &normal, const unsigned int state) const
        {
          const auto &fault=this->get_reconstructed_fault_manager().get_faults()[0];
          const double stations[]={0.,2500.,5000.,7500.,10000.,12500.,15000.,17500.,20000.,25000.,30000.,35000.};
          const auto path=this->get_output_directory()+"stations.csv";
          const bool header=!std::filesystem::exists(path);
          std::ofstream out(path,std::ios::app);
          if (header) out<<"step,time,dt,xd,V,Theta,slip,tau_total,sigma_n_total\n";
          out<<std::setprecision(17);
          for (const double xd : stations)
            {
              unsigned int segment=0;
              double xi=0.;
              bool found=false;
              for (; segment<fault.n_cells(); ++segment)
                {
                  const auto a=fault.vertex(segment), b=fault.vertex(segment+1);
                  const double s0=BP3::down_dip(a[0],a[1]), s1=BP3::down_dip(b[0],b[1]);
                  if (xd>=std::min(s0,s1)-1e-7 && xd<=std::max(s0,s1)+1e-7)
                    { xi=std::clamp((xd-s0)/(s1-s0),0.,1.); found=true; break; }
                }
              AssertThrow(found,ExcMessage("Official BP3 station is outside the represented fault."));
              const auto interpolate=[&](const std::vector<double> &v)
              { return (1.-xi)*v[segment]+xi*v[segment+1]; };
              const double theta=(1.-xi)*fault.get_properties(segment)[state]+xi*fault.get_properties(segment+1)[state];
              out<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<xd<<','
                 <<interpolate(V)<<','<<theta<<','<<interpolate(slip)<<','<<interpolate(shear)<<','<<interpolate(normal)<<'\n';
            }
          AssertThrow(out,ExcMessage("Cannot append BP3 station histories."));
        }

        void write_bulk_state() const
        {
          DataOut<dim> data;
          data.attach_dof_handler(this->get_dof_handler());
          std::vector<std::string> names(this->get_fe().n_components());
          for (unsigned int i=0;i<names.size();++i) names[i]="component_"+std::to_string(i);
          for (unsigned int d=0;d<dim;++d) names[this->introspection().component_indices.velocities[d]]="velocity_"+std::to_string(d);
          names[this->introspection().component_indices.pressure]="delta_pressure";
          for (unsigned int c=0;c<this->introspection().n_compositional_fields;++c)
            names[this->introspection().component_indices.compositional_fields[c]]=
              this->introspection().name_for_compositional_index(c);
          data.add_data_vector(this->get_solution(),names);
          data.build_patches(this->get_mapping());
          const std::string base="bulk_"+std::to_string(this->get_timestep_number());
          const unsigned int rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream out(this->get_output_directory()+base+"_"+std::to_string(rank)+".vtu");
          data.write_vtu(out);
          if (this->get_pcout().is_active())
            {
              std::vector<std::string> pieces;
              for (unsigned int r=0;r<Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());++r)
                pieces.push_back(base+"_"+std::to_string(r)+".vtu");
              std::ofstream master(this->get_output_directory()+base+".pvtu");
              data.write_pvtu_record(master,pieces);
            }
        }

        void write_audit_state() const
        {
          const auto tag=std::to_string(this->get_timestep_number())+"_rank"+
            std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv";
          std::ofstream bulk(this->get_output_directory()+"audit_bulk_"+tag);
          const auto &owned=this->get_dof_handler().locally_owned_dofs();
          std::vector<unsigned int> components(owned.n_elements());
          std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (!cell->is_artificial())
              {
                cell->get_dof_indices(indices);
                for (unsigned int j=0;j<indices.size();++j)
                  if (owned.is_element(indices[j]))
                    components[owned.index_within_set(indices[j])]=this->get_fe().system_to_component_index(j).first;
              }
          bulk<<std::setprecision(17)<<"dof,component,value\n";
          for (const auto i:owned) bulk<<i<<','<<components[owned.index_within_set(i)]<<','<<this->get_solution()[i]<<'\n';
          const auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
          std::ofstream out(this->get_output_directory()+"audit_particles_"+tag);
          out<<std::setprecision(17)<<"id,x,y,properties\n";
          for (const auto &particle:particles)
            {
              out<<particle.get_id()<<','<<particle.get_location()[0]<<','<<particle.get_location()[1];
              for (const auto v:particle.get_properties()) out<<','<<v;
              out<<'\n';
            }
        }

        bool audit_states=false;
        BP3::FirstEvent event;
        double last_profile_time=-std::numeric_limits<double>::max();
        double last_bulk_time=-std::numeric_limits<double>::max();
        std::vector<std::string> checkpoint_labels;
        std::vector<double> slip;
        std::vector<double> previous_theta;
        unsigned int last_step=numbers::invalid_unsigned_int;
    };
    ASPECT_REGISTER_POSTPROCESSOR(BP3Output,"reconstructed fault BP3","Accepted BP3 fault profiles and actual weak traction, without reevaluating committed stress.")
  }

  namespace TerminationCriteria
  {
    template<int dim> class BP3ReplayComplete : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override
        {
          const char *path=std::getenv("ASPECT_BP3_TIMESTEP_SEQUENCE");
          AssertThrow(path,ExcMessage("BP3 replay completion requires the saved comparison clock."));
          std::ifstream in(path);
          AssertThrow(in,ExcMessage("Cannot read BP3 replay completion clock."));
          std::string line;std::getline(in,line);
          unsigned int count=0;
          while (std::getline(in,line))
            {
              std::replace(line.begin(),line.end(),',',' ');
              std::istringstream row(line);unsigned int step;double time,dt;
              AssertThrow(row>>step>>time>>dt,ExcMessage("Invalid BP3 replay completion row."));
              AssertThrow(step==count++ && std::isfinite(time) && time>=0. && time>target,
                          ExcMessage("BP3 completion clock must have ordered accepted times."));
              target=time;
            }
          AssertThrow(count>0,ExcMessage("Empty BP3 replay completion clock."));
        }

        bool execute() override
        {
          // The postprocessor publishes this marker only after all accepted
          // history checks. A Newton trial or failed history audit cannot stop.
          if (!BP3::replay_time_reached(BP3Benchmark::audited_accepted_time,target)) return false;
          this->get_pcout()<<std::setprecision(17)<<"BP3 REPLAY COMPLETE: audited accepted step="
            <<BP3Benchmark::audited_accepted_step<<", time="<<BP3Benchmark::audited_accepted_time
            <<", target="<<target<<"; no end-time remainder step."<<std::endl;
          return true;
        }
      private:
        double target=-1.;
    };
    ASPECT_REGISTER_TERMINATION_CRITERION(BP3ReplayComplete,"BP3 replay complete",
      "Stop at the last audited accepted comparison time, allowing only floating-point roundoff in its representation.")

    template<int dim> class BP3FirstEvent : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        bool execute() override
        {
          if (BP3Benchmark::first_event_complete)
            {
              this->get_pcout()<<"BP3 FIRST EVENT COMPLETE: five consecutive accepted states below 1e-3 m/s."<<std::endl;
              return true;
            }
          if (this->get_time()>=1500.*31557600.)
            this->get_pcout()<<(BP3Benchmark::first_event_started
              ? "BP3 SAFETY END: first event started but termination criterion NOT completed."
              : "BP3 SAFETY END: no first seismic event; event criterion NOT satisfied.")<<std::endl;
          return false; // The independent standard end-time criterion enforces the safety limit.
        }
    };
    ASPECT_REGISTER_TERMINATION_CRITERION(BP3FirstEvent,"BP3 first event",
      "Stop after five accepted sub-threshold states following the first max(V)>=1e-3 m/s event.")
  }
}
