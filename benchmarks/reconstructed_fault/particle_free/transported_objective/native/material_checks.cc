#include <aspect/material_model/phase_field_fault/material_history_internal.h>
#include "../../../../../tests/phase_field_fault_test_access.h"
/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_history.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <aspect/particle/integrator/interface.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/phase_field.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Particle { namespace Integrator
  {
    template<int dim> class MaterialSamples : public Interface<dim>
    {
    public:
      static constexpr unsigned n_integrator_properties=0;
      void local_integrate_step(const typename ParticleHandler<dim>::particle_iterator &,
                               const typename ParticleHandler<dim>::particle_iterator &,
                               const std::vector<Tensor<1,dim>> &,
                               const std::vector<Tensor<1,dim>> &,double) override {}
      std::array<bool,3> required_solution_vectors() const override {return {{false,false,false}};}
    };
    ASPECT_REGISTER_PARTICLE_INTEGRATOR(MaterialSamples,"material diagnostic samples",
      "Fixed diagnostic sampling points. Physical history is material-owned DG, not these particles.")
  }}

  template<int dim> void connect_material_fixture(SimulatorSignals<dim> &signals)
  {
    signals.post_simulator_initialization.connect([](const SimulatorAccess<dim> &sim)
      {sim.get_reconstructed_fault_manager().register_property("material fixture tractions",2);});
    signals.post_advection_solver.connect([](const SimulatorAccess<dim> &sim,bool temperature,unsigned,const SolverControl &)
      {
        if(!temperature)return;
        auto &manager=sim.get_reconstructed_fault_manager();
        auto &model=const_cast<MaterialModel::PhaseFieldFault<dim> &>(
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
        AssertThrow(model.uses_transported_mature_history(),ExcMessage("This fixture requires the complete material selector."));
        const auto traction=manager.get_property_index("material fixture tractions");
        sim.get_reconstructed_fault_surface_system().enable_bulk_work_measure();
        if(sim.get_timestep_number()==0 && !sim.get_parameters().resume_computation)
          {
            auto &fault=manager.get_fault(0);
            const auto t=manager.get_property_information()[traction].position;
            const auto s=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
            for(unsigned i=0;i<fault.n_vertices();++i)
              {
                auto p=fault.get_properties(i);
                p[t]=2.7+.15*std::cos(3*fault.vertex(i)[0]);p[t+1]=5.;
                p[s]=10.+2.*std::sin(2*fault.vertex(i)[0]);
              }
            manager.initialize_slip_rate(0,std::vector<double>(fault.n_vertices(),.001));
          }
        model.set_reconstructed_fault_background_traction_property(traction);
        model.prepare_reconstructed_fault_mechanical_solve();
      });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_material_fixture<2>,connect_material_fixture<3>)

  namespace TimeStepping
  {
    template<int dim> class MaterialRejection : public Interface<dim>,public SimulatorAccess<dim>
    {
      bool rejected=false;
    public:
      double execute() override {return std::numeric_limits<double>::max();}
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        if(this->get_timestep_number()==2 && !rejected)
          {rejected=true;return {Reaction::repeat_step,.5*this->get_timestep()};}
        return {Reaction::advance,std::numeric_limits<double>::max()};
      }
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(MaterialRejection,"material transaction rejection",
      "Reject the second completed material trial once before publication.")
  }

  namespace Postprocess
  {
    // Same existing diagnostic friend as the other Stage C checker; these two
    // plugin libraries are alternative fixtures and are never loaded together.
    template<int dim> class MomentCycle : public Interface<dim>,public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      unsigned last_entry=numbers::invalid_unsigned_int,retries=0;
      std::string entry_record;
#ifdef MATERIAL_ISOLATION_CHECKS
#include "boundary_profile_checks.h"
#endif
      #include "true_stress_moving_checks.h"
      #include "true_stress_checks.h"
      std::string snapshot() const
      {
        const auto &m=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
        if(!m.fe_strain_state)return "uninitialized";
        const auto &s=*m.fe_strain_state;
        std::ostringstream o;o<<std::setprecision(17)<<s.accepted_publications<<' '<<s.geometry_publications<<' '<<s.geometry_version;
        for(auto v:MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::dg_history_values(m))o<<' '<<v;
        for(auto v:s.Z)o<<' '<<v;for(auto v:s.I)o<<' '<<v;
        const auto block=this->introspection().variable("phase_field").block_index;
        for(auto i:s.displacement_history.block(block).locally_owned_elements())o<<' '<<s.displacement_history.block(block)[i];
        const auto &f=this->get_reconstructed_fault_manager().get_fault(0);
        for(unsigned i=0;i<f.n_vertices();++i)
          {o<<' '<<f.vertex(i);for(auto p:f.get_properties(i))o<<' '<<p;}
        for(auto v:this->get_reconstructed_fault_manager().get_slip_rate(0))o<<' '<<v;
        return o.str();
      }
    public:
      void initialize() override
      {
        this->get_signals().start_timestep.connect([this](const auto &)
          {
            const auto now=snapshot();
            if(last_entry==this->get_timestep_number())
              {
                AssertThrow(Utilities::MPI::min(unsigned(now==entry_record),this->get_mpi_communicator()),
                            ExcMessage("Rejected material trial changed accepted complete state."));
                ++retries;
              }
            entry_record=now;last_entry=this->get_timestep_number();
          });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto &m=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
        const auto &s=*m.fe_strain_state;
        AssertThrow(!m.dg_material_trial,ExcMessage("Material trial remained live after acceptance."));
        const auto step=this->get_timestep_number();
#ifdef MATERIAL_ISOLATION_CHECKS
        if(step==0)boundary_profile_checks();
#endif
        if(step==0 && m.material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_CHECKS"))true_stress_checks();
        AssertThrow(s.accepted_publications==step && s.geometry_publications==step,
                    ExcMessage("Material histories/geometry did not publish exactly once per physical step."));
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        const auto stem=this->get_output_directory()+"material-"+std::to_string(step);
        {
          std::ofstream storage(stem+"-storage-rank"+std::to_string(rank)+".csv");
          const auto tensors=s.dg_history->residual ? 2u : 1u;
          storage<<"owned_tensor_coefficients,relevant_tensor_coefficients,surface_scalar_entries,tensor_vector_bytes,rank_peak_rss_kib,point_sample_cells,source_mass_cache_entries,accepted_stress_sample_entries,mechanical_cache_present,trial_anchor_present,trial_reaction_present\n";
          std::size_t bytes=0;for(const auto &v:s.dg_history->coefficients)bytes+=v.memory_consumption();
          if(s.dg_history->residual)for(const auto &v:s.dg_history->residual->coefficients)bytes+=v.memory_consumption();
          std::ifstream proc("/proc/self/status");std::string line;unsigned long peak=0;
          while(std::getline(proc,line))if(line.find("VmHWM:")==0){std::istringstream parser(line.substr(6));parser>>peak;}
          storage<<3*tensors*s.dg_history->space->dofs.locally_owned_dofs().n_elements()<<','
                 <<3*tensors*s.dg_history->space->relevant.n_elements()<<','<<s.Z.size()+s.I.size()<<','<<bytes<<','<<peak<<','<<s.dg_history->space->point_samples.size()<<','
                 <<s.dg_history->space->source_geometry.size()<<','<<s.stress_samples.size()<<','
                 <<bool(s.mechanical_samples)<<','<<bool(s.objective_anchor)<<','<<bool(m.dg_material_reaction)<<'\n';
        }
        std::ofstream fields(stem+"-rank"+std::to_string(rank)+".csv");
        fields<<std::setprecision(17)<<"cell,x,y,q,phi,vx,vy,Txx,Tyy,Txy,Rxx,Ryy,Rxy,tauxx,tauyy,tauxy,Lxx,Lxy,Lyx,Lyy,eta";
#ifdef MATERIAL_ENERGY_CHECKS
        fields<<",pressure";
#endif
        fields<<'\n';
        const QGauss<dim> rule(3);
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_gradients|update_quadrature_points);
        std::vector<double> phase(rule.size());std::vector<Tensor<1,dim>> velocity(rule.size());
        std::vector<Tensor<2,dim>> gradients(rule.size());
#ifdef MATERIAL_ENERGY_CHECKS
        std::vector<double> pressure(rule.size());
#endif
        for(const auto &entry:s.cells)
          {
            fe.reinit(entry.second);
            fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.displacement_history,phase);
            fe[this->introspection().extractors.velocities].get_function_values(this->get_solution(),velocity);
            fe[this->introspection().extractors.velocities].get_function_gradients(this->get_solution(),gradients);
#ifdef MATERIAL_ENERGY_CHECKS
            fe[this->introspection().extractors.pressure].get_function_values(this->get_solution(),pressure);
#endif
            const auto T=m.evaluate_history_tensor(*s.dg_history,entry.first,rule);
            const auto R=s.dg_history->residual ? m.evaluate_history_tensor(*s.dg_history->residual,entry.first,rule) : std::vector<SymmetricTensor<2,dim>>(rule.size());
            for(unsigned q=0;q<rule.size();++q)
              {
                const auto x=fe.quadrature_point(q);
                const auto tau=m.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,entry.first,x);
                if(m.material_direct_stress && std::getenv("ASPECT_TRUE_STRESS_CHECKS"))
                  {
                    auto diagnostic_Z=s.Z;for(auto &z:diagnostic_Z)z+=1.;
                    AssertThrow(tau==m.evaluate_dg_history_stress(*s.dg_history,diagnostic_Z,s.I,entry.first,x),
                                ExcMessage("Diagnostic Z changed direct physical stress."));
                  }
                fields<<entry.first.to_string()<<','<<x[0]<<','<<x[1]<<','<<q<<','<<phase[q]<<','<<velocity[q][0]<<','<<velocity[q][1];
                for(const auto &a:{T[q],R[q],tau})fields<<','<<a[0][0]<<','<<a[1][1]<<','<<a[0][1];
                fields<<','<<gradients[q][0][0]<<','<<gradients[q][0][1]<<','<<gradients[q][1][0]<<','<<gradients[q][1][1]<<','<<m.material_viscosity(x,s.eta);
#ifdef MATERIAL_ENERGY_CHECKS
                fields<<','<<pressure[q];
#endif
                fields<<'\n';
              }
          }
        if(rank==0)
          {
            const auto &manager=this->get_reconstructed_fault_manager();const auto &f=manager.get_fault(0);
            const auto age=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
            std::ofstream surf(stem+"-surface.csv");surf<<std::setprecision(17)<<"i,x,y,Theta_s,Z_Pa_m,I_m,V_m_s,width_m,eta\n";
            for(unsigned i=0;i<f.n_vertices();++i)surf<<i<<','<<f.vertex(i)[0]<<','<<f.vertex(i)[1]<<','<<f.get_properties(i)[age]<<','<<s.Z[i]<<','<<s.I[i]<<','<<manager.get_slip_rate(0)[i]<<','<<manager.get_projection_half_widths()[0][i]<<','<<m.material_viscosity(f.vertex(i),s.eta)<<'\n';
            std::ofstream metadata(this->get_output_directory()+"material-model.csv");
            const double mphi=(1./this->get_phase_field_handler().energetic_degradation(std::vector<double>{1.},.5)-1.)/3.;
            metadata<<std::setprecision(17)<<"G,reference_eta,m,sense,Dc,direct\n"<<s.G<<','<<s.eta<<','<<mphi<<','<<manager.get_shear_sense(0)<<",.008,"<<m.material_direct_stress<<'\n';
            const auto summary_path=this->get_output_directory()+"material-summary.csv";
            const bool header=step==0 || !std::ifstream(summary_path).good();
            std::ofstream summary(summary_path,step==0 ? std::ios::out : std::ios::app);
            if(header)summary<<"step,time,dt,history_publications,geometry_publications,retries\n";
            summary<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<s.accepted_publications<<','<<s.geometry_publications<<','<<retries<<'\n';
          }
        return {"Material transaction","complete endpoint recorded"};
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"material transaction checks","Bounded coupled mature material history/geometry lifecycle fixture.")
  }
}
