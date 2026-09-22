// Opt-in committing disturbance branches of an accepted checkpoint. All
// mechanics and stress updates remain production operations. Never loaded by
// the maintained BP3 library: only the separate fault_disturbance test target.
#include <aspect/time_stepping/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/adiabatic_conditions/compute_profile.h>
#include "../../../tests/phase_field_fault_test_access.h"

namespace aspect
{
  namespace BP3Disturbance
  {
    inline double amplitude() { return std::stod(std::getenv("ASPECT_DISTURBANCE_EPS")); }
    inline double wavelength()
    {
      static const double value=[]()
      {
        const char *input=std::getenv("ASPECT_DISTURBANCE_WAVELENGTH");
        const double length=input?std::stod(input):200.;
        AssertThrow(std::isfinite(length) && length>0.,ExcMessage("Disturbance wavelength must be positive and finite."));
        return length;
      }();
      return value;
    }
    inline double shape(const double xd)
    {
      if (std::abs(xd-16500.)>=1500.) return 0.;
      return std::pow(std::cos(numbers::PI*(xd-16500.)/3000.),2)
             *std::cos(2*numbers::PI*(xd-16500.)/wavelength());
    }
    inline void perturb_audit(std::vector<double> &theta,const std::vector<Point<2>> &geometry)
    {
      for (unsigned int j=0;j<theta.size();++j)
        theta[j]*=std::exp(amplitude()*shape(BP3::down_dip(geometry[j][0],geometry[j][1])));
    }
    inline std::vector<double> incoming;
    inline std::vector<double> reference_V;
    inline std::map<std::array<double,2>,double> reference_sigma;
    inline std::vector<double> physical_normal_load;
    inline bool first=true;
    inline bool state_control(){const char *c=std::getenv("ASPECT_DISTURBANCE_CONTROL");return c && std::string(c)=="state";}
    inline bool normal_control(){const char *c=std::getenv("ASPECT_DISTURBANCE_CONTROL");return c && std::string(c)=="normal";}
    inline double aging_rate(unsigned int j,double own){return state_control()?reference_V.at(j):own;}
    inline ReconstructedFaultSurfaceResidual physical_weak(const ReconstructedFaultSurfaceResidual &weak)
    {
      auto result=weak;
      if(normal_control())result.normal_traction[0]=physical_normal_load;
      return result;
    }

    template <int dim> void enable_normal_control(const SimulatorAccess<dim> &sim,bool temperature,unsigned int,const SolverControl &)
    {
      if(!temperature || !normal_control())return;
      auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
      // BP3 has just reattached its true-pressure work-measure/source mapping.
      // Only friction now uses the prescribed, evolving reference QP traction;
      // pressure is still a solved bulk unknown. Existing K/G semantics apply.
      MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::prescribed_friction_normal(model,true);
    }

    template <int dim> void begin(const SimulatorAccess<dim> &sim)
    {
      auto &manager=sim.get_reconstructed_fault_manager();auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      if (first)
        {
          AssertThrow(sim.get_timestep_number()==12,ExcMessage("Disturbance must branch accepted step 11."));
          for (unsigned int j=0;j<fault.n_vertices();++j)
            fault.get_properties(j)[state]*=std::exp(amplitude()*shape(BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1])));
          first=false;
        }
      incoming.resize(fault.n_vertices());
      for (unsigned int j=0;j<incoming.size();++j) incoming[j]=fault.get_properties(j)[state];
      if(state_control() || normal_control())
        {
          const std::string root=std::getenv("ASPECT_DISTURBANCE_REFERENCE");
          const unsigned int step=sim.get_timestep_number();
          std::ifstream nodes(root+"/disturbance_nodes_"+std::to_string(step)+".csv");
          AssertThrow(nodes,ExcMessage("Missing matched reference accepted state."));
          std::string line;std::getline(nodes,line);reference_V.clear();
          while(std::getline(nodes,line))
            {
              std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
              unsigned int j;double xd,t,dt,v;AssertThrow(row>>j>>xd>>t>>dt>>v,ExcMessage("Invalid reference node record."));
              AssertThrow(j==reference_V.size() && std::abs(t-sim.get_time())<1e-5 && dt==sim.get_timestep(),ExcMessage("Reference clock mismatch."));
              reference_V.push_back(v);
            }
          AssertThrow(reference_V.size()==fault.n_vertices(),ExcMessage("Reference fault grid mismatch."));
          if(normal_control())
            {
              auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
              MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::prescribed_friction_normal(model,false);
              std::map<std::pair<unsigned int,unsigned int>,double> samples;
              std::ifstream data(root+"/disturbance_qp_"+std::to_string(step)+"_rank"+std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".bin",std::ios::binary);
              AssertThrow(data,ExcMessage("Missing reference QP normal traction."));double r[12];
              while(data.read(reinterpret_cast<char*>(r),sizeof(r)))
                AssertThrow(samples.emplace(std::make_pair(static_cast<unsigned int>(r[0]),static_cast<unsigned int>(r[1])),r[8]).second,ExcMessage("Duplicate reference QP."));
              const auto &quad=sim.introspection().quadratures.velocities;
              FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,update_quadrature_points);
              reference_sigma.clear();
              for(const auto &cell:sim.get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
                {
                  fe.reinit(cell);
                  for(unsigned int q=0;q<quad.size();++q)
                    {
                      const auto it=samples.find({cell->active_cell_index(),q});
                      if(it!=samples.end())
                        {const auto p=fe.quadrature_point(q);AssertThrow(reference_sigma.emplace(std::array<double,2>{p[0],p[1]},it->second).second,ExcMessage("Duplicate physical reference QP."));}
                    }
                }
              AssertThrow(reference_sigma.size()==samples.size(),ExcMessage("Reference QP ownership changed."));
            }
        }
    }

    template <int dim> void accepted(const SimulatorAccess<dim> &sim)
    {
      const auto &intro=sim.introspection();const auto comm=sim.get_mpi_communicator();
      const unsigned int rank=Utilities::MPI::this_mpi_process(comm),step=sim.get_timestep_number();
      auto &manager=sim.get_reconstructed_fault_manager();auto &fault=manager.get_fault(0);
      const auto &V=manager.get_timestep_committed_slip_rate(0);
      const auto &weak=sim.get_reconstructed_fault_surface_system().get_linearization_residual();
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
      if(state_control())
        {
          // Verify the ordinary own-rate candidate, then replace it by the
          // control candidate evaluated from the SAME immutable incoming state.
          // No observer consumes the interim candidate, and aging is not applied
          // twice. Bulk/particle stress and every other history remain untouched.
          const auto &friction=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::fault_friction(model);
          std::vector<double> candidate(V.size());
          for(unsigned int j=0;j<V.size();++j)
            {
              const double own=static_cast<double>(BP3::aging_state_reference(V[j],incoming[j],sim.get_timestep(),model.characteristic_fault_slip_distance()));
              AssertThrow(std::abs(fault.get_properties(j)[state]/own-1)<1e-12,ExcMessage("Production own-rate aging candidate failed before state control."));
              candidate[j]=friction.update_state(reference_V[j],incoming[j],sim.get_timestep());
              AssertThrow(candidate[j]>0. && std::isfinite(candidate[j]),ExcMessage("Invalid controlled state candidate."));
            }
          for(unsigned int j=0;j<V.size();++j)fault.get_properties(j)[state]=candidate[j];
        }
      const double ratio=*std::max_element(V.begin(),V.end())*sim.get_timestep()/model.characteristic_fault_slip_distance();
      AssertThrow(ratio<=std::stod(std::getenv("ASPECT_DISTURBANCE_RATIO_LIMIT")),
                  ExcMessage("Accepted V dt/Dc exceeds diagnostic limit; preserve failure."));
      if (!rank)
        {
          std::ofstream out(sim.get_output_directory()+"disturbance_nodes_"+std::to_string(step)+".csv");
          out<<std::setprecision(17)<<"node,xd,time,dt,V,Theta_in,Theta_out,w,mass_diagonal,mass_upper,q_load,normal_load,residual_load\n";
          for(unsigned int j=0;j<V.size();++j)
            out<<j<<','<<BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1])<<','<<sim.get_time()<<','<<sim.get_timestep()<<','<<V[j]<<','<<incoming[j]<<','<<fault.get_properties(j)[state]<<','<<shape(BP3::down_dip(fault.vertex(j)[0],fault.vertex(j)[1]))<<','<<weak.mass_diagonal[0][j]<<','<<(j+1<V.size()?weak.mass_off_diagonal[0][j]:0.)<<','<<weak.shear_traction[0][j]<<','<<weak.normal_traction[0][j]<<','<<weak.values[0][j]<<'\n';
        }
      // Evaluate current mechanics with its actual incoming state and frozen
      // working FE stress, not the newly committed particle stress/state.
      // Restore the substituted surface state even if any diagnostic throws.
      struct Restore
      {
        ReconstructedFault<dim> &fault;unsigned int position;std::vector<double> values;
        ~Restore(){for(unsigned int j=0;j<values.size();++j)fault.get_properties(j)[position]=values[j];}
      } restore{fault,state,{}};
      for(unsigned int j=0;j<V.size();++j)
        {restore.values.push_back(fault.get_properties(j)[state]);fault.get_properties(j)[state]=incoming[j];}
      const auto &quad=intro.quadratures.velocities;
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
      const unsigned int nq=quad.size();
      std::vector<double> phi(nq),T(nq),p(nq);std::vector<SymmetricTensor<2,dim>> eps(nq);
      std::vector<std::vector<double>> comps(intro.n_compositional_fields,std::vector<double>(nq));
      std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> stress_fields;
      stress_fields.fill(numbers::invalid_unsigned_int);
      const auto &mapping=sim.get_parameters().mapped_particle_properties;
      for(const auto &m:mapping)
        if(m.second.first=="maxwell stress")stress_fields[m.second.second]=m.first;
      std::ofstream out(sim.get_output_directory()+"disturbance_qp_"+std::to_string(step)+"_rank"+std::to_string(rank)+".bin",std::ios::binary);
      // Fixed 12-double records; rank-local active cell index plus QP index.
      // Cell active indices are stable for these fixed-mesh same-rank branches.
      std::vector<double> residual(V.size(),0.);
      physical_normal_load.assign(V.size(),0.);
      const auto &working=sim.get_current_linearization_point();
      for(const auto &cell:sim.get_dof_handler().active_cell_iterators()) if(cell->is_locally_owned())
        {
          fe.reinit(cell);const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
          fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(working,phi);
          fe[intro.extractors.temperature].get_function_values(working,T);
          fe[intro.extractors.pressure].get_function_values(working,p);
          fe[intro.extractors.velocities].get_function_symmetric_gradients(working,eps);
          for(unsigned int c=0;c<comps.size();++c)fe[intro.extractors.compositional_fields[c]].get_function_values(working,comps[c]);
          for(unsigned int q=0;q<nq;++q)if(associations[q].active)
            {
              const auto &a=associations[q];const unsigned int j=a.segment_index;
              typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultPointInputs in;
              in.fault_index=0;in.segment_index=j;in.xi=a.xi;in.position=fe.quadrature_point(q);
              in.slip_rate=(1-a.xi)*V[j]+a.xi*V[j+1];in.phase_field=phi[q];in.previous_phase_field=phi[q];
              in.temperature=T[q];in.dynamic_pressure=p[q];in.strain_rate=eps[q];
              in.slip_tensor=symmetrize(outer_product(a.tangent,a.normal));in.normal_tensor=symmetrize(outer_product(a.normal,a.normal));
              for(unsigned int c=0;c<stress_fields.size();++c)in.old_maxwell_stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=comps[stress_fields[c]][q];
              std::vector<double> chemical;for(const auto c:intro.chemical_composition_field_indices())chemical.push_back(comps[c][q]);
              in.bulk_material_fractions=MaterialModel::MaterialUtilities::compute_composition_fractions(chemical);
              const auto r=model.evaluate_reconstructed_fault_point(in);const double weight=fe.JxW(q)*r.localization_factor;
              if(weight==0.)continue;
              double actual_sigma=r.normal_traction;
              if(normal_control())
                {
                  std::vector<double> all(comps.size());for(unsigned int c=0;c<all.size();++c)all[c]=comps[c][q];
                  const auto stress=2*r.kappa*(in.strain_rate-r.localization_factor*in.slip_rate*in.slip_tensor)
                                    +model.evaluate_frozen_maxwell_stress(in.temperature,all,in.old_maxwell_stress);
                  actual_sigma=r.background_normal_traction+in.dynamic_pressure-stress*in.normal_tensor;
                  AssertThrow(std::abs(r.normal_traction-reference_sigma.at({in.position[0],in.position[1]}))<1e-7,ExcMessage("Friction normal differs from the matched reference sample beyond roundoff."));
                  // The normal diagnostic must retain shear response but have
                  // zero pressure/normal-strain feedback through friction.
                  auto trial=in;trial.dynamic_pressure+=100.;
                  trial.strain_rate+=1e-12*in.normal_tensor;
                  const auto test=model.evaluate_reconstructed_fault_point(trial);
                  AssertThrow(std::abs(test.residual_density-r.residual_density)<1e-6,ExcMessage("Reference-normal friction retained bulk normal feedback."));
                }
              const double theta=(1-a.xi)*incoming[j]+a.xi*incoming[j+1];
              const double record[12]={static_cast<double>(cell->active_cell_index()),static_cast<double>(q),static_cast<double>(j),a.xi,weight,in.slip_rate,theta,r.shear_traction,actual_sigma,r.friction_coefficient,r.damping_traction,r.residual_density};
              out.write(reinterpret_cast<const char*>(record),sizeof(record));
              residual[j]+=weight*(1-a.xi)*r.residual_density;residual[j+1]+=weight*a.xi*r.residual_density;
              physical_normal_load[j]+=weight*(1-a.xi)*actual_sigma;physical_normal_load[j+1]+=weight*a.xi*actual_sigma;
            }
        }
      const auto local=residual;Utilities::MPI::sum(local,comm,residual);
      const auto local_normal=physical_normal_load;Utilities::MPI::sum(local_normal,comm,physical_normal_load);
      double error=0.;for(unsigned int j=0;j<V.size();++j)
        {
          const double rowmass=weak.mass_diagonal[0][j]+(j?weak.mass_off_diagonal[0][j-1]:0.)+(j+1<V.size()?weak.mass_off_diagonal[0][j]:0.);
          error=std::max(error,std::abs(residual[j]-weak.values[0][j])/rowmass);
        }
      AssertThrow(error<1e-5,ExcMessage("Diagnostic does not reproduce incoming-state production weak residual."));
      sim.get_pcout()<<std::setprecision(12)<<"Disturbance audit: step="<<step<<", Vdt/Dc="<<ratio<<", incoming-state weak error="<<error<<" Pa"<<std::endl;
    }
  }
  namespace AdiabaticConditions
  {
    // An adapter for the existing prescribed-normal constitutive path, not an
    // altered bulk pressure or gauge. Reference samples already include the
    // 50 MPa background; subtract it here because production adds it once.
    template <int dim> class DisturbanceNormal : public Interface<dim>
    {
      public:
        void parse_parameters(ParameterHandler &prm) override
        {background.initialize_simulator(this->get_simulator());background.parse_parameters(prm);}
        void initialize() override{background.initialize();}
        void update() override{background.update();}
        bool is_initialized() const override{return background.is_initialized();}
        double temperature(const Point<dim>&p) const override{return background.temperature(p);}
        double density(const Point<dim>&p) const override{return background.density(p);}
        double density_derivative(const Point<dim>&p) const override{return background.density_derivative(p);}
        double pressure(const Point<dim> &p) const override
        {
          const auto it=BP3Disturbance::reference_sigma.find({p[0],p[1]});
          // Zero-localization QPs have no work contribution and were not saved.
          return it==BP3Disturbance::reference_sigma.end()?background.pressure(p):it->second-BP3::sigma0;
        }
      private:
        ComputeProfile<dim> background;
    };
    ASPECT_REGISTER_ADIABATIC_CONDITIONS_MODEL(DisturbanceNormal,"disturbance normal reference","Benchmark-only matched QP normal traction for friction; bulk pressure remains solved.")
  }
  namespace TimeStepping
  {
    template <int dim> class DisturbanceClock : public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        double execute() override {return std::stod(std::getenv("ASPECT_DISTURBANCE_DT"));}
        std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override
        {
          AssertThrow(std::abs(info.next_time_step_size/execute()-1)<1e-10,
                      ExcMessage("Ordinary controller requires schedule cutback; stop matched comparison."));
          return {Reaction::advance,std::numeric_limits<double>::max()};
        }
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(DisturbanceClock,"disturbance clock","Matched short trajectory cap; production guards remain active.")
  }
}
