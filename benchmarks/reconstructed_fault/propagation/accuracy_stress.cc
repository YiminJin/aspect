// Fixed-mesh native history/transfer probe. Reuses the existing MomentCycle
// friendship; this standalone plugin must not be loaded with the BP5 plugin.
#include "recoverable_feedback.cc"
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/reconstructed_fault/manager.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
template<int dim> class MomentCycle : public Interface<dim>, public SimulatorAccess<dim> {
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 using BV=LinearAlgebra::BlockVector;
 // Three Q1-representable stress fields, including a mesh-scale nonconstant one.
 static SymmetricTensor<2,dim> pattern(unsigned int k,const Point<dim>&p) {
  SymmetricTensor<2,dim> s;
  if(k==0) {s[0][0]=700;s[1][1]=-300;s[0][1]=200;}
  if(k==1) {s[0][0]=700+400*p[0];s[1][1]=-300+200*p[1];s[0][1]=200+600*p[0]*p[1];}
  if(k==2) {
   auto wave=[](double x) {unsigned int i=std::min(31u,static_cast<unsigned int>(32*x));double t=32*x-i;
    return (1-t)*std::sin(.73*i)+t*std::sin(.73*(i+1));};
   s[0][0]=700+400*wave(p[0]);s[1][1]=-300+200*wave(p[1]);s[0][1]=200+600*wave(p[0])*wave(p[1]);
  }
  return s;
 }
public:
 std::pair<std::string,std::string> execute(TableHandler &) override {
  AssertThrow(dim==2 && this->get_timestep_number()==0 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
              ExcMessage("Initialization-only, serial fixed-mesh stress probe."));
  auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
  auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
  auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
  const auto sp=this->get_particle_manager(0).get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
  const auto saved_eta=F::viscosity_state(model);const BV solution(sim.solution),old(sim.old_solution),older(sim.old_old_solution),linear(sim.current_linearization_point);
  const auto saved_dt=sim.time_step,saved_time=sim.time;const auto saved_step=sim.timestep_number;
  std::map<types::particle_index,std::vector<double>> saved;
  for(const auto &p:particles)saved[p.get_id()]={p.get_properties().begin(),p.get_properties().end()};
  auto &rates=const_cast<std::vector<double>&>(this->get_reconstructed_fault_manager().get_slip_rate(0));
  const auto saved_rates=rates;
  const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
  std::vector<std::vector<double>> fault_saved;for(unsigned int j=0;j<fault.n_vertices();++j)fault_saved.emplace_back(fault.get_properties(j).begin(),fault.get_properties(j).end());
  auto restore=[&](){
   F::restore_viscosity(model,saved_eta);sim.solution=solution;sim.old_solution=old;sim.old_old_solution=older;sim.current_linearization_point=linear;
   sim.time_step=saved_dt;sim.time=saved_time;sim.timestep_number=saved_step;rates=saved_rates;
   for(auto &p:particles)std::copy(saved.at(p.get_id()).begin(),saved.at(p.get_id()).end(),p.get_properties().begin());
   for(unsigned int j=0;j<fault.n_vertices();++j)std::copy(fault_saved[j].begin(),fault_saved[j].end(),this->get_reconstructed_fault_manager().get_fault(0).get_properties(j).begin());
  };
  std::ofstream out(this->get_output_directory()+"bulk-stress-native.csv");
  out<<std::setprecision(17)<<"pattern,eta,dt,step,time,particle_error_Pa,FE_representation_relative,FE_additional_drift_Pa,particle_energy,FE_energy,analytic_energy,representation_energy_change\n";
  const auto &intro=this->introspection();const auto &quad=intro.quadratures.velocities;
  FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
  std::array<unsigned int,3> fields;
  for(const auto &e:this->get_parameters().mapped_particle_properties)if(e.second.first=="maxwell stress")fields[e.second.second]=e.first;
  const auto transfer=[&](){
   particles.exchange_ghost_particles();sim.interpolate_particle_properties({AdvectionField::composition(fields[0]),AdvectionField::composition(fields[1]),AdvectionField::composition(fields[2])});
   BV owned(intro.index_sets.system_partitioning,this->get_mpi_communicator());owned=sim.solution;
   this->get_current_constraints().distribute(owned);sim.solution=owned;
  };
  try {
   for(unsigned int k=0;k<3;++k)for(double eta:{1e8,1e30})for(double dt:{.125,.0625,.03125}) {
    restore();F::set_uniform_viscosity(model,eta);
    // Prescribe the zero-strain physical path; the native terminal updater,
    // not a mechanical equilibrium solve, consumes this diagnostic state.
    sim.solution.block(intro.block_indices.velocities)=0.;sim.old_solution=sim.solution;
    std::fill(rates.begin(),rates.end(),0.);
    for(auto &p:particles) {auto stress=pattern(k,p.get_location());for(unsigned int c=0;c<3;++c)p.get_properties()[sp+c]=stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)];}
    sim.timestep_number=1;sim.time_step=dt;transfer();const BV initial_FE(sim.solution);
    for(unsigned int step=0;step<=std::lround(1/dt);++step) {
     const double time=step*dt,decay=std::exp(-1e6*time/eta);sim.time=time;sim.timestep_number=step+1;
     if(step)model.commit_reconstructed_fault_mechanical_history(sim.solution);
     const auto hp=this->get_particle_manager(0).get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
     for(const auto &p:particles)AssertThrow(p.get_properties()[hp]==saved.at(p.get_id())[hp],ExcMessage("Zero-input stress probe changed H."));
     // Native publication must not consume the FE stress representation.
     transfer();double particle_error=0,particle_E=0,exact_E=0,FE_E=0,err2=0,scale2=0,drift=0;
     for(const auto &p:particles) {
      const auto stress=decay*pattern(k,p.get_location());SymmetricTensor<2,dim> actual;
      for(unsigned int c=0;c<3;++c)actual[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=p.get_properties()[sp+c];
      particle_error=std::max(particle_error,(actual-stress).norm());
      const auto domain=this->get_phase_field_handler().get_associated_particle_manager().get_particle_domain_handler().get_particle_domain(p.get_local_index());
      particle_E+=domain.volume()*(actual*actual)/(4e6);
     }
     for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned()) {
      fe.reinit(cell);std::array<std::vector<double>,3> values,initial;
      for(unsigned int c=0;c<3;++c) {values[c].resize(quad.size());initial[c].resize(quad.size());
       fe[intro.extractors.compositional_fields[fields[c]]].get_function_values(sim.solution,values[c]);
       fe[intro.extractors.compositional_fields[fields[c]]].get_function_values(initial_FE,initial[c]);}
      for(unsigned int j=0;j<quad.size();++j) {
       SymmetricTensor<2,dim> actual,baseline;for(unsigned int c=0;c<3;++c) {auto idx=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);actual[idx]=values[c][j];baseline[idx]=initial[c][j];}
       const auto exact=decay*pattern(k,fe.quadrature_point(j));const double w=fe.JxW(j);
       err2+=w*std::pow((actual-exact).norm(),2);scale2+=w*(exact*exact);drift=std::max(drift,(actual-decay*baseline).norm());
       FE_E+=w*(actual*actual)/(4e6);exact_E+=w*(exact*exact)/(4e6);
      }
     }
     AssertThrow(particle_error<2e-10 && drift<2e-10,ExcMessage("Native zero-input bulk history accumulates additional drift."));
     out<<k<<','<<eta<<','<<dt<<','<<step<<','<<time<<','<<particle_error<<','<<std::sqrt(err2/scale2)<<','<<drift<<','<<particle_E<<','<<FE_E<<','<<exact_E<<','<<FE_E-exact_E<<'\n';
    }
   }
   restore();
   // Exercise exceptional snapshot restoration without accepting a physical step.
   try {sim.solution.block(intro.block_indices.velocities)=0.;particles.begin()->get_properties()[sp]=-123.;throw std::runtime_error("restore probe");}
   catch(const std::runtime_error&) {restore();}
  } catch(...) {restore();throw;}
  for(const auto &p:particles)AssertThrow(std::equal(saved.at(p.get_id()).begin(),saved.at(p.get_id()).end(),p.get_properties().begin()),ExcMessage("Stress probe failed to restore particle histories."));
  BV difference(intro.index_sets.system_partitioning,this->get_mpi_communicator());
  for(const auto &pair:{std::make_pair(&sim.solution,&solution),std::make_pair(&sim.old_solution,&old),std::make_pair(&sim.old_old_solution,&older),std::make_pair(&sim.current_linearization_point,&linear)}) {
   difference=*pair.first;difference-=*pair.second;
   AssertThrow(difference.linfty_norm()==0.,ExcMessage("Stress probe failed to restore FE states."));
  }
  return {"Native zero-input bulk history:","particle relaxation and transfer drift tested"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"P2 accuracy stress","Native zero-input fixed-phase history and particle-to-FE transfer.")
}}
