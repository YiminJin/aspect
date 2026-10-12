#include "coupling_feedback.cc"
#include "../../../tests/phase_field_test_access.h"
#include <aspect/phase_field.h>
#include <aspect/particle/particle_domain.h>
#include <fstream>
#include <iomanip>
namespace aspect { namespace Postprocess {
template<int dim> class CouplingConsistency : public Interface<dim>, public SimulatorAccess<dim> {
public:
 std::pair<std::string,std::string> execute(TableHandler &) override {
  using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  AssertThrow(coupled_solve_converged,ExcMessage("Distributed control did not converge."));
  const auto &m=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
  auto &manager=this->get_reconstructed_fault_manager();auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
  const auto &props=manager.get_property_information();const auto &fault=manager.get_fault(0);
  const auto qp=props[manager.get_property_index("phase field fault cohesive traction")].position;
  const auto hp=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
  const auto I=F::current_normalization_integrals(m).at(0);const auto V=manager.get_timestep_committed_slip_rate(0);
  double H=0,maximum=0;
  for(const auto &p:pm.get_particle_handler())H+=p.get_properties()[hp]*pm.get_particle_domain_handler().get_particle_domain(p.get_local_index()).volume();
  const auto b=this->introspection().variable("phase_field").block_index;
  for(auto i:this->get_solution().block(b).locally_owned_elements())maximum=std::max(maximum,this->get_solution().block(b)[i]);
  const auto comm=this->get_mpi_communicator();H=Utilities::MPI::sum(H,comm);maximum=Utilities::MPI::max(maximum,comm);
  if(Utilities::MPI::this_mpi_process(comm)==0) {
   const auto step=this->get_timestep_number();std::ofstream out(this->get_output_directory()+"consistency.csv",step?std::ios::app:std::ios::out);
   if(!step)out<<"step,time,node,q,I,V,H_integral,phi_max\n";
   for(unsigned int j=0;j<fault.n_vertices();++j)out<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<j<<','<<fault.get_properties(j)[qp]<<','<<I[j]<<','<<V[j]<<','<<H<<','<<maximum<<'\n';
  }
  return {"Coupling consistency:","recorded"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(CouplingConsistency,"P2 coupling consistency","Small serial/MPI accepted-state comparison.")
}}
