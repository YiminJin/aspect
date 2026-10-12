// Read-only native Q1 -> particle values -> surface transfer diagnosis.
#include <aspect/phase_field.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/reconstructed_fault/manager.h>
#include <fstream>
#include <iomanip>
namespace aspect { namespace Postprocess {
template<int dim> class RecoverableProjection : public Interface<dim>, public SimulatorAccess<dim> {
public:
 std::pair<std::string,std::string> execute(TableHandler &) override {
  AssertThrow(this->get_timestep_number()==0,ExcMessage("Transfer experiment is initialization only."));
  auto &manager=this->get_reconstructed_fault_manager();const auto &fault=manager.get_fault(0);
  const auto &associations=manager.get_locally_owned_particle_fault_associations();
  const auto saved=fault.get_properties(0)[0];
  std::ofstream out(this->get_output_directory()+"recoverable-q-transfer.csv");
  out<<std::setprecision(17)<<"row,column,value\n";
  for(unsigned int column=0;column<fault.n_vertices();++column) {
   std::map<types::particle_index,double> values;
   for(const auto &a:associations) if(a.active)
    values.emplace(a.particle_id,(a.segment_index==column?1-a.xi:0.)+(a.segment_index+1==column?a.xi:0.));
   const auto projected=manager.project_particle_scalar(values).nodal_values.at(0);
   for(unsigned int row=0;row<projected.size();++row)out<<row<<','<<column<<','<<projected[row]<<'\n';
  }
  AssertThrow(fault.get_properties(0)[0]==saved,ExcMessage("Projection changed persistent fault state."));
  return {"Native traction transfer:","matrix captured without history publication"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(RecoverableProjection,"P2 recoverable projection","Isolate native repeated traction transfer.")
}}
