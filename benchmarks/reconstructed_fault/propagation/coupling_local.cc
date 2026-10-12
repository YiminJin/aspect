// Native surface history lifecycle; all diagnostic writes are restored.
#include "../../../tests/phase_field_fault_test_access.h"
#include <aspect/postprocess/interface.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <iomanip>
namespace aspect { namespace Postprocess {
template<int dim> class SurfaceHistory : public Interface<dim>, public SimulatorAccess<dim> {
public:
 std::pair<std::string,std::string> execute(TableHandler &) override {
  using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  auto &m=const_cast<MaterialModel::PhaseFieldFault<dim>&>(Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
  auto &manager=this->get_reconstructed_fault_manager();const auto &fault=manager.get_fault(0);
  AssertThrow(this->get_timestep_number()==0 && manager.get_faults().size()==1,ExcMessage("Initialization-only single-fault test."));
  const auto pos=manager.get_property_information()[manager.get_property_index("phase field fault cohesive traction")].position;
  std::vector<std::vector<double>> saved(1),v(1,std::vector<double>(fault.n_vertices(),0.));
  for(unsigned int j=0;j<fault.n_vertices();++j)saved[0].push_back(fault.get_properties(j)[pos]);
  const auto eta_saved=F::viscosity_state(m);const auto I=F::current_normalization_integrals(m).at(0);
  std::ofstream out(this->get_output_directory()+"surface-history.csv");
  out<<std::setprecision(17)<<"pattern,eta,dt,step,nodal_error,point_error\n";
  auto restore=[&](){F::restore_viscosity(m,eta_saved);F::commit_cohesive_state(m,saved);};
  try {
   for(unsigned int pattern=0;pattern<fault.n_vertices()+2;++pattern)
    for(double eta:{1e8,1e30}) for(double dt:{.125,.0625,.03125}) {
     F::set_uniform_viscosity(m,eta);auto initial=saved;
     for(unsigned int j=0;j<fault.n_vertices();++j)
      initial[0][j]=pattern<fault.n_vertices() ? (j==pattern ? 700. : 0.)
        : pattern==fault.n_vertices() ? 100.+40*j : 400.+200*std::sin(1.73*j);
     F::commit_cohesive_state(m,initial);
     for(unsigned int k=1;k<=std::lround(1/dt);++k) {
      const auto trial=F::surface_candidates(m,dt,v);F::commit_cohesive_state(m,trial);
      double error=0,point_error=0;const double decay=std::exp(-1e6*k*dt/eta);
      for(unsigned int j=0;j<fault.n_vertices();++j)
       error=std::max(error,std::abs(trial[0][j]-decay*initial[0][j]));
      for(unsigned int s=0;s<fault.n_cells();++s) for(double x:{0.,.17,.5,.83,1.}) {
       const double exact=decay*((1-x)*I[s]*initial[0][s]+x*I[s+1]*initial[0][s+1])/((1-x)*I[s]+x*I[s+1]);
       point_error=std::max(point_error,std::abs(F::committed_q(m,0,s,x)-exact));
      }
      AssertThrow(error<1e-10 && point_error<1e-10,ExcMessage("Nonconstant surface history was not preserved."));
      out<<pattern<<','<<eta<<','<<dt<<','<<k<<','<<error<<','<<point_error<<'\n';
     }
    }
   restore();
  } catch(...) {restore();throw;}
  for(unsigned int j=0;j<fault.n_vertices();++j) AssertThrow(fault.get_properties(j)[pos]==saved[0][j],ExcMessage("Surface history restoration failed."));
  return {"Surface history:","nonconstant Maxwell paths verified"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(SurfaceHistory,"P2 surface history","Native surface-owned history tests.")
}}
