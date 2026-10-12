// Read-only energy accounting for the explicitly selected unloaded P2 case.
#include "material_feedback.cc"
#include <deal.II/base/quadrature_lib.h>

namespace aspect { namespace Postprocess {
template<int dim> class RecoverableFeedback : public MaterialFeedback<dim> {
 using Base=MaterialFeedback<dim>;
 using A=internal::PhaseFieldTestAccess<dim>;
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 double Db=0,Df=0,Wb=0,Wf=0,Wfr=0,Jphase=0,initial_fracture=0;
public:
 std::pair<std::string,std::string> execute(TableHandler &table) override {
  const auto result=Base::execute(table);
  const auto step=this->get_timestep_number();const double dt=this->get_timestep();
  auto &manager=this->get_reconstructed_fault_manager();
  const auto &fault=manager.get_fault(0);
  const auto q=this->property("phase field fault cohesive traction");
  const auto I=F::current_normalization_integrals(this->material()).at(0);
  const auto &V=manager.get_timestep_committed_slip_rate(0);
  const double G=F::surface_shear_modulus(this->material(),0,0,0),eta=100*G;
  const auto c=F::maxwell_coefficients(eta,G,step?dt:1.);
  const double beta=c.first, one=-std::expm1(-dt*G/eta),lambda=G/eta;
  double Eb=0,Ef=0,fracture=0,dDb=0,dDf=0,dWb=0,dWf=0,dWfr=0,jump=0;
  double Ymax=0,ratio=0,Zerror=0,oldEb=0;
  const auto stresspos=this->pm().get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
  const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
  std::vector<Point<dim>> points;
  for(const auto &p:this->pm().get_particle_handler()) points.push_back(p.get_location());
  Utilities::MPI::RemotePointEvaluation<dim> cache;
  cache.reinit(this->get_phase_field_handler().get_grid_cache(),points);
  const auto gradients=VectorTools::point_gradients<dim>(cache,this->get_dof_handler(),this->get_solution(),VectorTools::EvaluationFlags::avg,this->introspection().extractors.velocities.first_vector_component);
  unsigned int index=0;
  for(const auto &p:this->pm().get_particle_handler()) {
   const auto domain=this->pm().get_particle_domain_handler().get_particle_domain(p.get_local_index());
   const double volume=domain.volume();double phi=0;Tensor<1,dim> grad;
   for(unsigned int j=0;j<domain.n_relevant_vertices();++j) {
    const double v=this->get_solution()[dofs[domain.relevant_vertex_index(j)]];
    phi+=v*domain.weighting_function_value(j);grad+=v*domain.weighting_function_gradient(j);
   }
   fracture+=volume*this->energy_scale*(phi+std::pow(this->get_phase_field_handler().get_length_scale(),2)*grad.norm_square());
   SymmetricTensor<2,dim> stress,old;
   for(unsigned int j=0;j<SymmetricTensor<2,dim>::n_independent_components;++j) {
    const auto component=SymmetricTensor<2,dim>::unrolled_to_component_indices(j);
    stress[component]=p.get_properties()[stresspos+j];old[component]=this->incoming_stress.at(p.get_id())[j];
   }
   Eb+=volume*(stress*stress)/(4*G);oldEb+=volume*(old*old)/(4*G);
   if(step) {
    const auto steady=(stress-beta*old)/one, transient=old-steady;
    const double square=(steady*steady)*dt+2*(steady*transient)*one/lambda+(transient*transient)*(-std::expm1(-2*lambda*dt))/(2*lambda);
    dDb+=volume*square/(2*eta);
    dWb+=volume*((steady*dt+transient*one/lambda)*symmetrize(gradients[index]));
   }
   const auto a=manager.project_to_normal_profiles(p.get_location());
   const double qq=F::uses_surface_history(this->material())
     ? F::committed_q(this->material(),a.fault_index,a.segment_index,a.xi)
     : (1-a.xi)*q[a.segment_index]+a.xi*q[a.segment_index+1];
   const double g=this->get_phase_field_handler().energetic_degradation({.5,.5},this->point_phase(p.get_location()[0],p.get_location()[1]));
   const double Y=F::recoverable_phase_driving_force(G,g,qq);
   Ymax=std::max(Ymax,Y);ratio=std::max(ratio,Y/this->initial_H.at(p.get_id()));
   if(!step) AssertThrow(qq==0 && Y==0 && stress.norm()==0 && Y<=p.get_properties()[this->hi()],ExcMessage("Unloaded initialization is inconsistent."));
   ++index;
  }
  if(!step) initial_fracture=fracture;
  QGauss<1> quadrature(8);
  for(unsigned int cell=0;cell<fault.n_cells();++cell) for(unsigned int k=0;k<quadrature.size();++k) {
   const double x=quadrature.point(k)[0],w=quadrature.weight(k)*(fault.vertex(cell+1)-fault.vertex(cell)).norm();
   const auto interp=[&](const auto &a){return (1-x)*a[cell]+x*a[cell+1];};
   const double ii=interp(I),v=interp(V);
   const double z=F::uses_surface_history(this->material())
     ? (1-x)*I[cell]*q[cell]+x*I[cell+1]*q[cell+1] : ii*interp(q);
   Ef+=w*z*z/(2*G*ii);
   if(step) {
    const double oldI=interp(this->previous_I);
    const double oldz=F::uses_surface_history(this->material())
      ? (1-x)*this->previous_I[cell]*this->previous_q[cell]
        + x*this->previous_I[cell+1]*this->previous_q[cell+1]
      : oldI*interp(this->previous_q);
    jump+=w*oldz*oldz/(2*G)*(1/ii-1/oldI);
    const double steady=eta*v/ii,transient=oldz/ii-steady;
    const double square=steady*steady*dt+2*steady*transient*one/lambda+transient*transient*(-std::expm1(-2*lambda*dt))/(2*lambda);
    dDf+=w*ii/eta*square;
    dWf+=w*v*(steady*dt+transient*one/lambda);
    dWfr+=w*(.1*.01*G)*v*dt;
    Zerror=std::max(Zerror,std::abs(z-(beta*oldz+c.second*v))/std::max(1.,std::abs(z)));
   }
  }
  Db+=dDb;Df+=dDf;Wb+=dWb;Wf+=dWf;Wfr+=dWfr;Jphase+=jump;
  std::ofstream out(this->get_output_directory()+"recoverable-energy.csv",step?std::ios::app:std::ios::out);
  if(!step)out<<"step,time,bulk_stored,fault_stored,fracture,fracture_change,bulk_viscous,fault_viscous,friction,bulk_input_work,cohesive_work,phase_fault_jump,bulk_crack_work_step,exchange_defect_step,energy_defect,Ymax,Y_over_H0_max,Z_update_relative_error\n";
  const double crack=dWb-(Eb-oldEb)-dDb;
  out<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<Eb<<','<<Ef<<','<<fracture<<','<<fracture-initial_fracture<<','<<Db<<','<<Df<<','<<Wfr<<','<<Wb<<','<<Wf<<','<<Jphase<<','<<crack<<','<<crack-dWf-dWfr<<','<<Wb-(Eb+Ef+Db+Df+Wfr-Jphase)<<','<<Ymax<<','<<ratio<<','<<Zerror<<'\n';
  return result;
 }
};
ASPECT_REGISTER_POSTPROCESSOR(RecoverableFeedback,"P2 recoverable feedback","Unloaded recoverable P2 history and separate energy accounting.")
}}
