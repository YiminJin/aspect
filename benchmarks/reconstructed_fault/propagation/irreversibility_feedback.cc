// Frozen-H replay of retained failed entries. No mechanical/history publication.
#include "coupling_feedback.cc"

namespace aspect { namespace Postprocess {
template<int dim> class IrreversibilityFeedback : public CouplingFeedback<dim> {
 using Base=CouplingFeedback<dim>;
 using A=internal::PhaseFieldTestAccess<dim>;
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 using BV=LinearAlgebra::BlockVector;
 using BM=LinearAlgebra::BlockSparseMatrix;
 double fracture(const BV &state) const {
  double result=0;const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
  const double ell=this->get_phase_field_handler().get_length_scale();
  for(const auto &p:this->pm().get_particle_handler()) {
   const auto d=this->pm().get_particle_domain_handler().get_particle_domain(p.get_local_index());
   double phi=0;Tensor<1,dim> grad;
   for(unsigned int j=0;j<d.n_relevant_vertices();++j) {
    const double v=state[dofs[d.relevant_vertex_index(j)]];
    phi+=v*d.weighting_function_value(j);grad+=v*d.weighting_function_gradient(j);
   }
   result+=d.volume()*this->energy_scale*(phi+ell*ell*grad.norm_square());
  }
  return result;
 }
 void normalization(const BV &state,const std::string &name,const bool force_failure=false) {
  auto &solution=const_cast<BV&>(this->get_solution());const BV saved(solution);
  auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(this->material());
  const auto saved_I=F::current_normalization_integrals(model);
  const auto restore=[&]() {
   solution.block(this->b())=saved.block(this->b());
   F::restore_diagnostic_normalization(model,saved_I);F::invalidate_normalization_cache(model);
  };
  try {
   solution.block(this->b())=state.block(this->b());F::invalidate_normalization_cache(model);
   const auto I=F::compute_normalization_integrals(model).at(0);
   if(force_failure)throw std::runtime_error("intentional normalization restoration probe");
   const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
   for(unsigned int v=0;v<dofs.size();++v)if(dofs[v]!=numbers::invalid_dof_index) {
    const auto p=this->get_triangulation().get_vertices()[v];
    this->nodal[33*std::lround(32*p[0])+std::lround(32*p[1])]=state[dofs[v]];
   }
   const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
   std::ofstream out(this->get_output_directory()+"replay-normalization-"+name+".csv");
   out<<std::setprecision(17)<<"x,direct,projected,normalized_source_integral,core_source\n";
   for(unsigned int k=0;k<=200;++k) {
    const double x=k/200.;const unsigned int s=std::min(9u,static_cast<unsigned int>(10*x));
    const double xi=(x-fault.vertex(s)[0])/(fault.vertex(s+1)[0]-fault.vertex(s)[0]);
    const double ii=(1-xi)*I[s]+xi*I[s+1],phi=this->point_phase(x,.5);
    AssertThrow(ii>0 && phi<1,ExcMessage("Replay normalization inadmissible."));
    const double h=1/this->get_phase_field_handler().energetic_degradation({.5,.5},phi)-1;
    out<<x<<','<<this->direct_integral(x)<<','<<ii<<','<<this->direct_integral(x)/ii<<','<<h/ii<<'\n';
   }
   restore();
  } catch(...) {restore();throw;}
 }
 void replay() {
  if(this->get_timestep_number()==0 || A::parameters(this->get_phase_field_handler()).enforce_irreversibility)return;
  double healing=0;const auto &state=this->get_solution();
  for(auto i:state.block(this->b()).locally_owned_elements())
   healing=std::max(healing,this->get_old_solution().block(this->b())[i]-state.block(this->b())[i]);
  if(healing<=4.922e-12)return;
  this->calibration();const BV bound(this->get_old_solution()),box(state);
  BV trial(bound),rhs(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
  BM matrix;const auto &original=this->get_system_matrix();matrix.reinit(original.n_block_rows(),original.n_block_cols());
  for(unsigned int i=0;i<original.n_block_rows();++i)for(unsigned int j=0;j<original.n_block_cols();++j)matrix.block(i,j).copy_from(original.block(i,j));
  matrix.collect_sizes();const auto &handler=this->get_phase_field_handler();
  const auto particles_before=this->incoming;const auto q_before=this->property("phase field fault cohesive traction");
  const auto I_before=this->property("phase field fault previous I h");
  A::bounded(handler,matrix,rhs,trial,&bound);
  A::assemble(handler,matrix,rhs,trial,true);
  double kkt=0,complementarity=0,free=0;unsigned int active=0,positive=0;
  for(auto i:rhs.block(this->b()).locally_owned_elements()) {
   const double x=trial.block(this->b())[i],l=bound.block(this->b())[i],r=rhs.block(this->b())[i];
   AssertThrow(x>=l && x<=1,ExcMessage("Replay irreversibility bounds failed."));
   const double defect=x==l?std::max(r,0.):(x==1?std::min(r,0.):r);
   kkt=std::max(kkt,std::abs(defect));
   complementarity=std::max(complementarity,std::abs((x-l)*std::max(-r,0.)));
   if(x==l){++active;positive+=l>0;}else if(x<1)free=std::max(free,std::abs(r));
  }
  AssertThrow(kkt<=1e-11,ExcMessage("Irreversible replay KKT failed."));
  // Exercise the public phase-publication boundary on deliberate nonconvergence.
  // This is a separate failure probe, not a different replay tolerance.
  auto &parameters=const_cast<PhaseField::SolverParameters&>(A::parameters(handler));
  const auto saved_parameters=parameters;BV failed_target(box);bool rejected=false;
  parameters.enforce_irreversibility=true;parameters.max_nonlinear_iterations=0;
  try {const_cast<PhaseFieldHandler<dim>&>(handler).evolve_phase_field(matrix,rhs,failed_target);}
  catch(const std::exception &e) {
   rejected=std::string(e.what()).find("Nonlinear solver failed to converge in the prescribed")!=std::string::npos;
   if(!rejected){parameters=saved_parameters;throw;}
  }
  catch(...) {parameters=saved_parameters;throw;}
  parameters=saved_parameters;
  AssertThrow(rejected,ExcMessage("Phase publication failure probe unexpectedly converged."));
  for(auto i:rhs.block(this->b()).locally_owned_elements())
   AssertThrow(failed_target.block(this->b())[i]==box.block(this->b())[i],ExcMessage("Failed phase solve published a trial."));
  BV repeat(trial);A::bounded(handler,matrix,rhs,repeat,&bound);
  double repeat_error=0;for(auto i:rhs.block(this->b()).locally_owned_elements())repeat_error=std::max(repeat_error,std::abs(repeat.block(this->b())[i]-trial.block(this->b())[i]));
  const auto &dofs=A::vertex_dofs(handler);
  std::ofstream out(this->get_output_directory()+"irreversibility-replay.csv");out<<std::setprecision(17)<<"x,y,box,irreversible,lower,residual,lower_active\n";
  for(unsigned int v=0;v<dofs.size();++v)if(dofs[v]!=numbers::invalid_dof_index) {
   const auto p=this->get_triangulation().get_vertices()[v];const auto global=dofs[v];
   const auto local=global-state.get_block_indices().block_start(this->b());
   out<<p[0]<<','<<p[1]<<','<<box[global]<<','<<trial[global]<<','<<bound[global]<<','<<rhs.block(this->b())[local]<<','<<(trial[global]==bound[global])<<'\n';
  }
  out.close();normalization(box,"box");normalization(trial,"irreversible");
  // Restoration is checked on the normal and exceptional diagnostic paths.
  try {normalization(trial,"exception",true);}catch(const std::runtime_error&) {}
  for(auto i:rhs.block(this->b()).locally_owned_elements()) {
   AssertThrow(state.block(this->b())[i]==box.block(this->b())[i] && bound.block(this->b())[i]==this->get_old_solution().block(this->b())[i],ExcMessage("Replay modified accepted or trial phase."));
  }
  for(const auto &p:this->pm().get_particle_handler())AssertThrow(p.get_properties()[this->hi()]==particles_before.at(p.get_id()),ExcMessage("Replay changed H."));
  AssertThrow(q_before==this->property("phase field fault cohesive traction") && I_before==this->property("phase field fault previous I h"),ExcMessage("Replay changed surface history."));
  std::ofstream metrics(this->get_output_directory()+"irreversibility-replay-metrics.csv");
  metrics<<std::setprecision(17)<<"time,original_healing,KKT,free_residual,complementarity,active_lower,positive_lower,repeat_error,fracture_lower,fracture_box,fracture_irreversible\n"<<this->get_time()<<','<<healing<<','<<kkt<<','<<free<<','<<complementarity<<','<<active<<','<<positive<<','<<repeat_error<<','<<fracture(bound)<<','<<fracture(box)<<','<<fracture(trial)<<'\n';metrics.close();
  AssertThrow(false,ExcMessage("P2 frozen-H irreversibility replay completed; no history advanced."));
 }
public:
 void initialize() override {
  this->get_signals().post_particle_management.connect([this](Particle::Manager<dim>&){replay();});Base::initialize();
 }
};
ASPECT_REGISTER_POSTPROCESSOR(IrreversibilityFeedback,"P2 irreversibility feedback","Frozen-H failed-entry replay and unchanged P2 diagnostics.")
}}
