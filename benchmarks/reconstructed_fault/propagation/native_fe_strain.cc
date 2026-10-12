// Restricted native P2 observer and stage-only representation probes.
// MomentCycle friendship is existing infrastructure; load this plugin alone.
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/reconstructed_fault/manager.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include "../../../tests/phase_field_test_access.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
template<int dim> class MomentCycle : public Interface<dim>, public SimulatorAccess<dim> {
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 using A=internal::PhaseFieldTestAccess<dim>;
 using BV=LinearAlgebra::BlockVector;
 using Model=MaterialModel::PhaseFieldFault<dim>;
 BV previous;
 std::vector<double> previous_I,previous_Z,slip;
 double previous_F=0,previous_bulk=0,previous_cohesive=0;
 double entry_energy=0,entry_kkt=0;
 std::map<types::particle_index,double> initial_H,last_H;
 const Model &model() const {return Plugins::get_plugin_as_type<const Model>(this->get_material_model());}
 unsigned int pb() const {return this->introspection().variable("phase_field").block_index;}
 std::vector<double> property(const std::string &name) const {
  const auto &m=this->get_reconstructed_fault_manager();const auto &f=m.get_fault(0);
  const auto info=m.get_property_information()[m.get_property_index(name)];
  std::vector<double> v(f.n_vertices());for(unsigned int i=0;i<v.size();++i)v[i]=f.get_properties(i)[info.position];return v;
 }
 double fracture(const BV &state) const {
  const auto &h=this->get_phase_field_handler();const auto &dofs=A::vertex_dofs(h);
  const auto &pm=h.get_associated_particle_manager();double e=0;
  const double scale=model().get_critical_energy_release_rates()[0]/((8./3.)*h.get_length_scale());
  for(const auto &p:pm.get_particle_handler()) {
   const auto d=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());double phi=0;Tensor<1,dim> grad;
   for(unsigned int j=0;j<d.n_relevant_vertices();++j) {const double x=state[dofs[d.relevant_vertex_index(j)]];phi+=x*d.weighting_function_value(j);grad+=x*d.weighting_function_gradient(j);}
   e+=d.volume()*scale*(phi+h.get_length_scale()*h.get_length_scale()*grad.norm_square());
  }return e;
 }
 void phase_entry() {
  if(!model().has_fe_strain_history() || this->get_timestep_number()==0)return;
  const auto e=model().evaluate_fe_phase_energy(this->get_solution(),false);
  entry_energy=e.energy;
  std::ofstream entry(this->get_output_directory()+"fe-entry-"+std::to_string(this->get_timestep_number())+".csv");
  entry<<std::setprecision(17)<<"x,y,phi,accepted_bound\n";
  const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
  for(unsigned int v=0;v<dofs.size();++v)if(dofs[v]!=numbers::invalid_dof_index) {
   const auto point=this->get_triangulation().get_vertices()[v];
   entry<<point[0]<<','<<point[1]<<','<<this->get_solution()[dofs[v]]<<','<<model().accepted_fe_strain_state()[dofs[v]]<<'\n';
  }
  // Actual native total-energy assembly; test residual signs at the immutable bound.
  const auto &original=this->get_system_matrix();LinearAlgebra::BlockSparseMatrix matrix;
  matrix.reinit(original.n_block_rows(),original.n_block_cols());
  for(unsigned int i=0;i<original.n_block_rows();++i)for(unsigned int j=0;j<original.n_block_cols();++j)matrix.block(i,j).copy_from(original.block(i,j));
  matrix.collect_sizes();BV rhs;rhs.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
  A::assemble(this->get_phase_field_handler(),matrix,rhs,this->get_solution(),true);
  entry_kkt=0;
  for(auto i:rhs.block(pb()).locally_owned_elements()) {
   const double x=this->get_solution().block(pb())[i],l=model().accepted_fe_strain_state().block(pb())[i],r=rhs.block(pb())[i];
   const double defect=x==l?std::max(r,0.):(x==1?std::min(r,0.):r);entry_kkt=std::max(entry_kkt,std::abs(defect));
  }
  const auto step=this->get_timestep_number();
  if(step==32 || step==40) {
   // Nontrivial native derivative check at the actual incoming FE anchor.
   // Explicit trial vectors never mutate accepted phase/R/Z or their caches.
   const auto mechanical=model().evaluate_fe_phase_energy(this->get_solution(),true);
   std::vector<double> direction(mechanical.gradient.size(),0.),Hd(direction);
   for(const auto &cell:this->get_dof_handler().active_cell_iterators()) {
    std::vector<types::global_dof_index> ids(this->get_fe().dofs_per_cell);cell->get_dof_indices(ids);
    const auto first=this->introspection().variable("phase_field").first_component_index;
    const auto offset=this->get_solution().get_block_indices().block_start(pb());
    for(const auto v:cell->vertex_indices()) {
     const auto i=ids[this->get_fe().component_to_system_index(first,v)]-offset;
     const double phi=this->get_solution().block(pb())[i];
     if(phi>.01 && phi<.95)direction[i]=std::sin(4.1*cell->vertex(v)[0]+2.3*cell->vertex(v)[1]);
    }
   }
   double predicted=0.;for(unsigned int i=0;i<direction.size();++i) {
    predicted+=mechanical.gradient[i]*direction[i];
    for(unsigned int j=0;j<direction.size();++j)Hd[i]+=mechanical.hessian(i,j)*direction[j];
   }
   std::ofstream derivatives(this->get_output_directory()+"phase-derivative-"+std::to_string(step)+".csv");
   derivatives<<std::setprecision(17)<<"epsilon,energy_derivative_relative_error,Hessian_direction_relative_error\n";
   for(double epsilon:{1.e-5,1.e-6,1.e-7}) {
    BV plus(this->get_solution()),minus(this->get_solution());
    for(auto i:plus.block(pb()).locally_owned_elements()){plus.block(pb())[i]+=epsilon*direction[i];minus.block(pb())[i]-=epsilon*direction[i];}
    const auto ep=model().evaluate_fe_phase_energy(plus,false),em=model().evaluate_fe_phase_energy(minus,false);
    AssertThrow(ep.admissible && em.admissible,ExcMessage("Invalid phase derivative diagnostic trial."));
    double error=0,scale=1.;for(unsigned int i=0;i<direction.size();++i){error=std::max(error,std::abs((ep.gradient[i]-em.gradient[i])/(2*epsilon)-Hd[i]));scale=std::max(scale,std::abs(Hd[i]));}
    derivatives<<epsilon<<','<<std::abs((ep.energy-em.energy)/(2*epsilon)-predicted)/std::max(1.,std::abs(predicted))<<','<<error/scale<<'\n';
   }
   const auto &current=this->get_solution();const auto &lower=model().accepted_fe_strain_state();
   std::vector<unsigned int> free;unsigned int strict_free=0;
   for(auto i:rhs.block(pb()).locally_owned_elements()) {
    const double gap=current.block(pb())[i]-lower.block(pb())[i];
    if(gap>0 && current.block(pb())[i]<1)++strict_free;
    if(gap>4.922e-12 && current.block(pb())[i]<1)free.push_back(i);
   }
   std::ofstream out(this->get_output_directory()+"phase-curvature-"+std::to_string(step)+".csv");
   out<<std::setprecision(17)<<"time,free_above_healing_allowance,strict_free,min_eigenvalue,negative_count,amplitude,energy_minus,energy_plus,quadratic_prediction,energy_arithmetic\n";
   if(!free.empty()) {
    LAPACKFullMatrix<double> H(free.size(),free.size());
    for(unsigned int i=0;i<free.size();++i)for(unsigned int j=0;j<free.size();++j)H(i,j)=matrix.block(pb(),pb()).el(free[i],free[j]);
    Vector<double> eigen;FullMatrix<double> vectors;
    H.compute_eigenvalues_symmetric(-1e100,1e100,2*std::numeric_limits<double>::min(),eigen,vectors);
    unsigned int negative=0;for(const auto v:eigen)if(v<0)++negative;
    double alpha=1e-4;
    for(unsigned int i=0;i<free.size();++i)if(vectors(i,0)!=0.)
     alpha=std::min(alpha,.25*std::min(current.block(pb())[free[i]]-lower.block(pb())[free[i]],1-current.block(pb())[free[i]])/std::abs(vectors(i,0)));
    const double base=e.energy+fracture(current);double delta[2]={};
    for(unsigned int k=0;k<2;++k) {
     BV trial(current);for(unsigned int i=0;i<free.size();++i)trial.block(pb())[free[i]]+=(k?1:-1)*alpha*vectors(i,0);
     const auto value=model().evaluate_fe_phase_energy(trial,false);
     AssertThrow(value.admissible,ExcMessage("Diagnostic curvature perturbation left the admissible normalization domain."));
     delta[k]=value.energy+fracture(trial)-base;
    }
    out<<this->get_time()<<','<<free.size()<<','<<strict_free<<','<<eigen[0]<<','<<negative<<','<<alpha<<','<<delta[0]<<','<<delta[1]<<','<<.5*alpha*alpha*eigen[0]<<','<<32*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(base))<<'\n';
   } else out<<this->get_time()<<",0,"<<strict_free<<",nan,0,0,0,0,0,0\n";
  }
 }
 void plane_strain_audit() {
  std::ofstream out(this->get_output_directory()+"plane-strain-algebra.csv");
  out<<std::setprecision(17)<<"case,helper_block_error,invariant_error,repeat_deviator_error,trace_2d\n";
  for(unsigned int k=0;k<3;++k) {
   SymmetricTensor<2,dim> raw;
   if(k==0)raw[0][1]=.7;
   if(k==1){raw[0][0]=.8;raw[1][1]=-.8;}
   if(k==2){raw[0][0]=.8;raw[1][1]=.2;raw[0][1]=.3;}
   SymmetricTensor<2,3> full;
   for(unsigned int i=0;i<2;++i)for(unsigned int j=i;j<2;++j)full[i][j]=raw[i][j];
   full=deviator(full);
   const auto block=Utilities::Tensors::consistent_deviator(raw);
   SymmetricTensor<2,dim> exact;
   for(unsigned int i=0;i<2;++i)for(unsigned int j=i;j<2;++j)exact[i][j]=full[i][j];
   const double block_error=(block-exact).norm();
   const double invariant_error=std::abs(Utilities::Tensors::consistent_second_invariant_of_deviatoric_tensor(block)+.5*(full*full));
   out<<k<<','<<block_error<<','<<invariant_error<<','<<(Utilities::Tensors::consistent_deviator(block)-block).norm()<<','<<trace(block)<<'\n';
   AssertThrow(block_error<1e-14 && invariant_error<1e-14,ExcMessage("Plane-strain tensor helper differs from explicit 3D embedding."));
  }
  // A zero-boundary, representable Q2 velocity with nonzero divergence.
  // It is an operator probe, not a physical compressible trajectory.
  const auto &intro=this->introspection();const auto quad=model().mechanical_quadrature();
  BV diagnostic(this->get_solution());diagnostic=0.;
  struct Field : Function<dim> {
   Field(unsigned int n):Function<dim>(n){}
   double value(const Point<dim>&p,unsigned int c=0)const override {return c==0?p[0]*(1-p[0])*p[1]*(1-p[1]):0.;}
  } field(this->get_fe().n_components());
  VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),field,diagnostic,intro.component_masks.velocities);
  FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_gradients|update_JxW_values);
  std::vector<SymmetricTensor<2,dim>> rate(quad.size());double raw_work=0.,plane_work=0.,div_work=0.;
  const auto c=F::maxwell_coefficients(1e8,1e6,this->get_timestep_number()?this->get_timestep():1.);
  for(const auto &cell:this->get_dof_handler().active_cell_iterators()) {
   fe.reinit(cell);fe[intro.extractors.velocities].get_function_symmetric_gradients(diagnostic,rate);
   for(unsigned int q=0;q<quad.size();++q) {
    const double w=fe.JxW(q),theta=trace(rate[q]);
    raw_work+=w*2*c.second*(rate[q]*rate[q]);
    plane_work+=w*2*c.second*((rate[q]*rate[q])-theta*theta/3.);
    div_work+=w*2*c.second*theta*theta/3.;
   }
  }
  BV action(diagnostic);action=0.;const auto vb=intro.block_indices.velocities;
  this->get_system_matrix().block(vb,vb).vmult(action.block(vb),diagnostic.block(vb));
  const double native=diagnostic.block(vb)*action.block(vb);
  std::ofstream matrix(this->get_output_directory()+"plane-strain-operator.csv");
  matrix<<std::setprecision(17)<<"native,raw_sym,plane_strain,missing_subtraction,native_minus_plane\n"<<native<<','<<raw_work<<','<<plane_work<<','<<div_work<<','<<native-plane_work<<'\n';
 }
 void local_probe() {
  auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());const auto dt=sim.time_step;const auto step=sim.timestep_number;
  const BV solution(sim.solution),old(sim.old_solution),older(sim.old_old_solution),linear(sim.current_linearization_point),accepted(model().accepted_fe_strain_state());
  auto &rates=const_cast<std::vector<double>&>(this->get_reconstructed_fault_manager().get_slip_rate(0));const auto saved_rates=rates;
  const auto I=F::current_normalization_integrals(model()).at(0);
  auto Z_anchor=property("phase field fault cohesive traction");
  for(unsigned int j=0;j<Z_anchor.size();++j)Z_anchor[j]*=I[j];
  const auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
  std::map<types::particle_index,std::vector<double>> saved_particles;
  for(const auto &particle:particles)saved_particles[particle.get_id()]={particle.get_properties().begin(),particle.get_properties().end()};
  const auto saved_q=property("phase field fault cohesive traction"),saved_previous_I=property("phase field fault previous I h");
  auto restore=[&](){sim.time_step=dt;sim.timestep_number=step;rates=saved_rates;sim.solution=solution;sim.old_solution=old;sim.old_old_solution=older;sim.current_linearization_point=linear;};
  std::ofstream out(this->get_output_directory()+"fe-local-"+std::to_string(step)+".csv");out<<std::setprecision(17)<<"dt,ratio,stress_error_Pa,stress_scale_Pa,relative_error,net_stress_scale_Pa,field_mode,max_raw_trace\n";
  try {
   BV prescribed(solution);
   struct Prescribed : Function<dim> { unsigned int mode=0; Prescribed(unsigned int n):Function<dim>(n){} double value(const Point<dim>&p,unsigned int c=0)const override {return c==0?1e-5*p[0]*p[0]*p[1]:(c==1?(mode?.5:-1.)*1e-5*p[0]*p[1]*p[1]:0.);} } f(this->get_fe().n_components());
   VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),f,prescribed,this->introspection().component_masks.velocities);
   for(unsigned int j=0;j<rates.size();++j)rates[j]=1e-5*(1+.4*std::sin(.37*j));
   const auto quad=model().mechanical_quadrature();FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points);
   std::vector<SymmetricTensor<2,dim>> epsilon(quad.size()),anchor_R(quad.size());std::vector<double> phi(quad.size());SymmetricTensor<2,dim>S;S[0][1]=.5;
   const auto &manager=this->get_reconstructed_fault_manager();const unsigned int ns=manager.get_fault(0).n_cells();
   for(unsigned int mode=0;mode<2;++mode) {
    f.mode=mode;VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),f,prescribed,this->introspection().component_masks.velocities);
   for(double probe_dt:{.125,100.,400.}) {
    sim.time_step=probe_dt;sim.timestep_number=step+1;const auto update=F::prepared_fe_state(model(),prescribed);
    const auto c=F::maxwell_coefficients(1e8,1e6,probe_dt);double error=0,scale=1,net_scale=1,max_trace=0.;
    for(const auto &cell:this->get_dof_handler().active_cell_iterators()) {
     fe.reinit(cell);fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(prescribed,epsilon);
     fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(accepted,anchor_R);
     fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(prescribed,phi);
     for(unsigned int q=0;q<quad.size();++q) {
      const auto x=fe.quadrature_point(q);const unsigned int s=std::min(ns-1,static_cast<unsigned int>(ns*x[0]));const double xi=ns*x[0]-s;
      const double ii=(1-xi)*I[s]+xi*I[s+1],V=(1-xi)*rates[s]+xi*rates[s+1];
      const double chi=(1/this->get_phase_field_handler().energetic_degradation({.5,.5},phi[q])-1)/ii;
      max_trace=std::max(max_trace,std::abs(trace(epsilon[q])));
      const auto old_stress=model().retained_history_stress(cell->id(),x,{});
      const auto expected=c.first*old_stress+2*c.second*(Utilities::Tensors::consistent_deviator(epsilon[q])-chi*V*S);
      double evaluation_scale=0.;
      const auto actual=F::prepared_fe_stress(model(),update,cell->id(),x,&evaluation_scale);
      error=std::max(error,(actual-expected).norm());net_scale=std::max(net_scale,expected.norm());
      // The FE recurrence reconstructs stress from two potentially large
      // operands. Measure arithmetic against those operands, keeping the
      // existing 256-epsilon factor and recording the net scale separately.
      const double Za=(1-xi)*Z_anchor[s]+xi*Z_anchor[s+1];
      const double operands=c.first*(2e6*anchor_R[q].norm()+2*std::abs(Za*chi)*S.norm())
                            +2*c.second*(epsilon[q].norm()+std::abs(chi*V)*S.norm());
      AssertThrow(std::isfinite(operands),ExcMessage("Nonfinite Maxwell probe operand scale."));
      // Include coefficient/basis-gradient operands before FE cancellation,
      // using the same estimate as terminal production publication. Keep the
      // existing epsilon multiplier; this is not a solver tolerance change.
      scale=std::max({scale,operands,evaluation_scale,expected.norm()});
     }
    }
    out<<probe_dt<<','<<probe_dt/100<<','<<error<<','<<scale<<','<<error/scale<<','<<net_scale<<','<<mode<<','<<max_trace<<'\n';
    AssertThrow(error<=256*std::numeric_limits<double>::epsilon()*scale,ExcMessage("Nonuniform FE state does not reproduce native Maxwell endpoint."));
   }
   }
   restore();
   // Independent derivative of normalization-dependent recoverable energy.
   BV trial(solution);for(auto i:trial.block(pb()).locally_owned_elements())trial.block(pb())[i]+=.01;
   const auto base=model().evaluate_fe_phase_energy(trial,true);std::vector<double>d(base.gradient.size());
   const auto &original=this->get_system_matrix();LinearAlgebra::BlockSparseMatrix matrix;
   matrix.reinit(original.n_block_rows(),original.n_block_cols());
   for(unsigned int i=0;i<original.n_block_rows();++i)for(unsigned int j=0;j<original.n_block_cols();++j)matrix.block(i,j).copy_from(original.block(i,j));
   matrix.collect_sizes();BV rhs;rhs.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
   A::assemble(this->get_phase_field_handler(),matrix,rhs,trial,true);
   const BV baseline_rhs(rhs);FullMatrix<double> baseline_matrix(d.size(),d.size());
   for(unsigned int i=0;i<d.size();++i)for(unsigned int j=0;j<d.size();++j)baseline_matrix(i,j)=matrix.block(pb(),pb()).el(i,j);
   for(unsigned int i=0;i<d.size();++i)d[i]=.3+.2*std::sin(.73*i);
   double exact=0;for(unsigned int i=0;i<d.size();++i)exact+=base.gradient[i]*d[i];
   std::ofstream derivative(this->get_output_directory()+"fe-derivative-"+std::to_string(step)+".csv");derivative<<std::setprecision(17)<<"h,energy_relative,hessian_relative\n";
   for(double h:{1e-4,1e-5,1e-6,1e-7}) {
    BV plus(trial),minus(trial);for(unsigned int i=0;i<d.size();++i){plus.block(pb())[i]+=h*d[i];minus.block(pb())[i]-=h*d[i];}
    const auto ep=model().evaluate_fe_phase_energy(plus,false),em=model().evaluate_fe_phase_energy(minus,false);
    double error=0,scale=0;for(unsigned int i=0;i<d.size();++i){double v=0;for(unsigned int j=0;j<d.size();++j)v+=base.hessian(i,j)*d[j];error+=std::pow((ep.gradient[i]-em.gradient[i])/(2*h)-v,2);scale+=v*v;}
    derivative<<h<<','<<std::abs((ep.energy-em.energy)/(2*h)-exact)/std::max(1.,std::abs(exact))<<','<<std::sqrt(error/std::max(1.,scale))<<'\n';
   }
   sim.old_solution.block(pb())=.13;sim.old_old_solution.block(pb())=.27;sim.current_linearization_point.block(pb())=.41;sim.solution.block(pb())=.19;
   sim.old_solution.block(this->introspection().block_indices.velocities)=123.;
   const auto poisoned=model().evaluate_fe_phase_energy(trial,true);
   AssertThrow(poisoned.energy==base.energy && poisoned.gradient==base.gradient,ExcMessage("Explicit phase energy reads an unrelated predictor/history."));
   for(unsigned int i=0;i<d.size();++i)for(unsigned int j=0;j<d.size();++j)AssertThrow(poisoned.hessian(i,j)==base.hessian(i,j),ExcMessage("Phase Hessian reads an unrelated predictor/history."));
   A::assemble(this->get_phase_field_handler(),matrix,rhs,trial,true);
   for(unsigned int i=0;i<d.size();++i) {
    AssertThrow(rhs.block(pb())[i]==baseline_rhs.block(pb())[i],ExcMessage("Complete phase residual reads unrelated history."));
    for(unsigned int j=0;j<d.size();++j)AssertThrow(matrix.block(pb(),pb()).el(i,j)==baseline_matrix(i,j),ExcMessage("Complete phase Jacobian reads unrelated history."));
   }
   restore();try {sim.old_solution.block(pb())=.9;throw std::runtime_error("restore probe");}catch(const std::runtime_error&){restore();}
  }catch(...){restore();throw;}
  for(const auto &particle:particles)AssertThrow(std::equal(saved_particles.at(particle.get_id()).begin(),saved_particles.at(particle.get_id()).end(),particle.get_properties().begin()),ExcMessage("Stage-only probe changed particle histories."));
  AssertThrow(saved_q==property("phase field fault cohesive traction") && saved_previous_I==property("phase field fault previous I h"),ExcMessage("Stage-only probe changed surface histories."));
  BV difference(accepted);difference-=model().accepted_fe_strain_state();AssertThrow(difference.linfty_norm()==0,ExcMessage("Stage-only probe advanced the accepted FE state."));
 }
public:
 void initialize() override {this->get_signals().post_particle_management.connect([this](Particle::Manager<dim>&){phase_entry();});}
 std::pair<std::string,std::string> execute(TableHandler &) override {
  const auto step=this->get_timestep_number();AssertThrow(dim==2 && model().has_fe_strain_history(),ExcMessage("Restricted native FE observer requires initialized serial state."));
  const auto &intro=this->introspection();const auto &state=model().accepted_fe_strain_state();const auto &solution=this->get_solution();
  const auto I=F::current_normalization_integrals(model()).at(0),q=property("phase field fault cohesive traction");
  const auto &V=this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0);const unsigned int ns=I.size()-1;
  std::vector<double>Z(I.size());for(unsigned int j=0;j<I.size();++j)Z[j]=I[j]*q[j];
  if(step==0){plane_strain_audit();slip.assign(I.size(),0.);}else for(unsigned int j=0;j<slip.size();++j)slip[j]+=this->get_timestep()*V[j];
  const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());double healing=0,increment=0,maximum=0;Point<dim>maxpoint;
  std::ofstream phase(this->get_output_directory()+"fe-phase-"+std::to_string(step)+".csv");phase<<std::setprecision(17)<<"x,y,phi\n";
  for(unsigned int v=0;v<dofs.size();++v)if(dofs[v]!=numbers::invalid_dof_index){const double phi=solution[dofs[v]];const auto p=this->get_triangulation().get_vertices()[v];
   phase<<p[0]<<','<<p[1]<<','<<phi<<'\n';AssertThrow(phi>=0 && phi<1,ExcMessage("Phase bounds or finite normalization failed."));if(phi>maximum){maximum=phi;maxpoint=p;}
   if(step){healing=std::max(healing,previous[dofs[v]]-phi);increment=std::max(increment,phi-previous[dofs[v]]);}
   AssertThrow(state[dofs[v]]==phi,ExcMessage("Accepted FE phase was not published."));}
  AssertThrow(healing<=4.922e-12,ExcMessage("Resolved healing in native FE run."));
  std::ofstream surface(this->get_output_directory()+"fe-surface-"+std::to_string(step)+".csv");surface<<std::setprecision(17)<<"x,I,q,Z,V_interval,slip\n";
  for(unsigned int j=0;j<I.size();++j)surface<<double(j)/ns<<','<<I[j]<<','<<q[j]<<','<<Z[j]<<','<<V[j]<<','<<slip[j]<<'\n';
  std::map<types::global_dof_index,Point<dim>> support;
  DoFTools::map_dofs_to_support_points(this->get_mapping(),this->get_dof_handler(),support);
  std::map<types::global_dof_index,unsigned int> component;
  std::vector<types::global_dof_index> ids(this->get_fe().dofs_per_cell);
  for(const auto &cell:this->get_dof_handler().active_cell_iterators()) {cell->get_dof_indices(ids);for(unsigned int j=0;j<ids.size();++j)component[ids[j]]=this->get_fe().system_to_component_index(j).first;}
  std::ofstream encoding(this->get_output_directory()+"fe-encoding-"+std::to_string(step)+".csv");encoding<<std::setprecision(17)<<"dof,x,y,d_R,u,component\n";
  const unsigned int vb=intro.block_indices.velocities;const auto offset=state.get_block_indices().block_start(vb);
  for(auto i:state.block(vb).locally_owned_elements())encoding<<i<<','<<support.at(i+offset)[0]<<','<<support.at(i+offset)[1]<<','<<state.block(vb)[i]<<','<<solution.block(vb)[i]<<','<<component.at(i+offset)<<'\n';
  const double Fnow=fracture(solution);const auto e=model().evaluate_fe_phase_energy(solution,false);
  double Wbulk=0,Wboundary=0,Db=0,Dc=0,Wcohesive=0,Wfriction=0,Eold=0,Ejump=0,Enew=0,stress_error=0,Ymax=0;
  const double dt=this->get_timestep(),G=1e6,eta=1e8;const auto quad=model().mechanical_quadrature();const QGauss<1> time_quad(12);
  FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
  std::vector<double>phi(quad.size()),oldphi(phi);std::vector<SymmetricTensor<2,dim>>r(quad.size()),oldr(r),strain(r);
  SymmetricTensor<2,dim>S;S[0][1]=.5;SymmetricTensor<2,dim>affine;affine[0][1]=.5*.256*std::max(0.,this->get_time()-.5*dt);
  const bool dump=(step==0 || step==1 || step%8==0 || std::abs(this->get_time()-.25)<1e-12);
  std::ofstream bulk;if(dump){bulk.open(this->get_output_directory()+"fe-bulk-"+std::to_string(step)+".csv");bulk<<std::setprecision(17)<<"x,y,w,tau_xx,tau_yy,tau_xy,epsilon_xx,epsilon_yy,epsilon_xy\n";}
  for(const auto &cell:this->get_dof_handler().active_cell_iterators()) {
   fe.reinit(cell);fe[intro.extractors.velocities].get_function_symmetric_gradients(state,r);fe[intro.extractors.velocities].get_function_symmetric_gradients(solution,strain);
   const FEValuesExtractors::Scalar ph(intro.variable("phase_field").first_component_index);fe[ph].get_function_values(solution,phi);
   if(step){fe[intro.extractors.velocities].get_function_symmetric_gradients(previous,oldr);fe[ph].get_function_values(previous,oldphi);}
   for(unsigned int j=0;j<quad.size();++j) {
    const auto p=fe.quadrature_point(j);const unsigned int s=std::min(ns-1,static_cast<unsigned int>(ns*p[0]));const double xi=ns*p[0]-s,w=fe.JxW(j);
    const auto interp=[&](const auto &a){return (1-xi)*a[s]+xi*a[s+1];};
    const double chi=(1/this->get_phase_field_handler().energetic_degradation({.5,.5},phi[j])-1)/interp(I),v=interp(V);
    const auto tau=2*G*Utilities::Tensors::consistent_deviator(r[j])-2*interp(Z)*chi*S;Enew+=w*((tau*tau)+std::pow(trace(tau),2))/(4*G);
    const double g=this->get_phase_field_handler().energetic_degradation({.5,.5},phi[j]);Ymax=std::max(Ymax,F::recoverable_phase_driving_force(G,g,interp(Z)/interp(I)));
    if(dump)bulk<<p[0]<<','<<p[1]<<','<<w<<','<<tau[0][0]<<','<<tau[1][1]<<','<<tau[0][1]<<','<<strain[j][0][0]<<','<<strain[j][1][1]<<','<<strain[j][0][1]<<'\n';
    if(!step)continue;
    const double oldchi=(1/this->get_phase_field_handler().energetic_degradation({.5,.5},oldphi[j])-1)/interp(previous_I);
    const auto tauold=2*G*Utilities::Tensors::consistent_deviator(oldr[j])-2*interp(previous_Z)*oldchi*S;
    const auto jump=2*G*Utilities::Tensors::consistent_deviator(oldr[j])-2*interp(previous_Z)*chi*S;
    Eold+=w*((tauold*tauold)+std::pow(trace(tauold),2))/(4*G);Ejump+=w*((jump*jump)+std::pow(trace(jump),2))/(4*G);
    const auto coeff=F::maxwell_coefficients(eta,G,dt);const auto expected=coeff.first*jump+2*coeff.second*(Utilities::Tensors::consistent_deviator(strain[j])-chi*v*S);
    stress_error=std::max(stress_error,(tau-expected).norm());
    for(unsigned int k=0;k<time_quad.size();++k) {
     const auto c=F::maxwell_coefficients(eta,G,dt*time_quad.point(k)[0]);const auto t=c.first*jump+2*c.second*(Utilities::Tensors::consistent_deviator(strain[j])-chi*v*S);const double wt=w*dt*time_quad.weight(k);
     Db+=wt*((t*t)+std::pow(trace(t),2))/(2*eta);Wbulk+=wt*(t*strain[j]);Wboundary+=wt*(t*affine);
    }
   }
  }
  const QGauss<1>sq(3);double Ecoh_old=0,Ecoh_jump=0;
  if(step)for(unsigned int s=0;s<ns;++s)for(unsigned int j=0;j<sq.size();++j) {
   const double x=sq.point(j)[0],w=sq.weight(j)/ns;const auto interp=[&](const auto&a){return (1-x)*a[s]+x*a[s+1];};const double ii=interp(I),z0=interp(previous_Z),v=interp(V);
   Ecoh_old+=w*z0*z0/(2*G*interp(previous_I));Ecoh_jump+=w*z0*z0/(2*G*ii);
   Wfriction+=w*dt*1000*v;
   for(unsigned int k=0;k<time_quad.size();++k){const auto c=F::maxwell_coefficients(eta,G,dt*time_quad.point(k)[0]);const double z=c.first*z0+c.second*v,wt=w*dt*time_quad.weight(k);Dc+=wt*z*z/(eta*ii);Wcohesive+=wt*z*v/ii;}
  }
  const double release=step?previous_bulk+previous_cohesive+previous_F-entry_energy-Fnow:0;
  const double mech_error=step?Wboundary-(Enew+e.cohesive-previous_bulk-previous_cohesive)-Db-Dc-Wfriction-release-(Fnow-previous_F):0;
  double Hmax=0,Hgain=0;
  const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
  const auto hi=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
  for(const auto &p:pm.get_particle_handler()) {
   const double H=p.get_properties()[hi];if(!step)initial_H[p.get_id()]=H;
   else AssertThrow(H>=last_H.at(p.get_id()),ExcMessage("Accepted diagnostic peak H decreased."));
   Hmax=std::max(Hmax,H);Hgain+=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index()).volume()*(H-initial_H.at(p.get_id()));last_H[p.get_id()]=H;
  }
  std::ofstream metrics(this->get_output_directory()+"fe-metrics.csv",step?std::ios::app:std::ios::out);
  if(!step)metrics<<"step,time,dt,phi_max,max_x,max_y,increment,healing,KKT,bulk_energy,cohesive_energy,fracture,D_relax,mechanical_work_error,bulk_work,boundary_work,bulk_viscous,cohesive_viscous,friction,cohesive_work,bulk_profile_change,cohesive_profile_change,Maxwell_error_Pa,energy_reconstruction_error,Hmax_Pa,Hgain_J_m,Ymax_Pa\n";
  metrics<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<dt<<','<<maximum<<','<<maxpoint[0]<<','<<maxpoint[1]<<','<<increment<<','<<healing<<','<<entry_kkt<<','<<Enew<<','<<e.cohesive<<','<<Fnow<<','<<release<<','<<mech_error<<','<<Wbulk<<','<<Wboundary<<','<<Db<<','<<Dc<<','<<Wfriction<<','<<Wcohesive<<','<<Ejump-Eold<<','<<Ecoh_jump-Ecoh_old<<','<<stress_error<<','<<Enew-e.bulk<<','<<Hmax<<','<<Hgain<<','<<Ymax<<'\n';
  AssertThrow(stress_error<1e-7 && std::abs(Enew-e.bulk)<1e-8,ExcMessage("Native accepted-state Maxwell/energy consistency failed."));
  if(step==0 || step==1 || step==8 || this->get_time()>=1.)local_probe();
  previous.reinit(state);previous=state;previous_I=I;previous_Z=Z;previous_F=Fnow;previous_bulk=Enew;previous_cohesive=e.cohesive;
  return {"Restricted FE strain:","accepted-state and energy diagnostics retained"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"P2 native FE strain","Restricted accepted FE state, nonuniform native Maxwell and energy probes.")
}}
