// Same-field, same-quadrature mechanical work audit. No history writes.
#include "recoverable_feedback.cc"
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

namespace aspect { namespace Postprocess {
template<int dim> class CouplingFeedback : public RecoverableFeedback<dim> {
 using Base=RecoverableFeedback<dim>;
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 using FV=ReconstructedFaultVector;
 double adjoint_error=0, K_error=0, G_error=0, B_error=0;
 unsigned int derivative_step=numbers::invalid_unsigned_int;
public:
 void initialize() override {
  this->get_signals().post_particle_management.connect([this](Particle::Manager<dim>&) {
   if(this->get_timestep_number()==0 || this->get_reconstructed_fault_manager().get_faults().empty())return;
   double healing=0;
   for(auto i:this->get_solution().block(this->b()).locally_owned_elements())
    healing=std::max(healing,this->get_old_solution().block(this->b())[i]-this->get_solution().block(this->b())[i]);
   if(healing<=4.922e-12)return;
   using A=internal::PhaseFieldTestAccess<dim>;
   using BV=LinearAlgebra::BlockVector;
   const auto &original=this->get_system_matrix();LinearAlgebra::BlockSparseMatrix matrix;
   matrix.reinit(original.n_block_rows(),original.n_block_cols());
   for(unsigned int i=0;i<original.n_block_rows();++i)for(unsigned int j=0;j<original.n_block_cols();++j)
    matrix.block(i,j).copy_from(original.block(i,j));
   matrix.collect_sizes();
   BV rhs,correction;rhs.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());correction.reinit(rhs);
   const auto &phase=this->get_phase_field_handler();const auto &state=this->get_solution();
   A::assemble(phase,matrix,rhs,state,true);
   double kkt=0;unsigned int positive=0,negative=0;
   for(auto i:rhs.block(this->b()).locally_owned_elements()) {
    const double x=state.block(this->b())[i],r=rhs.block(this->b())[i];
    kkt=std::max(kkt,std::abs(x==0?std::max(0.,r):(x==1?std::min(0.,r):r)));
    for(auto it=matrix.block(this->b(),this->b()).begin(i);it!=matrix.block(this->b(),this->b()).end(i);++it)
     if(it->column()!=i) {positive+=it->value()>1e-14;negative+=it->value()<-1e-14;}
   }
   A::constrain(phase,matrix,rhs,state);A::solve(phase,matrix,rhs,correction);
   BV repeat(state);std::vector<double> errors;
   for(unsigned int k=0;k<2;++k) {
    A::bounded(phase,matrix,rhs,repeat);double error=0;
    for(auto i:state.block(this->b()).locally_owned_elements())error=std::max(error,std::abs(repeat.block(this->b())[i]-state.block(this->b())[i]));
    errors.push_back(error);
   }
   double remaining=0;for(auto i:state.block(this->b()).locally_owned_elements())remaining=std::max(remaining,this->get_old_solution().block(this->b())[i]-repeat.block(this->b())[i]);
   std::ofstream out(this->get_output_directory()+"healing-diagnosis.csv");
   out<<std::setprecision(17)<<"time,healing,KKT,correction_inf,repeat1,repeat2,repeated_healing,positive_offdiagonals,negative_offdiagonals\n"<<this->get_time()<<','<<healing<<','<<kkt<<','<<correction.block(this->b()).linfty_norm()<<','<<errors[0]<<','<<errors[1]<<','<<remaining<<','<<positive<<','<<negative<<'\n';
   this->phase_check(); // Includes unchanged incoming particle and surface histories.
  });
  Base::initialize();
  this->get_signals().post_reconstructed_fault_linear_solver.connect(
   [this](const auto &,const auto &,const auto &,const auto &,const auto &,const auto &,double,unsigned int,double) {
    // A globally indexed, homogeneous constrained bulk direction and a
    // nonconstant surface direction exercise the actual frozen B and G.
    LinearAlgebra::BlockVector owned;
    owned.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
    const auto vb=this->introspection().block_indices.velocities;
    for(auto i:owned.block(vb).locally_owned_elements()) owned.block(vb)[i]=std::sin(.73*i);
    owned.compress(VectorOperation::insert);this->get_current_constraints().set_zero(owned);
    LinearAlgebra::BlockVector ghost(this->get_solution());ghost=owned;
    FV v(1,std::vector<double>(this->get_reconstructed_fault_manager().get_fault(0).n_vertices()));
    for(unsigned int i=0;i<v[0].size();++i)v[0][i]=std::cos(.97*i);
    LinearAlgebra::BlockVector B;B.reinit(owned);FV G;
    this->get_reconstructed_fault_stokes_coupling().apply_B(v,B);
    this->get_reconstructed_fault_surface_system().apply_G(ghost,G);
    double vg=0;for(unsigned int i=0;i<v[0].size();++i)vg+=v[0][i]*G[0][i];
    const double ub=owned*B;
    adjoint_error=std::max(adjoint_error,std::abs(ub-vg)/std::max({1.,std::abs(ub),std::abs(vg)}));
    const auto &rates=this->get_reconstructed_fault_manager().get_slip_rate(0);
    if (derivative_step!=this->get_timestep_number() && *std::max_element(rates.begin(),rates.end())>1e-5) {
     auto &surface=this->get_reconstructed_fault_surface_system();
     auto &bulk=this->get_reconstructed_fault_stokes_coupling();
     const auto &state=this->get_current_linearization_point();
     FV rate(1,rates),d=rate,plus=rate,minus=rate,K;
     const double epsilon=1e-4;
     for(unsigned int i=0;i<rates.size();++i) {
      d[0][i]*=.2+.1*std::sin(i);plus[0][i]+=epsilon*d[0][i];minus[0][i]-=epsilon*d[0][i];
     }
     surface.apply_surface_jacobian(d,K);
     const auto rp=surface.evaluate_surface_residual(state,plus).values;
     const auto rm=surface.evaluate_surface_residual(state,minus).values;
     double error=0,scale=0;
     for(unsigned int i=0;i<rates.size();++i) {error+=std::pow((rp[0][i]-rm[0][i])/(2*epsilon)+K[0][i],2);scale+=K[0][i]*K[0][i];}
     K_error=std::sqrt(error/std::max(1e-30,scale));
     LinearAlgebra::BlockVector xp(state),xm(state);xp.add(epsilon,ghost);xm.add(-epsilon,ghost);
     const auto gp=surface.evaluate_surface_residual(xp,rate).values,gm=surface.evaluate_surface_residual(xm,rate).values;
     error=scale=0;
     for(unsigned int i=0;i<rates.size();++i) {error+=std::pow((gp[0][i]-gm[0][i])/(2*epsilon)-G[0][i],2);scale+=G[0][i]*G[0][i];}
     G_error=std::sqrt(error/std::max(1e-30,scale));
     LinearAlgebra::BlockVector bp,bm,bd;bp.reinit(owned);bm.reinit(owned);bd.reinit(owned);
     bulk.apply_B(d,bd);bulk.evaluate_slip_dependent_bulk_residual(state,plus,bp);bulk.evaluate_slip_dependent_bulk_residual(state,minus,bm);
     bp-=bm;bp*=.5/epsilon;bp+=bd;B_error=bp.l2_norm()/std::max(1e-30,bd.l2_norm());
     AssertThrow(std::max({K_error,G_error,B_error})<1e-7,ExcMessage("Coupled residual/tangent finite difference failed."));
     derivative_step=this->get_timestep_number();
    }
    if(this->material().uses_cohesive_work_measure())
     AssertThrow(adjoint_error<1e-11,ExcMessage("Native B/G are not work adjoints."));
   });
 }
 std::pair<std::string,std::string> execute(TableHandler &table) override {
  const auto result=Base::execute(table);const auto step=this->get_timestep_number();
  const auto &intro=this->introspection();const auto &quad=intro.quadratures.velocities;
  auto &manager=this->get_reconstructed_fault_manager();auto &pm=this->pm();
  const auto &fault=manager.get_fault(0);const auto &v=manager.get_timestep_committed_slip_rate(0);
  const auto I=F::current_normalization_integrals(this->material()).at(0);
  const auto qnodes=this->property("phase field fault cohesive traction");
  const double dt=step?this->get_timestep():1.,G=1e6,eta=1e8,lambda=G/eta;
  const auto coefficients=F::maxwell_coefficients(eta,G,dt);
  const double beta=coefficients.first,kappa=coefficients.second,one=-std::expm1(-lambda*dt);
  FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
  std::vector<double> phi(quad.size()),oldphi(phi),temp(phi);
  std::vector<SymmetricTensor<2,dim>> strain(quad.size());
  std::vector<std::vector<double>> compositions(intro.n_compositional_fields,phi);
  std::array<unsigned int,SymmetricTensor<2,dim>::n_independent_components> stress_fields;
  for(const auto &entry:this->get_parameters().mapped_particle_properties)
   if(entry.second.first=="maxwell stress")stress_fields[entry.second.second]=entry.first;
  const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
  ComponentMask mask(pm.get_particle_handler().n_properties_per_particle(),false);
  for(unsigned int c=0;c<stress_fields.size();++c)mask.set(sp+c,true);
  double oldE=0,newE=0,mappedE=0,Db=0,W=0,exchange=0,end_exchange=0;
  double transfer2=0,stress2=0,chi_measure=0,q_point_error=0,profile_work=0,profile_integral=0;
  std::vector<double> adjoint(v.size(),0.);
  const auto &working=this->get_current_linearization_point();
  for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned()) {
   fe.reinit(cell);
   const auto &a=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
   const FEValuesExtractors::Scalar phase(intro.variable("phase_field").first_component_index);
   fe[phase].get_function_values(working,phi);fe[phase].get_function_values(this->get_old_solution(),oldphi);
   fe[intro.extractors.velocities].get_function_symmetric_gradients(working,strain);
   for(unsigned int c=0;c<compositions.size();++c)fe[intro.extractors.compositional_fields[c]].get_function_values(working,compositions[c]);
   const auto mapped=pm.get_interpolator().properties_at_points(pm.get_particle_handler(),fe.get_quadrature_points(),mask,cell);
   for(unsigned int j=0;j<quad.size();++j) {
    AssertThrow(a[j].active,ExcMessage("Work audit requires full cross-section association."));
    const auto s=a[j].segment_index;const double x=a[j].xi,w=fe.JxW(j);
    const auto interp=[&](const auto &f){return (1-x)*f[s]+x*f[s+1];};
    const double ii=interp(I),oldI=interp(this->previous_I),V=interp(v);
    const double oldZ=(1-x)*this->previous_I[s]*this->previous_q[s]+x*this->previous_I[s+1]*this->previous_q[s+1];
    const double gg=this->get_phase_field_handler().energetic_degradation({.5,.5},phi[j]);
    const double oldg=this->get_phase_field_handler().energetic_degradation({.5,.5},step?oldphi[j]:phi[j]);
    const auto cohesive=F::compute_cohesive_response(beta,kappa,ii,oldI,oldZ/oldI,V,1/gg-1,1/oldg-1);
    const auto S=manager.get_shear_sense(0)*symmetrize(outer_product(a[j].tangent,a[j].normal));
    SymmetricTensor<2,dim> old,mapped_stress;
    for(unsigned int c=0;c<stress_fields.size();++c) {
     const auto component=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
     old[component]=compositions[stress_fields[c]][j];mapped_stress[component]=mapped[j][sp+c];
    }
    const auto rate=strain[j]-cohesive.crack_strain_rate*S;
    const auto stress=beta*old+2*kappa*rate;
    const auto steady=2*eta*rate,transient=old-steady;
    const auto integral=steady*dt+transient*one/lambda;
    const double square=(steady*steady)*dt+2*(steady*transient)*one/lambda+(transient*transient)*(-std::expm1(-2*lambda*dt))/(2*lambda);
    oldE+=w*(old*old)/(4*G);newE+=w*(stress*stress)/(4*G);mappedE+=w*(mapped_stress*mapped_stress)/(4*G);
    Db+=w*square/(2*eta);W+=w*(integral*strain[j]);exchange+=w*(integral*S)*cohesive.crack_strain_rate;
    profile_work+=w*(integral*S)*cohesive.history_correction;
    profile_integral+=w*cohesive.history_correction;
    const double chi=cohesive.localization_factor;
    end_exchange+=w*(stress*S)*chi*V;adjoint[s]+=w*chi*(1-x)*(stress*S);adjoint[s+1]+=w*chi*x*(stress*S);
    chi_measure+=w*chi;transfer2+=w*std::pow((mapped_stress-stress).norm(),2);stress2+=w*(stress*stress);
    const double published=((1-x)*I[s]*qnodes[s]+x*I[s+1]*qnodes[s+1])/ii;
    if(step)q_point_error=std::max(q_point_error,std::abs(published-cohesive.cohesive_traction));
   }
  }
  const auto comm=this->get_mpi_communicator();
  for(double *x:{&oldE,&newE,&mappedE,&Db,&W,&exchange,&end_exchange,&transfer2,&stress2,&chi_measure,&profile_work,&profile_integral})*x=Utilities::MPI::sum(*x,comm);
  double surface_pair=0;for(unsigned int i=0;i<v.size();++i)surface_pair+=v[i]*Utilities::MPI::sum(adjoint[i],comm);
  double cohesive_work=0,cohesive_end=0,friction_work=0,fault_phase_jump=0;
  const QGauss<1> surface_quad(8);
  for(unsigned int s=0;s<fault.n_cells();++s)for(unsigned int j=0;j<surface_quad.size();++j) {
   const double x=surface_quad.point(j)[0],w=surface_quad.weight(j)*fault.vertex(s).distance(fault.vertex(s+1));
   const auto interp=[&](const auto &f){return (1-x)*f[s]+x*f[s+1];};
   const double ii=interp(I),V=interp(v),oldZ=(1-x)*this->previous_I[s]*this->previous_q[s]+x*this->previous_I[s+1]*this->previous_q[s+1];
   fault_phase_jump+=w*oldZ*oldZ/(2*G)*(1/ii-1/interp(this->previous_I));
   const double steady=eta*V/ii,transient=oldZ/ii-steady;
   cohesive_work+=w*V*(steady*dt+transient*one/lambda);
   cohesive_end+=w*V*(beta*oldZ+kappa*V)/ii;
   friction_work+=w*1000.*V*dt;
  }
  std::ofstream out(this->get_output_directory()+"coupling-work.csv",step?std::ios::app:std::ios::out);
  if(!step)out<<"step,time,old_FE_energy,native_QP_energy,mapped_particle_energy,stress_transfer_relative,bulk_viscous,bulk_work,bulk_crack_work,bulk_identity_defect,cohesive_work,friction_work,time_integrated_exchange_defect,endpoint_exchange_defect,chi_measure,work_adjoint_identity,operator_adjoint_error,q_publication_error,K_error,G_error,B_error,profile_work,profile_integral,fault_phase_jump\n";
  out<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<oldE<<','<<newE<<','<<mappedE<<','<<std::sqrt(transfer2/std::max(1.,stress2))<<','<<Db<<','<<W<<','<<exchange<<','<<W-exchange-(newE-oldE)-Db<<','<<cohesive_work<<','<<friction_work<<','<<exchange-profile_work-cohesive_work-friction_work<<','<<end_exchange-cohesive_end-friction_work/dt<<','<<chi_measure<<','<<end_exchange-surface_pair<<','<<adjoint_error<<','<<q_point_error<<','<<K_error<<','<<G_error<<','<<B_error<<','<<profile_work<<','<<profile_integral<<','<<fault_phase_jump<<'\n';
  return result;
 }
};
ASPECT_REGISTER_POSTPROCESSOR(CouplingFeedback,"P2 coupling feedback","Matching quadrature work and stress-transfer diagnosis.")
}}
