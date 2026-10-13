// Fixed-state native operators and compact fresh-trajectory observer. No history import.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include "../../../tests/phase_field_test_access.h"
#include "../../../tests/phase_field_fault_test_access.h"
#include <fstream>
#include <iomanip>
#include <cstring>
#include <chrono>

namespace aspect { namespace Postprocess {
template<int dim> class NativeEndpoint : public Interface<dim>, public SimulatorAccess<dim>
{
 using BV=LinearAlgebra::BlockVector;
 using Model=MaterialModel::PhaseFieldFault<dim>;
 using A=internal::PhaseFieldTestAccess<dim>;
 using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
 std::string input,initial_history,fixed_entries,schedule_file;
 std::vector<double> expected_times;
 bool all_step_energies=false,complete_fixed_entry=false,direct_benchmark=false;
 bool initialized_H=false;
 bool recovery_test=false,recovery_done=false,inject_failure=false;
 unsigned int failure_step=1;
 BV previous;
 BV previous_strain;
 std::vector<double> previous_Z;
 double previous_energy=0.;
 std::vector<double> slip;
 unsigned int pb()const{return this->introspection().variable("phase_field").block_index;}
 std::string root_file(const std::string &path)const {
  return Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0?path:"/dev/null";
 }
 std::vector<Point<dim>> phase_points()const {
  const auto &state=this->get_solution();const auto &dofs=A::vertex_dofs(this->get_phase_field_handler());
  const auto offset=state.get_block_indices().block_start(pb());const unsigned int n=state.block(pb()).size();
  std::vector<double> data((dim+1)*n,0.);
  for(unsigned int v=0;v<dofs.size();++v)if(dofs[v]!=numbers::invalid_dof_index) {
   const auto i=dofs[v]-offset;if(!state.block(pb()).locally_owned_elements().is_element(i))continue;
   const auto point=this->get_triangulation().get_vertices()[v];
   for(unsigned int d=0;d<dim;++d)data[(dim+1)*i+d]=point[d];
   data[(dim+1)*i+dim]=1.;
  }
  aspect::internal::endpoint_sum(data,this->get_mpi_communicator());
  std::vector<Point<dim>> points(n);
  for(unsigned int i=0;i<n;++i){AssertThrow(data[(dim+1)*i+dim]==1.,ExcMessage("Missing/duplicate phase owner."));
   for(unsigned int d=0;d<dim;++d)points[i][d]=data[(dim+1)*i+d];}
  return points;
 }
 unsigned int tangential_cells()const {
  auto cell=this->get_dof_handler().begin_active();
  while(cell!=this->get_dof_handler().end() && !cell->is_locally_owned())++cell;
  AssertThrow(cell!=this->get_dof_handler().end(),ExcMessage("Diagnostic rank has no bulk cells."));
  return std::lround(1./(cell->vertex(1)[0]-cell->vertex(0)[0]));
 }

 Model &model()const{return const_cast<Model&>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
 struct Profile: public Function<dim>
 {
  const std::vector<double> &phi,&R;
  unsigned int phase_component,ux_component;
  Profile(unsigned int nc,unsigned int pc,unsigned int uc,const std::vector<double>&p,const std::vector<double>&r)
   :Function<dim>(nc),phi(p),R(r),phase_component(pc),ux_component(uc){}
  double value(const Point<dim>&x,unsigned int c=0)const override {
   const unsigned int n=phi.size()-1,j=std::min<unsigned int>(n-1,x[1]*n);
   const double s=x[1]*n-j;
   if(c==phase_component)return (1-s)*phi[j]+s*phi[j+1];
   if(c==ux_component)return (1-3*s+2*s*s)*R[2*j]+(4*s-4*s*s)*R[2*j+1]+(-s+2*s*s)*R[2*j+2];
   return 0.;
  }
 };
 BV field(const std::vector<double>&p,const std::vector<double>&R)const {
  const auto &intro=this->introspection();
  BV owned(intro.index_sets.system_partitioning,this->get_mpi_communicator());owned=this->get_solution();
  Profile function(intro.n_components,intro.variable("phase_field").first_component_index,
                   intro.component_indices.velocities[0],p,R);
  std::vector<bool> mask(intro.n_components,false);mask[intro.variable("phase_field").first_component_index]=true;
  for(unsigned int d=0;d<dim;++d)mask[intro.component_indices.velocities[d]]=true;
  VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),function,owned,ComponentMask(mask));
  BV ghost(this->get_solution());ghost=owned;return ghost;
 }
 void set_phase(const BV&value) {
  const_cast<BV&>(this->get_solution())=value;
  const_cast<BV&>(this->get_current_linearization_point())=value;
 }
 struct SavedSurface {
  unsigned int q,I;
  std::vector<double> qvalue,Ivalue;
 };
 SavedSurface surface()const {
  auto &m=this->get_reconstructed_fault_manager();const auto &info=m.get_property_information();
  SavedSurface s;s.q=info[m.get_property_index("phase field fault cohesive traction")].position;
  s.I=info[m.get_property_index("phase field fault previous I h")].position;
  const auto &fault=m.get_fault(0);
  for(unsigned int i=0;i<fault.n_vertices();++i){s.qvalue.push_back(fault.get_properties(i)[s.q]);s.Ivalue.push_back(fault.get_properties(i)[s.I]);}
  return s;
 }
 void set_surface(const SavedSurface&s) {
  auto &fault=const_cast<ReconstructedFaultManager<dim>&>(this->get_reconstructed_fault_manager()).get_fault(0);
  for(unsigned int i=0;i<fault.n_vertices();++i){fault.get_properties(i)[s.q]=s.qvalue[i];fault.get_properties(i)[s.I]=s.Ivalue[i];}
 }
 void benchmark_surface_schur(const std::string &name,Simulator<dim> &simulator,
                              const BV &bulk,const ReconstructedFaultVector &rates) {
  using Lin=typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization;
  auto &manager=this->get_reconstructed_fault_manager();auto &surface_system=this->get_reconstructed_fault_surface_system();
  const auto endpoint=model().make_fe_endpoint_trial(bulk,rates);
  const auto points=phase_points();
  StokesSolver::ReconstructedFaultCondensedSystem<dim> owner(simulator);
  auto linearization=std::make_unique<Lin>(owner.linearize(this->get_system_matrix(),bulk,rates));
  auto active=manager.prescribed_slip_rate_mask();
  for(unsigned int f=0;f<rates.size();++f)for(unsigned int i=0;i<rates[f].size();++i)
   if(rates[f][i]<=model().minimum_fault_slip_rate() && linearization->surface_residual().values[f][i]<=0.)active[f][i]=true;
  std::shared_ptr<ReconstructedFaultSurfaceLinearSolve<dim>> restricted;
  auto restrict_surface=[&](){restricted=surface_system.create_restricted_linear_solve(active);
   linearization=std::make_unique<Lin>(linearization->with_surface_solve(*restricted));};
  if(std::any_of(active[0].begin(),active[0].end(),[](bool x){return x;}))restrict_surface();
  std::ofstream out(root_file(this->get_output_directory()+"surface-schur.csv"));
  out<<std::setprecision(17)<<"case,contact,direction,solver,seconds,linear_calls,Krylov,rhs,bulk_residual,free_contact,velocity_error,pressure_error,rate_error,action_error,action_scale,rss_kib,peak_kib\n";
  std::ofstream actions(root_file(this->get_output_directory()+"surface-schur-actions.csv"));
  actions<<std::setprecision(17)<<"contact,direction,solver,x,y,action\n";
  // Normal, translation-like and genuinely tangential phase forcings exercise
  // the native normalization derivative and all u/p/V response components.
  for(unsigned int contact=0;contact<2;++contact) {
   if(contact==1){active[0][0]=true;restrict_surface();}
   for(unsigned int k=0;k<8;++k) {
    const unsigned int mode=k%4;
    std::vector<double>d(points.size());
    for(unsigned int i=0;i<d.size();++i){const double y=points[i][1]-.5,x=points[i][0];
     d[i]=std::exp(-y*y/.04)*(mode==1?y/.2:(mode==2?std::cos(2*numbers::PI*x):1.));}
    auto forcing=model().evaluate_fe_endpoint_action(bulk,endpoint,d,nullptr,nullptr,true);
    BV load(forcing.bulk_rhs);ReconstructedFaultVector surface_load=forcing.surface_rhs;
    if(mode==3) {
     // A known full-system mechanical right-hand side, independent of phase
     // differentiation. Contains velocity, pressure and nonconstant surface modes.
     BV probe(this->introspection().index_sets.stokes_partitioning,this->get_mpi_communicator());
     for(const auto i:probe.block(0).locally_owned_elements())probe.block(0)[i]=1e-5*std::sin(.13*i);
     for(const auto i:probe.block(1).locally_owned_elements())probe.block(1)[i]=1e-6*std::cos(.17*i);
     ReconstructedFaultVector v=rates;for(unsigned int i=0;i<v[0].size();++i)v[0][i]=active[0][i]?0.:1e-5*std::cos(.23*i);
     BV applied(probe);linearization->apply_uncondensed_jacobian(probe,v,applied,surface_load);
     load.block(0)=applied.block(0);load.block(1)=applied.block(1);
     for(auto &f:surface_load)for(auto &value:f)value=-value;
    }
    BV condensed(this->introspection().index_sets.stokes_partitioning,this->get_mpi_communicator());
    linearization->build_condensed_rhs(load,surface_load,condensed);
    BV reference(condensed);ReconstructedFaultVector reference_V;Vector<double>reference_action;
    // Run FGMRES first. The direct first call below includes full bulk setup,
    // all 65 interface solves and the small surface factorization.
    for(unsigned int method=0;method<2;++method) {
     const auto start=std::chrono::steady_clock::now();BV response(condensed);response=0.;
     unsigned int iterations=A::endpoint_linear_solve(simulator,*linearization,active,condensed,bulk,response,method==1),calls=1;
     ReconstructedFaultVector v;linearization->recover_slip_rate_increment(response,surface_load,v);
     auto gradient=[&](){const auto physical=linearization->make_physical_bulk_direction(response);
      const auto result=model().evaluate_fe_endpoint_action(bulk,endpoint,d,&physical,&v);
      Vector<double>g(result.gradient.size());for(unsigned int i=0;i<g.size();++i)g[i]=result.gradient[i];return g;};
     auto action=gradient();
     BV error(condensed);ReconstructedFaultVector ev;
     auto residual=[&](){linearization->apply_uncondensed_jacobian(response,v,error,ev);
      error.block(0)-=load.block(0);error.block(1)-=load.block(1);
      for(unsigned int f=0;f<v.size();++f)for(unsigned int i=0;i<v[f].size();++i)ev[f][i]=active[f][i]?0.:ev[f][i]+surface_load[f][i];};
     if(method==0 && mode!=3) {
      bool accurate=false;
      for(unsigned int repeat=0;repeat<4;++repeat){residual();BV delta(condensed),r(condensed),b(load);b.block(0)=error.block(0);b.block(1)=error.block(1);b*=-1.;
       linearization->build_condensed_rhs(b,ev,r);
       iterations+=A::endpoint_linear_solve(simulator,*linearization,active,r,bulk,delta,false);++calls;
       ReconstructedFaultVector dv;linearization->recover_slip_rate_increment(delta,ev,dv);response+=delta;
       for(unsigned int f=0;f<v.size();++f)for(unsigned int i=0;i<v[f].size();++i)v[f][i]+=dv[f][i];
       auto updated=gradient(),change(updated);change-=action;action=updated;
       if(change.l2_norm()<=this->get_phase_field_handler().get_solver_parameters().linear_solver_tolerance*action.l2_norm()){accurate=true;break;}}
      AssertThrow(accurate,ExcMessage("Benchmark FGMRES action uncertainty unresolved."));
     }
     residual();double contact_error=0.;for(const auto &f:ev)for(double value:f)contact_error=std::max(contact_error,std::abs(value));
     double uerror=0.,perror=0.,verror=0.,aerror=0.;
     if(method==0){reference=response;reference_V=v;reference_action=action;}
     else {BV diff(response);diff-=reference;uerror=diff.block(0).linfty_norm();perror=diff.block(1).linfty_norm();
      for(unsigned int f=0;f<v.size();++f)for(unsigned int i=0;i<v[f].size();++i)verror=std::max(verror,std::abs(v[f][i]-reference_V[f][i]));
      Vector<double> diff_action(action);diff_action-=reference_action;aerror=diff_action.l2_norm();
      AssertThrow(mode==3 || aerror<=this->get_phase_field_handler().get_solver_parameters().linear_solver_tolerance*action.l2_norm(),ExcMessage("Direct Schur phase action disagrees with resolved FGMRES."));}
     Utilities::System::MemoryStats memory;Utilities::System::get_memory_stats(memory);
     if(k<3)for(unsigned int i=0;i<points.size();++i)actions<<contact<<','<<k<<','<<(method==0?"FGMRES":"direct")<<','<<points[i][0]<<','<<points[i][1]<<','<<action[i]<<'\n';
     actions.flush();
     out<<name<<','<<contact<<','<<k<<','<<(method==0?"FGMRES":"direct")<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        <<','<<calls<<','<<iterations<<','<<condensed.l2_norm()<<','<<error.l2_norm()<<','<<contact_error<<','<<uerror<<','<<perror<<','<<verror<<','<<aerror<<','<<action.l2_norm()<<','<<memory.VmRSS<<','<<memory.VmHWM<<'\n';out.flush();
    }
   }
  }
 }
 void write_fixed_stress(const std::string &name,const BV &state,const typename Model::FEEndpointTrial &endpoint,const std::vector<double> &I)const {
  const QGauss<1>normal(12);std::vector<Point<dim>>points;for(unsigned int q=0;q<normal.size();++q){Point<dim>x;x[0]=.5;x[1]=normal.point(q)[0];points.push_back(x);}
  FEValues<dim>fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(points),update_values|update_gradients|update_quadrature_points);
  std::vector<double>phi(points.size());std::vector<SymmetricTensor<2,dim>>R(points.size()),du(points.size());
  const auto scalar=FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index);
  SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
  std::ostringstream local;local<<std::setprecision(17);
  for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned()){
   const unsigned int nx=tangential_cells(),column=std::lround(nx*cell->vertex(0)[0]);if(column!=0 && column!=nx/2-1 && column!=nx-1)continue;
   fe.reinit(cell);fe[scalar].get_function_values(state,phi);
   fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(endpoint.strain,R);
   fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(state,du);
   for(unsigned int q=0;q<points.size();++q){const auto x=fe.quadrature_point(q);const unsigned int j=std::min<unsigned int>(63,64*x[0]);const double xi=64*x[0]-j;
    const double ih=(1-xi)*I[j]+xi*I[j+1],z=(1-xi)*endpoint.Z[j]+xi*endpoint.Z[j+1];
    const double chi=(1./this->get_phase_field_handler().energetic_degradation({.5,.5},phi[q])-1.)/ih;
    const auto tau=2e6*Utilities::Tensors::consistent_deviator(R[q])-2*z*chi*S;
    local<<x[0]<<','<<x[1]<<','<<phi[q]<<','<<chi<<','<<tau[0][0]<<','<<tau[0][1]<<','<<tau[1][1]<<','<<R[q][0][0]<<','<<R[q][0][1]<<','<<R[q][1][1]<<','<<du[q][0][0]<<','<<du[q][0][1]<<','<<du[q][1][1]<<'\n';
   }
  }
  std::ofstream out(root_file(this->get_output_directory()+name+"-stress.csv"));out<<"x,y,phase,chi,tau_xx,tau_xy,tau_yy,R_xx,R_xy,R_yy,rate_xx,rate_xy,rate_yy\n";
  for(const auto &rows:Utilities::MPI::all_gather(this->get_mpi_communicator(),local.str()))out<<rows;
 }
 void fixed_entry_diagnostics() {
  auto &simulator=const_cast<Simulator<dim>&>(this->get_simulator());
  const BV saved(this->get_solution()),linear(this->get_current_linearization_point()),accepted(model().accepted_fe_strain_state());
  const auto saved_surface=surface();const auto saved_I=F::current_normalization_integrals(model());
  auto &manager=const_cast<ReconstructedFaultManager<dim>&>(this->get_reconstructed_fault_manager());
  auto &pm=const_cast<Particle::Manager<dim>&>(this->get_particle_manager(0));
  std::vector<std::vector<double>> saved_particles,saved_properties;
  for(const auto &particle:pm.get_particle_handler()){const auto data=particle.get_properties();saved_particles.emplace_back(data.begin(),data.end());}
  for(unsigned int i=0;i<manager.get_fault(0).n_vertices();++i){const auto data=manager.get_fault(0).get_properties(i);saved_properties.emplace_back(data.begin(),data.end());}
  auto restore_properties=[&](){unsigned int k=0;for(auto &particle:pm.get_particle_handler()){
   auto data=particle.get_properties();std::copy(saved_particles[k].begin(),saved_particles[k].end(),data.begin());++k;}
   for(unsigned int i=0;i<saved_properties.size();++i){auto data=manager.get_fault(0).get_properties(i);std::copy(saved_properties[i].begin(),saved_properties[i].end(),data.begin());}};
  const auto saved_committed=manager.get_timestep_committed_slip_rate(0),saved_current=manager.get_slip_rate(0);
  std::ifstream list(fixed_entries+"/files.txt");std::string name;
  AssertThrow(bool(list),ExcMessage("Missing fixed-entry diagnostic list."));
  std::ofstream report(root_file(this->get_output_directory()+"fixed-entries.csv"));
  report<<std::setprecision(17)<<"case,time,dt,initial_potential,final_potential,KKT,phase_error,reflected_phase_error,bulk_residual,contact_rms,increment,healing,I_min,centroid,reference_centroid,velocity_error_selected,q_center,V_interval,slip_candidate,bulk,cohesive,fracture\n";
  while(list>>name) {
   std::ifstream in(fixed_entries+"/"+name);unsigned int n;double time,dt,Za,slip_a,Va,Zb,Vb,Vstart;
   in>>n>>time>>dt>>Za>>slip_a>>Va>>Zb>>Vb>>Vstart;
   auto values=[&](unsigned int size){std::vector<double>v(size);for(auto &x:v)in>>x;return v;};
   const auto lower=values(n+1),Ra=values(2*n+1),start=values(n+1),expected=values(n+1),
              u_start=values(2*n+1),u_expected=values(2*n+1);
   AssertThrow(bool(in) && n==512,ExcMessage("Malformed fixed-entry reference."));
   const BV incoming=field(lower,Ra),guess=field(start,u_start);
   typename Model::FEEndpointTrial old;old.strain.reinit(incoming);old.strain=incoming;old.Z.assign(65,Za);
   const auto old_energy=model().evaluate_fe_endpoint_energy(incoming,old);
   auto restore_clock=A::diagnostic_endpoint_clock(simulator,time,dt);
   auto restore_reference=F::diagnostic_fe_reference(model(),incoming,std::vector<double>(65,Za),old_energy.normalization);
   auto properties=saved_surface;
   properties.Ivalue=old_energy.normalization;
   for(unsigned int i=0;i<65;++i)properties.qvalue[i]=Za/properties.Ivalue[i];
   try {
    set_surface(properties);set_phase(guess);A::diagnostic_old_phase(simulator,incoming);
    auto &committed=const_cast<std::vector<double>&>(manager.get_timestep_committed_slip_rate(0));
    auto &current=const_cast<std::vector<double>&>(manager.get_slip_rate(0));
    committed.assign(65,Va);current.assign(65,Va);
    auto total=[&](const BV &bulk,const ReconstructedFaultVector &rates) {
     const auto endpoint=model().make_fe_endpoint_trial(bulk,rates);
     auto energy=model().evaluate_fe_endpoint_energy(bulk,endpoint);
     BV rhs(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());LinearAlgebra::BlockSparseMatrix matrix;
     double fracture=0.,allowance=0.;this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,bulk,fracture,allowance);
     const auto grhs=aspect::internal::endpoint_values(rhs.block(pb()),this->get_mpi_communicator());
     const auto phi=aspect::internal::endpoint_values(bulk.block(pb()),this->get_mpi_communicator());
     const auto obstacle=aspect::internal::endpoint_values(incoming.block(pb()),this->get_mpi_communicator());
     double kkt=0.;for(unsigned int i=0;i<energy.gradient.size();++i) {
      const double g=energy.gradient[i]-grhs[i];
      kkt=std::max(kkt,std::abs(phi[i]==obstacle[i]?std::min(g,0.):g));
     }
     return std::make_tuple(fracture+energy.energy+endpoint.friction_potential,kkt,energy.normalization);
    };
    // Initial mechanical state is independently supplied for this explicit
    // phase. Neither it nor the starting phase replaces the incoming anchor.
    const auto initial=total(guess,ReconstructedFaultVector(1,std::vector<double>(65,Vstart)));
    ReconstructedFaultVector rates;
    this->get_pcout()<<"Fixed-entry diagnostic: "<<name<<", time="<<time<<", dt="<<dt<<std::endl;
    const auto step_started=std::chrono::steady_clock::now();
    if(direct_benchmark) {
     set_phase(field(expected,u_expected));rates={std::vector<double>(65,Vb)};
     A::mechanical_trial(simulator,rates);
     benchmark_surface_schur(name,simulator,this->get_solution(),rates);
    } else if(complete_fixed_entry) {
     model().prepare_reconstructed_fault_mechanical_solve();
     if(std::getenv("ASPECT_BENCHMARK_MPI_PROFILE"))MPI_Pcontrol(101);
     if(recovery_test && !recovery_done) {
      recovery_done=true;const BV before(this->get_solution()),before_linear(this->get_current_linearization_point()),
       before_old(this->get_old_solution()),before_older(this->get_old_old_solution()),before_R(model().accepted_fe_strain_state());
      const auto before_I=F::current_normalization_integrals(model());const auto metadata=A::endpoint_bookkeeping(simulator);
      std::vector<std::vector<double>> before_particles;for(const auto &particle:pm.get_particle_handler()){
       const auto data=particle.get_properties();before_particles.emplace_back(data.begin(),data.end());}
      std::vector<std::vector<double>> before_fault;for(unsigned int i=0;i<manager.get_fault(0).n_vertices();++i){
       const auto data=manager.get_fault(0).get_properties(i);before_fault.emplace_back(data.begin(),data.end());}
      const auto failure_started=std::chrono::steady_clock::now();
      A::endpoint_failure_hook(simulator,[this](){
       if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())+1==Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()))
        throw std::runtime_error("controlled complete endpoint failure");});
      bool caught=false;try{A::complete_step(simulator);}catch(const std::runtime_error &e){caught=std::string(e.what())=="controlled complete endpoint failure";if(!caught)throw;}
      A::endpoint_failure_hook(simulator,{});
      this->get_pcout()<<"   Fixed-entry failure attempt seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-failure_started).count()<<std::endl;
      AssertThrow(caught && this->get_solution()==before && this->get_current_linearization_point()==before_linear
       && this->get_old_solution()==before_old && this->get_old_old_solution()==before_older
       && model().accepted_fe_strain_state()==before_R && F::current_normalization_integrals(model())==before_I
       && A::endpoint_bookkeeping(simulator)==metadata && surface().qvalue==properties.qvalue && surface().Ivalue==properties.Ivalue
       && manager.get_timestep_committed_slip_rate(0)==std::vector<double>(65,Va)
       && manager.get_slip_rate(0)==std::vector<double>(65,Va),ExcMessage("Loaded complete-step MPI recovery failed."));
      bool particles_unchanged=true;unsigned int k=0;for(const auto &particle:pm.get_particle_handler()){
       const auto data=particle.get_properties();particles_unchanged=particles_unchanged && data.size()==before_particles[k].size()
        && std::memcmp(data.data(),before_particles[k].data(),data.size()*sizeof(double))==0;++k;}
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(particles_unchanged),this->get_mpi_communicator())==1,
       ExcMessage("Loaded MPI failure changed particle histories."));
      for(unsigned int i=0;i<before_fault.size();++i){const auto data=manager.get_fault(0).get_properties(i);
       AssertThrow(data.size()==before_fault[i].size() && std::memcmp(data.data(),before_fault[i].data(),data.size()*sizeof(double))==0,
        ExcMessage("Loaded MPI failure changed surface property bytes."));}
      std::ofstream recovery(root_file(this->get_output_directory()+"loaded-complete-recovery.txt"));
      recovery<<"One rank failed after convergence, before publication. All accepted FE/surface/particle histories, phase histories and bookkeeping restored exactly. Retry follows.\n";
      if(std::getenv("ASPECT_BENCHMARK_MPI_PROFILE"))MPI_Pcontrol(101);
     }
     const auto accepted_started=std::chrono::steady_clock::now();
     A::complete_step(simulator);rates={manager.get_timestep_committed_slip_rate(0)};
     this->get_pcout()<<"   Fixed-entry accepted-step seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-accepted_started).count()<<std::endl;
     if(std::getenv("ASPECT_BENCHMARK_MPI_PROFILE"))MPI_Pcontrol(101);
    } else A::endpoint_trial(simulator,guess,rates);
    this->get_pcout()<<"   Fixed-entry whole-step seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-step_started).count()
     <<", complete publication="<<complete_fixed_entry<<std::endl;
    const BV candidate(this->get_solution());const auto published_surface=surface();
    // Evaluate the completed state against the immutable incoming reference,
    // rather than advancing its newly published Maxwell history a second time.
    std::function<void()> restore_published;
    if(complete_fixed_entry){restore_published=F::diagnostic_fe_reference(model(),incoming,std::vector<double>(65,Za),old_energy.normalization);set_surface(properties);}
    const auto final=total(candidate,rates);
    const auto residual=A::endpoint_defects(simulator,candidate,rates);
    double error=0.,mirror=0.,increment=0.,healing=0.,centroid_n=0.,centroid_d=0.,reference_n=0.,reference_d=0.;
    std::ofstream fields(root_file(this->get_output_directory()+name+"-phase.csv"));fields<<std::setprecision(17)<<"x,y,phase,lower,reference\n";
    const auto points=phase_points();
    const auto phase=aspect::internal::endpoint_values(candidate.block(pb()),this->get_mpi_communicator());
    const auto obstacle=aspect::internal::endpoint_values(incoming.block(pb()),this->get_mpi_communicator());
    for(unsigned int v=0;v<points.size();++v) {
     const auto point=points[v];const unsigned int j=std::lround(n*point[1]);
     const double phi=phase[v],lo=obstacle[v];
     error=std::max(error,std::abs(phi-expected[j]));mirror=std::max(mirror,std::abs(phi-expected[n-j]));
     increment=std::max(increment,phi-lo);healing=std::max(healing,lo-phi);
     fields<<point[0]<<','<<point[1]<<','<<phi<<','<<lo<<','<<expected[j]<<'\n';
     if(point[0]==.5) {
      const double weight=(j==0 || j==n)?.5:1.;centroid_n+=weight*point[1]*phi;centroid_d+=weight*phi;
      reference_n+=weight*point[1]*expected[j];reference_d+=weight*expected[j];
     }
    }
    const auto &I=std::get<2>(final);
    auto mirror_phi=expected,mirror_u=u_expected;
    std::reverse(mirror_phi.begin(),mirror_phi.end());std::reverse(mirror_u.begin(),mirror_u.end());
    for(auto &u:mirror_u)u=-u;
    const auto expected_bulk=error<=mirror?field(expected,u_expected):field(mirror_phi,mirror_u);
    auto velocity_difference=candidate.block(this->introspection().block_indices.velocities);
    velocity_difference-=expected_bulk.block(this->introspection().block_indices.velocities);
    const auto endpoint=model().make_fe_endpoint_trial(candidate,rates);
    const auto stored=model().evaluate_fe_endpoint_energy(candidate,endpoint);
    write_fixed_stress(name,candidate,endpoint,I);
    std::ofstream surface_out(root_file(this->get_output_directory()+name+"-surface.csv"));
    surface_out<<std::setprecision(17)<<"x,I,q,V_interval,slip\n";
    for(unsigned int i=0;i<65;++i)surface_out<<double(i)/64<<','<<I[i]<<','<<endpoint.Z[i]/I[i]<<','<<rates[0][i]<<','<<slip_a+dt*rates[0][i]<<'\n';
    report<<name<<','<<time<<','<<dt<<','<<std::get<0>(initial)<<','<<std::get<0>(final)<<','<<std::get<1>(final)
          <<','<<error<<','<<mirror<<','<<residual.first<<','<<residual.second<<','<<increment<<','<<healing
          <<','<<*std::min_element(I.begin(),I.end())<<','<<centroid_n/centroid_d<<','<<reference_n/reference_d
          <<','<<velocity_difference.linfty_norm()<<','<<endpoint.Z[32]/I[32]<<','<<rates[0][32]<<','<<slip_a+dt*rates[0][32]<<','<<stored.bulk<<','<<stored.cohesive<<','<<std::get<0>(final)-stored.energy-endpoint.friction_potential<<'\n';report.flush();
    if(complete_fixed_entry) {
     restore_published();set_surface(published_surface);
     AssertThrow(model().accepted_fe_strain_state()==endpoint.strain
       && manager.get_timestep_committed_slip_rate(0)==rates[0],ExcMessage("Complete fixed-entry FE state did not publish once."));
     for(unsigned int i=0;i<65;++i)AssertThrow(std::abs(published_surface.qvalue[i]*published_surface.Ivalue[i]-endpoint.Z[i])
       <=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(endpoint.Z[i])),ExcMessage("Complete fixed-entry surface publication mismatch."));
     this->get_pcout()<<"   Complete retained-entry publication verified; diagnostic histories restored afterward."<<std::endl;
    }
    AssertThrow(healing==0. && (complete_fixed_entry || (model().accepted_fe_strain_state()==incoming
                && manager.get_timestep_committed_slip_rate(0)==std::vector<double>(65,Va)
                && surface().qvalue==properties.qvalue && surface().Ivalue==properties.Ivalue)),
                ExcMessage("Fixed-entry solve changed the obstacle or accepted history."));
   } catch(...) {
    restore_properties();restore_reference();restore_clock();set_phase(saved);const_cast<BV&>(this->get_current_linearization_point())=linear;
    set_surface(saved_surface);F::restore_diagnostic_normalization(model(),saved_I);
    const_cast<std::vector<double>&>(manager.get_timestep_committed_slip_rate(0))=saved_committed;
    const_cast<std::vector<double>&>(manager.get_slip_rate(0))=saved_current;throw;
   }
   restore_properties();restore_reference();restore_clock();set_phase(saved);const_cast<BV&>(this->get_current_linearization_point())=linear;
   set_surface(saved_surface);F::restore_diagnostic_normalization(model(),saved_I);
   const_cast<std::vector<double>&>(manager.get_timestep_committed_slip_rate(0))=saved_committed;
   const_cast<std::vector<double>&>(manager.get_slip_rate(0))=saved_current;
  }
  AssertThrow(model().accepted_fe_strain_state()==accepted,ExcMessage("Diagnostic reference was not restored."));
 }
 void operators() {
  AssertThrow(dim==2 && this->get_triangulation().n_global_active_cells()==8192,ExcMessage("Native operator fixture is 16x512."));
  const BV saved(this->get_solution()),linear(this->get_current_linearization_point()),accepted(model().accepted_fe_strain_state());
  const auto history=surface();const auto committed=this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0);
  const auto coordinates=phase_points();
  std::ifstream names(input+"/files.txt");std::string name;
  AssertThrow(bool(names),ExcMessage("Missing native operator input list."));
  std::ofstream report(root_file(this->get_output_directory()+"operators.csv"));
  report<<std::setprecision(17)<<"case,bulk_error,cohesive_error,fracture_error,potential_error,gradient_max_error,I_relative_error,DI_error,weak_force,contact,point_stress_error,full_gradient_error\n";
  std::ofstream gradients(root_file(this->get_output_directory()+"operator-gradients.csv"));gradients<<std::setprecision(17)<<"case,y,native,reference\n";
  std::ofstream adjoint(root_file(this->get_output_directory()+"operator-adjoint.csv"));adjoint<<std::setprecision(17)<<"case,eps,energy_derivative,gradient_pairing,error,I_action_relative_error\n";
  std::ofstream quadratic(root_file(this->get_output_directory()+"operator-fracture-quadratic.csv"));
  quadratic<<std::setprecision(17)<<"case,eps,energy_error,gradient_error\n";
  std::ofstream tangent(root_file(this->get_output_directory()+"operator-linear-tangent.csv"));
  tangent<<std::setprecision(17)<<"case,direction,eps,maximum_error,action_scale\n";
  try {
   while(names>>name) {
    std::ifstream in(input+"/"+name);unsigned int n;double time,dt,Za,Zb,V,Iexpected,eb,ec,fracture,potential,DIexpected;
    in>>n>>time>>dt>>Za>>Zb>>V>>Iexpected>>eb>>ec>>fracture>>potential>>DIexpected;
    AssertThrow(n==512 && dt==this->get_timestep(),ExcMessage("Operator state/step mismatch."));
    auto values=[&](unsigned int size){std::vector<double>x(size);for(auto &v:x)in>>v;AssertThrow(bool(in),ExcMessage("Malformed operator state."));return x;};
    const auto phi=values(n+1),lower=values(n+1),Ra=values(2*n+1),Rb=values(2*n+1),u=values(2*n+1),reference=values(n+1),direction=values(n+1);
    const BV state=field(phi,u),strain=field(phi,Rb),incoming=field(lower,Ra);
    typename Model::FEEndpointTrial endpoint;endpoint.strain.reinit(strain);endpoint.strain=strain;endpoint.Z.assign(65,Zb);
    typename Model::FEEndpointTrial old;old.strain.reinit(incoming);old.strain=incoming;old.Z.assign(65,Za);
    std::vector<double> vector(coordinates.size());
    for(unsigned int i=0;i<vector.size();++i)vector[i]=direction[std::lround(coordinates[i][1]*n)];
    const auto e=model().evaluate_fe_endpoint_energy(state,endpoint,&vector);
    const auto ea=model().evaluate_fe_endpoint_energy(incoming,old);
    set_phase(state);const auto actual_I=F::compute_normalization_integrals(model()).at(0);
    auto synthetic=history;
    for(unsigned int i=0;i<65;++i){synthetic.Ivalue[i]=ea.normalization[i];synthetic.qvalue[i]=Za/ea.normalization[i];}
    set_surface(synthetic);
    LinearAlgebra::BlockSparseMatrix matrix;
    BV rhs(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
    double Fvalue=0.,allowance=0.;
    AssertThrow(this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,state,Fvalue,allowance),ExcInternalError());
    std::vector<double> collapsed(n+1,0.);
    double Ierror=0.,DIerror=0.;
    for(unsigned int i=0;i<e.normalization.size();++i){Ierror=std::max({Ierror,std::abs(e.normalization[i]/Iexpected-1.),std::abs(actual_I[i]/Iexpected-1.)});DIerror=std::max(DIerror,std::abs(e.normalization_action[i]-DIexpected));}
    const double scale=model().get_critical_energy_release_rates()[0]/((8./3.)*this->get_phase_field_handler().get_length_scale());
    const auto rhs_values=aspect::internal::endpoint_values(rhs.block(pb()),this->get_mpi_communicator());
    double full_gerror=0.;
    for(unsigned int i=0;i<coordinates.size();++i) {
     const unsigned int j=std::lround(coordinates[i][1]*n);const double g=(e.gradient[i]-rhs_values[i])/scale;
     const double tangential_measure=(coordinates[i][0]==0. || coordinates[i][0]==1.)?.5/tangential_cells():1./tangential_cells();
     full_gerror=std::max(full_gerror,std::abs(g-tangential_measure*reference[j]));collapsed[j]+=g;
    }
    double gerror=0.;for(unsigned int i=0;i<=n;++i){gerror=std::max(gerror,std::abs(collapsed[i]-reference[i]));gradients<<name<<','<<double(i)/n<<','<<collapsed[i]<<','<<reference[i]<<'\n';}
    if(name.find("-+0.txt")!=std::string::npos) {
     // Full native velocity/pressure/surface Schur action, with the explicit
     // fixed-state endpoint. Coulomb/prescribed-P A/B/G/K do not depend on the
     // accepted stress anchor; nonconstant phase directions still refresh I.
     auto &simulator=const_cast<Simulator<dim>&>(this->get_simulator());
     const ReconstructedFaultVector rates(1,std::vector<double>(65,std::max(V,model().minimum_fault_slip_rate())));
     auto linear_action=A::endpoint_tangent(simulator,state,rates,strain,endpoint.Z);
     Vector<double> native_direction(vector.size());
     for(unsigned int i=0;i<vector.size();++i)native_direction[i]=vector[i];
     const auto linear_result=linear_action(native_direction);
     const double configured=A::set_stokes_tolerance(simulator,1e-10);
     Vector<double> tight;
     try{tight=linear_action(native_direction);A::set_stokes_tolerance(simulator,configured);}
     catch(...){A::set_stokes_tolerance(simulator,configured);throw;}
     double inner_error=0.,inner_scale=0.;
     for(unsigned int i=0;i<vector.size();++i){inner_error=std::max(inner_error,std::abs(tight[i]-linear_result[i]));inner_scale=std::max(inner_scale,std::abs(tight[i]));}
     tangent<<name<<",inner-accuracy,"<<configured<<','<<inner_error<<','<<inner_scale<<'\n';tangent.flush();
     AssertThrow(inner_error<1e-6*std::max(1.,inner_scale),ExcMessage("Inner accuracy obscures tangent action."));
     for(double eps:{1e-5,5e-6}) {
      auto read_perturbed=[&](double offset) {
       std::ostringstream file;file<<input<<'/'<<name.substr(0,name.find("-+0.txt"))<<'-'<<std::showpos<<offset<<".txt";
       std::ifstream data(file.str());unsigned int count;double t,step,za,zb,v,ih,eb,ec,f,pot,di;
       data>>count>>t>>step>>za>>zb>>v>>ih>>eb>>ec>>f>>pot>>di;
       auto vals=[&](unsigned int size){std::vector<double>a(size);for(auto &x:a)data>>x;return a;};
       const auto ph=vals(count+1);vals(count+1);vals(2*count+1);
       const auto rr=vals(2*count+1),uu=vals(2*count+1);
       AssertThrow(bool(data),ExcMessage("Missing fixed-state tangent perturbation."));
       typename Model::FEEndpointTrial ep;const auto rrfield=field(ph,rr);ep.strain.reinit(rrfield);ep.strain=rrfield;ep.Z.assign(65,zb);
       return model().evaluate_fe_endpoint_energy(field(ph,uu),ep);
      };
      const auto plus=read_perturbed(eps),minus=read_perturbed(-eps);
      double error=0.,magnitude=0.;
      for(unsigned int i=0;i<vector.size();++i) {
       error=std::max(error,std::abs(linear_result[i]-(plus.gradient[i]-minus.gradient[i])/(2*eps)));
       magnitude=std::max(magnitude,std::abs(linear_result[i]));
      }
      tangent<<name<<",normal,"<<eps<<','<<error<<','<<magnitude<<'\n';tangent.flush();
      AssertThrow(error<1e-6*std::max(1.,magnitude),ExcMessage("Linearized native reduced tangent disagrees with finite differences."));
     }
     LinearAlgebra::BlockSparseMatrix stiffness;
     const auto &native_matrix=this->get_system_matrix();
     stiffness.reinit(native_matrix.n_block_rows(),native_matrix.n_block_cols());
     for(unsigned int i=0;i<stiffness.n_block_rows();++i)for(unsigned int j=0;j<stiffness.n_block_cols();++j)
      stiffness.block(i,j).copy_from(native_matrix.block(i,j));
     stiffness.collect_sizes();
     double Fa=0.,arithmetic=0.;
     A::assemble(this->get_phase_field_handler(),stiffness,rhs,state,true,&arithmetic,&Fa,true);
     BV ga(rhs);ga.block(pb())*=-1.;
     LinearAlgebra::Vector nd(this->introspection().index_sets.system_partitioning[pb()],this->get_mpi_communicator()),nk(nd);
     aspect::internal::endpoint_assign(nd,native_direction);
     stiffness.block(pb(),pb()).vmult(nk,nd);
     const auto global_nk=aspect::internal::endpoint_values(nk,this->get_mpi_communicator());
     std::vector<double>collapsed_action(n+1,0.);
     for(unsigned int i=0;i<vector.size();++i)collapsed_action[std::lround(coordinates[i][1]*n)]+=(linear_result[i]+global_nk[i])/scale;
     std::ifstream expected(input+"/"+name.substr(0,name.find("-+0.txt"))+"-action.txt");
     double prototype_error=0.,prototype_scale=0.,expected_value;
     for(unsigned int j=0;j<=n;++j){expected>>expected_value;prototype_error=std::max(prototype_error,std::abs(collapsed_action[j]-expected_value));prototype_scale=std::max(prototype_scale,std::abs(expected_value));}
     AssertThrow(bool(expected),ExcMessage("Missing exact prototype tangent."));
     tangent<<name<<",normal-prototype-exact,0,"<<prototype_error<<','<<prototype_scale<<'\n';tangent.flush();
     AssertThrow(prototype_error<1e-7,ExcMessage("Reduced native tangent disagrees with exact matching prototype."));
     // A tangentially varying direction exercises the real surface projection
     // and its adjoint, rather than only constant reproduction along the fault.
     std::vector<double>d(vector.size());double pairing=0.;
     for(unsigned int i=0;i<d.size();++i){d[i]=vector[i]*std::sin(numbers::PI*coordinates[i][0]);pairing+=(e.gradient[i]-rhs_values[i])*d[i];}
     const auto exact=model().evaluate_fe_endpoint_energy(state,endpoint,&d);
     const auto analytic=model().evaluate_fe_endpoint_action(state,endpoint,d);
     Vector<double> td(d.size());for(unsigned int i=0;i<d.size();++i)td[i]=d[i];
     const auto tangential_reduced=linear_action(td);
     LinearAlgebra::Vector distributed_mode(this->introspection().index_sets.system_partitioning[pb()],this->get_mpi_communicator()),mode_image(distributed_mode);
     aspect::internal::endpoint_assign(distributed_mode,td);stiffness.block(pb(),pb()).vmult(mode_image,distributed_mode);
     const auto mode_fracture=aspect::internal::endpoint_values(mode_image,this->get_mpi_communicator());
     BV owned_mode(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());owned_mode=0.;aspect::internal::endpoint_assign(owned_mode.block(pb()),td);
     BV ghost_mode(state);ghost_mode=owned_mode;
     FEValues<dim>mode_fe(this->get_mapping(),this->get_fe(),QGauss<dim>(2),update_values|update_JxW_values);
     std::vector<double>sampled(mode_fe.n_quadrature_points);double l2=0.,fracture_pairing=0.;
     for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned()){
      mode_fe.reinit(cell);mode_fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(ghost_mode,sampled);
      for(unsigned int q=0;q<sampled.size();++q)l2+=mode_fe.JxW(q)*sampled[q]*sampled[q];}
     l2=Utilities::MPI::sum(l2,this->get_mpi_communicator());
     for(unsigned int i=0;i<td.size();++i)fracture_pairing+=td[i]*mode_fracture[i];
     std::ofstream curvature(root_file(this->get_output_directory()+"tangential-curvature.csv"),std::ios::app);
     if(curvature.tellp()==0)curvature<<"case,phase_L2_squared,elastic_curvature,total_curvature\n";
     curvature<<std::setprecision(17)<<name<<','<<l2<<','<<(td*tangential_reduced)/l2<<','<<(td*tangential_reduced+fracture_pairing)/l2<<'\n';
     tangent<<name<<",tangential-reduced,0,0,"<<tangential_reduced.linfty_norm()<<'\n';tangent.flush();
     const double symmetry=std::abs(native_direction*tangential_reduced-td*linear_result);
     tangent<<name<<",mixed-symmetry,0,"<<symmetry<<','<<std::max(std::abs(native_direction*tangential_reduced),std::abs(td*linear_result))<<'\n';tangent.flush();
     for(double eps:{1e-5,5e-6}) {
      BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());owned=state;
      const auto base_phase=aspect::internal::endpoint_values(state.block(pb()),this->get_mpi_communicator());
      Vector<double> trial_phase(base_phase);for(unsigned int i=0;i<d.size();++i)trial_phase[i]+=eps*d[i];
      aspect::internal::endpoint_assign(owned.block(pb()),trial_phase);BV plus(state);plus=owned;
      for(unsigned int i=0;i<d.size();++i)trial_phase[i]=base_phase[i]-eps*d[i];
      aspect::internal::endpoint_assign(owned.block(pb()),trial_phase);BV minus(state);minus=owned;
      const auto ep=model().evaluate_fe_endpoint_energy(plus,endpoint),em=model().evaluate_fe_endpoint_energy(minus,endpoint);
      double tangent_error=0.,tangent_scale=0.;
      for(unsigned int i=0;i<d.size();++i) {
       tangent_error=std::max(tangent_error,std::abs(analytic.gradient[i]-(ep.gradient[i]-em.gradient[i])/(2*eps)));
       tangent_scale=std::max(tangent_scale,std::abs(analytic.gradient[i]));
      }
      tangent<<name<<",tangential-fixed-endpoint,"<<eps<<','<<tangent_error<<','<<tangent_scale<<'\n';tangent.flush();
      AssertThrow(tangent_error<1e-6*std::max(1.,tangent_scale),ExcMessage("Tangential normalization Hessian pullback failed."));
      double fp=0.,fm=0.,roundoff=0.;
      this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,plus,fp,roundoff);
      this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,minus,fm,roundoff);
      const double fd=(ep.energy+fp-em.energy-fm)/(2*eps);double derr=0.,dscale=1.;
      LinearAlgebra::Vector delta(this->introspection().index_sets.system_partitioning[pb()],this->get_mpi_communicator()),kd(delta);
      Vector<double> increment(d.size());for(unsigned int i=0;i<d.size();++i)increment[i]=eps*d[i];
      aspect::internal::endpoint_assign(delta,increment);
      stiffness.block(pb(),pb()).vmult(kd,delta);
      const double predicted=Fa+ga.block(pb())*delta+.5*(delta*kd);
      double actual=0.,ignored=0.;
      this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,plus,actual,ignored);
      const auto g0=aspect::internal::endpoint_values(ga.block(pb()),this->get_mpi_communicator()),
                 gk=aspect::internal::endpoint_values(kd,this->get_mpi_communicator()),
                 gr=aspect::internal::endpoint_values(rhs.block(pb()),this->get_mpi_communicator());
      double g_error=0.;for(unsigned int i=0;i<d.size();++i)g_error=std::max(g_error,std::abs(g0[i]+gk[i]+gr[i]));
      quadratic<<name<<','<<eps<<','<<std::abs(predicted-actual)<<','<<g_error<<'\n';quadratic.flush();
      AssertThrow(std::abs(predicted-actual)<1e-10 && g_error<1e-10,
                  ExcMessage("Native AT1 quadratic reuse differs from the independently integrated CPDI operator."));
      for(unsigned int i=0;i<65;++i){derr=std::max(derr,std::abs((ep.normalization[i]-em.normalization[i])/(2*eps)-exact.normalization_action[i]));dscale=std::max(dscale,std::abs(exact.normalization_action[i]));}
      adjoint<<name<<','<<eps<<','<<fd<<','<<pairing<<','<<std::abs(fd-pairing)<<','<<derr/dscale<<'\n';adjoint.flush();
      AssertThrow(std::abs(fd-pairing)<1e-5 && derr/dscale<1e-6,ExcMessage("Native normalization/energy adjoint derivative failed."));
     }
     // Explicit trial phase/reference must ignore unrelated predictor and
     // rotated phase blocks. All physical vectors are restored immediately.
     BV old(this->get_old_solution()),older(this->get_old_old_solution());
     auto &old_mutable=const_cast<BV&>(this->get_old_solution());auto &older_mutable=const_cast<BV&>(this->get_old_old_solution());
     auto &predictor=const_cast<BV&>(this->get_current_linearization_point());const BV before(predictor);
     try {
      old_mutable.block(pb())=.13;older_mutable.block(pb())=.23;predictor.block(pb())=.37;
      const auto poisoned=model().evaluate_fe_endpoint_energy(state,endpoint,&vector);
      AssertThrow(poisoned.energy==e.energy && poisoned.gradient==e.gradient && poisoned.normalization_action==e.normalization_action,
                  ExcMessage("Explicit native endpoint energy reads unrelated phase history."));
     } catch(...){old_mutable=old;older_mutable=older;predictor=before;throw;}
     old_mutable=old;older_mutable=older;predictor=before;
    }
    const auto quad=model().mechanical_quadrature();
    FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
    std::vector<double>ph(quad.size()),pha(quad.size());std::vector<SymmetricTensor<2,dim>>du(quad.size()),ra(quad.size());
    const auto scalar=FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index);
    Vector<double> force(this->get_dof_handler().n_dofs());
    std::vector<types::global_dof_index>indices(this->get_fe().n_dofs_per_cell());
    SymmetricTensor<2,dim>S,N;S[0][1]=.5;N[1][1]=1.;
    double bulk_pair=0.,strength=0.,stress_error=0.;
    for(const auto &cell:this->get_dof_handler().active_cell_iterators()) if(cell->is_locally_owned()) {
     fe.reinit(cell);cell->get_dof_indices(indices);
     fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(state,du);
     fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(incoming,ra);
     fe[scalar].get_function_values(state,ph);fe[scalar].get_function_values(incoming,pha);
     for(unsigned int q=0;q<quad.size();++q) {
      const auto x=fe.quadrature_point(q);const auto segment=std::min<unsigned int>(63,x[0]*64);const double xi=x[0]*64-segment;
      const double I=(1-xi)*actual_I[segment]+xi*actual_I[segment+1];
      const double Ia=(1-xi)*ea.normalization[segment]+xi*ea.normalization[segment+1];
      const double chi=(1./this->get_phase_field_handler().energetic_degradation({.5,.5},ph[q])-1.)/I;
      const double chia=(1./this->get_phase_field_handler().energetic_degradation({.5,.5},pha[q])-1.)/Ia;
      typename Model::ReconstructedFaultPointInputs request;
      request.fault_index=0;request.segment_index=segment;request.xi=xi;request.position=x;
      request.slip_rate=std::max(V,model().minimum_fault_slip_rate());request.phase_field=ph[q];request.previous_phase_field=pha[q];request.temperature=293;request.dynamic_pressure=0;
      request.bulk_material_fractions={.5,.5};request.strain_rate=du[q];
      request.old_maxwell_stress=2e6*Utilities::Tensors::consistent_deviator(ra[q])-2*Za*chia*S;
      request.slip_tensor=S;request.normal_tensor=N;
      const auto response=model().evaluate_reconstructed_fault_point(request);
      // Compare native point update with independent FE endpoint reconstruction.
      const double beta=std::exp(-dt/100.),kappa=-1e8*std::expm1(-dt/100.);
      const auto expected=2e6*Utilities::Tensors::consistent_deviator(beta*ra[q]+kappa/1e6*du[q])-2*Zb*chi*S;
      stress_error=std::max(stress_error,(response.stress-expected).norm());
      bulk_pair+=fe.JxW(q)*chi*(response.stress*S);strength=response.friction_traction+response.cohesive_traction;
      for(unsigned int i=0;i<indices.size();++i) {
       const auto c=this->get_fe().system_to_component_index(i).first;
       if(c==this->introspection().component_indices.velocities[0] || c==this->introspection().component_indices.velocities[1])
        force[indices[i]]+=fe.JxW(q)*response.stress*fe[this->introspection().extractors.velocities].symmetric_gradient(i,q);
       if(c==this->introspection().component_indices.pressure)
        force[indices[i]]+=fe.JxW(q)*trace(du[q])*fe[this->introspection().extractors.pressure].value(i,q);
      }
     }
    }
    std::vector<double> global_force(force.begin(),force.end());aspect::internal::endpoint_sum(global_force,this->get_mpi_communicator());
    std::copy(global_force.begin(),global_force.end(),force.begin());
    bulk_pair=Utilities::MPI::sum(bulk_pair,this->get_mpi_communicator());
    stress_error=Utilities::MPI::max(stress_error,this->get_mpi_communicator());
    strength=Utilities::MPI::max(strength,this->get_mpi_communicator());
    std::vector<double> constrained(force.size(),0.);
    for(const auto &line:this->get_current_constraints().get_lines())constrained[line.index]=1.;
    aspect::internal::endpoint_sum(constrained,this->get_mpi_communicator());
    double free_force=0.;for(unsigned int i=0;i<force.size();++i)if(constrained[i]==0.)free_force=std::max(free_force,std::abs(force[i]));
    const double contact=V>0?std::abs(strength-bulk_pair):std::max(bulk_pair-strength,0.);
    const double jp=e.energy+Fvalue+(-1e8*std::expm1(-dt/100.))/1e6*1000.*V;
    report<<name<<','<<e.bulk-eb<<','<<e.cohesive-ec<<','<<Fvalue-fracture<<','<<jp-potential<<','<<gerror<<','<<Ierror<<','<<DIerror<<','<<free_force<<','<<contact<<','<<stress_error<<','<<full_gerror<<'\n';report.flush();
    AssertThrow(gerror<1e-8 && full_gerror<1e-10 && Ierror<1e-9 && DIerror<1e-7 && free_force<1e-7 && contact<1e-6 && stress_error<1e-6,
                ExcMessage("Native fixed-state operator equivalence failed; do not advance physical history."));
   }
  } catch(...) {set_phase(saved);const_cast<BV&>(this->get_current_linearization_point())=linear;set_surface(history);F::compute_normalization_integrals(model());throw;}
  set_phase(saved);const_cast<BV&>(this->get_current_linearization_point())=linear;set_surface(history);F::compute_normalization_integrals(model());
  AssertThrow(model().accepted_fe_strain_state()==accepted && this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0)==committed,
              ExcMessage("Operator diagnostics changed accepted physical history."));
 }
 void path_ledger(const BV &state,const std::vector<double>&Z,const std::vector<double>&V,
                  const double total_energy,const unsigned int time_points=8) {
  // Diagnostic smooth path: linear phase, exponential R and Z, fixed interval
  // velocity. Intermediate imbalance is measured, not assumed to vanish.
  const double dt=this->get_timestep();const auto &intro=this->introspection();
  const unsigned int vb=intro.block_indices.velocities;
  const auto phase=aspect::internal::endpoint_values(state.block(pb()),this->get_mpi_communicator());
  const auto incoming_phase=aspect::internal::endpoint_values(previous.block(pb()),this->get_mpi_communicator());
  std::vector<double>pdot(phase.size());for(unsigned int i=0;i<pdot.size();++i)pdot[i]=(phase[i]-incoming_phase[i])/dt;
  BV interval_rate(intro.index_sets.system_partitioning,this->get_mpi_communicator());interval_rate=state;
  const auto quad=model().mechanical_quadrature();const QGauss<1>time_quad(time_points);
  FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
  const auto scalar=FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index);
  std::vector<double>phi(quad.size()),dp(quad.size()),pressure(quad.size());std::vector<SymmetricTensor<2,dim>>R(quad.size()),du(quad.size());
  SymmetricTensor<2,dim>S,lift;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
  lift[0][1]=.5*.256*(this->get_time()-.5*dt);
  double bulk_work=0.,lifting_work=0.,boundary_work=0.,Db=0.,Dc=0.,Df=0.,phase_work=0.,friction_work=0.,mechanical_work=0.,profile_bulk=0.,profile_cohesive=0.;
  double phase_absolute=0.,friction_absolute=0.,mechanical_absolute=0.;
  for(unsigned int k=0;k<time_quad.size();++k) {
   const double s=dt*time_quad.point(k)[0],wt=dt*time_quad.weight(k),beta=std::exp(-s/100.),kappa=-1e8*std::expm1(-s/100.);
   BV owned(intro.index_sets.system_partitioning,this->get_mpi_communicator());owned=state;
   Vector<double> trial_phase(phase.size());for(unsigned int i=0;i<pdot.size();++i)trial_phase[i]=incoming_phase[i]+s*pdot[i];
   aspect::internal::endpoint_assign(owned.block(pb()),trial_phase);BV trial(state);trial=owned;
   owned=previous_strain;owned.block(vb)*=beta;owned.block(vb).add(kappa/1e6,interval_rate.block(vb));owned.block(pb())=trial.block(pb());
   BV strain(state);strain=owned;
   typename Model::FEEndpointTrial endpoint;endpoint.strain.reinit(strain);endpoint.strain=strain;endpoint.Z.resize(Z.size());
   for(unsigned int i=0;i<Z.size();++i)endpoint.Z[i]=beta*previous_Z[i]+kappa*V[i];
   const auto energy=model().evaluate_fe_endpoint_energy(trial,endpoint,&pdot);
   BV rhs(intro.index_sets.system_partitioning,this->get_mpi_communicator());LinearAlgebra::BlockSparseMatrix matrix;double fracture=0.,allowance=0.;
   this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,trial,fracture,allowance);
   const auto grhs=aspect::internal::endpoint_values(rhs.block(pb()),this->get_mpi_communicator());
   double phase_power=0.;for(unsigned int i=0;i<pdot.size();++i)phase_power+=(energy.gradient[i]-grhs[i])*pdot[i];
   phase_work+=wt*phase_power;phase_absolute+=wt*std::abs(phase_power);
   double slip_work=0.,cohesive_work=0.,friction=0.,continuity=0.,bw=0.,lw=0.,reaction=0.;
   std::vector<types::global_dof_index>ids(this->get_fe().n_dofs_per_cell());
   for(const auto &cell:this->get_dof_handler().active_cell_iterators()) if(cell->is_locally_owned()) {
    fe.reinit(cell);fe[scalar].get_function_values(trial,phi);fe[intro.extractors.pressure].get_function_values(state,pressure);
    cell->get_dof_indices(ids);
    fe[intro.extractors.velocities].get_function_symmetric_gradients(strain,R);fe[intro.extractors.velocities].get_function_symmetric_gradients(state,du);
    for(unsigned int q=0;q<quad.size();++q) {
     const auto x=fe.quadrature_point(q);const unsigned int j=std::min<unsigned int>(63,64*x[0]);const double xi=64*x[0]-j;
     auto interp=[&](const auto &a){return (1-xi)*a[j]+xi*a[j+1];};
     const double I=interp(energy.normalization),Idot=interp(energy.normalization_action),z=interp(endpoint.Z),v=interp(V);
     double p_rate=0.;
     for(unsigned int vertex=0;vertex<GeometryInfo<dim>::vertices_per_cell;++vertex){const auto local=this->get_fe().component_to_system_index(scalar.component,vertex);p_rate+=fe.shape_value_component(local,q,scalar.component)*pdot[ids[local]-state.get_block_indices().block_start(pb())];}
     const double h=128.00000000016001*phi[q]*(1+phi[q])/std::pow(1-phi[q],2),hp=128.00000000016001*(4/std::pow(1-phi[q],3)-3/std::pow(1-phi[q],2));
     const double chi=h/I,chidot=hp*p_rate/I-h*Idot/(I*I),w=wt*fe.JxW(q);
     const auto tau=2e6*Utilities::Tensors::consistent_deviator(R[q])-2*z*chi*S;
     Db+=w*((tau*tau)+std::pow(trace(tau),2))/2e8;bw+=w*(tau*du[q]);lw+=w*(tau*lift);
     continuity+=w*pressure[q]*trace(du[q]);slip_work+=w*chi*v*(tau*S);profile_bulk-=w*z/1e6*(tau*S)*chidot;
     // Weak Dirichlet reaction from the same bulk operator and quadrature.
     // No strong-traction extrapolation or arbitrary interior lifting.
     if(cell->at_boundary())for(unsigned int local=0;local<ids.size();++local) {
      const auto c=this->get_fe().system_to_component_index(local).first;
      if(c>=dim || !this->get_current_constraints().is_constrained(ids[local]) || state[ids[local]]==0.)continue;
      const auto strain_test=fe[intro.extractors.velocities].symmetric_gradient(local,q);
      reaction+=w*state[ids[local]]*(tau*strain_test-pressure[q]*trace(strain_test));
     }
    }
   }
   const QGauss<1>surface_quad(3);
   for(unsigned int j=0;j<64;++j)for(unsigned int q=0;q<surface_quad.size();++q) {
    const double xi=surface_quad.point(q)[0],w=wt*surface_quad.weight(q)/64;
    auto interp=[&](const auto&a){return (1-xi)*a[j]+xi*a[j+1];};
    const double I=interp(energy.normalization),z=interp(endpoint.Z),v=interp(V);
    Dc+=w*z*z/(1e8*I);friction+=w*1000*v;cohesive_work+=w*z*v/I;
    profile_cohesive-=w*z*z*interp(energy.normalization_action)/(2e6*I*I);
   }
   bw=Utilities::MPI::sum(bw,this->get_mpi_communicator());lw=Utilities::MPI::sum(lw,this->get_mpi_communicator());
   reaction=Utilities::MPI::sum(reaction,this->get_mpi_communicator());continuity=Utilities::MPI::sum(continuity,this->get_mpi_communicator());
   slip_work=Utilities::MPI::sum(slip_work,this->get_mpi_communicator());
   bulk_work+=bw;lifting_work+=lw;boundary_work+=reaction;Df+=friction;
   const double rf=cohesive_work+friction-slip_work,rm=bw-reaction-continuity;
   friction_work+=rf;mechanical_work+=rm;friction_absolute+=std::abs(rf);mechanical_absolute+=std::abs(rm);
  }
  Db=Utilities::MPI::sum(Db,this->get_mpi_communicator());profile_bulk=Utilities::MPI::sum(profile_bulk,this->get_mpi_communicator());
  const double residual=total_energy-previous_energy-boundary_work+Db+Dc+Df;
  std::ofstream out(root_file(this->get_output_directory()+(time_points==8?"path-energy.csv":"path-energy-refined.csv")),std::ios::app);
  if(out.tellp()==0)out<<"time,storage_change,bulk_rate_work,lifting_work,boundary_reaction_work,bulk_viscous,cohesive_viscous,friction,mechanical_residual_work,friction_residual_work,phase_residual_work,profile_bulk_work,profile_cohesive_work,external_remainder,identity_remainder,absolute_mechanical_work,absolute_friction_work,absolute_phase_work\n";
  out<<std::setprecision(17)<<this->get_time()<<','<<total_energy-previous_energy<<','<<bulk_work<<','<<lifting_work<<','<<boundary_work<<','<<Db<<','<<Dc<<','<<Df<<','<<mechanical_work<<','<<friction_work<<','<<phase_work<<','<<profile_bulk<<','<<profile_cohesive<<','<<residual<<','<<residual-mechanical_work-friction_work-phase_work<<','<<mechanical_absolute<<','<<friction_absolute<<','<<phase_absolute<<'\n';
 }
public:
 static void declare_parameters(ParameterHandler&prm) {
  prm.enter_subsection("Postprocess");prm.enter_subsection("P2 native endpoint");
  prm.declare_entry("Verify complete fixed entry","false",Patterns::Bool(),"Execute the native complete solve/publication at a scoped retained entry; restore diagnostic state afterward, never a fresh trajectory.");
  prm.declare_entry("Benchmark surface Schur solve","false",Patterns::Bool(),"Fixed-growing-state native mechanical/sensitivity batch; includes direct setup, full residuals, active surface elimination and FGMRES comparator. Does not execute outer phase Newton.");
  prm.declare_entry("Operator input directory","",Patterns::DirectoryName(),"Fixed-state FE operator inputs; never initial physical histories.");
  prm.declare_entry("Fixed entry input directory","",Patterns::DirectoryName(),"Noncommitting retained-entry solver/escape diagnostics; scoped incoming FE reference and obstacle are restored.");
  prm.declare_entry("Record all step energies","false",Patterns::Bool(),"Record audited residual works also over the frozen/evolving loading prefix.");
  prm.declare_entry("Expected timestep schedule file","",Patterns::FileName(),"Optional diagnostic list of exact reference end times, checked before every physical step.");
  prm.declare_entry("Initial history file","",Patterns::FileName(),"Shared physical seeded-H evaluation at initial particle coordinates only.");
  prm.declare_entry("Force complete step failure and retry","false",Patterns::Bool(),"One controlled prepublication failure of a complete coupled timestep; verify restoration and let the native driver retry.");
  prm.declare_entry("Complete failure timestep","1",Patterns::Integer(1),"Physical timestep of the controlled failure. The onset control uses step three, with nonzero and distinct accepted histories.");
  prm.leave_subsection();prm.leave_subsection();
 }
 void parse_parameters(ParameterHandler&prm)override {
  prm.enter_subsection("Postprocess");prm.enter_subsection("P2 native endpoint");input=prm.get("Operator input directory");initial_history=prm.get("Initial history file");fixed_entries=prm.get("Fixed entry input directory");all_step_energies=prm.get_bool("Record all step energies");recovery_test=prm.get_bool("Force complete step failure and retry");failure_step=prm.get_integer("Complete failure timestep");prm.leave_subsection();prm.leave_subsection();
  prm.enter_subsection("Postprocess");prm.enter_subsection("P2 native endpoint");complete_fixed_entry=prm.get_bool("Verify complete fixed entry");direct_benchmark=prm.get_bool("Benchmark surface Schur solve");schedule_file=prm.get("Expected timestep schedule file");prm.leave_subsection();prm.leave_subsection();
 }
 void initialize()override {
  if(!schedule_file.empty()) {
   std::ifstream file(schedule_file);AssertThrow(file.good(),ExcMessage("Reference timestep schedule cannot be read."));
   double time=0.;while(file>>time) {
    AssertThrow(std::isfinite(time) && time>0. && (expected_times.empty() || time>expected_times.back()),ExcMessage("Reference end times must strictly increase."));
    expected_times.push_back(time);
   }
   AssertThrow(file.eof() && !expected_times.empty(),ExcMessage("Invalid reference timestep schedule."));
   this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&){
    const unsigned int step=this->get_timestep_number();if(!step)return;
    AssertThrow(step<=expected_times.size(),ExcMessage("Native trajectory exceeds its reference schedule."));
    const double previous=step>1?expected_times[step-2]:0.;
    AssertThrow(this->get_time()==expected_times[step-1] && this->get_timestep()==expected_times[step-1]-previous,
                ExcMessage("Native timestep differs from reference schedule; check timestep-growth limits before qualifying loading or fields."));
   });
  }
  this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&){
   if(!recovery_test || recovery_done || this->get_timestep_number()!=failure_step)return;
   recovery_done=true;
   auto &simulator=const_cast<Simulator<dim>&>(this->get_simulator());
   const BV solution(this->get_solution()),linear(this->get_current_linearization_point()),
            old(this->get_old_solution()),older(this->get_old_old_solution()),accepted(model().accepted_fe_strain_state());
   const auto histories=surface();const auto I=F::current_normalization_integrals(model());
   const auto energy_before=model().evaluate_fe_phase_energy(solution,false);
   std::vector<std::vector<double>>fault_properties;
   const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
   for(unsigned int i=0;i<fault.n_vertices();++i){const auto p=fault.get_properties(i);fault_properties.emplace_back(p.begin(),p.end());}
   const auto V=this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0);
   const auto metadata=A::endpoint_bookkeeping(simulator);
   auto &pm=const_cast<Particle::Manager<dim>&>(this->get_particle_manager(0));
   std::vector<std::vector<double>>properties;
   std::vector<Point<dim>>locations;
   for(const auto &p:pm.get_particle_handler()) {
    const auto data=p.get_properties();properties.emplace_back(data.begin(),data.end());locations.push_back(p.get_location());
   }
   inject_failure=true;
   A::endpoint_failure_hook(simulator,[this](){
    if(inject_failure){inject_failure=false;
     if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())+1==Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()))
      throw std::runtime_error("controlled complete endpoint failure");}
   });
   bool caught=false;
   try{A::complete_step(simulator);}
   catch(const std::runtime_error &e){caught=std::string(e.what())=="controlled complete endpoint failure";if(!caught)throw;}
   A::endpoint_failure_hook(simulator,{});
   AssertThrow(caught,ExcMessage("The controlled complete-step failure was not reached."));
   const auto after=surface();
   const auto energy_after=model().evaluate_fe_phase_energy(this->get_solution(),false);
   AssertThrow(this->get_solution()==solution && this->get_current_linearization_point()==linear
               && this->get_old_solution()==old && this->get_old_old_solution()==older
               && model().accepted_fe_strain_state()==accepted && A::endpoint_bookkeeping(simulator)==metadata
               && after.qvalue==histories.qvalue && after.Ivalue==histories.Ivalue
               && this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0)==V
               && F::current_normalization_integrals(model())==I,
               ExcMessage("Complete-step failure did not restore state/bookkeeping."));
   AssertThrow(energy_after.energy==energy_before.energy && energy_after.gradient==energy_before.gradient,
               ExcMessage("Failed step changed accepted elastic/normalization data."));
   std::ofstream property_check(root_file(this->get_output_directory()+"complete-step-properties.csv"));
   property_check<<std::setprecision(17)<<"vertex,property,component,before,after,same_bytes\n";
   unsigned int sentinels=0,differences=0;
   for(unsigned int i=0;i<fault.n_vertices();++i){const auto p=fault.get_properties(i);
    for(const auto &info:this->get_reconstructed_fault_manager().get_property_information())
     for(unsigned int c=0;c<info.n_components;++c) {
      const unsigned int j=info.position+c;
      const bool same=std::memcmp(&p[j],&fault_properties[i][j],sizeof(double))==0;
      if(std::isnan(fault_properties[i][j]))++sentinels;
      if(!same)++differences;
      property_check<<i<<','<<info.name<<','<<c<<','<<fault_properties[i][j]<<','<<p[j]<<','<<same<<'\n';
     }
   }
   property_check.close();
   // Unused properties retain signaling-NaN sentinels. Numeric vector equality
   // rejects even an unchanged sentinel; byte equality checks all stored values.
   AssertThrow(differences==0,ExcMessage("Failed step changed surface property bytes; see complete-step-properties.csv."));
   unsigned int j=0;
   for(const auto &p:pm.get_particle_handler()) {
    const auto data=p.get_properties();
    AssertThrow(data.size()==properties[j].size()
                && std::memcmp(data.data(),properties[j].data(),data.size()*sizeof(double))==0
                && p.get_location()==locations[j],
                ExcMessage("Failed complete step changed particle histories/positions."));++j;
   }
   std::ofstream out(root_file(this->get_output_directory()+"complete-step-recovery.txt"));
   out<<"Complete outer solve_timestep failed after coupled convergence and before publication.\n"
      <<"All solution/phase histories, accepted d_R, surface q/I/V, particle properties/positions, clock, pressure bookkeeping and statistics restored exactly.\n"
      <<"Surface property differences (including sentinel bytes): "<<differences<<"; unchanged NaN sentinel components: "<<sentinels<<".\n"
      <<"Normal native driver retries the same physical timestep next; metrics.csv verifies its successful publication.\n";
  });
  this->get_signals().post_particle_management.connect([this](Particle::Manager<dim>&pm){
   if(initialized_H || initial_history.empty() || this->get_timestep_number()!=0)return;
   AssertThrow(!model().has_fe_strain_history(),ExcMessage("Shared H must precede dependent FE initialization."));
   std::ifstream in(initial_history);AssertThrow(bool(in),ExcMessage("Missing shared seeded-H input."));
   std::vector<std::pair<double,double>>values;double y,H;while(in>>y>>H)values.emplace_back(y,H);
   const auto hi=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
   double correction=0.;
   for(auto &p:pm.get_particle_handler()) {
    auto right=std::lower_bound(values.begin(),values.end(),p.get_location()[1],[](const auto&a,double b){return a.first<b;});
    if(right==values.end() || (right!=values.begin() && std::abs((right-1)->first-p.get_location()[1])<std::abs(right->first-p.get_location()[1])))--right;
    AssertThrow(std::abs(right->first-p.get_location()[1])<1e-12,ExcMessage("Shared H particle coordinates mismatch."));
    correction=std::max(correction,std::abs(p.get_properties()[hi]-right->second));p.get_properties()[hi]=right->second;
   }
   correction=Utilities::MPI::max(correction,this->get_mpi_communicator());
   initialized_H=true;std::ofstream out(root_file(this->get_output_directory()+"initial-H-correction.txt"));out<<std::setprecision(17)<<correction<<'\n';
  });
 }
 std::pair<std::string,std::string>execute(TableHandler&)override {
  AssertThrow(model().uses_consistent_endpoint_phase() && model().has_fe_strain_history(),ExcMessage("Native endpoint observer requires its explicit option."));
  const auto &state=this->get_solution();
  const auto step=this->get_timestep_number();double maximum=0.,healing=0.,increment=0.;
  std::ofstream fields(root_file(this->get_output_directory()+"phase-"+std::to_string(step)+".csv"));fields<<std::setprecision(17)<<"x,y,phase\n";
  const auto points=phase_points();
  const auto phase=aspect::internal::endpoint_values(state.block(pb()),this->get_mpi_communicator());
  const auto accepted_phase=aspect::internal::endpoint_values(model().accepted_fe_strain_state().block(pb()),this->get_mpi_communicator());
  const auto incoming_phase=step?aspect::internal::endpoint_values(previous.block(pb()),this->get_mpi_communicator()):phase;
  for(unsigned int v=0;v<points.size();++v) {
   const auto x=points[v];const double p=phase[v];maximum=std::max(maximum,p);
   AssertThrow(p>=0. && p<1.,ExcMessage("Native phase bounds failed."));
   if(step){healing=std::max(healing,incoming_phase[v]-p);increment=std::max(increment,p-incoming_phase[v]);}
   fields<<x[0]<<','<<x[1]<<','<<p<<'\n';
   AssertThrow(accepted_phase[v]==p,ExcMessage("Accepted phase publication mismatch."));
  }
  AssertThrow(healing<=4.922e-12,ExcMessage("Resolved native phase healing."));
  const auto e=model().evaluate_fe_phase_energy(state,false);
  BV rhs(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());LinearAlgebra::BlockSparseMatrix matrix;
  double fracture=0.,allowance=0.;this->get_phase_field_handler().evaluate_fe_fracture(matrix,rhs,state,fracture,allowance);
  const auto rhs_values=aspect::internal::endpoint_values(rhs.block(pb()),this->get_mpi_communicator());
  double kkt=0.;for(unsigned int i=0;i<e.gradient.size();++i) {
   const double g=e.gradient[i]-rhs_values[i],lower=incoming_phase[i];
   kkt=std::max(kkt,std::abs(phase[i]==lower?std::min(g,0.):g));
  }
  auto history=surface();const auto I=F::current_normalization_integrals(model()).at(0),V=this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0);
  std::vector<double>Z(I.size());for(unsigned int i=0;i<Z.size();++i)Z[i]=I[i]*history.qvalue[i];
  if(step && (all_step_energies || increment>1e-6)) {
   path_ledger(state,Z,V,e.bulk+e.cohesive+fracture);
   if(this->get_time()==.53125)path_ledger(state,Z,V,e.bulk+e.cohesive+fracture,16);
  }
  double recurrence=0.;
  if(step) {
   const auto vb=this->introspection().block_indices.velocities;
   const double beta=std::exp(-this->get_timestep()/100.),kappa=-1e8*std::expm1(-this->get_timestep()/100.);
   for(const auto i:state.block(vb).locally_owned_elements())
    recurrence=std::max(recurrence,std::abs(model().accepted_fe_strain_state().block(vb)[i]-beta*previous_strain.block(vb)[i]-kappa/1e6*state.block(vb)[i]));
   recurrence=Utilities::MPI::max(recurrence,this->get_mpi_communicator());
   AssertThrow(recurrence<1e-15,ExcMessage("Native endpoint FE recurrence/publication failed."));
  }
  // Independent normal sampling, away from the six-point assembly rule.
  // Three tangential sections reveal, rather than suppress, additional modes.
  if(!step || this->get_time()==.5 || this->get_time()>=.52734375) {
   const QGauss<1>normal(12);std::vector<Point<dim>>points;std::vector<double>weights;
   for(unsigned int q=0;q<normal.size();++q){Point<dim>p;p[0]=.5;p[1]=normal.point(q)[0];points.push_back(p);weights.push_back(normal.weight(q));}
   const Quadrature<dim>quad(points,weights);
   FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_gradients|update_quadrature_points);
   std::vector<double>phi(quad.size());std::vector<SymmetricTensor<2,dim>>R(quad.size()),du(quad.size());
   const auto scalar=FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index);
   SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
   std::ofstream out(root_file(this->get_output_directory()+"stress-"+std::to_string(step)+".csv"));
   out<<std::setprecision(17)<<"x,y,phase,chi,tau_xx,tau_xy,tau_yy,R_xx,R_xy,R_yy,rate_xx,rate_xy,rate_yy\n";
   std::ostringstream local_stress;local_stress<<std::setprecision(17);
   for(const auto &cell:this->get_dof_handler().active_cell_iterators()) if(cell->is_locally_owned()) {
    const unsigned int nx=tangential_cells(),column=std::lround(nx*cell->vertex(0)[0]);if(column!=0 && column!=nx/2-1 && column!=nx-1)continue;
    fe.reinit(cell);fe[scalar].get_function_values(state,phi);
    fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(model().accepted_fe_strain_state(),R);
    fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(state,du);
    for(unsigned int q=0;q<quad.size();++q){const auto x=fe.quadrature_point(q);const unsigned int j=std::min<unsigned int>(63,64*x[0]);const double xi=64*x[0]-j;
     const double ih=(1-xi)*I[j]+xi*I[j+1],Z=(1-xi)*I[j]*history.qvalue[j]+xi*I[j+1]*history.qvalue[j+1];
     const double chi=(1./this->get_phase_field_handler().energetic_degradation({.5,.5},phi[q])-1.)/ih;
     const auto tau=2e6*Utilities::Tensors::consistent_deviator(R[q])-2*Z*chi*S;
     local_stress<<x[0]<<','<<x[1]<<','<<phi[q]<<','<<chi<<','<<tau[0][0]<<','<<tau[0][1]<<','<<tau[1][1]<<','<<R[q][0][0]<<','<<R[q][0][1]<<','<<R[q][1][1]<<','<<du[q][0][0]<<','<<du[q][0][1]<<','<<du[q][1][1]<<'\n';
    }
   }
   for(const auto &rows:Utilities::MPI::all_gather(this->get_mpi_communicator(),local_stress.str()))out<<rows;
  }
  if(!step)slip.assign(V.size(),0.);else for(unsigned int i=0;i<V.size();++i)slip[i]+=this->get_timestep()*V[i];
  std::ofstream surface_out(root_file(this->get_output_directory()+"surface-"+std::to_string(step)+".csv"));surface_out<<std::setprecision(17)<<"x,I,q,V_interval,slip\n";
  for(unsigned int i=0;i<I.size();++i)surface_out<<double(i)/(I.size()-1)<<','<<I[i]<<','<<history.qvalue[i]<<','<<V[i]<<','<<slip[i]<<'\n';
  std::ofstream metrics(root_file(this->get_output_directory()+"metrics.csv"),step?std::ios::app:std::ios::out);
  if(!step)metrics<<"step,time,dt,phase_max,increment,healing,KKT,bulk,cohesive,fracture,FE_recurrence\n";
  metrics<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<maximum<<','<<increment<<','<<healing<<','<<kkt<<','<<e.bulk<<','<<e.cohesive<<','<<fracture<<','<<recurrence<<'\n';
  if(step==1 && !input.empty())operators();
  if(!step && !fixed_entries.empty())fixed_entry_diagnostics();
  unsigned long long peak=0,resident=0;std::ifstream process("/proc/self/status");std::string line;
  while(std::getline(process,line)){std::istringstream data(line);std::string key;unsigned long long value=0;data>>key>>value;
   if(key=="VmHWM:")peak=value;
   if(key=="VmRSS:")resident=value;}
  const auto peaks=Utilities::MPI::all_gather(this->get_mpi_communicator(),peak),residents=Utilities::MPI::all_gather(this->get_mpi_communicator(),resident);
  std::ofstream memory(root_file(this->get_output_directory()+"mpi-memory.csv"),step?std::ios::app:std::ios::out);
  if(!step)memory<<"step,time,rank,peak_RSS_KiB,current_RSS_KiB\n";
  for(unsigned int rank=0;rank<peaks.size();++rank)memory<<step<<','<<std::setprecision(17)<<this->get_time()<<','<<rank<<','<<peaks[rank]<<','<<residents[rank]<<'\n';
  previous.reinit(state);previous=state;
  previous_strain.reinit(state);previous_strain=model().accepted_fe_strain_state();
  previous_Z=Z;previous_energy=e.bulk+e.cohesive+fracture;
  return {"Native endpoint P2","compact fields, interval rates, accepted-state and operator checks"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(NativeEndpoint,"P2 native endpoint","Default-off endpoint FE operator and fresh-history verification.")
}}
