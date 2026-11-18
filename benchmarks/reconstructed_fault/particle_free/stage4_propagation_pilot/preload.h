// Synthetic native initial state: prescribed affine elastic boundary loading,
// equilibrated bulk tau/A and a verified RSF operating point. Not a restart
// conversion or a change to the material law/physical-step solver.
void prepare_slipping_background()
{
  using Vector=LinearAlgebra::BlockVector;
  using Tensor=MaterialModel::BulkCohesive::Tensor;
  namespace Tip=MaterialModel::BulkCohesive::Tip;
  auto &state=*model().bulk_cohesive_state;
  const auto comm=this->get_mpi_communicator();
  const auto rank=Utilities::MPI::this_mpi_process(comm);
  const auto directory=std::string(std::getenv("ASPECT_WAVELENGTH_INPUT"));
  const unsigned int ns=state.sections.size();
  std::ifstream planes(directory+"/sections.txt"),initial(directory+"/background.txt");
  unsigned int count;planes>>count;AssertDimension(count,ns);initial>>count;AssertDimension(count,ns);
  std::vector<double> V(ns),theta(ns),force(ns);
  std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
  const auto &law=model().fault_friction;
  const double pressure=this->get_parameters().surface_pressure,radiation=model().radiation_damping_coefficients[0];
  // Use the signed section API, which admits zero and arbitrarily small rate.
  // Its public operation advances age: invert that BE age map analytically so
  // this diagnostic evaluates precisely the specified, unchanged age. This is
  // a fixed-state evaluation, not a fictitious physical initialization interval.
  auto resistance=[&](double v,double age)
    {
      const double duration=.25*age;
      const double incoming=age*(1.+std::abs(v)*duration/law.get_characteristic_slip_distance())-duration;
      const auto r=law.evaluate_resistance(fractions,v,incoming,duration,pressure,radiation,
                                         MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
      AssertThrow(std::abs(r.age/age-1.)<1e-14,ExcMessage("Fixed-age signed resistance evaluation mismatch."));
      return r.resistance;
    };
  for(unsigned int k=0;k<ns;++k)
    {
      double coordinate,angle,other;unsigned int formed;
      planes>>coordinate>>angle>>formed;initial>>other>>V[k]>>theta[k];
      AssertThrow(std::abs(coordinate-state.sections[k].coordinate)<1e-12 && std::abs(other-coordinate)<1e-12,ExcMessage("Preload label mismatch."));
      auto &s=state.sections[k];s.angle=angle;s.formed=formed;s.weight/=std::cos(angle);s.age=theta[k];
      force[k]=resistance(V[k],theta[k]);
    }
  AssertThrow(planes && initial,ExcMessage("Incomplete preload inputs."));
  for(auto &entry:state.normal_weights)for(unsigned int q=0;q<entry.second.size();++q)
    entry.second[q]*=std::cos(state.sections[state.section_ids.at(entry.first)[q]].angle);
  sim().bulk_tip_coupling=std::make_shared<internal::BulkTipCoupling<dim>>(sim(),model(),sim().current_constraints);
  auto &op=*sim().bulk_tip_coupling;
  const AffineConstraints<double> saved_constraints(sim().current_constraints);
  const bool saved_system=sim().assemble_newton_stokes_system,saved_matrix=sim().assemble_newton_stokes_matrix;
  const double saved_tolerance=sim().parameters.linear_stokes_solver_tolerance;
  struct Restore{std::function<void()> f;~Restore(){f();}} restore{[&]()
    {
      sim().current_constraints.copy_from(saved_constraints);
      sim().assemble_newton_stokes_system=saved_system;sim().assemble_newton_stokes_matrix=saved_matrix;
      sim().parameters.linear_stokes_solver_tolerance=saved_tolerance;
      sim().bulk_tip_coupling.reset();sim().set_assemblers();
    }};
  for(const auto &line:saved_constraints.get_lines())sim().current_constraints.set_inhomogeneity(line.index,0.);
  sim().current_constraints.close();sim().assemble_newton_stokes_system=false;sim().assemble_newton_stokes_matrix=false;
  sim().set_assemblers();sim().pressure_scaling=sim().compute_pressure_scaling_factor();
  sim().rebuild_stokes_matrix=true;sim().rebuild_stokes_preconditioner=true;
  sim().assemble_stokes_system();sim().build_stokes_preconditioner();
  sim().parameters.linear_stokes_solver_tolerance=1e-4;
  auto x=op.coordinates(sim().solution);x=0.;x.block(op.pb)=op.lower;
  const auto &fe=this->get_fe();std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  const double gamma=std::stod(std::getenv("ASPECT_PRELOAD_GAMMA")),extension=.41954981558863999*gamma;
  for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
    {
      cell->get_dof_indices(ids);
      for(unsigned int c=0;c<dim;++c)
        {
          const auto component=this->introspection().component_indices.velocities[c];
          const auto &element=fe.base_element(fe.component_to_base_index(component).first);
          for(unsigned int j=0;j<element.n_dofs_per_cell();++j)
            {
              const auto i=ids[fe.component_to_system_index(component,j)];
              if(!x.locally_owned_elements().is_element(i))continue;
              const auto p=this->get_mapping().transform_unit_to_real_cell(cell,element.get_generalized_support_points()[j]);
              // Localized divergence-free shear from streamfunction a*f(x)*(y-2)^2/2.
              const double amplitude=std::stod(std::getenv("ASPECT_PILOT_LOAD_AMPLITUDE"));
              const double f=1., df=0.; // Additional uniform shear, fixed normal preload.
              x[i]=(c==0 ? gamma*(p[1]-2)-extension*(p[0]-4)+amplitude*f*(p[1]-2)
                          : extension*(p[1]-2)-.5*amplitude*df*(p[1]-2)*(p[1]-2))/op.beta;
            }
        }
    }
  x.compress(VectorOperation::insert);
  const auto base=op.evaluate(x,true);op.section_slopes.assign(ns,0.);
  auto prescribed_residual=[&](const auto &evaluation)
    {
      Vector raw(evaluation->raw);raw.block(op.pb)=0.;
      for(unsigned int k=0;k<ns;++k)
        for(const auto &entry:op.section_adjoints[k])if(entry.first<op.nv)
          raw[entry.first]+=entry.second*(force[k]-evaluation->section_solutions[k].resistance_response.resistance);
      raw.compress(VectorOperation::add);return raw;
    };
  Vector raw=prescribed_residual(base),rhs(raw),answer(x);op.homogeneous.set_zero(rhs);rhs*=-1.;answer=0.;
  const double load_norm=rhs.l2_norm();
  struct Action{std::function<void(Vector &,const Vector &)> f;void vmult(Vector &d,const Vector &s)const{f(d,s);}};
  const Action matrix{[&](Vector &dst,const Vector &src){Vector restricted(src);restricted.block(op.pb)=0.;op.vmult(dst,restricted);dst.block(op.pb)=src.block(op.pb);}};
  const Action inverse{[&](Vector &dst,const Vector &src){dst=src;sim().apply_bulk_tip_stokes_inverse(dst,src);}};
  unsigned int iterations=0;
  for(unsigned int refinement=0;refinement<4;++refinement)
    {
      Vector defect(rhs);matrix.vmult(defect,answer);defect-=rhs;
      if(defect.l2_norm()/load_norm<2e-9)break;
      const double scale=defect.l2_norm();defect*=-1./scale;Vector correction(answer);correction=0.;
      SolverControl control(600,1e-11);SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(80));
      solver.solve(matrix,correction,defect,inverse);iterations+=control.last_step();answer.add(scale,correction);
    }
  x+=answer;
  if(!std::getenv("ASPECT_PILOT_EQUILIBRIUM_SCREEN"))settle_loaded_seed(op,x,force);
  const auto final=op.evaluate(x,false);raw=prescribed_residual(final);
  Vector free_residual(raw);op.homogeneous.set_zero(free_residual);
  const double mechanical=free_residual.l2_norm()/load_norm;
  gate("preload native mixed equilibrium",mechanical<2e-9);
  const auto readable=op.ghost(x);
  FEValues<dim> values(this->get_mapping(),fe,model().mechanical_quadrature(),update_values|update_gradients|update_JxW_values);
  std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);
  std::vector<double> prepared_phi(values.n_quadrature_points);
  std::vector<unsigned int> prepared_formed(ns,0);
  double energy=0.,force_work=0.,point_error=0.;
  for(auto &entry:state.history)
    {
      values.reinit(state.native_cells.at(entry.first));
      values[this->introspection().extractors.velocities].get_function_symmetric_gradients(readable,strains);
      values[FEValuesExtractors::Scalar(op.pc)].get_function_values(readable,prepared_phi);
      for(unsigned int q=0;q<values.n_quadrature_points;++q)
        {
          const auto k=state.section_ids.at(entry.first)[q];Tensor E,S;
          const auto d=Utilities::Tensors::consistent_deviator(strains[q]);
          E[0][0]=d[0][0];E[1][1]=d[1][1];E[2][2]=-d[0][0]-d[1][1];E[0][1]=d[0][1];
          S[0][0]=-.5*std::sin(2*op.section_angles[k]);S[1][1]=-S[0][0];S[0][1]=.5*std::cos(2*op.section_angles[k]);
          const double phi=prepared_phi[q];
          if(phi>0.)prepared_formed[k]=1;
          const auto r=Tip::update(state.parameters,{Tensor(),Tensor(),phi},E,phi,op.dt,S,force[k]);
          auto &h=entry.second[q];h.stress=r.state.stress;h.A=r.state.cohesive_strain;h.Q=r.Q;h.q=r.Q*S;
          energy+=values.JxW(q)*(r.bulk_energy+r.cohesive_energy);
          force_work+=values.JxW(q)*force[k]*2*(h.A*S);
          point_error=std::max(point_error,std::abs((h.stress-h.Q)*S-force[k]));
        }
    }
  Utilities::MPI::max(prepared_formed,comm,prepared_formed);
  for(unsigned int k=0;k<ns;++k)state.sections[k].formed=state.sections[k].formed || prepared_formed[k];
  state.tip_phase=x.block(op.pb);
  double boundary_work=0.;
  for(const auto &line:saved_constraints.get_lines())if(line.index<op.nv && x.locally_owned_elements().is_element(line.index))
    boundary_work+=op.G*op.beta*x[line.index]*raw[line.index];
  energy=Utilities::MPI::sum(energy,comm);force_work=Utilities::MPI::sum(force_work,comm);boundary_work=Utilities::MPI::sum(boundary_work,comm);
  point_error=Utilities::MPI::max(point_error,comm);
  const double work_error=std::abs(2*energy+force_work-boundary_work)/std::max(energy,std::abs(boundary_work));
  gate("preload static work identity",work_error<2e-8);gate("preload local traction consistency",point_error<1e-8);
  // The same physical age perturbation is applied after the matched preload.
  // Invert the owned friction law to observe its instantaneous section rate.
  const double perturbation=std::getenv("ASPECT_PILOT_AGE_PERTURBATION") ? std::stod(std::getenv("ASPECT_PILOT_AGE_PERTURBATION")) : 0.;
  double rate_error=0.,law_error=0.;
  for(unsigned int k=0;k<ns;++k)
    {
      auto &s=state.sections[k];const double coordinate=s.coordinate;
      const double shape=coordinate>3.0 && coordinate<6.0 ? std::pow(std::sin(numbers::PI*(coordinate-3.0)/3.),2) : 0.;
      s.age=theta[k]*(1.-perturbation*shape);s.force=force[k];s.slip=0.;
      double lo=0.,hi=2*V[k];
      if(V[k]>0.)for(unsigned int j=0;j<80;++j){const double mid=.5*(lo+hi);if(resistance(mid,s.age)>force[k])hi=mid;else lo=mid;}
      s.rate=.5*(lo+hi);
      rate_error=std::max(rate_error,std::abs(s.rate-V[k]));law_error=std::max(law_error,std::abs(resistance(s.rate,s.age)-force[k]));
    }
  gate("preload section friction inverse",law_error<1e-10);
  if(perturbation==0.)gate("preload intended initial section rate",rate_error<1e-14);
  for(auto *field:{&sim().solution,&sim().old_solution,&sim().old_old_solution,&sim().current_linearization_point})
    {Vector owned(this->introspection().index_sets.system_partitioning,comm);owned=*field;owned.block(1)=x.block(1);owned.block(1)*=op.G;owned.block(op.pb)=x.block(op.pb);*field=owned;}
  if(rank==0)
    {
      std::ofstream file(this->get_output_directory()+"preload-checks.csv");
      file<<"gamma,extension,iterations,relative_equilibrium_residual,storage,force_work,boundary_work,relative_work_error,point_traction_error,rate_change,friction_inverse_error,age_perturbation\n"<<std::setprecision(17)
          <<gamma<<','<<extension<<','<<iterations<<','<<mechanical<<','<<energy<<','<<force_work<<','<<boundary_work<<','<<work_error<<','<<point_error<<','<<rate_error<<','<<law_error<<','<<perturbation<<'\n';
    }
}

void observe_slipping_pilot()
{
  const auto &state=*model().bulk_cohesive_state;
  if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())!=0)return;
  const double dt=state.step ? this->get_timestep() : 0.;
  std::ofstream sections(this->get_output_directory()+"pilot-sections-"+Utilities::int_to_string(state.step,4)+".csv");
  sections<<"section,x,ds,angle,age_endpoint,rate_stage,slip_endpoint,resistance_stage\n"<<std::setprecision(17);
  double peak=0.,integral=0.,slip=0.,traction=0.,measure=0.;
  for(unsigned int k=0;k<state.sections.size();++k)
    {
      const auto &s=state.sections[k];sections<<k<<','<<s.coordinate<<','<<s.weight<<','<<s.angle<<','<<s.age<<','<<s.rate<<','<<s.slip<<','<<s.force<<'\n';
      peak=std::max(peak,std::abs(s.rate));integral+=s.weight*s.rate;slip+=s.weight*s.slip;
      if(s.coordinate>2.5 && s.coordinate<5.5){traction+=s.weight*s.force;measure+=s.weight;}
    }
  std::ofstream file(this->get_output_directory()+"pilot-summary.csv",std::ios::app);
  if(file.tellp()==0)file<<"step,time,stage_time,dt,peak_rate,integrated_rate,integrated_slip,mean_patch_resistance\n";
  file<<std::setprecision(17)<<state.step<<','<<state.time<<','<<state.time-.5*dt<<','<<dt<<','<<peak<<','<<integral<<','<<slip<<','<<traction/measure<<'\n';
}
