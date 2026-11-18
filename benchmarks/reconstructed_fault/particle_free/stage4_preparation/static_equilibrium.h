// Benchmark preparation at held displacement and prescribed section resistance.
// Native mechanics, fracture matrix, Bernstein bounds and obstacle solver are
// reused. This minimization has no elapsed slip/aging/Maxwell clock.
void settle_loaded_seed(internal::BulkTipCoupling<dim> &op,
                        LinearAlgebra::BlockVector &x,
                        const std::vector<double> &force)
{
  using Vector=LinearAlgebra::BlockVector;
  using Scalar=LinearAlgebra::Vector;
  namespace Tip=MaterialModel::BulkCohesive::Tip;
  using Tensor=MaterialModel::BulkCohesive::Tensor;
  const auto comm=this->get_mpi_communicator();
  const auto rank=Utilities::MPI::this_mpi_process(comm);
  const auto &sets=this->introspection().index_sets;
  const auto &fe=this->get_fe();
  const auto &parameters=op.parent->parameters;
  const auto &settings=this->get_phase_field_handler().get_solver_parameters();
  const auto po=op.po;
  const bool settle=std::getenv("ASPECT_PREP_SETTLE")!=nullptr;
  Scalar physical_upper(op.lower);physical_upper=1.;
  // In fixed-phase runs the benchmark signal supplies an artificial upper
  // obstacle. Static preparation always tests the physical upper bound.
  op.upper=physical_upper;
  struct Action {std::function<void(Vector &,const Vector &)> f;
    void vmult(Vector &d,const Vector &s)const {f(d,s);}};
  auto prescribed_raw=[&](const auto &e)
    {
      Vector r(e->raw);r.block(op.pb)=0.;
      for(unsigned int k=0;k<force.size();++k)
        for(const auto &a:op.section_adjoints[k])if(a.first<op.nv)
          r[a.first]+=a.second*(force[k]-e->section_solutions[k].resistance_response.resistance);
      r.compress(VectorOperation::add);return r;
    };
  unsigned int linear_iterations=0;
  auto mechanical=[&](Vector &u)
    {
      auto e=op.evaluate(u,true);op.section_slopes.assign(force.size(),0.);
      Vector rhs=prescribed_raw(e),answer(u);op.homogeneous.set_zero(rhs);rhs*=-1.;answer=0.;
      const double scale=rhs.l2_norm();
      if(scale>1e-16)
        {
          rhs*=1./scale;
          Action matrix{[&](Vector &d,const Vector &s){Vector r(s);r.block(op.pb)=0.;op.vmult(d,r);d.block(op.pb)=s.block(op.pb);}};
          Action inverse{[&](Vector &d,const Vector &s){d=s;sim().apply_bulk_tip_stokes_inverse(d,s);}};
          SolverControl control(600,1e-11);
          SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(80));
          solver.solve(matrix,answer,rhs,inverse);linear_iterations+=control.last_step();u.add(scale,answer);
        }
      e=op.evaluate(u,false);auto r=prescribed_raw(e);op.homogeneous.set_zero(r);
      return r.l2_norm();
    };
  struct StaticResponse {Scalar gradient;double potential=0.,kkt=0.,fracture=0.;};
  auto response=[&](const Vector &u)
    {
      StaticResponse result;result.gradient.reinit(op.lower);
      op.fracture_matrix.vmult(result.gradient,u.block(op.pb));result.gradient+=op.fracture_linear;
      const auto field=op.ghost(u);
      Scalar native_gradient;
      result.fracture=this->get_phase_field_handler().assemble_dg_fracture(field,native_gradient);
      double storage=0.;
      FEValues<dim> values(this->get_mapping(),fe,model().mechanical_quadrature(),update_values|update_gradients|update_JxW_values);
      std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);
      std::vector<double> phi(values.n_quadrature_points);
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      for(const auto &entry:op.parent->native_cells)
        {
          values.reinit(entry.second);entry.second->get_dof_indices(ids);
          values[this->introspection().extractors.velocities].get_function_symmetric_gradients(field,strains);
          values[FEValuesExtractors::Scalar(op.pc)].get_function_values(field,phi);
          for(unsigned int q=0;q<values.n_quadrature_points;++q)
            {
              const auto k=op.parent->section_ids.at(entry.first)[q];Tensor E,S;
              const auto d=Utilities::Tensors::consistent_deviator(strains[q]);
              E[0][0]=d[0][0];E[1][1]=d[1][1];E[2][2]=-d[0][0]-d[1][1];E[0][1]=d[0][1];
              S[0][0]=-.5*std::sin(2*op.section_angles[k]);S[1][1]=-S[0][0];S[0][1]=.5*std::cos(2*op.section_angles[k]);
              const auto r=Tip::update(parameters,{Tensor(),Tensor(),0.},E,phi[q],op.dt,S,force[k]);
              storage+=values.JxW(q)*(r.bulk_energy+r.cohesive_energy+force[k]*2.*(r.state.cohesive_strain*S));
              for(unsigned int j=0;j<op.basis.expansion.m();++j)
                result.gradient[ids[fe.component_to_system_index(op.pc,j)]-po]-=values.JxW(q)*op.phase_values(q,j)*r.driving;
            }
        }
      result.gradient.compress(VectorOperation::add);result.gradient*=1./op.Kf;
      result.potential=Utilities::MPI::sum(storage,comm)+result.fracture;
      for(const auto i:result.gradient.locally_owned_elements())
        result.kkt=std::max(result.kkt,std::abs(PhaseField::box_constraint(u.block(op.pb)[i],result.gradient[i],op.mass[i],op.lower[i],1.).projected_residual));
      result.kkt=Utilities::MPI::max(result.kkt,comm);return result;
    };
  std::ofstream history;
  if(rank==0)
    {
      history.open(this->get_output_directory()+"preparation-iterations.csv");
      history<<"iteration,potential,fracture,physical_phase_KKT,mechanical_residual,step_length,linear_iterations\n"<<std::setprecision(17);
    }
  double mechanics=mechanical(x),alpha=0.;
  auto current=response(x);
  bool converged=false;
  const auto original_lower=op.lower;
  for(unsigned int iteration=0;iteration<=settings.max_nonlinear_iterations;++iteration)
    {
      if(rank==0)
        {
          history<<iteration<<','<<current.potential<<','<<current.fracture<<','<<current.kkt<<','<<mechanics<<','<<alpha<<','<<linear_iterations<<std::endl;
          this->get_pcout()<<"   Static preparation "<<iteration<<": phase KKT="<<current.kkt<<", mechanics="<<mechanics<<", potential="<<current.potential<<std::endl;
        }
      if(current.kkt<=settings.nonlinear_solver_tolerance && mechanics<=1e-9){converged=true;break;}
      AssertThrow(settle,ExcMessage("Imported prepared phase does not pass the native physical KKT test."));
      Scalar phase(x.block(op.pb)),load(op.lower),direction(op.lower);
      LinearAlgebra::SparseMatrix metric;op.phase_search_matrix(metric,1.);
      metric.vmult(load,phase);load-=current.gradient;
      double minimum_mass=std::numeric_limits<double>::max();
      for(const auto i:op.mass.locally_owned_elements())minimum_mass=std::min(minimum_mass,double(op.mass[i]));
      minimum_mass=Utilities::MPI::min(minimum_mass,comm);
      PhaseField::solve_obstacle(metric,load,op.lower,physical_upper,phase,sets.system_relevant_partitioning[op.pb],comm,1e-10*minimum_mass,100,1000);
      direction=phase;direction-=x.block(op.pb);
      const double slope=op.Kf*(current.gradient*direction);
      AssertThrow(slope<0.,ExcMessage("Static preparation obstacle direction is not descending."));
      bool accepted=false;alpha=1.;
      for(unsigned int search=0;search<settings.max_newton_line_search_iterations;++search,alpha*=.5)
        {
          Vector trial(x);trial.block(op.pb).add(alpha,direction);
          const double trial_mechanics=mechanical(trial);auto next=response(trial);
          const double roundoff=64.*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(current.potential));
          if(next.potential<=current.potential+1e-4*alpha*slope+roundoff)
            {x=trial;current=std::move(next);mechanics=trial_mechanics;accepted=true;break;}
        }
      AssertThrow(accepted,ExcMessage("Static coupled preparation line search failed; no physical clock was advanced."));
    }
  gate("loaded seed physical phase KKT",converged);
  gate("loaded seed accepted clock unchanged",model().bulk_cohesive_state->time==0. && model().bulk_cohesive_state->step==0 && model().bulk_cohesive_state->publications==0);
  double minimum=0.,maximum=0.;
  for(const auto i:op.lower.locally_owned_elements())
    {minimum=std::min(minimum,double(x.block(op.pb)[i]-original_lower[i]));maximum=std::max(maximum,double(x.block(op.pb)[i]));}
  gate("loaded seed irreversible physical bounds",Utilities::MPI::min(minimum,comm)>=-1e-14 && Utilities::MPI::max(maximum,comm)<=1.+1e-14);
}
