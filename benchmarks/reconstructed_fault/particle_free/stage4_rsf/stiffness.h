/* Included inside the existing benchmark-only MomentCycle class.
 * Read-only, fixed-phase elastic compliance: no friction tangent is included.
 * Reuse the native material derivatives, section adjoints, mixed FE assembly,
 * distributed Krylov operator and ASPECT Stokes inverse. */
void measure_restoring_response()
{
  if (!std::getenv("ASPECT_TIP_RESTORING_RESPONSE")) return;
  using Vector=LinearAlgebra::BlockVector;
  namespace Tip=MaterialModel::BulkCohesive::Tip;
  using Tensor=MaterialModel::BulkCohesive::Tensor;
  auto &op=*sim().bulk_tip_coupling;
  const auto before=persistent();
  const auto original_coordinates=op.accepted_coordinates;
  const auto original_evaluation=op.accepted_evaluation;
  const auto &state=*model().bulk_cohesive_state;
  gate("restoring diagnostic requires frozen phase",bool(std::getenv("ASPECT_TIP_FIXED_PHASE")));
  const auto comm=this->get_mpi_communicator();
  const unsigned int rank=Utilities::MPI::this_mpi_process(comm),ns=state.sections.size(),nm=7;
  const double amplitude=100.; // Pa sqrt(m); linear diagnostic, not a physical loading.
  auto x=op.accepted_coordinates;
  x.block(op.pb)=op.lower;
  const auto evaluation=op.evaluate(x,true);
  // The physical solve restores its boundary constraints before postprocessing.
  // Its Stokes inverse requires the same homogeneous increment context here.
  const AffineConstraints<double> saved_constraints(sim().current_constraints);
  const bool saved_newton_system=sim().assemble_newton_stokes_system;
  const bool saved_newton_matrix=sim().assemble_newton_stokes_matrix;
  const double saved_linear_tolerance=sim().parameters.linear_stokes_solver_tolerance;
  struct RestoreContext
  {
    std::function<void()> restore;
    ~RestoreContext(){restore();}
  } restore_context{[&]()
    {
      sim().current_constraints.copy_from(saved_constraints);
      sim().assemble_newton_stokes_system=saved_newton_system;
      sim().assemble_newton_stokes_matrix=saved_newton_matrix;
      sim().parameters.linear_stokes_solver_tolerance=saved_linear_tolerance;
    }};
  for(const auto &line:saved_constraints.get_lines())
    sim().current_constraints.set_inhomogeneity(line.index,0.);
  sim().current_constraints.close();
  sim().assemble_newton_stokes_system=false;
  sim().assemble_newton_stokes_matrix=false;
  sim().parameters.linear_stokes_solver_tolerance=1e-4;
  const auto slopes=op.section_slopes;
  op.section_slopes.assign(ns,0.); // Remove ONLY condensed friction response.
  std::vector<double> C0(ns,0.),Ccoh(ns,0.);
  FEValues<dim> values(this->get_mapping(),this->get_fe(),model().mechanical_quadrature(),update_gradients|update_JxW_values);
  for (const auto &entry:state.native_cells)
    {
      values.reinit(entry.second);
      for (unsigned int q=0;q<values.n_quadrature_points;++q)
        {
          const auto k=state.section_ids.at(entry.first)[q];
          const auto &point=evaluation->points.at(entry.first)[q];
          const double phi=point.state.phi,dn=values.JxW(q)/op.section_measures[k];
          AssertThrow(phi<1.,ExcMessage("Finite cohesive-bound diagnostic requires phi<1; no cap."));
          C0[k]+=dn*point.fraction/op.G;
          Ccoh[k]+=dn*state.parameters.m*phi*(1.+phi)/((1.-phi)*(1.-phi)*op.G);
        }
    }
  Utilities::MPI::sum(C0,comm,C0); Utilities::MPI::sum(Ccoh,comm,Ccoh);
  std::vector<std::vector<double>> pattern(nm,std::vector<double>(ns,0.)),response=pattern;
  // Force basis restricted to established flat seed; every section response,
  // including outside this patch, remains in the mechanical equilibrium solve.
  for (unsigned int j=0;j<nm;++j)
    {
      double norm=0.;
      for (unsigned int k=0;k<ns;++k)
        {
          const double coordinate=state.sections[k].coordinate;
          if (coordinate>2.5 && coordinate<5.5)
            pattern[j][k]=std::sin((j<6?j+1:48)*numbers::PI*(coordinate-2.5)/3.);
          norm+=op.section_measures[k]*pattern[j][k]*pattern[j][k];
        }
      for (auto &value:pattern[j])value/=std::sqrt(norm);
    }
  std::ofstream modes,summary,sections;
  if (rank==0)
    {
      modes.open(this->get_output_directory()+"restoring-modes.csv");
      modes<<"mode,section,x,ds,force_basis_per_sqrt_m,slip_response_per_force_coefficient\n"<<std::setprecision(17);
      summary.open(this->get_output_directory()+"restoring-checks.csv");
      summary<<"mode,iterations,relative_equilibrium_residual,energy_J_m,half_force_slip_J_m,relative_power_error,finite_difference_relative_error,seconds\n"<<std::setprecision(17);
      sections.open(this->get_output_directory()+"restoring-sections.csv");
      sections<<"section,x,ds,angle,formed,age,stage_age,slip,V,force,clamped_compliance_m_Pa,cohesive_compliance_bound_m_Pa\n"<<std::setprecision(17);
      for (unsigned int k=0;k<ns;++k)
        {
          const auto &s=state.sections[k];
          sections<<k<<','<<s.coordinate<<','<<op.section_measures[k]<<','<<op.section_angles[k]<<','<<s.formed<<','<<s.age<<','<<evaluation->section_solutions[k].resistance_response.age<<','<<s.slip<<','<<s.rate<<','<<s.force<<','<<C0[k]<<','<<Ccoh[k]<<'\n';
        }
    }
  if(rank==0)
    {
      std::ofstream friction(this->get_output_directory()+"restoring-friction.csv");
      friction<<"V,theta,resistance,fV,ftheta\n"<<std::setprecision(17);
      std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
      const auto &law=model().fault_friction;
      const double pressure=this->get_parameters().surface_pressure, radiation=model().radiation_damping_coefficients[0];
      for (const double V:{1e-6,1e-4,2e-4,.001,.02})
        for(const double age:{.2,law.get_characteristic_slip_distance()/V})
          friction<<V<<','<<age<<','<<pressure*law.friction_coefficient(fractions,V,age)+radiation*V
                  <<','<<pressure*law.friction_coefficient_derivative_wrt_slip_rate(fractions,V,age)+radiation
                  <<','<<pressure*law.friction_coefficient_derivative_wrt_state(fractions,V,age)<<'\n';
      std::ofstream parameters(this->get_output_directory()+"restoring-parameters.csv");
      parameters<<"G,eta,m,stage_dt,beta,time,physical_dt\n"<<std::setprecision(17)
                <<op.G<<','<<state.parameters.eta<<','<<state.parameters.m<<','<<op.dt<<','<<op.beta<<','<<state.time<<','<<op.physical_dt<<'\n';
    }
  struct Action
  {
    std::function<void(Vector &,const Vector &)> function;
    void vmult(Vector &dst,const Vector &src)const {function(dst,src);}
  };
  const Action matrix{[&](Vector &dst,const Vector &src)
    {
      Vector restricted(src); restricted.block(op.pb)=0.;
      op.vmult(dst,restricted); dst.block(op.pb)=src.block(op.pb);
    }};
  const Action inverse{[&](Vector &dst,const Vector &src)
    {dst=src;sim().apply_bulk_tip_stokes_inverse(dst,src);}};
  for (unsigned int j=0;j<nm;++j)
    {
      Timer timer;
      Vector rhs(x),answer(x),residual(x); rhs=0.;answer=0.;
      for (unsigned int k=0;k<ns;++k)
        for (const auto &entry:op.section_adjoints[k])
          if (entry.first<op.nv && entry.second!=0.)rhs[entry.first]-=amplitude*pattern[j][k]*entry.second;
      rhs.compress(VectorOperation::add); op.homogeneous.set_zero(rhs);
      const double rhs_norm=rhs.l2_norm();
      double relative=1.;unsigned int total_iterations=0;
      // Verify the true residual and correct its defect. Arnoldi's estimated
      // residual can under-report roundoff in this scaled mixed system.
      for(unsigned int refinement=0;refinement<4;++refinement)
        {
          matrix.vmult(residual,answer);residual-=rhs;
          relative=residual.l2_norm()/rhs_norm;
          this->get_pcout()<<"   Restoring true residual "<<refinement<<": "<<relative<<std::endl;
          if(relative<2e-9)break;
          const double scale=residual.l2_norm();
          Vector load(residual),correction(answer);load*=-1./scale;correction=0.;
          SolverControl control(600,1e-11);
          SolverFGMRES<Vector> solver(control,typename SolverFGMRES<Vector>::AdditionalData(80));
          solver.solve(matrix,correction,load,inverse);
          total_iterations+=control.last_step();
          answer.add(scale,correction);
        }
      matrix.vmult(residual,answer);residual-=rhs;
      relative=residual.l2_norm()/rhs_norm;
      this->get_pcout()<<"   Restoring final residual: "<<relative<<", rhs="<<rhs_norm<<", iterations="<<total_iterations<<std::endl;
      gate("restoring equilibrated mixed residual",relative<2e-9);
      const auto readable=op.ghost(answer);
      std::vector<double> mechanical(ns,0.),finite_difference(ns,0.);
      for (unsigned int k=0;k<ns;++k)
        for (const auto &entry:op.section_integrals[k])
          if (entry.first<op.nv)mechanical[k]+=entry.second*readable[entry.first];
      Utilities::MPI::sum(mechanical,comm,mechanical);
      for (unsigned int k=0;k<ns;++k)
        response[j][k]=C0[k]*pattern[j][k]-op.beta*op.dt*mechanical[k]/amplitude;
      double energy=0.;
      std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);
      for (const auto &entry:state.native_cells)
        {
          values.reinit(entry.second);
          values[this->introspection().extractors.velocities].get_function_symmetric_gradients(readable,strains);
          for (unsigned int q=0;q<values.n_quadrature_points;++q)
            {
              const auto k=state.section_ids.at(entry.first)[q];
              Tensor E,S;
              const auto d=Utilities::Tensors::consistent_deviator(strains[q]);
              E[0][0]=d[0][0];E[1][1]=d[1][1];E[2][2]=-d[0][0]-d[1][1];E[0][1]=d[0][1];
              S[0][0]=-.5*std::sin(2.*op.section_angles[k]);S[1][1]=-S[0][0];S[0][1]=.5*std::cos(2.*op.section_angles[k]);
              const auto &point=evaluation->points.at(entry.first)[q];
              const double force=amplitude*pattern[j][k],phi=point.state.phi;
              const auto variation=Tip::derivative(state.parameters,point,E,0.,op.dt,S,force);
              energy+=values.JxW(q)*((variation.stress*variation.stress)/(4.*op.G)
                         +(phi>0. ? op.G*(variation.cohesive_strain*variation.cohesive_strain)*(1.-phi)*(1.-phi)/(state.parameters.m*phi*(1.+phi)) : 0.));
              // Independent symmetric point updates, zero history. Fixed phi
              // makes this exactly linear; amplitude need not be infinitesimal.
              const Tip::State zero{Tensor(),Tensor(),phi};
              const auto plus=Tip::update(state.parameters,zero,E,phi,op.dt,S,force);
              const auto minus=Tip::update(state.parameters,zero,-E,phi,op.dt,S,-force);
              finite_difference[k]-=values.JxW(q)/op.section_measures[k]*op.beta*op.dt*(plus.rate-minus.rate)/(2.*amplitude);
            }
        }
      energy=Utilities::MPI::sum(energy,comm);
      Utilities::MPI::sum(finite_difference,comm,finite_difference);
      double work=0.,error=0.,scale=0.;
      for (unsigned int k=0;k<ns;++k)
        {
          work+=.5*amplitude*amplitude*op.section_measures[k]*pattern[j][k]*response[j][k];
          error+=op.section_measures[k]*std::pow(response[j][k]-finite_difference[k],2);
          scale+=op.section_measures[k]*response[j][k]*response[j][k];
          if(rank==0)modes<<j<<','<<k<<','<<state.sections[k].coordinate<<','<<op.section_measures[k]<<','<<pattern[j][k]<<','<<response[j][k]<<'\n';
        }
      const double power_error=std::abs(energy-work)/std::max(energy,std::abs(work));
      const double fd_error=std::sqrt(error/scale);
      gate("restoring energy and force-slip work",power_error<2e-8);
      gate("restoring independent symmetric point update",fd_error<2e-8);
      if(rank==0)
        {
          summary<<j<<','<<total_iterations<<','<<relative<<','<<energy<<','<<work<<','<<power_error<<','<<fd_error<<','<<timer.wall_time()<<std::endl;
          this->get_pcout()<<"   Restoring mode "<<j<<": residual="<<relative<<", power="<<power_error<<", seconds="<<timer.wall_time()<<std::endl;
        }
    }
  op.section_slopes=slopes;
  op.evaluate(original_coordinates,true);
  gate("restoring diagnostic preserves accepted state",before==persistent()
       && original_coordinates==op.accepted_coordinates && original_evaluation==op.accepted_evaluation);
}
