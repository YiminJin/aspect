/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator/solver/bulk_tip.h>
#include <aspect/simulator.h>
#include <aspect/phase_field.h>
#include <aspect/phase_field/box_constraints.h>
#include <aspect/phase_field/obstacle.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>

namespace aspect
{
  namespace internal
  {
    namespace
    {
      template <int dim>
      MaterialModel::BulkCohesive::Tensor embed_tip_strain(const SymmetricTensor<2,dim> &strain)
      {
        const auto d = Utilities::Tensors::consistent_deviator(strain);
        MaterialModel::BulkCohesive::Tensor result;
        result[0][0]=d[0][0];
        result[1][1]=d[1][1];
        result[2][2]=-d[0][0]-d[1][1];
        result[0][1]=d[0][1];
        return result;
      }
    }

    template <int dim>
    BulkTipCoupling<dim>::BulkTipCoupling(Simulator<dim> &simulator, Model &material,
                                          const AffineConstraints<double> &constraints)
      : model(material), parent(material.bulk_cohesive_state),
        basis(material.get_fe().base_element(material.get_fe().component_to_base_index(
                                               material.introspection().phase_field().first_component_index).first))
    {
      this->initialize_simulator(simulator);
      AssertThrow(dim==2 && parent && parent->direct_tip() && model.uses_integrated_bulk_tip(),
                  ExcMessage("Distributed bulk-tip operator requires its physical native history."));
      AssertThrow(this->get_geometry_model().get_periodic_boundary_pairs().empty()
                  && parent->mesh==this->get_triangulation().get_vertices()
                  && (this->get_timestep_number()==0 || parent->step+1==this->get_timestep_number())
                  && this->get_phase_field_handler().uses_dg_phase(),
                  ExcMessage("Bulk tip requires its accepted clock, fixed nonperiodic mesh and native DG phase."));
      const auto &intro=this->introspection();
      const auto &sets=intro.index_sets;
      const auto &fe=this->get_fe();
      const auto comm=this->get_mpi_communicator();
      pb=intro.phase_field().block_index;
      pc=intro.phase_field().first_component_index;
      po=this->get_solution().get_block_indices().block_start(pb);
      nv=this->get_solution().block(0).size();
      pressure_begin=nv;
      pressure_end=nv+this->get_solution().block(1).size();
      physical_dt=this->get_timestep_number()>0 ? this->get_timestep() : model.initial_time_step;
      midpoint=model.uses_bulk_tip_midpoint() && this->get_timestep_number()>0;
      dt=midpoint ? .5*physical_dt : physical_dt;
      AssertThrow(!midpoint || physical_dt*parent->parameters.G/parent->parameters.eta<=2.,
                  ExcMessage("Midpoint control requires dt*G/eta<=2 to exclude sign-changing free Maxwell relaxation; no BE fallback."));
      G=parent->parameters.G;
      beta=MaterialModel::BulkCohesive::Tip::relaxation_factor(G,parent->parameters.eta,dt);
      const double ell=this->get_phase_field_handler().get_length_scale();
      Kf=model.critical_energy_release_rates[0]/((8./3.)*ell);
      boundary.copy_from(constraints);
      const auto boundary_velocity=DoFTools::extract_boundary_dofs(this->get_dof_handler(),intro.component_masks.velocities);
      for (const auto i:boundary_velocity)
        if (sets.system_partitioning[0].is_element(i))
          AssertThrow(boundary.is_constrained(i),
                      ExcMessage("The integrated tip control requires prescribed velocity on the complete boundary."));
      for (const auto &line:constraints.get_lines())
        if (line.index<nv) boundary.set_inhomogeneity(line.index,dt*line.inhomogeneity);
      // Fixed-reference closed velocity control: p is only a multiplier. The
      // reference pressure used in friction never enters this gauge.
      if (boundary.can_store_line(pressure_begin) && !boundary.is_constrained(pressure_begin))
        boundary.add_line(pressure_begin);
      boundary.close();
      homogeneous.copy_from(boundary);
      for (const auto &line:boundary.get_lines()) homogeneous.set_inhomogeneity(line.index,0.);
      homogeneous.close();
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      const unsigned int nphase=basis.expansion.m();
      AssertThrow(basis.native_bernstein,ExcMessage("Bulk tip requires native Bernstein phase coordinates."));
      lower.reinit(sets.system_partitioning[pb],comm);
      auto parent_field=ghost(this->get_solution());
      parent_field.block(pb)=parent->tip_phase;
      auto initial=coordinates(parent_field);
      lower=initial.block(pb);
      upper.reinit(sets.system_partitioning[pb],comm);
      upper=1.;
      if(midpoint) for(const auto i:upper.locally_owned_elements()) upper[i]=.5*(1.+lower[i]);
      upper.compress(VectorOperation::insert);
      mass.reinit(sets.system_partitioning[pb],comm);
      const auto quadrature = model.mechanical_quadrature();
      phase_values.reinit(quadrature.size(), nphase);
      const auto &phase_element = fe.base_element(fe.component_to_base_index(pc).first);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int j = 0; j < nphase; ++j)
          phase_values(q,j) = basis.native_bernstein
                              ? phase_element.shape_value(j, quadrature.point(q))
                              : basis.basis_value(j, quadrature.point(q));
      LinearAlgebra::Vector old_coefficients(sets.system_partitioning[pb], sets.system_relevant_partitioning[pb], comm);
      old_coefficients = lower;
      FEValues<dim> values(this->get_mapping(), fe, quadrature, update_JxW_values);
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(ids);
            values.reinit(cell);
            auto &samples = parent_phase_values[cell->id()];
            samples.assign(quadrature.size(), 0.);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              for (unsigned int j = 0; j < nphase; ++j)
                {
                  const auto index = ids[fe.component_to_system_index(pc,j)]-po;
                  mass[index] += values.JxW(q)*phase_values(q,j);
                  samples[q] += phase_values(q,j)*old_coefficients[index];
                }
          }
      mass.compress(VectorOperation::add);
      LinearAlgebra::Vector native_gradient;
      LinearAlgebra::SparseMatrix native_hessian;
      this->get_phase_field_handler().assemble_dg_fracture(parent_field,native_gradient,&native_hessian);
      fracture_matrix.copy_from(native_hessian);
      fracture_linear.reinit(sets.system_partitioning[pb], comm);
      fracture_linear = native_gradient;
      LinearAlgebra::Vector action(lower);
      fracture_matrix.vmult(action,lower);
      fracture_linear-=action;
      LinearAlgebra::BlockDynamicSparsityPattern pattern;
      pattern.reinit(sets.system_partitioning,sets.system_partitioning,sets.system_relevant_partitioning,comm);
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
          {
            cell->get_dof_indices(ids);
            std::vector<types::global_dof_index> selected;
            for (const auto i:ids) if (i<pressure_end || (i>=po && i<po+lower.size())) selected.push_back(i);
            homogeneous.add_entries_local_to_global(selected,pattern,true);
          }
      for (const auto i:sets.system_partitioning[pb])
        for (auto entry=fracture_matrix.begin(i); entry!=fracture_matrix.end(i); ++entry)
          pattern.add(po+i,po+entry->column());
      for (const auto &line:homogeneous.get_lines())pattern.add(line.index,line.index);
      pattern.compress();
      local_matrix.reinit(pattern);
    }

    template <int dim>
    typename BulkTipCoupling<dim>::Vector BulkTipCoupling<dim>::ghost(const Vector &v) const
    {
      Vector result(this->introspection().index_sets.system_partitioning,
                    this->introspection().index_sets.system_relevant_partitioning,this->get_mpi_communicator());
      result=v;
      return result;
    }

    template <int dim>
    typename BulkTipCoupling<dim>::Vector BulkTipCoupling<dim>::coordinates(const Vector &v) const
    {
      Vector result(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
      result=v;
      result.block(0)*=dt;
      result.block(1)/=G;
      return result;
    }

    template <int dim>
    typename BulkTipCoupling<dim>::Vector BulkTipCoupling<dim>::physical_field(const Vector &v) const
    {
      Vector result(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
      result=v;
      result.block(0)/=dt;
      result.block(1)*=G;
      if(midpoint) { result.block(pb)*=2.; result.block(pb)-=lower; }
      return ghost(result);
    }

    template <int dim>
    typename BulkTipCoupling<dim>::Vector BulkTipCoupling<dim>::initial_guess(const Vector &v) const
    {
      auto result=coordinates(v);
      result.block(pb)=lower;
      // Subtract the same constant everywhere before pinning its pressure row.
      const double shift=Utilities::MPI::sum(result.block(1).locally_owned_elements().is_element(0)?double(result.block(1)[0]):0.,this->get_mpi_communicator());
      for (const auto i:result.block(1).locally_owned_elements()) result.block(1)[i]-=shift;
      result.compress(VectorOperation::insert);
      boundary.distribute(result);
      return result;
    }

    template <int dim>
    std::shared_ptr<typename BulkTipCoupling<dim>::Evaluation>
    BulkTipCoupling<dim>::evaluate(const Vector &x,const bool tangent)
    {
      TimerOutput::Scope timer(this->get_computing_timer(), tangent ? "Tip tangent evaluation" : "Tip trial evaluation");
      ++evaluations;
      if (tangent)++tangent_assemblies;
      namespace Tip=MaterialModel::BulkCohesive::Tip;
      using Tensor=MaterialModel::BulkCohesive::Tensor;
      const auto comm=this->get_mpi_communicator();
      const auto &fe=this->get_fe();
      const auto &intro=this->introspection();
      const auto &sets=intro.index_sets;
      const auto &parameters=parent->parameters;
      Tensor S;
      S[0][1]=.5;
      auto e=std::make_shared<Evaluation>();
      e->raw.reinit(sets.system_partitioning,comm);
      e->residual.reinit(sets.system_partitioning,comm);
      const auto ns=parent->sections.size();
      e->laws.resize(ns);
      for (auto *v:
           {
             &e->rate,&e->absolute_rate,&e->positive_rate,&e->negative_rate
           })v->assign(ns,0.);
      std::vector<double> free_rate(ns,0.),mobility(ns,0.);
      // Velocity and pressure already have solver coordinates. Phase is
      // sampled from Bernstein coordinates below on both compatibility paths;
      // no nodal copy or second ghost exchange is needed for this evaluation.
      const auto field=ghost(x);
      FEValues<dim> values(this->get_mapping(),fe,model.mechanical_quadrature(),
                           update_values|update_gradients|update_JxW_values);
      const unsigned int nq=values.n_quadrature_points,dpc=fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> ids(dpc);
      std::vector<SymmetricTensor<2,dim>> strains(nq);
      std::vector<double> phis(nq),old_phis(nq),pressures(nq);
      auto sample=[&](const auto &cell)
      {
        values.reinit(cell);
        cell->get_dof_indices(ids);
        values[intro.extractors.velocities].get_function_symmetric_gradients(field,strains);
        std::fill(phis.begin(),phis.end(),0.);
        old_phis = parent_phase_values.at(cell->id());
        for (unsigned int q=0; q<nq; ++q)for (unsigned int j=0; j<basis.expansion.m(); ++j)
            {
              const auto index=ids[fe.component_to_system_index(pc,j)];
              phis[q]+=phase_values(q,j)*field[index];
            }
        values[intro.extractors.pressure].get_function_values(field,pressures);
      };
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
          {
            sample(cell);
            for (unsigned int q=0; q<nq; ++q)
              {
                const auto &h=parent->history.at(cell->id())[q];
                const auto r=Tip::update(parameters, {h.stress,h.A,old_phis[q]},embed_tip_strain(strains[q]),phis[q],dt,S,0.);
                const auto k=parent->section_ids.at(cell->id())[q];
                const auto dn=parent->normal_weights.at(cell->id())[q];
                free_rate[k]+=dn*r.rate;
                mobility[k]+=dn*r.fraction/(G*beta*dt);
              }
          }
      Utilities::MPI::sum(free_rate,comm,free_rate);
      Utilities::MPI::sum(mobility,comm,mobility);
      std::vector<double> fractions(model.elastic_shear_moduli.size(),0.);
      fractions[0]=1.;
      for (unsigned int k=0; k<ns; ++k)
        e->laws[k]=model.fault_friction.section_endpoint(fractions,free_rate[k],mobility[k],parent->sections[k].age,
                                                         dt,this->get_parameters().surface_pressure,model.radiation_damping_coefficients[0],
            midpoint ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                     : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
      if (tangent)
        {
          local_matrix=0.;
          section_integrals.assign(ns, {});
          section_adjoints.assign(ns, {});
          section_slopes.resize(ns);
          for (unsigned int k=0; k<ns; ++k)section_slopes[k]=e->laws[k].force_derivative;
        }
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
          {
            sample(cell);
            auto &points=e->points[cell->id()];
            points.resize(nq);
            FullMatrix<double> cell_matrix(dpc,dpc);
            dealii::Vector<double> cell_residual(dpc);
            for (unsigned int q=0; q<nq; ++q)
              {
                const auto &h=parent->history.at(cell->id())[q];
                const auto k=parent->section_ids.at(cell->id())[q];
                const double f=e->laws[k].law.traction;
                auto &r=points[q];
                const Tip::State incoming
                {
                  h.stress,h.A,old_phis[q]
                };
                const auto increment=embed_tip_strain(strains[q]);
                r=Tip::update(parameters,incoming,increment,phis[q],dt,S,f);
                const double weight=values.JxW(q),dn=parent->normal_weights.at(cell->id())[q];
                e->rate[k]+=dn*r.rate;
                e->absolute_rate[k]+=dn*std::abs(r.rate);
                e->positive_rate[k]+=dn*std::max(0.,r.rate);
                e->negative_rate[k]+=dn*std::min(0.,r.rate);
                const auto ledger=midpoint ? Tip::midpoint_account(parameters,incoming,r,2.*increment,physical_dt,f)
                                           : Tip::account(parameters,incoming,r,increment,dt,f);
                const auto endpoint=midpoint ? Tip::midpoint_endpoint(incoming,r) : r.state;
                e->stage_storage+=weight*(r.bulk_energy+r.cohesive_energy);
                e->bulk+=weight*(midpoint ? (endpoint.stress*endpoint.stress)/(4.*G) : r.bulk_energy);
                e->cohesive+=weight*(midpoint ? Tip::cohesive_energy(parameters,endpoint) : r.cohesive_energy);
                e->Db+=weight*physical_dt*r.bulk_viscosity;
                e->Dc+=weight*physical_dt*r.cohesive_viscosity;
                e->Df+=weight*physical_dt*f*r.rate;
                e->storage_change+=weight*ledger.storage_change;
                e->internal_work+=weight*ledger.internal_work;
                e->stress_loss+=weight*ledger.stress_loss;
                e->traction_loss+=weight*ledger.traction_loss;
                e->compliance_remainder+=weight*(ledger.compliance_work-r.driving*(endpoint.phi-old_phis[q]));
                std::vector<Tensor> E(dpc);
                std::vector<double> div(dpc),P(dpc),N(dpc,0.);
                for (unsigned int i=0; i<dpc; ++i)
                  {
                    E[i]=embed_tip_strain(values[intro.extractors.velocities].symmetric_gradient(i,q));
                    div[i]=values[intro.extractors.velocities].divergence(i,q);
                    P[i]=values[intro.extractors.pressure].value(i,q);
                  }
                for (unsigned int j=0; j<basis.expansion.m(); ++j)
                  N[fe.component_to_system_index(pc,j)]=phase_values(q,j);
                for (unsigned int i=0; i<dpc; ++i)
                  {
                    cell_residual[i]+=weight*((r.state.stress*E[i])/G-pressures[q]*div[i]-P[i]*trace(strains[q])-N[i]*r.driving/Kf);
                    if (tangent)
                      {
                        const auto force_action=Tip::derivative(parameters,r, {},0.,dt,S,1.);
                        section_adjoints[k][ids[i]]+=weight*((force_action.stress*E[i])/G-N[i]*force_action.driving/Kf);
                      }
                  }
                if (tangent)for (unsigned int j=0; j<dpc; ++j)
                    {
                      const auto action=Tip::derivative(parameters,r,E[j],N[j],dt,S,0.);
                      section_integrals[k][ids[j]]+=dn*action.rate;
                      for (unsigned int i=0; i<dpc; ++i)
                        cell_matrix(i,j)+=weight*((action.stress*E[i])/G-P[j]*div[i]-P[i]*div[j]-N[i]*action.driving/Kf);
                    }
              }
            // Native local assembly and distributed add/compress; no global gather.
            for (unsigned int i=0; i<dpc; ++i)if (cell_residual[i]!=0.)e->raw[ids[i]] += cell_residual[i];
            if (tangent)homogeneous.distribute_local_to_global(cell_matrix,ids,local_matrix);
          }
      e->raw.compress(VectorOperation::add);
      for (double *value:
           {
             &e->stage_storage,&e->bulk,&e->cohesive,&e->Db,&e->Dc,&e->Df,&e->storage_change,&e->internal_work,
             &e->stress_loss,&e->traction_loss,&e->compliance_remainder
           })
        *value=Utilities::MPI::sum(*value,comm);
      for (auto *v:
           {
             &e->rate,&e->absolute_rate,&e->positive_rate,&e->negative_rate
           })Utilities::MPI::sum(*v,comm,*v);
      LinearAlgebra::Vector gradient(lower),action(lower),delta(lower);
      delta=x.block(pb);
      delta-=lower;
      fracture_matrix.vmult(action,x.block(pb));
      gradient=action;
      gradient+=fracture_linear;
      e->fracture=.5*(x.block(pb)*action)+x.block(pb)*fracture_linear;
      fracture_matrix.vmult(action,delta);
      e->stage_storage+=e->fracture;
      e->fracture_loss=midpoint ? 0. : .5*(delta*action);
      if(midpoint)
        {
          // Quadratic fracture chain rule uses the stage gradient and full change.
          e->fracture+=gradient*delta+.5*(delta*action);
          delta*=2.;
        }
      fracture_matrix.vmult(action,lower);
      const double old_fracture=.5*(lower*action)+lower*fracture_linear;
      e->storage_change+=e->fracture-old_fracture;
      for (const auto i:sets.system_partitioning[pb])e->raw.block(pb)[i]+=gradient[i]/Kf;
      e->raw.compress(VectorOperation::insert);
      if (tangent)
        {
          for (const auto i:sets.system_partitioning[pb])
            for (auto entry=fracture_matrix.begin(i); entry!=fracture_matrix.end(i); ++entry)
              local_matrix.block(pb,pb).add(i,entry->column(),entry->value()/Kf);
          local_matrix.compress(VectorOperation::add);
          linearization=e;
        }
      e->residual=e->raw;
      homogeneous.set_zero(e->residual);
      e->mechanical=std::hypot(e->residual.block(0).l2_norm(),e->residual.block(1).l2_norm());
      double kkt=0.,external=0.;
      for (const auto i:sets.system_partitioning[pb])
        kkt=std::max(kkt,std::abs(PhaseField::box_constraint(x.block(pb)[i],e->residual.block(pb)[i],mass[i],lower[i],upper[i]).projected_residual));
      e->kkt=Utilities::MPI::max(kkt,comm);
      for (const auto &line:boundary.get_lines())
        if (line.index<nv && sets.system_partitioning[0].is_element(line.index))external+=G*x[line.index]*e->raw[line.index];
      e->external_work=(midpoint ? 2. : 1.)*Utilities::MPI::sum(external,comm);
      e->phase_work=Kf*(delta*e->residual.block(pb));
      e->functional=midpoint ? e->stage_storage : e->bulk+e->cohesive+e->fracture;
      double section_work=0.;
      for (unsigned int k=0; k<ns; ++k)
        {
          const auto &law=e->laws[k];
          const auto ds=parent->sections[k].weight;
          section_work+=physical_dt*ds*law.law.traction*e->rate[k];
          e->friction+=physical_dt*ds*law.law.friction*law.rate;
          e->radiation+=physical_dt*ds*law.law.radiation*law.rate;
          e->section_defect=std::max(e->section_defect,std::abs(e->rate[k]-law.rate));
          e->functional+=beta*dt*ds*model.fault_friction.section_primitive(fractions,law.rate,parent->sections[k].age,
                                                                           dt,this->get_parameters().surface_pressure,model.radiation_damping_coefficients[0],
            midpoint ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                     : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
        }
      e->adjoint_error=e->Df-section_work;
      e->ledger_error=e->internal_work-e->storage_change-e->Db-e->Dc-e->Df-e->stress_loss-e->traction_loss
                      -e->compliance_remainder-e->fracture_loss+e->phase_work;
      return e;
    }

    template <int dim>
    void BulkTipCoupling<dim>::vmult(Vector &destination,const Vector &source) const
    {
      local_matrix.vmult(destination,source);
      const auto readable=ghost(source);
      std::vector<double> actions(section_integrals.size(),0.);
      for (unsigned int k=0; k<actions.size(); ++k)
        for (const auto &entry:section_integrals[k])actions[k]+=entry.second*readable[entry.first];
      Utilities::MPI::sum(actions,this->get_mpi_communicator(),actions);
      for (unsigned int k=0; k<actions.size(); ++k)
        for (const auto &entry:section_adjoints[k])
          if (entry.second!=0.)destination[entry.first] += entry.second*section_slopes[k]*actions[k];
      destination.compress(VectorOperation::add);
      homogeneous.set_zero(destination);
      for (const auto &line:homogeneous.get_lines())
        if (source.locally_owned_elements().is_element(line.index))destination[line.index]=source[line.index];
      for (unsigned int b=2; b<source.n_blocks(); ++b)if (b!=pb)destination.block(b)=source.block(b);
      destination.compress(VectorOperation::insert);
    }

    template <int dim>
    void BulkTipCoupling<dim>::phase_search_matrix(LinearAlgebra::SparseMatrix &matrix,const double shift) const
    {
      matrix.copy_from(fracture_matrix);
      matrix*=1./Kf;
      for (const auto i:mass.locally_owned_elements())matrix.add(i,i,shift*mass[i]);
      matrix.compress(VectorOperation::add);
    }

    template <int dim>
    double BulkTipCoupling<dim>::directional_work(const Evaluation &e,const Vector &direction) const
    {
      return beta*G*(e.residual.block(0)*direction.block(0))+Kf*(e.residual.block(pb)*direction.block(pb));
    }

    template <int dim>
    std::shared_ptr<typename BulkTipCoupling<dim>::State>
    BulkTipCoupling<dim>::candidate(const Vector &x,const Evaluation &e) const
    {
      std::vector<double> forces(e.laws.size());
      for (unsigned int k=0; k<forces.size(); ++k)forces[k]=e.laws[k].law.traction;
      return model.prepare_bulk_tip_candidate(parent,e.points,e.rate,forces,physical_field(x).block(pb));
    }

    template <int dim>
    void BulkTipCoupling<dim>::equilibrate_seed(Vector &x)
    {
      LinearAlgebra::SparseMatrix search;
      phase_search_matrix(search,1.);
      LinearAlgebra::Vector load(lower),upper(lower),state(lower),action(lower);
      upper=1.;
      state=lower;
      const auto &settings=this->get_phase_field_handler().get_solver_parameters();
      for (unsigned int iteration=0; iteration<settings.max_nonlinear_iterations; ++iteration)
        {
          for (const auto i:load.locally_owned_elements())load[i]=mass[i]*state[i]-fracture_linear[i]/Kf;
          load.compress(VectorOperation::insert);
          PhaseField::solve_obstacle(search,load,lower,upper,state,
                                     this->introspection().index_sets.system_relevant_partitioning[pb],this->get_mpi_communicator(),
                                     settings.nonlinear_solver_tolerance*mass.linfty_norm()*.05,100,1000);
          fracture_matrix.vmult(action,state);
          action+=fracture_linear;
          double kkt=0.;
          for (const auto i:state.locally_owned_elements())
            kkt=std::max(kkt,std::abs(PhaseField::box_constraint(state[i],action[i]/Kf,mass[i],lower[i],1.).projected_residual));
          if (Utilities::MPI::max(kkt,this->get_mpi_communicator())<=settings.nonlinear_solver_tolerance)
            {
              x.block(pb)=state;
              return;
            }
        }
      AssertThrow(false,ExcMessage("Zero-clock seed equilibration failed its physical fracture KKT test."));
    }
  }
#define INSTANTIATE(dim) template class internal::BulkTipCoupling<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
