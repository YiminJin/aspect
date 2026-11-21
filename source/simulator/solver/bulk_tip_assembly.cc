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
      MaterialModel::BulkCohesive::Tensor strip_direction(const double angle)
      {
        MaterialModel::BulkCohesive::Tensor S;
        if (angle==0.) { S[0][1]=.5; return S; }
        S[0][0]=-.5*std::sin(2.*angle);
        S[1][1]=-S[0][0];
        S[0][1]=.5*std::cos(2.*angle);
        return S;
      }
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
      for (const auto &section:parent->sections)
        {
          section_angles.push_back(section.angle);
          section_measures.push_back(section.weight);
          formed_sections.push_back(section.formed);
        }
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
      if (midpoint) for (const auto i:upper.locally_owned_elements()) upper[i]=.5*(1.+lower[i]);
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
      if (parent->material_strips)
        {
          std::vector<unsigned int> formed(formed_sections.size(),0);
          for (const auto &cell:parent_phase_values)for (unsigned int q=0;q<cell.second.size();++q)
              if (cell.second[q]>0.)formed[parent->section_ids.at(cell.first)[q]]=1;
          Utilities::MPI::max(formed,comm,formed);
          for (unsigned int k=0;k<formed.size();++k)formed_sections[k]=formed_sections[k] || formed[k];
        }
      if (model.tip_section_closure==Model::TipSectionClosure::coulomb_formation && this->get_timestep_number()>0)
        prepare_coulomb_planes();
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
      if (midpoint)
        {
          result.block(pb)*=2.;
          result.block(pb)-=lower;
        }
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
      auto e=std::make_shared<Evaluation>();
      e->raw.reinit(sets.system_partitioning,comm);
      e->residual.reinit(sets.system_partitioning,comm);
      const auto ns=parent->sections.size();
      e->section_solutions.resize(ns);
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
                const auto k=parent->section_ids.at(cell->id())[q];
                const auto S=strip_direction(section_angles[k]);
                const auto r=Tip::update(parameters, {h.stress,h.A,old_phis[q]},embed_tip_strain(strains[q]),phis[q],dt,S,0.);
                const auto dn=parent->material_strips ? values.JxW(q)/section_measures[k] : parent->normal_weights.at(cell->id())[q];
                free_rate[k]+=dn*r.rate;
                mobility[k]+=dn*r.fraction/(G*beta*dt);
              }
          }
      Utilities::MPI::sum(free_rate,comm,free_rate);
      Utilities::MPI::sum(mobility,comm,mobility);
      std::vector<double> fractions(model.elastic_shear_moduli.size(),0.);
      fractions[0]=1.;
      for (unsigned int k=0; k<ns; ++k)
        e->section_solutions[k]=model.fault_friction.solve_section_slip_rate(fractions,free_rate[k],mobility[k],parent->sections[k].age,
                                                                             dt,this->get_parameters().surface_pressure,model.radiation_damping_coefficients[0],
                                                                             midpoint ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                                                             : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
      if (tangent)
        {
          local_matrix=0.;
          section_integrals.assign(ns, {});
          section_adjoints.assign(ns, {});
          section_slopes.resize(ns);
          for (unsigned int k=0; k<ns; ++k)section_slopes[k]=e->section_solutions[k].resistance_derivative_wrt_free_rate;
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
                const auto S=strip_direction(section_angles[k]);
                const double f=e->section_solutions[k].resistance_response.resistance;
                auto &r=points[q];
                const Tip::State incoming
                {
                  h.stress,h.A,old_phis[q]
                };
                const auto increment=embed_tip_strain(strains[q]);
                r=Tip::update(parameters,incoming,increment,phis[q],dt,S,f);
                const double weight=values.JxW(q),dn=parent->material_strips ? weight/section_measures[k] : parent->normal_weights.at(cell->id())[q];
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
      if (midpoint)
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
          const auto &section_solution=e->section_solutions[k];
          const auto ds=section_measures[k];
          section_work+=physical_dt*ds*section_solution.resistance_response.resistance*e->rate[k];
          e->friction+=physical_dt*ds*section_solution.resistance_response.friction*section_solution.slip_rate;
          e->radiation+=physical_dt*ds*section_solution.resistance_response.radiation*section_solution.slip_rate;
          e->section_defect=std::max(e->section_defect,std::abs(e->rate[k]-section_solution.slip_rate));
          e->functional+=beta*dt*ds*model.fault_friction.compute_resistance_potential(fractions,section_solution.slip_rate,parent->sections[k].age,
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
    std::array<double,3> BulkTipCoupling<dim>::trace_functional_difference(const Vector &increment,
                                                                       const Evaluation &from,
                                                                       const Evaluation &to) const
    {
      const auto &fe=this->get_fe();
      const auto field=ghost(increment);
      FEValues<dim> values(this->get_mapping(),fe,model.mechanical_quadrature(),update_values|update_JxW_values);
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      long double storage=0.,driving=0.,scale=0.;
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())if (cell->is_locally_owned())
        {
          values.reinit(cell);cell->get_dof_indices(ids);
          for (unsigned int q=0;q<values.n_quadrature_points;++q)
            {
              const auto &a=from.points.at(cell->id())[q],&b=to.points.at(cell->id())[q];
              const double weight=values.JxW(q);
              storage+=weight*((static_cast<long double>(b.bulk_energy)-a.bulk_energy)
                               +(static_cast<long double>(b.cohesive_energy)-a.cohesive_energy));
              scale+=weight*(std::abs(a.bulk_energy)+std::abs(b.bulk_energy)+std::abs(a.cohesive_energy)+std::abs(b.cohesive_energy));
              double dphi=0.;
              for (unsigned int j=0;j<basis.expansion.m();++j)
                dphi+=phase_values(q,j)*field[ids[fe.component_to_system_index(pc,j)]];
              driving+=weight*a.driving*dphi;
            }
        }
      const auto comm=this->get_mpi_communicator();
      const double bulk=Utilities::MPI::sum(static_cast<double>(storage),comm);
      const double drive=Utilities::MPI::sum(static_cast<double>(driving),comm);
      scale=Utilities::MPI::sum(static_cast<double>(scale),comm);
      LinearAlgebra::Vector action(lower);
      fracture_matrix.vmult(action,increment.block(pb));
      // Reconstruct the fracture gradient from the raw phase gradient plus
      // its independently integrated mechanical driving. This avoids subtracting
      // two O(1) fracture energies to observe a tiny change.
      const double fracture=Kf*(from.raw.block(pb)*increment.block(pb))+drive
                            +.5*(increment.block(pb)*action);
      std::vector<double> fractions(model.elastic_shear_moduli.size(),0.);fractions[0]=1.;
      std::array<double,2> potential{};
      for (unsigned int rule=0;rule<2;++rule)
        {
          const QGauss<1> quadrature(rule==0?32:64);
          long double sum=0.;
          for (unsigned int k=0;k<parent->sections.size();++k)
            {
              const double a=from.section_solutions[k].slip_rate,b=to.section_solutions[k].slip_rate;
              if (a==b)continue;
              long double integral=0.;
              for (unsigned int q=0;q<quadrature.size();++q)
                integral+=quadrature.weight(q)*model.fault_friction.evaluate_resistance(fractions,a+quadrature.point(q)[0]*(b-a),
                      parent->sections[k].age,dt,this->get_parameters().surface_pressure,model.radiation_damping_coefficients[0],
                      midpoint?MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact:
                               MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler).resistance;
              sum+=beta*dt*section_measures[k]*(b-a)*integral;
            }
          potential[rule]=static_cast<double>(sum);
        }
      return {{bulk+fracture+potential[1],std::abs(potential[1]-potential[0]),
               64.*std::numeric_limits<double>::epsilon()*std::max(1.,static_cast<double>(scale))}};
    }

    template <int dim>
    bool BulkTipCoupling<dim>::finish_strip_active_bounds(Vector &x,const Evaluation &e)
    {
      if (!parent->material_strips)return false;
      auto trial=x;
      unsigned int changed=0;
      double largest=0.;
      for (const auto i:x.block(pb).locally_owned_elements())
        {
          const auto bound=PhaseField::box_constraint(x.block(pb)[i],e.residual.block(pb)[i],mass[i],lower[i],upper[i]);
          if (bound.status!=0 && x.block(pb)[i]!=bound.target)
            {
              largest=std::max(largest,std::abs(x.block(pb)[i]-bound.target));
              trial.block(pb)[i]=bound.target;
              ++changed;
            }
        }
      trial.block(pb).compress(VectorOperation::insert);
      changed=Utilities::MPI::sum(changed,this->get_mpi_communicator());
      largest=Utilities::MPI::max(largest,this->get_mpi_communicator());
      if (changed==0)return false;
      // Cold-creep RSF can make a small positive damage solution very different
      // from exact intactness. Classification at the original point alone is
      // not sufficient: reject a bound candidate that fails the actual equations.
      const auto response=evaluate(trial,false);
      const bool admissible=response->mechanical<=this->get_parameters().nonlinear_tolerance
                            && response->kkt<=this->get_phase_field_handler().get_solver_parameters().nonlinear_solver_tolerance;
      this->get_pcout()<<"   Tip strip active-bound candidate: rows="<<changed<<", max stage change="<<largest
                       <<", mechanics="<<response->mechanical<<", phase KKT="<<response->kkt
                       <<", admissible="<<admissible<<std::endl;
      if (admissible)x=trial;
      return admissible;
    }

    template <int dim>
    void BulkTipCoupling<dim>::prepare_coulomb_planes()
    {
      TimerOutput::Scope timer(this->get_computing_timer(),"Tip Coulomb plane selection");
      // Integrate signed stress before forming a Coulomb score. The positive
      // strip area and the angle-independent isotropic pressure cancel in argmax.
      const auto count=parent->sections.size();
      std::vector<double> moments(2*count,0.);
      FEValues<dim> values(this->get_mapping(),this->get_fe(),model.mechanical_quadrature(),update_JxW_values);
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            const auto &indices=parent->section_ids.at(cell->id());
            const auto &history=parent->history.at(cell->id());
            for (unsigned int q=0;q<values.n_quadrature_points;++q)
              {
                const auto k=indices[q];
                if (formed_sections[k])continue;
                AssertThrow(history[q].A.norm()==0. && history[q].absolute_shear==0.,
                            ExcMessage("A Coulomb formation plane cannot reorient a section with fault history."));
                const auto &tau=history[q].stress;
                moments[2*k]+=values.JxW(q)*.5*(tau[1][1]-tau[0][0]);
                moments[2*k+1]+=values.JxW(q)*tau[0][1];
              }
          }
      Utilities::MPI::sum(moments,this->get_mpi_communicator(),moments);
      const double mu=model.fault_friction.reference_friction_coefficient(0);
      for (unsigned int k=0;k<count;++k)
        {
          const double u=moments[2*k],v=moments[2*k+1];
          if (formed_sections[k] || (u==0. && v==0.))continue;
          const auto &incoming=parent->sections[k];
          double best_distance=numbers::PI;
          // Each signed sinusoid has one maximum modulo pi. The maxima have
          // equal height. Use the nearest forward branch, positive shear on a tie.
          for (const double sign:{1.,-1.})
            {
              const double angle=.5*std::atan2(sign*u-mu*v,sign*v+mu*u);
              const double distance=std::abs(angle-incoming.angle);
              if (std::abs(angle)<numbers::PI/4. && distance<best_distance)
                {
                  best_distance=distance;
                  section_angles[k]=angle;
                }
            }
          AssertThrow(best_distance<numbers::PI,
                      ExcMessage("No forward Coulomb formation plane in section "+Utilities::to_string(k)+"."));
          section_measures[k]=incoming.weight*std::cos(incoming.angle)/std::cos(section_angles[k]);
        }
    }

    template <int dim>
    bool BulkTipCoupling<dim>::update_virgin_planes(const Vector &x)
    {
      if (model.tip_section_closure!=Model::TipSectionClosure::intact_driving || this->get_timestep_number()==0)return false;
      TimerOutput::Scope timer(this->get_computing_timer(),"Tip virgin plane selection");
      const auto ns=parent->sections.size();
      // Moments of the actual native physical-stress predictor and phase.
      // All points in a section participate, including quiescent intact bulk.
      // For virgin A=0: tau_m:S-f=(1-F)*(predictor:S-f).
      std::vector<double> moments(9*ns,0.);
      const auto field=ghost(x);
      const auto &fe=this->get_fe();
      FEValues<dim> values(this->get_mapping(),fe,model.mechanical_quadrature(),
                           update_values|update_gradients|update_JxW_values);
      std::vector<SymmetricTensor<2,dim>> strains(values.n_quadrature_points);
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())if (cell->is_locally_owned())
          {
            values.reinit(cell);
            cell->get_dof_indices(ids);
            values[this->introspection().extractors.velocities].get_function_symmetric_gradients(field,strains);
            for (unsigned int q=0;q<values.n_quadrature_points;++q)
              {
                const auto k=parent->section_ids.at(cell->id())[q];
                if (formed_sections[k])continue;
                const auto &old=parent->history.at(cell->id())[q];
                AssertThrow(old.A.norm()==0. && old.absolute_shear==0.,
                            ExcMessage("Virgin plane selection cannot reassign existing fault tensor history."));
                double phi=0.;
                for (unsigned int j=0;j<basis.expansion.m();++j)
                  phi+=phase_values(q,j)*field[ids[fe.component_to_system_index(pc,j)]];
                const double c=parent->parameters.m*phi*(1.+phi),d=(1.-phi)*(1.-phi),F=c/(c+d);
                const auto predictor=beta*(old.stress+2.*G*embed_tip_strain(strains[q]));
                const double u=-.5*(predictor[0][0]-predictor[1][1]),v=predictor[0][1];
                const double w=values.JxW(q),a=w*(1.-F)*(1.-F);
                const std::array<double,9> local{{w*F*u,w*F*v,w*F,a*u*u,a*v*v,a*u*v,a*u,a*v,a}};
                for (unsigned int j=0;j<9;++j)moments[9*k+j]+=local[j];
              }
          }
      Utilities::MPI::sum(moments,this->get_mpi_communicator(),moments);
      std::vector<double> fractions(model.elastic_shear_moduli.size(),0.);
      fractions[0]=1.;
      auto angles=section_angles;
      double largest_change=0.,largest_gain=0.;
      unsigned int changed=0;
      for (unsigned int k=0;k<ns;++k)if (!formed_sections[k])
          {
            const auto m=&moments[9*k];
            if (m[3]+m[4]==0.)continue; // No force: retain the incoming preference.
            const double dx=parent->sections[k].weight*std::cos(parent->sections[k].angle);
            const auto objective=[&](const double angle)
            {
              const double s=std::sin(2.*angle),c=std::cos(2.*angle),ds=dx/std::cos(angle);
              const double free=(m[0]*s+m[1]*c)/(G*beta*dt*ds),mobility=m[2]/(G*beta*dt*ds);
              const auto law=model.fault_friction.solve_section_slip_rate(fractions,free,mobility,parent->sections[k].age,
                                                                         dt,this->get_parameters().surface_pressure,
                                                                         model.radiation_damping_coefficients[0],
                                                                         MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact);
              const double f=law.resistance_response.resistance;
              return m[3]*s*s+m[4]*c*c+2.*m[5]*s*c-2.*f*(m[6]*s+m[7]*c)+m[8]*f*f;
            };
            // A single forward graph: choose the conjugate sector closest to
            // the declared horizontal seed. No projection of old tensor memory.
            const unsigned int intervals=128;
            const double limit=numbers::PI/4.,spacing=2.*limit/intervals;
            unsigned int best=0;
            double maximum=-1.;
            for (unsigned int j=0;j<=intervals;++j)
              {
                const double value=objective(-limit+j*spacing);
                if (value>maximum) { maximum=value; best=j; }
              }
            if (best==0 || best==intervals)
              this->get_pcout()<<"   Virgin branch conflict: section="<<k<<", x="<<parent->sections[k].coordinate
                               <<", angle="<<(-limit+best*spacing)<<", dx="<<dx
                               <<", Y0 integral="<<maximum*parent->parameters.m/(2.*G)
                               <<", fraction volume="<<m[2]<<", age="<<parent->sections[k].age
                               <<", slip="<<parent->sections[k].slip<<std::endl;
            AssertThrow(best>0 && best<intervals,
                        ExcMessage("Virgin section "+Utilities::to_string(k)+" reaches the forward-graph conjugate boundary; no plane admitted."));
            double lo=-limit+(best-1)*spacing,hi=-limit+(best+1)*spacing;
            const double ratio=(std::sqrt(5.)-1.)/2.;
            double left=hi-ratio*(hi-lo),right=lo+ratio*(hi-lo),fl=objective(left),fr=objective(right);
            while (hi-lo>1e-9)
              {
                if (fl<fr) { lo=left; left=right; fl=fr; right=lo+ratio*(hi-lo); fr=objective(right); }
                else { hi=right; right=left; fr=fl; left=hi-ratio*(hi-lo); fl=objective(left); }
              }
            double angle=.5*(lo+hi);
            // Resolve roundoff-level angular ties by the retained preference.
            const double incoming=objective(section_angles[k]),selected=objective(angle);
            if (selected-incoming<=128.*std::numeric_limits<double>::epsilon()*(m[3]+m[4]))angle=section_angles[k];
            const double change=std::abs(angle-section_angles[k]);
            largest_change=std::max(largest_change,change);
            largest_gain=std::max(largest_gain,(selected-incoming)*parent->parameters.m/(2.*G));
            if (change>1e-7) { angles[k]=angle; ++changed; }
          }
      this->get_pcout()<<"   Tip virgin planes: changed="<<changed<<", max angle change="<<largest_change
                       <<", integrated Y0 gain="<<largest_gain<<std::endl;
      if (changed==0)return false;
      AssertThrow(++orientation_updates<=12,ExcMessage("Virgin-plane/mechanics outer iteration failed to converge; accepted histories preserved."));
      section_angles=angles;
      for (unsigned int k=0;k<ns;++k)if (!formed_sections[k])
          section_measures[k]=parent->sections[k].weight*std::cos(parent->sections[k].angle)/std::cos(section_angles[k]);
      return true;
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
      std::vector<double> forces(e.section_solutions.size());
      for (unsigned int k=0; k<forces.size(); ++k)forces[k]=e.section_solutions[k].resistance_response.resistance;
      auto next=model.prepare_bulk_tip_candidate(parent,e.points,e.rate,forces,physical_field(x).block(pb));
      if (parent->material_strips)
        {
          std::vector<unsigned int> formed(formed_sections.size(),0);
          for (const auto &cell:e.points)for (unsigned int q=0;q<cell.second.size();++q)
              {
                const auto k=parent->section_ids.at(cell.first)[q];
                if (cell.second[q].state.phi>0.)formed[k]=1;
                next->history.at(cell.first)[q].q=cell.second[q].Q*strip_direction(section_angles[k]);
                next->normal_weights.at(cell.first)[q]*=parent->sections[k].weight/section_measures[k];
              }
          Utilities::MPI::max(formed,this->get_mpi_communicator(),formed);
          for (unsigned int k=0;k<formed.size();++k)
            {
              next->sections[k].angle=section_angles[k];
              next->sections[k].weight=section_measures[k];
              next->sections[k].formed=formed_sections[k] || formed[k];
            }
        }
      return next;
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
              if (parent->material_strips)
                {
                  unsigned int finish=0;
                  for (const auto i:state.locally_owned_elements())
                    {
                      const auto bound=PhaseField::box_constraint(state[i],action[i]/Kf,mass[i],lower[i],1.);
                      if (bound.status!=0 && state[i]!=bound.target) { state[i]=bound.target; ++finish; }
                    }
                  state.compress(VectorOperation::insert);
                  if (Utilities::MPI::sum(finish,this->get_mpi_communicator()))continue;
                }
              x.block(pb)=state;
              if (parent->material_strips)
                {
                  const auto field=ghost(x);
                  std::vector<unsigned int> formed(parent->sections.size(),0);
                  const auto &fe=this->get_fe();
                  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
                  for (const auto &cell:parent->section_ids)
                    {
                      parent->native_cells.at(cell.first)->get_dof_indices(ids);
                      for (unsigned int q=0;q<cell.second.size();++q)
                        {
                          double phi=0.;
                          for (unsigned int j=0;j<basis.expansion.m();++j)
                            phi+=phase_values(q,j)*field[ids[fe.component_to_system_index(pc,j)]];
                          if (phi>0.)formed[cell.second[q]]=1;
                        }
                    }
                  Utilities::MPI::max(formed,this->get_mpi_communicator(),formed);
                  for (unsigned int k=0;k<formed.size();++k)model.bulk_cohesive_state->sections[k].formed=formed[k];
                }
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
