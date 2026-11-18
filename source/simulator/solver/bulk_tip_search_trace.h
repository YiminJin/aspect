/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_INTERNAL_BULK_TIP_SEARCH_TRACE_H
#define ASPECT_INTERNAL_BULK_TIP_SEARCH_TRACE_H
#include <fstream>
#include <iomanip>

namespace aspect { namespace internal {
  /** Compare the coupled directional derivative to feasible nonlinear probes.
   * evaluate(false) re-solves the sections without replacing the assembled tangent. */
  template <int dim>
  void probe_bulk_tip_tangent(BulkTipCoupling<dim> &op,
                              const typename BulkTipCoupling<dim>::Vector &x,
                              const typename BulkTipCoupling<dim>::Evaluation &current,
                              const typename BulkTipCoupling<dim>::Vector &direction,
                              const types::global_dof_index row)
  {
    const auto pb=op.phase_block();
    const auto &owned=x.block(pb).locally_owned_elements();
    const auto comm=op.get_mpi_communicator();
    AssertThrow(row<x.block(pb).size(),ExcMessage("Phase tangent probe row out of range."));
    for (const double alpha:{1e-2,1e-3,1e-4,1e-5,1e-6})
      {
        auto trial=x;
        trial.add(alpha,direction);
        unsigned int feasible=1;
        for (const auto i:owned)
          {
            const double value=PhaseField::box_trial_value(x.block(pb)[i],direction.block(pb)[i],
                                  op.obstacle()[i],op.upper_bound()[i],alpha);
            feasible &= value>=op.obstacle()[i] && value<=op.upper_bound()[i];
            trial.block(pb)[i]=value;
          }
        AssertThrow(Utilities::MPI::min(feasible,comm),ExcMessage("Infeasible tangent probe."));
        trial.compress(VectorOperation::insert);
        const auto actual=op.evaluate(trial,false);
        auto delta=trial;
        delta-=x;
        auto predicted=delta;
        op.vmult(predicted,delta);
        auto measured=actual->residual;
        measured-=current.residual;
        const auto component=[&](const auto &v)
        {
          return Utilities::MPI::sum(owned.is_element(row)?v.block(pb)[row]/op.phase_mass()[row]:0.,comm);
        };
        op.get_pcout()<<"   Tip coupled tangent probe: row="<<row<<", alpha="<<alpha
                      <<", measured="<<component(measured)<<", predicted="<<component(predicted);
        measured-=predicted;
        for (const auto block:{0u,1u,pb})
          op.get_pcout()<<", block "<<block<<" error="<<measured.block(block).l2_norm()
                        <<", prediction="<<predicted.block(block).l2_norm();
        op.get_pcout()<<std::endl;
      }
  }

  /** Sparse, read-only diagnostics of a fixed-plane nonlinear search.
   * Never changes a tangent, trial, constraint or accepted history. */
  template <int dim>
  void trace_bulk_tip_rows(BulkTipCoupling<dim> &op,
                          const typename BulkTipCoupling<dim>::Vector &x,
                          const typename BulkTipCoupling<dim>::Evaluation &current,
                          const typename BulkTipCoupling<dim>::Vector &direction,
                          const typename BulkTipCoupling<dim>::Vector &trial,
                          const typename BulkTipCoupling<dim>::Evaluation &candidate,
                          const unsigned int component,const unsigned int iteration,
                          const double shift,const double alpha)
  {
    const auto comm=op.get_mpi_communicator();
    const auto pb=op.phase_block();
    const auto &mass=op.phase_mass();
    const auto &lower=op.obstacle(),&upper=op.upper_bound();
    auto displacement=trial; displacement-=x;
    auto action=displacement; op.vmult(action,displacement);
    auto predicted=current.residual; predicted+=action;
    const auto functional=op.trace_functional_difference(displacement,current,candidate);
    op.get_pcout()<<"   Tip independent functional: iteration="<<iteration<<", shift="<<shift<<", alpha="<<alpha
                  <<", difference="<<functional[0]<<", resistance quadrature difference="<<functional[1]
                  <<", arithmetic scale="<<functional[2]<<", slope increment="<<op.directional_work(current,displacement)<<std::endl;
    std::vector<types::global_dof_index> selected;
    const auto row_residual=[&](const auto i,const bool incoming)
    {
      return PhaseField::box_constraint(incoming?x.block(pb)[i]:trial.block(pb)[i],
                                        incoming?current.residual.block(pb)[i]:candidate.residual.block(pb)[i],
                                        mass[i],lower[i],upper[i]).projected_residual;
    };
    for (unsigned int kind=0;kind<4;++kind)for (unsigned int leading=0;leading<2;++leading)
      {
        double maximum=-1.; types::global_dof_index index=numbers::invalid_dof_index;
        for (const auto i:mass.locally_owned_elements())
          {
            if (std::find(selected.begin(),selected.end(),i)!=selected.end())continue;
            const double value=kind<2?std::abs(row_residual(i,kind==0)):
              kind==2?std::abs((candidate.residual.block(pb)[i]-predicted.block(pb)[i])/mass[i]):
              std::abs(PhaseField::box_constraint(trial.block(pb)[i],predicted.block(pb)[i],mass[i],lower[i],upper[i]).projected_residual);
            if (value>maximum) {maximum=value;index=i;}
          }
        const double global=Utilities::MPI::max(maximum,comm);
        if (maximum!=global)index=numbers::invalid_dof_index;
        selected.push_back(Utilities::MPI::min(index,comm));
        if (leading==0)op.get_pcout()<<"   Tip row leader: iteration="<<iteration<<", shift="<<shift<<", alpha="<<alpha<<", kind="<<kind<<", row="<<selected.back()<<", value="<<global<<std::endl;
      }
    const auto rank=Utilities::MPI::this_mpi_process(comm);
    std::ofstream rows(op.get_output_directory()+"tip-search-rows-r"+Utilities::to_string(rank)+".csv",std::ios::app);
    std::ofstream points(op.get_output_directory()+"tip-search-points-r"+Utilities::to_string(rank)+".csv",std::ios::app);
    if (rows.tellp()==0)rows<<"iteration,shift,alpha,row,cell,i,x,y,lower,upper,current,trial,gap,mass,current_gradient_over_mass,trial_gradient_over_mass,physical_trial_gradient_over_mass,current_projected,trial_projected,predicted_projected,current_status,trial_status,direction,tangent_over_mass,nonlinear_defect_over_mass\n";
    if (points.tellp()==0)points<<"iteration,shift,alpha,row,cell,q,section,formed,weight,shape,parent_phi,current_phi,trial_phi,current_rate,trial_rate,current_force,trial_force,current_df_dfree,trial_df_dfree,current_driving,trial_driving,weighted_trial_driving\n";
    rows<<std::setprecision(17); points<<std::setprecision(17);
    const auto &fe=op.get_fe();
    const auto parent=op.accepted_parent();
    FEValues<dim> values(op.get_mapping(),fe,QGauss<dim>(parent->quadrature),update_values|update_JxW_values);
    std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
    const auto offset=x.get_block_indices().block_start(pb);
    unsigned int nphase=1; for (unsigned int d=0;d<dim;++d)nphase*=3;
    for (const auto &cell:op.get_dof_handler().active_cell_iterators())if (cell->is_locally_owned())
      {
        cell->get_dof_indices(ids);
        for (unsigned int j=0;j<nphase;++j)
          {
            const auto local=fe.component_to_system_index(component,j),global=ids[local];
            const auto i=global-offset;
            if (std::find(selected.begin(),selected.end(),i)==selected.end())continue;
            dealii::Point<dim> reference;
            unsigned int digits=j;
            for (unsigned int d=0;d<dim;++d,digits/=3)reference[d]=.5*(digits%3);
            const auto position=op.get_mapping().transform_unit_to_real_cell(cell,reference);
            const auto incoming=PhaseField::box_constraint(x.block(pb)[i],current.residual.block(pb)[i],mass[i],lower[i],upper[i]);
            const auto actual=PhaseField::box_constraint(trial.block(pb)[i],candidate.residual.block(pb)[i],mass[i],lower[i],upper[i]);
            const auto prediction=PhaseField::box_constraint(trial.block(pb)[i],predicted.block(pb)[i],mass[i],lower[i],upper[i]);
            rows<<iteration<<','<<shift<<','<<alpha<<','<<i<<','<<cell->id().to_string()<<','<<j<<','<<position[0]<<','<<position[1]<<','<<lower[i]<<','<<upper[i]<<','<<x.block(pb)[i]<<','<<trial.block(pb)[i]<<','<<x.block(pb)[i]-lower[i]<<','<<mass[i]<<','<<current.residual.block(pb)[i]/mass[i]<<','<<candidate.residual.block(pb)[i]/mass[i]<<','<<op.fracture_scale()*candidate.residual.block(pb)[i]/mass[i]<<','<<incoming.projected_residual<<','<<actual.projected_residual<<','<<prediction.projected_residual<<','<<incoming.status<<','<<actual.status<<','<<direction.block(pb)[i]<<','<<action.block(pb)[i]/mass[i]<<','<<(candidate.residual.block(pb)[i]-predicted.block(pb)[i])/mass[i]<<'\n';
            values.reinit(cell);
            for (unsigned int q=0;q<values.n_quadrature_points;++q)
              {
                const auto k=parent->section_ids.at(cell->id())[q];
                const auto &a=current.points.at(cell->id())[q],&b=candidate.points.at(cell->id())[q];
                const auto &sa=current.section_solutions[k],&sb=candidate.section_solutions[k];
                const double N=values.shape_value_component(local,q,component);
                double parent_phi=0.;
                for (unsigned int l=0;l<nphase;++l)
                  { const auto li=fe.component_to_system_index(component,l); parent_phi+=values.shape_value_component(li,q,component)*parent->tip_phase[ids[li]-offset]; }
                points<<iteration<<','<<shift<<','<<alpha<<','<<i<<','<<cell->id().to_string()<<','<<q<<','<<k<<','<<parent->sections[k].formed<<','<<values.JxW(q)<<','<<N<<','<<parent_phi<<','<<a.state.phi<<','<<b.state.phi<<','<<sa.slip_rate<<','<<sb.slip_rate<<','<<sa.resistance_response.resistance<<','<<sb.resistance_response.resistance<<','<<sa.resistance_derivative_wrt_free_rate<<','<<sb.resistance_derivative_wrt_free_rate<<','<<a.driving<<','<<b.driving<<','<<values.JxW(q)*N*b.driving/op.fracture_scale()/mass[i]<<'\n';
              }
          }
      }
  }
}}
#endif
