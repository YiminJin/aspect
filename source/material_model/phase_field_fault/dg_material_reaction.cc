/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect { namespace MaterialModel
{
  template<int dim> std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
  PhaseFieldFault<dim>::make_material_reaction(const LinearAlgebra::BlockVector &bulk,
                                             const std::vector<double> &rates) const
  {
    AssertThrow(dg_material_transport && material_variable_viscosity && fe_strain_state->objective_anchor,
                ExcMessage("Projected material reaction requires its complete transported anchor."));
    const auto &s=*fe_strain_state;const auto &anchor=*s.objective_anchor;
    const auto &space=*anchor.space;
    const auto comm=this->get_mpi_communicator();
    const double dt=this->get_timestep_number()>0 ? this->get_timestep() : initial_time_step;
    auto update=std::make_shared<FEStrainUpdate>();
    update->displacement_history=s.displacement_history;update->I=s.I;update->Z.resize(s.Z.size());
    AssertDimension(rates.size(),s.Z.size());
    for(unsigned j=0;j<s.Z.size();++j)
      {
        const auto c=compute_maxwell_coefficients(material_viscosity(s.vertices[j],s.eta),s.G,dt);
        update->Z[j]=c.beta*s.Z[j]+c.eta_ve*rates[j];
      }
    update->dg_history=std::make_shared<DGHistory>();update->dg_history->space=anchor.space;
    update->dg_history->residual=std::make_shared<DGHistory>();update->dg_history->residual->space=anchor.space;
    constexpr unsigned components=SymmetricTensor<2,dim>::n_independent_components;
    std::array<LinearAlgebra::Vector,2*components> owned;
    for(auto &v:owned)v.reinit(space.dofs.locally_owned_dofs(),comm);
    const QGauss<dim> rule(std::max(8u,fe_bulk_normal_quadrature_points));
    FEValues<dim> native(this->get_mapping(),this->get_fe(),rule,update_gradients|update_values|update_quadrature_points);
    FEValues<dim> dg(this->get_mapping(),space.element,rule,update_values|update_JxW_values);
    std::vector<SymmetricTensor<2,dim>> strain(rule.size());std::vector<double> phase(rule.size());
    const unsigned n=space.element.n_dofs_per_cell();std::vector<types::global_dof_index> ids(n);
    for(const auto &entry:s.cells)
      {
        native.reinit(entry.second);dg.reinit(space.cells.at(entry.first));space.cells.at(entry.first)->get_dof_indices(ids);
        native[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk,strain);
        native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.displacement_history,phase);
        const auto T=evaluate_dg_trial_tensor(anchor,entry.first,rule),R=evaluate_dg_trial_tensor(*anchor.residual,entry.first,rule);
        FullMatrix<double> mass(n,n);std::array<Vector<double>,2*components> rhs;
        for(auto &v:rhs)v.reinit(n);
        for(unsigned q=0;q<rule.size();++q)
          {
            const auto x=native.quadrature_point(q);
            const auto c=compute_maxwell_coefficients(material_viscosity(x,s.eta),s.G,dt);
            const auto B0=evaluate_mature_dg_memory(x,phase[q],s.Z,s.I);
            const auto B1=evaluate_mature_dg_memory(x,phase[q],update->Z,update->I);
            const auto twice_LV=evaluate_mature_dg_memory(x,phase[q],rates,s.I);
            const std::array<SymmetricTensor<2,dim>,2> values={
              c.beta*T[q]+2*c.eta_ve*Utilities::Tensors::consistent_deviator(strain[q]),
              c.beta*(R[q]+B0)+c.eta_ve*twice_LV-B1};
            for(unsigned i=0;i<n;++i)
              {
                const double w=dg.JxW(q)*dg.shape_value(i,q);
                for(unsigned j=0;j<n;++j)mass(i,j)+=w*dg.shape_value(j,q);
                for(unsigned k=0;k<2;++k)for(unsigned cpt=0;cpt<components;++cpt)
                  rhs[k*components+cpt][i]+=w*values[k][SymmetricTensor<2,dim>::unrolled_to_component_indices(cpt)];
              }
          }
        mass.gauss_jordan();
        for(unsigned c=0;c<rhs.size();++c)
          {Vector<double> coefficients(n);mass.vmult(coefficients,rhs[c]);for(unsigned i=0;i<n;++i)owned[c][ids[i]]=coefficients[i];}
      }
    for(unsigned c=0;c<owned.size();++c)
      {
        owned[c].compress(VectorOperation::insert);
        auto &v=c<components ? update->dg_history->trial_stress[c] : update->dg_history->residual->trial_stress[c-components];
        v.reinit(space.dofs.locally_owned_dofs(),space.relevant,comm);v=owned[c];
      }
    return update;
  }

  template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::material_reaction_correction(
    const CellId &id,const Point<dim> &x) const
  {
    if(!dg_material_reaction)return {};
    const auto &s=*fe_strain_state;const auto &r=*dg_material_reaction;
    const auto cell=s.evaluation_cells.at(id);
    const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,x);
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_gradients|update_values);
    fe.reinit(cell);
    std::vector<SymmetricTensor<2,dim>> strain(1);std::vector<double> phi(1);
    fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(r.bulk,strain);
    fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(s.displacement_history,phi);
    const double dt=this->get_timestep_number()>0 ? this->get_timestep() : initial_time_step;
    const auto c=compute_maxwell_coefficients(material_viscosity(x,s.eta),s.G,dt);
    const auto physical=evaluate_dg_history_stress(*r.candidate->dg_history,r.candidate->Z,s.I,id,x);
    const auto anchor=evaluate_dg_history_stress(*s.objective_anchor,s.Z,s.I,id,x);
    const auto direct=c.beta*anchor+2*c.eta_ve*Utilities::Tensors::consistent_deviator(strain[0])
                      -c.eta_ve*evaluate_mature_dg_memory(x,phi[0],r.rates,s.I);
    return physical-direct;
  }

  template<int dim> double PhaseFieldFault<dim>::refresh_material_reaction(
    const LinearAlgebra::BlockVector &bulk,const std::vector<double> &rates)
  {
    auto next=std::make_shared<DGMaterialReaction>();next->bulk=bulk;next->rates=rates;
    next->candidate=make_material_reaction(bulk,rates);
    auto previous=dg_material_reaction;
    double difference=0.,scale=1.;
    const QGauss<dim> rule(std::max(8u,fe_bulk_normal_quadrature_points));
    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_quadrature_points);
    for(const auto &entry:fe_strain_state->cells)
      {
        fe.reinit(entry.second);
        for(const auto &x:fe.get_quadrature_points())
          {
            dg_material_reaction=previous;const auto before=material_reaction_correction(entry.first,x);
            dg_material_reaction=next;const auto after=material_reaction_correction(entry.first,x);
            difference=std::max(difference,(after-before).norm());
            scale=std::max({scale,before.norm(),after.norm(),evaluate_dg_history_stress(*next->candidate->dg_history,next->candidate->Z,fe_strain_state->I,entry.first,x).norm()});
          }
      }
    dg_material_reaction=next;
    fe_strain_state->stress_samples.clear();fe_strain_state->mechanical_samples.reset();
    difference=Utilities::MPI::max(difference,this->get_mpi_communicator());
    scale=Utilities::MPI::max(scale,this->get_mpi_communicator());
    return difference/scale;
  }

#define INSTANTIATE(dim) \
  template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::make_material_reaction(const LinearAlgebra::BlockVector &,const std::vector<double> &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::material_reaction_correction(const CellId &,const Point<dim> &) const; \
  template double PhaseFieldFault<dim>::refresh_material_reaction(const LinearAlgebra::BlockVector &,const std::vector<double> &);
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}}
