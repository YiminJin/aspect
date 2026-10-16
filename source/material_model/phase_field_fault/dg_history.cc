/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT, distributed under the GNU General Public License
  version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/utilities.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &state) const
    {
      AssertThrow(dg_trial_stress_history && dim==2 && this->get_parameters().stokes_velocity_degree==2
                  && state.Z==std::vector<double>(state.Z.size(),0.),
                  ExcMessage("PF2 DG history initializes only the unloaded cohesive seed."));
      auto history=std::make_shared<DGHistory>();
      history->space=std::make_shared<typename DGHistory::Space>(this->get_triangulation());
      for (auto &component : history->trial_stress)
        {
          component.reinit(history->space->dofs.locally_owned_dofs(),history->space->relevant,
                           this->get_mpi_communicator());
          component=0.;
        }
      state.dg_history=std::move(history);
      this->get_pcout()<<"   Accepted DGQ2 trial stress: independent native source; fixed phase/geometry; "
                       <<(dg_accepted_history ? "PF5 prescribed transport/restart." : "no transport/restart.")<<std::endl;
    }


    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::make_dg_history_trial(
      const LinearAlgebra::BlockVector &bulk,const MaxwellCoefficients &coefficients) const
    {
      AssertThrow(fe_strain_state && fe_strain_state->dg_history,
                  ExcMessage("Missing authoritative DG trial-stress history."));
      return make_dg_history_trial(*fe_strain_state->dg_history,bulk,coefficients);
    }


    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGHistory>
    PhaseFieldFault<dim>::make_dg_history_trial(
      const DGHistory &incoming,const LinearAlgebra::BlockVector &bulk,
      const MaxwellCoefficients &coefficients) const
    {
      const auto &state=*fe_strain_state;
      AssertThrow(dg_trial_stress_history && incoming.space==state.dg_history->space,
                  ExcMessage("Native DG source requires the accepted fixed space."));
      auto trial=std::make_shared<DGHistory>();
      trial->space=incoming.space;
      const auto &space=*trial->space;
      std::array<LinearAlgebra::Vector,SymmetricTensor<2,dim>::n_independent_components> owned;
      for (auto &component : owned)
        component.reinit(space.dofs.locally_owned_dofs(),this->get_mpi_communicator());

      // On the restricted affine Cartesian Q2 mesh, every component of the
      // native deviatoric symmetric gradient is in DGQ2. Three Gauss points
      // integrate both its scalar mass and source products exactly. This
      // projection stores the trial tensor, never the narrow chi*Z correction.
      const QGauss<dim> quadrature(3);
      FEValues<dim> source(this->get_mapping(),this->get_fe(),quadrature,update_gradients);
      FEValues<dim> dg(this->get_mapping(),space.element,quadrature,update_values|update_JxW_values);
      const unsigned int n=space.element.n_dofs_per_cell();
      std::vector<types::global_dof_index> indices(n);
      std::vector<SymmetricTensor<2,dim>> gradient(quadrature.size());
      for (const auto &entry : state.cells)
        {
          source.reinit(entry.second);
          const auto dg_cell=space.cells.at(entry.first);
          dg.reinit(dg_cell);
          dg_cell->get_dof_indices(indices);
          source[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk,gradient);
          FullMatrix<double> mass(n,n);
          std::array<Vector<double>,SymmetricTensor<2,dim>::n_independent_components> rhs;
          for (auto &component : rhs)
            component.reinit(n);
          for (unsigned int q=0; q<quadrature.size(); ++q)
            {
              const auto strain=Utilities::Tensors::consistent_deviator(gradient[q]);
              for (unsigned int i=0; i<n; ++i)
                {
                  const double weight=dg.JxW(q)*dg.shape_value(i,q);
                  for (unsigned int j=0; j<n; ++j)
                    mass(i,j)+=weight*dg.shape_value(j,q);
                  for (unsigned int c=0; c<rhs.size(); ++c)
                    rhs[c][i]+=weight*strain[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)];
                }
            }
          mass.gauss_jordan();
          Vector<double> projected(n);
          for (unsigned int c=0; c<rhs.size(); ++c)
            {
              mass.vmult(projected,rhs[c]);
              for (unsigned int i=0; i<n; ++i)
                {
                  const double value=coefficients.beta*incoming.trial_stress[c][indices[i]]
                                     +2*coefficients.eta_ve*projected[i];
                  AssertThrow(std::isfinite(value),ExcMessage("Nonfinite DG trial-stress candidate."));
                  owned[c][indices[i]]=value;
                }
            }
        }
      for (unsigned int c=0; c<owned.size(); ++c)
        {
          owned[c].compress(VectorOperation::insert);
          trial->trial_stress[c].reinit(space.dofs.locally_owned_dofs(),space.relevant,
                                        this->get_mpi_communicator());
          trial->trial_stress[c]=owned[c];
        }
      return trial;
    }


    template <int dim>
    std::vector<SymmetricTensor<2,dim>> PhaseFieldFault<dim>::evaluate_dg_trial_tensor(
      const DGHistory &history,const CellId &id,const Quadrature<dim> &quadrature) const
    {
      FEValues<dim> fe(this->get_mapping(),history.space->element,quadrature,update_values);
      fe.reinit(history.space->cells.at(id));
      std::vector<SymmetricTensor<2,dim>> tensor(quadrature.size());
      std::vector<double> component(quadrature.size());
      for (unsigned int c=0; c<history.trial_stress.size(); ++c)
        {
          fe.get_function_values(history.trial_stress[c],component);
          const auto ij=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
          for (unsigned int q=0; q<quadrature.size(); ++q)
            tensor[q][ij]=component[q];
        }
      return tensor;
    }


    template <int dim>
    SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(
      const DGHistory &history,const std::vector<double> &Z,const std::vector<double> &I,
      const CellId &id,const Point<dim> &point,double *evaluation_scale,
      const LinearAlgebra::BlockVector *phase_trial) const
    {
      const auto &state=*fe_strain_state;
      const auto cell=history.space->cells.at(id);
      const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,point);
      FEValues<dim> dg(this->get_mapping(),history.space->element,Quadrature<dim>({unit}),update_values);
      dg.reinit(cell);
      SymmetricTensor<2,dim> trial;
      double scale=0.;
      std::vector<types::global_dof_index> indices(history.space->element.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      for (unsigned int c=0; c<history.trial_stress.size(); ++c)
        {
          const auto ij=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
          for (unsigned int i=0; i<indices.size(); ++i)
            {
              const double value=history.trial_stress[c][indices[i]]*dg.shape_value(i,0);
              trial[ij]+=value;
              scale+=std::abs(value);
            }
        }

      // Only the immutable phase block of this shared vector is sampled.
      // Neither its unused velocity block nor any mapped/particle stress is read.
      const auto native_cell=state.evaluation_cells.at(id);
      FEValues<dim> phase(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_values);
      phase.reinit(native_cell);
      std::vector<double> phi(1);
      phase[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
      .get_function_values(phase_trial ? *phase_trial : state.displacement_history,phi);
      const unsigned int segment=std::min<unsigned int>(Z.size()-2,point[0]*(Z.size()-1));
      const double xi=point[0]*(Z.size()-1)-segment;
      const double z=(1-xi)*Z[segment]+xi*Z[segment+1];
      const double normalization=(1-xi)*I[segment]+xi*I[segment+1];
      AssertThrow(normalization>0. && std::isfinite(normalization) && phi[0]>=0. && phi[0]<1.,
                  ExcMessage("Undefined fixed DG localization."));
      const double h=state.m*phi[0]*(1+phi[0])/std::pow(1-phi[0],2);
      SymmetricTensor<2,dim> S;
      S[0][1]=.5*state.shear_sense;
      const double correction=2*z*h/normalization;
      if (evaluation_scale)
        *evaluation_scale=scale+std::abs(correction)*S.norm();
      return trial-correction*S;
    }


    template <int dim>
    std::vector<double> PhaseFieldFault<dim>::dg_history_values() const
    {
      AssertThrow(dg_trial_stress_history && fe_strain_state && fe_strain_state->dg_history,
                  ExcMessage("Missing DG history for the diagnostic snapshot."));
      const auto &history=*fe_strain_state->dg_history;
      std::vector<double> values;
      for (const auto &component : history.trial_stress)
        for (const auto i : component.locally_owned_elements())
          values.push_back(component[i]);
      return values;
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const LinearAlgebra::BlockVector &,const MaxwellCoefficients &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const DGHistory &,const LinearAlgebra::BlockVector &,const MaxwellCoefficients &) const; \
  template std::vector<SymmetricTensor<2,dim>> PhaseFieldFault<dim>::evaluate_dg_trial_tensor(const DGHistory &,const CellId &,const Quadrature<dim> &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(const DGHistory &,const std::vector<double> &,const std::vector<double> &,const CellId &,const Point<dim> &,double *,const LinearAlgebra::BlockVector *) const; \
  template std::vector<double> PhaseFieldFault<dim>::dg_history_values() const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
