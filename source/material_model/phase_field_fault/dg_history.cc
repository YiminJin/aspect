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
#include <aspect/reconstructed_fault/linear_performance.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <mutex>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      long double q2_roundoff_product(std::initializer_list<long double> factors)
      {
        long double result=0.;
        for (const auto error:factors)result+=error+result*error;
        return result;
      }
    }

    template<int dim> struct PhaseFieldFault<dim>::FEStrainState::MechanicalSamples
    {
      explicit MechanicalSamples(const Quadrature<dim> &q):points(q.get_points()),weights(q.get_weights()) {}
      std::vector<Point<dim>> points;
      std::vector<double> weights;
      std::map<CellId,std::vector<Point<dim>>> units;
      std::map<CellId,std::vector<double>> phase_values;
      std::map<CellId,std::vector<SymmetricTensor<2,dim>>> stress;
      std::weak_ptr<const DGHistory> history;
      unsigned int geometry=numbers::invalid_unsigned_int;
    };

    template<int dim>
    const std::vector<SymmetricTensor<2,dim>> &PhaseFieldFault<dim>::retained_dg_qp_stress(
      const CellId &id,const Quadrature<dim> &quadrature,const std::vector<Point<dim>> &positions) const
    {
      AssertThrow(uses_stationary_mature_dg_optimizations(),ExcMessage("Stationary DG QP cache used outside its scope."));
      // WorkStream may call from multiple workers. Publication is outside the
      // assembly, so returned cell arrays remain immutable during each assembly.
      static std::mutex mutex;std::lock_guard<std::mutex> guard(mutex);
      const auto &s=*fe_strain_state;
      auto &manager=this->get_reconstructed_fault_manager();
      const unsigned int generation=manager.get_stokes_qp_cache_diagnostics().rebuild_count;
      auto &pointer=s.mechanical_samples;
      if(!pointer || pointer->geometry!=generation)
        {
          pointer=std::make_shared<typename FEStrainState::MechanicalSamples>(quadrature);
          pointer->geometry=generation;
        }
      auto &cache=*pointer;
      AssertThrow(cache.points==quadrature.get_points() && cache.weights==quadrature.get_weights(),
                  ExcMessage("Stationary DG mechanical cache requires its original quadrature."));
      if(cache.history.lock()!=s.dg_history){cache.stress.clear();cache.history=s.dg_history;}
      const auto found=cache.stress.find(id);if(found!=cache.stress.end())return found->second;
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::history_cache_misses);
      const auto cell=s.evaluation_cells.at(id);
      // Associations are already prepared by the Stokes owner. No segment
      // search or coordinate-keyed stress map is needed at these known QPs.
      const auto &association=manager.get_stokes_qp_fault_associations(id,quadrature,positions);
      auto phase=cache.phase_values.find(id);
      if(phase==cache.phase_values.end())
        {
          auto &units=cache.units[id];
          for(const auto &x:positions)units.push_back(this->get_mapping().transform_real_to_unit_cell(cell,x));
          FEValues<dim> phase_fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(units),update_values);
          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
          phase_fe.reinit(cell);std::vector<double> values(quadrature.size());
          phase_fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
            .get_function_values(s.displacement_history,values);
          phase=cache.phase_values.emplace(id,std::move(values)).first;
        }
      FEValues<dim> dg(this->get_mapping(),s.dg_history->space->element,Quadrature<dim>(cache.units.at(id)),update_values);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      const auto dg_cell=s.dg_history->space->cells.at(id);dg.reinit(dg_cell);
      std::vector<types::global_dof_index> indices(s.dg_history->space->element.n_dofs_per_cell());dg_cell->get_dof_indices(indices);
      std::vector<SymmetricTensor<2,dim>> stress(quadrature.size());
      for(unsigned int c=0;c<s.dg_history->trial_stress.size();++c)
        {
          const auto ij=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
          for(unsigned int q=0;q<stress.size();++q)
            for(unsigned int i=0;i<indices.size();++i)
              stress[q][ij]+=s.dg_history->trial_stress[c][indices[i]]*dg.shape_value(i,q);
        }
      for(unsigned int q=0;q<stress.size();++q)if(association[q].active)
        {
          const auto &a=association[q];const auto j=a.segment_index;const double xi=a.xi;
          const double z=(1-xi)*s.Z.at(j)+xi*s.Z.at(j+1),I=(1-xi)*s.I.at(j)+xi*s.I.at(j+1);
          const auto localization=evaluate_reconstructed_fault_localization(a.fault_index,j,xi,phase->second[q],phase->second[q],"mature DG QP stress");
          const auto S=manager.get_shear_sense(a.fault_index)*symmetrize(outer_product(a.tangent,a.normal));
          stress[q]-=(2*z*localization.current_h/I)*S;
        }
      return cache.stress.emplace(id,std::move(stress)).first->second;
    }

    template <int dim>
    void PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &state) const
    {
      AssertThrow(dg_trial_stress_history && dim==2 && this->get_parameters().stokes_velocity_degree==2
                  && state.Z==std::vector<double>(state.Z.size(),0.),
                  ExcMessage("Independent DG history initializes only an unloaded seed."));
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
                       <<(dg_accepted_history ? "accepted-history lifecycle/restart." : "no transport/restart.")<<std::endl;
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
      const MaxwellCoefficients &coefficients,const bool estimate_source_roundoff,const bool reference) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(),"DG native source/update");
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
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      FEValues<dim> source(this->get_mapping(),this->get_fe(),quadrature,update_gradients);
      FEValues<dim> dg(this->get_mapping(),space.element,quadrature,update_values|update_JxW_values);
      const unsigned int n=space.element.n_dofs_per_cell();
      std::vector<types::global_dof_index> indices(n);
      std::vector<types::global_dof_index> native_indices(this->get_fe().n_dofs_per_cell());
      std::vector<SymmetricTensor<2,dim>> gradient(quadrature.size());
      for (const auto &entry : state.cells)
        {
          source.reinit(entry.second);
          const auto dg_cell=space.cells.at(entry.first);
          dg.reinit(dg_cell);
          dg_cell->get_dof_indices(indices);
          entry.second->get_dof_indices(native_indices);
          source[this->introspection().extractors.velocities].get_function_symmetric_gradients(bulk,gradient);
          typename DGHistory::Space::SourceGeometry *geometry=nullptr;
          if(uses_stationary_mature_dg_optimizations() && !reference)
            {
              std::array<long double,9+dim> key;
              for(unsigned int q=0;q<9;++q)key[q]=dg.JxW(q);
              for(unsigned int d=0;d<dim;++d)key[9+d]=static_cast<long double>(entry.second->vertex(1u<<d)[d])-entry.second->vertex(0)[d];
              geometry=&space.source_geometry[key];
            }
          const bool cached_mass=geometry && geometry->inverse.m()==n;
          FullMatrix<double> mass(n,n);
          if(cached_mass)mass=geometry->inverse;
          std::array<Vector<double>,SymmetricTensor<2,dim>::n_independent_components> rhs;
          for (auto &component : rhs)
            component.reinit(n);
          for (unsigned int q=0; q<quadrature.size(); ++q)
            {
              const auto strain=Utilities::Tensors::consistent_deviator(gradient[q]);
              for (unsigned int i=0; i<n; ++i)
                {
                  const double weight=dg.JxW(q)*dg.shape_value(i,q);
                  if(!cached_mass)
                    for (unsigned int j=0; j<n; ++j)
                      mass(i,j)+=weight*dg.shape_value(j,q);
                  for (unsigned int c=0; c<rhs.size(); ++c)
                    rhs[c][i]+=weight*strain[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)];
                }
            }
          if(!cached_mass)
            {
              aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::mass_inverse);
              mass.gauss_jordan();if(geometry)geometry->inverse=mass;
            }
          const auto gamma=[](unsigned int count)
          {
            const long double u=std::numeric_limits<double>::epsilon()/2.;
            return count*u/(1-count*u);
          };
          SymmetricTensor<2,dim> strain_scale;
          std::vector<long double> projection_factor;
          if (estimate_source_roundoff)
            {
              // Absolute operands of Q2 differentiation, rather than the small
              // resulting gradient. |L_i|<=1, |L_i'|<=4 on [0,1]. In particular,
              // a nearly constant velocity in a thin cell is ill conditioned.
              auto &roundoff=trial->source_roundoff[entry.first];
              std::array<long double,dim> velocity_sum {};
              for (unsigned int i=0; i<native_indices.size(); ++i)
                {
                  const auto c=this->get_fe().system_to_component_index(i).first;
                  for (unsigned int a=0; a<dim; ++a)
                    if (c==this->introspection().component_indices.velocities[a])
                      velocity_sum[a]+=std::abs(bulk[native_indices[i]]);
                }
              long double volume=1.;
              for (unsigned int b=0; b<dim; ++b)
                {
                  const long double width=static_cast<long double>(entry.second->vertex(1u<<b)[b])-entry.second->vertex(0)[b];
                  volume*=width;
                  for (unsigned int a=0; a<dim; ++a)
                    roundoff.gradient_scale[a][b]=std::nextafter(static_cast<double>(4*velocity_sum[a]/width
                                                                                     *(1+2*n*std::numeric_limits<long double>::epsilon())),std::numeric_limits<double>::infinity());
                }
              for (unsigned int a=0; a<dim; ++a)
                for (unsigned int b=0; b<dim; ++b)
                  if (a!=b)strain_scale[a][b]=.5*(roundoff.gradient_scale[a][b]+roundoff.gradient_scale[b][a]);
                  else
                    for (unsigned int d=0; d<dim; ++d)
                      strain_scale[a][a]+=(a==d ? 2./3. : 1./3.)*roundoff.gradient_scale[d][d];

              if(geometry && !geometry->projection_factor.empty())
                projection_factor=geometry->projection_factor;
              else
                {
                  aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::roundoff_geometry);
                  // A posteriori inverse defect against the analytic tensor-product
                  // Q2 mass (not against its rounded assembly). This accounts for
                  // assembly/inversion without assuming a backward-stable inverse.
                  const auto gamma_long=[](unsigned int count)
                  {
                    const long double u=std::numeric_limits<long double>::epsilon()/2.;
                    return count*u/(1-count*u);
                  };
                  const int mass_1d[3][3]= {{4,2,-1},{2,16,2},{-1,2,4}};
                  FullMatrix<long double> exact_mass(n,n),defect(n,n);
                  const auto &support=space.element.get_unit_support_points();
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int j=0; j<n; ++j)
                      {
                        long double value=volume;
                        for (unsigned int d=0; d<dim; ++d)
                          value*=mass_1d[std::lround(2*support[i][d])][std::lround(2*support[j][d])]/30.L;
                        exact_mass(i,j)=value;
                      }
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int j=0; j<n; ++j)
                      {
                        long double product=0.,absolute_product=0.;
                        for (unsigned int k=0; k<n; ++k)
                          {
                            product+=mass(i,k)*exact_mass(k,j);
                            absolute_product+=std::abs(mass(i,k)*exact_mass(k,j));
                          }
                        defect(i,j)=std::abs(product-(i==j ? 1.L : 0.L))
                                    +gamma_long(2*n+2*dim+4)*absolute_product;
                      }
                  std::vector<long double> rhs_weight(n,0.);
                  projection_factor.assign(n,0.);
                  // Monomial absolute coefficient sums are <=8 for L and <=12
                  // for L'; |L|<=1, |L'|<=4, |L''|<=8. Six operations bound the
                  // quadratic Horner/product evaluations (also the simultaneous
                  // derivative recurrence). Include rounded Gauss abscissae here.
                  const long double value_error=8*gamma(6)+4*gamma(4);
                  const long double derivative_relative_error=3*gamma(6)+2*gamma(4);
                  const long double gradient_relative_error=q2_roundoff_product({gamma(2*n+10),value_error,derivative_relative_error});
                  const long double shape_error=q2_roundoff_product({value_error,value_error});
                  const long double accumulation=gamma(2*quadrature.size()+10);
                  const long double rhs_relative_error=q2_roundoff_product({gradient_relative_error,accumulation});
                  long double weight_total=0.;
                  for (unsigned int q=0; q<quadrature.size(); ++q)weight_total+=std::abs(dg.JxW(q));
                  for (unsigned int j=0; j<n; ++j)
                    for (unsigned int q=0; q<quadrature.size(); ++q)
                      rhs_weight[j]+=std::abs(dg.JxW(q)*dg.shape_value(j,q));
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int j=0; j<n; ++j)
                      projection_factor[i]+=defect(i,j)+std::abs(mass(i,j))
                                            *(rhs_relative_error*rhs_weight[j]+weight_total*shape_error*(1+rhs_relative_error));
                  if(geometry)geometry->projection_factor=projection_factor;
                }
            }
          Vector<double> projected(n);
          for (unsigned int c=0; c<rhs.size(); ++c)
            {
              mass.vmult(projected,rhs[c]);
              const auto ij=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
              for (unsigned int i=0; i<n; ++i)
                {
                  if (estimate_source_roundoff)
                    {
                      long double multiply_scale=0.;
                      for (unsigned int j=0; j<n; ++j)
                        multiply_scale+=std::abs(mass(i,j)*rhs[c][j]);
                      // Inverse defect, RHS operand errors, and inverse/RHS dot.
                      const long double projection_error=projection_factor[i]*strain_scale[ij]+gamma(2*n)*multiply_scale;
                      trial->source_roundoff.at(entry.first).projection[ij]=std::max(trial->source_roundoff.at(entry.first).projection[ij],std::nextafter(static_cast<double>(projection_error
                                                                                     *(1+4*n*std::numeric_limits<long double>::epsilon())),std::numeric_limits<double>::infinity()));
                    }
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
    double PhaseFieldFault<dim>::dg_source_publication_allowance(
      const DGHistory &trial,const CellId &id,const Point<dim> &point) const
    {
      const auto &source=trial.source_roundoff.at(id);
      const auto cell=trial.space->cells.at(id);
      const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,point);
      const auto &element=trial.space->element;
      long double lebesgue=0.;
      for (unsigned int i=0; i<element.n_dofs_per_cell(); ++i)lebesgue+=std::abs(element.shape_value(i,unit));
      const long double u=std::numeric_limits<double>::epsilon()/2.;
      const auto gamma=[&](unsigned int count)
      {
        return count*u/(1-count*u);
      };
      const long double gradient_relative_error=q2_roundoff_product({gamma(2*element.n_dofs_per_cell()+10),8*gamma(6),3*gamma(6)});
      Tensor<2,dim> native_error;
      for (unsigned int a=0; a<dim; ++a)
        for (unsigned int b=0; b<dim; ++b)
          {
            long double coordinate_error=0.;
            for (unsigned int d=0; d<dim; ++d)
              {
                const long double width=static_cast<long double>(cell->vertex(1u<<d)[d])-cell->vertex(0)[d];
                // Subtraction/division in both real-to-unit maps;
                // |L_i''|<=8, |L_i' L_j'|<=16.
                coordinate_error+=(d==b ? 2 : 4)*gamma(4)
                                  *(std::abs(point[d])+std::abs(cell->vertex(0)[d]))/width;
              }
            native_error[a][b]=(gradient_relative_error+2*coordinate_error)*source.gradient_scale[a][b];
          }
      SymmetricTensor<2,dim> strain_error;
      for (unsigned int a=0; a<dim; ++a)
        for (unsigned int b=0; b<dim; ++b)
          if (a!=b)strain_error[a][b]=.5*(native_error[a][b]+native_error[b][a]);
          else
            for (unsigned int d=0; d<dim; ++d)strain_error[a][a]+=(a==d ? 2./3. : 1./3.)*native_error[d][d];
      const auto source_error=2*compute_maxwell_coefficients(fe_strain_state->eta,fe_strain_state->G,this->get_timestep()).eta_ve*(strain_error+static_cast<double>(lebesgue)*source.projection);
      return std::sqrt(std::pow(source_error.norm(),2)+std::pow(trace(source_error),2));
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


    template<int dim> void PhaseFieldFault<dim>::prepare_dg_point_samples(
      const CellId &id,const std::vector<Point<dim>> &points) const
    {
      if(!uses_stationary_mature_dg_optimizations())return;
      const auto &s=*fe_strain_state;auto &space=*s.dg_history->space;
      const auto &manager=this->get_reconstructed_fault_manager();
      const auto generation=manager.get_stokes_qp_cache_diagnostics().rebuild_count;
      if(space.point_geometry_generation!=generation)
        {space.point_samples.clear();space.point_geometry_generation=generation;}
      auto &data=space.point_samples[id];
      // Keep the few fixed, cell-owned layouts (publication particles and dense
      // diagnostic QPs) together. Adding a diagnostic layout must not evict the
      // publication layout or attach accepted stress values to geometry.
      std::vector<Point<dim>> missing;
      for(const auto &x:points)
        if(std::none_of(data.begin(),data.end(),[&](const auto &p){return p.point==x;}))missing.push_back(x);
      if(missing.empty())return;
      const auto cell=s.evaluation_cells.at(id);std::vector<Point<dim>> units;
      for(const auto &x:missing)units.push_back(this->get_mapping().transform_real_to_unit_cell(cell,x));
      const Quadrature<dim> quadrature(units);
      FEValues<dim> dg(this->get_mapping(),space.element,quadrature,update_values);
      FEValues<dim> native(this->get_mapping(),this->get_fe(),quadrature,update_values);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
      aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
      dg.reinit(space.cells.at(id));native.reinit(cell);
      std::vector<double> phi(missing.size());
      native[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
        .get_function_values(s.displacement_history,phi);
      const auto old_size=data.size();data.resize(old_size+missing.size());
      for(unsigned int q=0;q<missing.size();++q)
        {
          auto &sample=data[old_size+q];sample.point=missing[q];sample.phase=phi[q];
          for(unsigned int i=0;i<9;++i)sample.shape[i]=dg.shape_value(i,q);
          sample.source=manager.project_to_bulk_source(missing[q]);
        }
    }

    template <int dim>
    SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(
      const DGHistory &history,const std::vector<double> &Z,const std::vector<double> &I,
      const CellId &id,const Point<dim> &point,double *evaluation_scale,
      const LinearAlgebra::BlockVector *phase_trial) const
    {
      const auto &state=*fe_strain_state;
      const auto cell=history.space->cells.at(id);
      const typename DGHistory::Space::PointSample *sample=nullptr;
      if(uses_stationary_mature_dg_optimizations() && phase_trial==nullptr)
        {
          const auto found=history.space->point_samples.find(id);
          if(found!=history.space->point_samples.end())
            for(const auto &p:found->second)if(p.point==point){sample=&p;break;}
        }
      SymmetricTensor<2,dim> trial;double scale=0.;std::vector<double> phi(1);
      std::vector<types::global_dof_index> indices(history.space->element.n_dofs_per_cell());cell->get_dof_indices(indices);
      if(sample)
        {
          for(unsigned int c=0;c<history.trial_stress.size();++c)
            {
              const auto ij=SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
              for(unsigned int i=0;i<indices.size();++i)
                {const double value=history.trial_stress[c][indices[i]]*sample->shape[i];trial[ij]+=value;scale+=std::abs(value);}
            }
          phi[0]=sample->phase;
        }
      else
        {
      const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,point);
      FEValues<dim> dg(this->get_mapping(),history.space->element,Quadrature<dim>({unit}),update_values);
      dg.reinit(cell);
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
      phase[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
      .get_function_values(phase_trial ? *phase_trial : state.displacement_history,phi);

          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::dg_fevalues);
          aspect::internal::FaultSamplingCounters::add(aspect::internal::FaultSamplingCounters::native_fevalues);
        }
      if (mature_frictional_fault)
        {
          // General oblique map, including the retained constant endpoint
          // continuation. Z is auxiliary Maxwell/slip memory, not I*C.
          const auto &manager=this->get_reconstructed_fault_manager();
          const auto source=sample ? sample->source : manager.project_to_bulk_source(point);
          SymmetricTensor<2,dim> correction;
          if (source.active)
            {
              const auto &fault=manager.get_fault(source.fault_index);
              const auto j=source.segment_index;
              const double t=source.xi;
              const double z=(1-t)*Z.at(j)+t*Z.at(j+1);
              const double normalization=(1-t)*I.at(j)+t*I.at(j+1);
              AssertThrow(std::isfinite(z) && std::isfinite(normalization) && normalization>0.,
                          ExcMessage("Mature DG stress requires finite memory and positive native normalization."));
              auto tangent=fault.vertex(j+1)-fault.vertex(j);
              tangent/=tangent.norm();
              Tensor<1,dim> normal;
              normal[0]=-tangent[1];
              normal[1]=tangent[0];
              const auto S=manager.get_shear_sense(source.fault_index)*symmetrize(outer_product(tangent,normal));
              const auto localization=evaluate_reconstructed_fault_localization(
                                        source.fault_index,j,t,phi[0],phi[0],"mature DG stress");
              correction=(2*z*localization.current_h/normalization)*S;
            }
          if (evaluation_scale)
            *evaluation_scale=scale+correction.norm();
          return trial-correction;
        }
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
  template const std::vector<SymmetricTensor<2,dim>> &PhaseFieldFault<dim>::retained_dg_qp_stress(const CellId &,const Quadrature<dim> &,const std::vector<Point<dim>> &) const; \
  template void PhaseFieldFault<dim>::initialize_dg_history(FEStrainState &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const LinearAlgebra::BlockVector &,const MaxwellCoefficients &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::DGHistory> PhaseFieldFault<dim>::make_dg_history_trial(const DGHistory &,const LinearAlgebra::BlockVector &,const MaxwellCoefficients &,bool,bool) const; \
  template std::vector<SymmetricTensor<2,dim>> PhaseFieldFault<dim>::evaluate_dg_trial_tensor(const DGHistory &,const CellId &,const Quadrature<dim> &) const; \
  template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_dg_history_stress(const DGHistory &,const std::vector<double> &,const std::vector<double> &,const CellId &,const Point<dim> &,double *,const LinearAlgebra::BlockVector *) const; \
  template void PhaseFieldFault<dim>::prepare_dg_point_samples(const CellId &,const std::vector<Point<dim>> &) const; \
  template double PhaseFieldFault<dim>::dg_source_publication_allowance(const DGHistory &,const CellId &,const Point<dim> &) const; \
  template std::vector<double> PhaseFieldFault<dim>::dg_history_values() const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
