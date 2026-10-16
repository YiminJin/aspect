/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later.
 * Compact whole-polynomial reconstruction adapted from
 * ../aspect-weno/source/simulator/limiters.cc, apply_WENO_limiter_to_dg_solutions.
 * Tensor RMS detectors and the Q2 reference smoothness are PF4 experiments. */
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <chrono>
#include <algorithm>
#include <limits>
#include <numeric>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::DGLimiter>
    PhaseFieldFault<dim>::make_dg_limiter(const DGTransport &op,
                                          const LinearAlgebra::BlockVector &velocity) const
    {
      AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                  ExcMessage("PF4 limiting requires serial fixed Cartesian 2D DGQ2."));
      auto result=std::make_shared<DGLimiter>();
      auto &data=*result;
      data.space=op.space;
      data.mesh_vertices=op.mesh_vertices;
      const auto &element=op.space->element;
      const auto &support=element.get_unit_support_points();
      const unsigned int n=element.n_dofs_per_cell();
      const QGauss<dim> rule(3);
      const QGaussLobatto<dim> scaling_rule(9);
      data.values.reinit(rule.size(),n);
      data.scaling_values.reinit(scaling_rule.size(),n);
      for (unsigned int q=0; q<scaling_rule.size(); ++q)
        for (unsigned int i=0; i<n; ++i)
          data.scaling_values(q,i)=element.shape_value(i,scaling_rule.point(q));
      data.mean.reinit(n);
      for (auto &matrix:data.derivatives) matrix.reinit(rule.size(),n);
      for (unsigned int q=0; q<rule.size(); ++q)
        {
          data.quadrature_weights.push_back(rule.weight(q));
          for (unsigned int i=0; i<n; ++i)
            {
              data.values(q,i)=element.shape_value(i,rule.point(q));
              data.mean[i]+=rule.weight(q)*data.values(q,i);
              const auto gradient=element.shape_grad(i,rule.point(q));
              const auto hessian=element.shape_grad_grad(i,rule.point(q));
              data.derivatives[0](q,i)=gradient[0];
              data.derivatives[1](q,i)=gradient[1];
              data.derivatives[2](q,i)=hessian[0][0];
              data.derivatives[3](q,i)=hessian[0][1];
              data.derivatives[4](q,i)=hessian[1][1];
            }
        }
      Point<dim> lo=data.mesh_vertices.front(),hi=lo;
      for (const auto &p:data.mesh_vertices)
        for (unsigned int d=0; d<dim; ++d)
          {
            lo[d]=std::min(lo[d],p[d]);
            hi[d]=std::max(hi[d],p[d]);
          }
      const double reference_length=lo.distance(hi);
      const QGauss<dim-1> face_rule(op.quadrature_order);
      FEFaceValues<dim> face(this->get_mapping(),element,face_rule,
                             update_values|update_quadrature_points|update_normal_vectors|update_JxW_values);
      FEFaceValues<dim> flow(this->get_mapping(),this->get_fe(),face_rule,update_values);
      std::vector<Tensor<1,dim>> face_velocity(face_rule.size());
      // Reference coordinates outside a neighbor are intentional extrapolation.
      const auto reference_point=[](const auto &cell,const Point<dim> &p)
      {
        Point<dim> r;
        for (unsigned int d=0; d<dim; ++d)
          r[d]=(p[d]-cell->vertex(0)[d])/(cell->vertex(1U<<d)[d]-cell->vertex(0)[d]);
        return r;
      };
      for (const auto &cell:op.cells)
        {
          const auto target=op.space->cells.at(cell.id);
          typename DGLimiter::Cell stencil;
          stencil.h_power=std::pow(target->diameter()/reference_length,1.5);
          stencil.measure=target->measure();
          const auto append=[&](const auto &neighbor)
          {
            typename DGLimiter::Candidate candidate;
            candidate.indices.resize(n);
            neighbor->get_dof_indices(candidate.indices);
            candidate.extrapolate.reinit(n,n);
            for (unsigned int i=0; i<n; ++i)
              {
                Point<dim> physical=target->vertex(0);
                for (unsigned int d=0; d<dim; ++d)
                  physical[d]+=support[i][d]*(target->vertex(1U<<d)[d]-target->vertex(0)[d]);
                const auto r=reference_point(neighbor,physical);
                for (unsigned int j=0; j<n; ++j)
                  candidate.extrapolate(i,j)=element.shape_value(j,r);
              }
            stencil.candidates.push_back(std::move(candidate));
          };
          append(target);
          for (const auto f:target->face_indices())
            {
              AssertThrow(!target->has_periodic_neighbor(f),ExcMessage("PF4 limiter excludes periodic faces."));
              const bool boundary=target->at_boundary(f);
              if (!boundary)
                {
                  AssertThrow(target->neighbor(f)->is_active() && !target->neighbor_is_coarser(f),
                              ExcMessage("PF4 limiter excludes AMR."));
                  append(target->neighbor(f));
                }
              face.reinit(target,f);
              flow.reinit(fe_strain_state->cells.at(cell.id),f);
              flow[this->introspection().extractors.velocities].get_function_values(velocity,face_velocity);
              for (unsigned int q=0; q<face_rule.size(); ++q)
                if (face_velocity[q]*face.normal_vector(q)<0.)
                  {
                    typename DGLimiter::FacePoint point;
                    point.point=face.quadrature_point(q);
                    point.weight=face.JxW(q);
                    point.inside.reinit(n);
                    point.outside.reinit(n);
                    for (unsigned int i=0; i<n; ++i) point.inside[i]=face.shape_value(i,q);
                    if (!boundary)
                      {
                        const auto neighbor=target->neighbor(f);
                        point.neighbor.resize(n);
                        neighbor->get_dof_indices(point.neighbor);
                        const auto r=reference_point(neighbor,point.point);
                        for (unsigned int i=0; i<n; ++i) point.outside[i]=element.shape_value(i,r);
                      }
                    stencil.inflow.push_back(std::move(point));
                  }
            }
          data.cells.push_back(std::move(stencil));
        }
      return result;
    }


    template <int dim>
    void PhaseFieldFault<dim>::limit_dg_transport(const DGLimiter &data,
                                                  const DGComponents &incoming,DGComponents &result,
                                                  const DGField &boundary,const double time,
                                                  DGLimiterStats &statistics) const
    {
      AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1
                  && this->get_triangulation().get_vertices()==data.mesh_vertices
                  && &incoming!=&result && std::isfinite(data.threshold) && data.threshold>0.,
                  ExcMessage("PF4 limiter requires a separate output, serial fixed mesh and positive threshold."));
      using TensorType=SymmetricTensor<2,dim>;
      for (const auto &component:incoming)
        AssertThrow(component.size()==data.space->dofs.n_dofs(),ExcMessage("PF4 limiter field/space mismatch."));
      const auto norm_square=[](const TensorType &a)
      {
        return a*a+trace(a)*trace(a);
      };
      const auto now=[]()
      {
        return std::chrono::steady_clock::now();
      };
      const auto seconds=[](const auto a,const auto b)
      {
        return std::chrono::duration<double>(b-a).count();
      };
      result=incoming;
      statistics.last.clear();
      AssertThrow(!statistics.capture_first_call || statistics.measure_norm_changes,
                  ExcMessage("PF4 snapshot capture requires norm-call measurement."));
      const bool capture=statistics.capture_first_call && statistics.norm_changes.empty();
      if (capture) statistics.first_incoming=incoming;
      typename DGLimiterStats::NormChange norm_change;
      norm_change.time=time;
      const unsigned int n=data.mean.size(),nq=data.quadrature_weights.size();
      for (const auto &cell:data.cells)
        {
          const auto begin=now();
          typename DGLimiterStats::Indicator indicator;
          const unsigned int count=cell.candidates.size();
          if (data.detector==DGLimiter::Detector::scaling)
            {
              const double reference_square=norm_square(data.scaling_reference);
              AssertThrow(std::isfinite(reference_square) && reference_square>0.
                          && std::isfinite(data.scaling_lower) && std::isfinite(data.scaling_upper)
                          && data.scaling_lower<data.scaling_upper,
                          ExcMessage("PF4 scaling requires an explicit diagnostic tensor reference and scalar envelope."));
              std::vector<TensorType> values(n);
              TensorType mean;
              double scale=0.;
              for (unsigned int i=0; i<n; ++i)
                {
                  for (unsigned int c=0; c<incoming.size(); ++c)
                    {
                      const double a=incoming[c][cell.candidates[0].indices[i]];
                      AssertThrow(std::isfinite(a),ExcMessage("Nonfinite PF4 scaling input."));
                      values[i][TensorType::unrolled_to_component_indices(c)]=a;
                      scale=std::max(scale,std::abs(a));
                    }
                  mean+=data.mean[i]*values[i];
                }
              const auto amplitude=[&](const TensorType &a)
              {
                return (a*data.scaling_reference+trace(a)*trace(data.scaling_reference))/reference_square;
              };
              const double average=amplitude(mean);
              const double mean_excess=std::max({0.,average-data.scaling_upper,data.scaling_lower-average});
              norm_change.means_outside_envelope+=mean_excess>0.;
              norm_change.maximum_mean_excess=std::max(norm_change.maximum_mean_excess,mean_excess);
              const double allowance=128.*std::numeric_limits<double>::epsilon()
                                     *std::max({1.,std::abs(data.scaling_lower),std::abs(data.scaling_upper)});
              AssertThrow(average>=data.scaling_lower-allowance && average<=data.scaling_upper+allowance,
                          ExcMessage("PF4 scaling cannot preserve an inadmissible cell mean; candidate rejected."));
              // A mean outside by roundoff is retained, never clipped.
              const double lower=std::min(data.scaling_lower,average),upper=std::max(data.scaling_upper,average);
              for (unsigned int q=0; q<data.scaling_values.m(); ++q)
                {
                  TensorType a;
                  for (unsigned int i=0; i<n; ++i) a+=data.scaling_values(q,i)*values[i];
                  const double scalar=amplitude(a);
                  if (scalar>upper+allowance) indicator.theta=std::min(indicator.theta,(upper-average)/(scalar-average));
                  if (scalar<lower-allowance) indicator.theta=std::min(indicator.theta,(lower-average)/(scalar-average));
                }
              indicator.limited=indicator.theta<1.;
              ++statistics.visits;
              statistics.detector_seconds+=seconds(begin,now());
              if (indicator.limited)
                {
                  const auto reconstruction_begin=now();
                  ++statistics.limited;
                  TensorType reconstructed_mean;
                  for (unsigned int i=0; i<n; ++i)
                    {
                      const auto a=mean+indicator.theta*(values[i]-mean);
                      reconstructed_mean+=data.mean[i]*a;
                      for (unsigned int c=0; c<result.size(); ++c)
                        result[c][cell.candidates[0].indices[i]]=a[TensorType::unrolled_to_component_indices(c)];
                    }
                  indicator.mean_error=scale>0. ? std::sqrt(norm_square((reconstructed_mean-mean)/scale)) : 0.;
                  statistics.maximum_mean_error=std::max(statistics.maximum_mean_error,indicator.mean_error);
                  statistics.reconstruction_seconds+=seconds(reconstruction_begin,now());
                }
            }
          else
            {
              std::vector<std::vector<TensorType>> own(count,std::vector<TensorType>(n)),candidate=own;
              double scale=0.;
              for (unsigned int j=0; j<count; ++j)
                for (unsigned int i=0; i<n; ++i)
                  for (unsigned int c=0; c<incoming.size(); ++c)
                    {
                      const double value=incoming[c][cell.candidates[j].indices[i]];
                      AssertThrow(std::isfinite(value),ExcMessage("Nonfinite PF4 limiter input."));
                      own[j][i][TensorType::unrolled_to_component_indices(c)]=value;
                      scale=std::max(scale,std::abs(value));
                    }
              // Boundary traces enter KXRCF only; include their scale before ratios,
              // but not in the patch RMS definition.
              std::vector<TensorType> boundary_values(cell.inflow.size());
              if (data.detector==DGLimiter::Detector::kxrcf)
                for (unsigned int q=0; q<cell.inflow.size(); ++q)
                  if (cell.inflow[q].neighbor.empty())
                    {
                      boundary_values[q]=boundary(cell.inflow[q].point,time);
                      for (unsigned int c=0; c<incoming.size(); ++c)
                        {
                          const double value=boundary_values[q][TensorType::unrolled_to_component_indices(c)];
                          AssertThrow(std::isfinite(value),ExcMessage("Nonfinite PF4 limiter inflow."));
                          scale=std::max(scale,std::abs(value));
                        }
                    }
              std::vector<TensorType> means(count);
              double amplitude_square=0.;
              if (scale!=0.)
                for (unsigned int j=0; j<count; ++j)
                  {
                    for (auto &a:own[j]) a/=scale;
                    double square=0.;
                    for (unsigned int q=0; q<nq; ++q)
                      {
                        TensorType value;
                        for (unsigned int i=0; i<n; ++i) value+=data.values(q,i)*own[j][i];
                        square+=data.quadrature_weights[q]*norm_square(value);
                      }
                    amplitude_square=std::max(amplitude_square,square);
                    for (unsigned int i=0; i<n; ++i)
                      {
                        candidate[j][i]=own[j][0];
                        for (unsigned int k=0; k<n; ++k)
                          candidate[j][i]+=cell.candidates[j].extrapolate(i,k)*(own[j][k]-own[j][0]);
                        means[j]+=data.mean[i]*candidate[j][i];
                      }
                  }
              const double amplitude=std::sqrt(amplitude_square);
              if (amplitude>0. && data.detector!=DGLimiter::Detector::kxrcf)
                {
                  for (unsigned int j=1; j<count; ++j)
                    {
                      indicator.mean+=std::sqrt(norm_square(means[0]-means[j]))/amplitude;
                      double square=0.;
                      for (unsigned int q=0; q<nq; ++q)
                        {
                          TensorType difference;
                          for (unsigned int i=0; i<n; ++i)
                            difference+=data.values(q,i)*(candidate[0][i]-candidate[j][i]);
                          square+=data.quadrature_weights[q]*norm_square(difference);
                        }
                      indicator.polynomial=std::max(indicator.polynomial,std::sqrt(square)/amplitude);
                    }
                }
              if (data.detector==DGLimiter::Detector::kxrcf && scale>0.)
                {
                  TensorType jump;
                  double area=0.;
                  for (unsigned int q=0; q<cell.inflow.size(); ++q)
                    {
                      const auto &point=cell.inflow[q];
                      TensorType inside,outside;
                      for (unsigned int i=0; i<n; ++i) inside+=point.inside[i]*own[0][i];
                      if (point.neighbor.empty()) outside=boundary_values[q]/scale;
                      else
                        for (unsigned int i=0; i<n; ++i)
                          for (unsigned int c=0; c<incoming.size(); ++c)
                            outside[TensorType::unrolled_to_component_indices(c)]+=
                              point.outside[i]*(incoming[c][point.neighbor[i]]/scale);
                      jump+=point.weight*(inside-outside);
                      area+=point.weight;
                    }
                  // A zero interior patch with nonzero boundary inflow has an
                  // unbounded relative jump, rather than an artificial stress floor.
                  if (area>0. && norm_square(jump)>0.)
                    indicator.kxrcf=amplitude>0. ? std::sqrt(norm_square(jump))/(area*cell.h_power*amplitude)
                                    : std::numeric_limits<double>::infinity();
                }
              indicator.limited=data.detector==DGLimiter::Detector::force
                                || (data.detector==DGLimiter::Detector::kxrcf ? indicator.kxrcf>data.threshold
                                    : std::max(indicator.mean,indicator.polynomial)>data.threshold);
              ++statistics.visits;
              statistics.mean_flagged+=indicator.mean>data.threshold;
              statistics.safeguard_only+=indicator.polynomial>data.threshold && indicator.mean<=data.threshold;
              statistics.detector_seconds+=seconds(begin,now());
              if (indicator.limited)
                {
                  const auto reconstruction_begin=now();
                  ++statistics.limited;
                  indicator.beta.assign(count,0.);
                  indicator.weights.assign(count,.001);
                  indicator.weights[0]=1.-.001*(count-1);
                  // Smoothness in target reference coordinates. The common tensor
                  // first/second derivative sum is a Q2 adaptation, not full P4 JS.
                  if (amplitude>0.)
                    for (unsigned int j=0; j<count; ++j)
                      for (unsigned int q=0; q<nq; ++q)
                        for (unsigned int d=0; d<5; ++d)
                          {
                            TensorType derivative;
                            for (unsigned int i=0; i<n; ++i)
                              derivative+=data.derivatives[d](q,i)*(candidate[j][i]-candidate[j][0]);
                            derivative/=amplitude;
                            indicator.beta[j]+=data.quadrature_weights[q]*(d==3?2.:1.)*norm_square(derivative);
                          }
                  if (*std::max_element(indicator.beta.begin(),indicator.beta.end())>0.)
                    {
                      const double minimum=*std::min_element(indicator.beta.begin(),indicator.beta.end())+1e-6;
                      for (unsigned int j=0; j<count; ++j)
                        indicator.weights[j]*=std::pow(minimum/(indicator.beta[j]+1e-6),2);
                    }
                  const double sum=std::accumulate(indicator.weights.begin(),indicator.weights.end(),0.);
                  for (auto &w:indicator.weights) w/=sum;
                  TensorType reconstructed_mean;
                  for (unsigned int i=0; i<n; ++i)
                    {
                      TensorType value;
                      for (unsigned int j=0; j<count; ++j)
                        value+=indicator.weights[j]*(candidate[j][i]-means[j]+means[0]);
                      reconstructed_mean+=data.mean[i]*value;
                      for (unsigned int c=0; c<result.size(); ++c)
                        result[c][cell.candidates[0].indices[i]]=scale*value[TensorType::unrolled_to_component_indices(c)];
                    }
                  indicator.mean_error=std::sqrt(norm_square(reconstructed_mean-means[0]));
                  statistics.maximum_mean_error=std::max(statistics.maximum_mean_error,indicator.mean_error);
                  statistics.reconstruction_seconds+=seconds(reconstruction_begin,now());
                }
            }
          if (statistics.measure_norm_changes)
            {
              const auto measurement_begin=now();
              double cell_before=0.,cell_change=0.;
              for (unsigned int q=0; q<nq; ++q)
                {
                  TensorType a,difference;
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int c=0; c<incoming.size(); ++c)
                      {
                        const auto component=TensorType::unrolled_to_component_indices(c);
                        const auto index=cell.candidates[0].indices[i];
                        a[component]+=data.values(q,i)*incoming[c][index];
                        difference[component]+=data.values(q,i)*(result[c][index]-incoming[c][index]);
                      }
                  const double weight=cell.measure*data.quadrature_weights[q];
                  cell_before+=weight*norm_square(a);
                  cell_change+=weight*(2.*(a*difference+trace(a)*trace(difference))+norm_square(difference));
                }
              norm_change.before+=cell_before;
              norm_change.change+=cell_change;
              const double allowance=128.*std::numeric_limits<double>::epsilon()*cell_before;
              norm_change.positive_cells+=cell_change>allowance;
              norm_change.negative_cells+=cell_change<-allowance;
              statistics.norm_measurement_seconds+=seconds(measurement_begin,now());
            }
          statistics.last.push_back(std::move(indicator));
        }
      if (statistics.measure_norm_changes)
        {
          norm_change.after=norm_change.before+norm_change.change;
          AssertThrow(std::isfinite(norm_change.after) && std::isfinite(norm_change.change),
                      ExcMessage("Nonfinite PF4 limiter norm measurement."));
          statistics.norm_changes.push_back(norm_change);
        }
      if (capture) statistics.first_outgoing=result;
    }

#define INSTANTIATE(dim) \
  template std::shared_ptr<PhaseFieldFault<dim>::DGLimiter> PhaseFieldFault<dim>::make_dg_limiter(const DGTransport &,const LinearAlgebra::BlockVector &) const; \
  template void PhaseFieldFault<dim>::limit_dg_transport(const DGLimiter &,const DGComponents &,DGComponents &,const DGField &,double,DGLimiterStats &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
