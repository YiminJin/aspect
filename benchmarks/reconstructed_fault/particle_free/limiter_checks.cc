/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/reconstructed_fault/manager.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <chrono>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        using History=typename Model::DGHistory;
        using Transport=typename Model::DGTransport;
        using Field=typename Model::DGField;
        using BV=LinearAlgebra::BlockVector;
        using TensorField=SymmetricTensor<2,dim>;
        using Limiter=typename Model::DGLimiter;
        using Stats=typename Model::DGLimiterStats;
        using Components=typename Model::DGComponents;
        bool calibration_only=false;
        unsigned int nx,ny;
        std::ofstream metrics;
        const double final_time=.2;

        static TensorField basis(const double scalar)
        {
          TensorField value;
          value[0][0]=1000*scalar;
          value[1][1]=-400*scalar;
          value[0][1]=700*scalar;
          return value;
        }
        static double pulse(const double x,const double y)
        {
          return std::exp(-std::pow((x-.3)/.13,2)-std::pow((y-.5)/.16,2));
        }
        static double chi(const double y)
        {
          return std::exp(-std::pow((y-.5)/.12,2))/(.12*std::sqrt(numbers::PI)*std::erf(.5/.12));
        }
        static double chi_prime(const double y)
        {
          return -2*(y-.5)/(.12*.12)*chi(y);
        }
        static double slip(const double x)
        {
          return 1+.25*std::sin(2*numbers::PI*x);
        }
        static TensorField shear(const double value)
        {
          TensorField tensor;
          tensor[0][1]=value;
          return tensor;
        }
        static double full_norm_square(const TensorField &value)
        {
          return value*value+trace(value)*trace(value);
        }
        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+name);
          out<<std::setprecision(17);
          return out;
        }
        BV velocity(const std::function<Tensor<1,dim>(const Point<dim>&)> &field) const
        {
          struct NativeField : Function<dim>
          {
            NativeField(const unsigned int n,const std::function<Tensor<1,dim>(const Point<dim>&)> &f)
              : Function<dim>(n),function(f) {}
            double value(const Point<dim> &p,const unsigned int c) const override
            {
              return c<dim?function(p)[c]:0.;
            }
            std::function<Tensor<1,dim>(const Point<dim>&)> function;
          } native(this->get_fe().n_components(),field);
          BV result(this->get_solution());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),native,result,
                                   this->introspection().component_masks.velocities);
          return result;
        }
        std::shared_ptr<History> evolve(Model &model,const Transport &op,
                                        std::shared_ptr<History> state,const Field &inflow,
                                        const Field &source,const unsigned int steps) const
        {
          const double dt=final_time/steps;
          for (unsigned int step=0; step<steps; ++step)
            state=model.advance_dg_transport(op,*state,inflow,source,step*dt,dt);
          return state;
        }
        double difference(Model &model,const History &a,const History &b) const
        {
          double error=0.;
          const QGauss<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(),a.space->element,rule,update_JxW_values);
          for (const auto &entry:a.space->cells)
            {
              fe.reinit(entry.second);
              const auto va=model.evaluate_dg_trial_tensor(a,entry.first,rule);
              const auto vb=model.evaluate_dg_trial_tensor(b,entry.first,rule);
              for (unsigned int q=0; q<rule.size(); ++q) error+=fe.JxW(q)*full_norm_square(va[q]-vb[q]);
            }
          return std::sqrt(error);
        }
        void record(Model &model,const std::string &name,const History &state,const Field &exact,
                    const unsigned int steps,const double initial_error=0.)
        {
          const QGauss<dim> rule(8);
          const QGaussLobatto<dim> extrema_rule(9);
          FEValues<dim> fe(this->get_mapping(),state.space->element,rule,update_JxW_values|update_quadrature_points);
          double error=0.,norm=0.,reference_norm=0.,maximum=0.,minimum_xx=std::numeric_limits<double>::max(),maximum_xx=-minimum_xx;
          for (const auto &entry:state.space->cells)
            {
              fe.reinit(entry.second);
              const auto values=model.evaluate_dg_trial_tensor(state,entry.first,rule);
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  const auto reference=exact(fe.quadrature_point(q),final_time);
                  error+=fe.JxW(q)*full_norm_square(values[q]-reference);
                  norm+=fe.JxW(q)*full_norm_square(values[q]);
                  reference_norm+=fe.JxW(q)*full_norm_square(reference);
                  maximum=std::max(maximum,std::sqrt(full_norm_square(values[q]-reference)));
                }
              for (const auto &value:model.evaluate_dg_trial_tensor(state,entry.first,extrema_rule))
                {
                  minimum_xx=std::min(minimum_xx,value[0][0]);
                  maximum_xx=std::max(maximum_xx,value[0][0]);
                }
            }
          metrics << name << ',' << nx << ',' << ny << ',' << final_time << ',' << final_time/steps << ',' << steps << ','
                  << std::sqrt(error) << ',' << std::sqrt(reference_norm) << ',' << std::sqrt(norm) << ',' << maximum << ','
                  << minimum_xx << ',' << maximum_xx << ',' << initial_error << '\n';
          auto coefficients=output("field-"+name+".csv");
          coefficients << "i,xx,yy,xy\n";
          for (unsigned int i=0; i<state.space->dofs.n_dofs(); ++i)
            coefficients << i << ',' << state.trial_stress[0][i] << ',' << state.trial_stress[1][i] << ',' << state.trial_stress[2][i] << '\n';
        }
        void export_geometry(const Transport &op) const
        {
          auto cells=output("cells.csv"),nodes=output("basis.csv");
          cells << "cell,x0,y0,dx,dy";
          for (unsigned int i=0; i<9; ++i) cells << ",i" << i;
          cells << '\n';
          nodes << "i,r,s\n";
          const auto &support=op.space->element.get_unit_support_points();
          for (unsigned int i=0; i<support.size(); ++i) nodes << i << ',' << support[i][0] << ',' << support[i][1] << '\n';
          unsigned int c=0;
          for (const auto &cell:op.cells)
            {
              const auto iterator=op.space->cells.at(cell.id);
              cells << c++ << ',' << iterator->vertex(0)[0] << ',' << iterator->vertex(0)[1] << ','
                    << iterator->vertex(1)[0]-iterator->vertex(0)[0] << ',' << iterator->vertex(2)[1]-iterator->vertex(0)[1];
              for (const auto i:cell.indices) cells << ',' << i;
              cells << '\n';
            }
        }
        Components components(const History &state) const
        {
          Components result;
          for (unsigned int c=0; c<result.size(); ++c)
            {
              result[c].reinit(state.space->dofs.n_dofs());
              for (unsigned int i=0; i<result[c].size(); ++i) result[c][i]=state.trial_stress[c][i];
            }
          return result;
        }
        Components nodal(const Transport &op,const std::function<TensorField(const Point<dim>&,const Point<dim>&,double,double)> &f) const
        {
          Components result;
          for (auto &c:result) c.reinit(op.space->dofs.n_dofs());
          for (const auto &cell:op.cells)
            {
              const auto native=op.space->cells.at(cell.id);
              const Point<dim> origin=native->vertex(0);
              const double dx=native->vertex(1)[0]-origin[0],dy=native->vertex(2)[1]-origin[1];
              const auto &support=op.space->element.get_unit_support_points();
              for (unsigned int i=0; i<support.size(); ++i)
                {
                  Point<dim> p=origin;
                  p[0]+=support[i][0]*dx;
                  p[1]+=support[i][1]*dy;
                  const auto a=f(p,native->center(),dx,dy);
                  for (unsigned int c=0; c<result.size(); ++c)
                    result[c][cell.indices[i]]=a[TensorField::unrolled_to_component_indices(c)];
                }
            }
          return result;
        }
        void coefficients(const std::string &name,const Components &state) const
        {
          auto out=output(name+".csv");
          out << "i,xx,yy,xy\n";
          for (unsigned int i=0; i<state[0].size(); ++i)
            out << i << ',' << state[0][i] << ',' << state[1][i] << ',' << state[2][i] << '\n';
        }
        void indicators(const std::string &name,const Stats &statistics) const
        {
          auto out=output(name+".csv");
          out << "cell,mean,poly,kxrcf,limited,mean_error";
          for (unsigned int j=0; j<5; ++j) out << ",w" << j << ",b" << j;
          out << '\n';
          for (unsigned int i=0; i<statistics.last.size(); ++i)
            {
              const auto &r=statistics.last[i];
              out << i << ',' << r.mean << ',' << r.polynomial << ',' << r.kxrcf << ',' << r.limited << ',' << r.mean_error;
              for (unsigned int j=0; j<5; ++j)
                out << ',' << (j<r.weights.size()?r.weights[j]:0.) << ',' << (j<r.beta.size()?r.beta[j]:0.);
              out << '\n';
            }
        }
        static TensorField rotate(const TensorField &a)
        {
          const double c=std::cos(.37),s=std::sin(.37);
          TensorField b;
          b[0][0]=c*c*a[0][0]+s*s*a[1][1]-2*c*s*a[0][1];
          b[1][1]=s*s*a[0][0]+c*c*a[1][1]+2*c*s*a[0][1];
          b[0][1]=c*s*(a[0][0]-a[1][1])+(c*c-s*s)*a[0][1];
          return b;
        }
        void local_checks(Model &model,const Transport &op,const BV &v)
        {
          auto limiter=model.make_dg_limiter(op,v);
          limiter->detector=Limiter::Detector::force;
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          auto summary=output("local.csv");
          summary << "case,relative_change,maximum_mean_error,order_error,basis_error,weight_sum_error,min_weight\n";
          const std::vector<std::string> names= {"zero","constant","piecewise_constant","affine","q2","mixed_xy","zero_crossing","nonproportional","orientation","near_constant","tiny","huge","background"};
          for (const auto &name:names)
            {
              const auto field=[name](const Point<dim> &p,const Point<dim> &center,double dx,double dy)
              {
                if (name=="zero") return TensorField();
                if (name=="constant") return basis(1.);
                if (name=="piecewise_constant") return basis(center[0]<.5?1.:-.7);
                if (name=="affine") return basis(1.+p[0]-2*p[1]);
                if (name=="q2") return basis(1.+p[0]-p[1]+p[0]*p[1]+p[0]*p[0]*p[1]+p[0]*p[1]*p[1]+p[0]*p[0]*p[1]*p[1]);
                if (name=="mixed_xy") return basis(1.+(p[0]-center[0])*(p[1]-center[1])/(dx*dy));
                if (name=="zero_crossing") return basis(std::sin(2*numbers::PI*(p[0]-.5)));
                if (name=="near_constant") return basis(1.+1e-12*p[0]*p[1]);
                TensorField value;
                if (name=="orientation")
                  {
                    value[0][0]=std::cos(8*p[0]);
                    value[1][1]=-value[0][0];
                    value[0][1]=std::sin(8*p[0]);
                    return value;
                  }
                value[0][0]=1.+(p[0]-center[0])*(p[1]-center[1])/(dx*dy);
                value[1][1]=std::sin(4*p[1])+p[0]*p[0]*p[1];
                value[0][1]=p[0]<.5?-.8:1.2;
                if (name=="tiny") value*=1e-250;
                if (name=="huge") value*=1e250;
                if (name=="background") value+=basis(1e5);
                return value;
              };
              const auto incoming=nodal(op,field),snapshot=incoming;
              Components result,reordered,rotated_input=incoming,rotated_result;
              Stats stats,order_stats,rotated_stats;
              model.limit_dg_transport(*limiter,incoming,result,zero,0.,stats);
              auto reversed=*limiter;
              std::reverse(reversed.cells.begin(),reversed.cells.end());
              for (auto &c:reversed.cells) std::reverse(c.candidates.begin()+1,c.candidates.end());
              model.limit_dg_transport(reversed,incoming,reordered,zero,0.,order_stats);
              for (unsigned int i=0; i<incoming[0].size(); ++i)
                {
                  TensorField a;
                  for (unsigned int c=0; c<incoming.size(); ++c) a[TensorField::unrolled_to_component_indices(c)]=incoming[c][i];
                  const auto b=rotate(a);
                  for (unsigned int c=0; c<incoming.size(); ++c) rotated_input[c][i]=b[TensorField::unrolled_to_component_indices(c)];
                }
              model.limit_dg_transport(*limiter,rotated_input,rotated_result,zero,0.,rotated_stats);
              for (unsigned int k=0; k<stats.last.size(); ++k)
                AssertThrow(std::abs(stats.last[k].mean-rotated_stats.last[k].mean)<1e-11
                            && std::abs(stats.last[k].polynomial-rotated_stats.last[k].polynomial)<1e-11,
                            ExcMessage("Tensor detector basis covariance failed."));
              double scale=0.,change=0.,order_error=0.,basis_error=0.,weight_error=0.,min_weight=1.;
              for (unsigned int i=0; i<incoming[0].size(); ++i)
                {
                  TensorField a;
                  for (unsigned int c=0; c<incoming.size(); ++c) a[TensorField::unrolled_to_component_indices(c)]=result[c][i];
                  const auto b=rotate(a);
                  for (unsigned int c=0; c<incoming.size(); ++c)
                    {
                      scale=std::max(scale,std::abs(incoming[c][i]));
                      change=std::max(change,std::abs(incoming[c][i]-result[c][i]));
                      order_error=std::max(order_error,std::abs(result[c][i]-reordered[c][i]));
                      basis_error=std::max(basis_error,std::abs(rotated_result[c][i]-b[TensorField::unrolled_to_component_indices(c)]));
                      AssertThrow(incoming[c][i]==snapshot[c][i] && std::isfinite(result[c][i]),ExcInternalError());
                    }
                }
              for (const auto &r:stats.last)
                {
                  double sum=0.;
                  for (const auto w:r.weights)
                    {
                      AssertThrow(std::isfinite(w) && w>=0.,ExcMessage("Invalid WENO weight."));
                      sum+=w;
                      min_weight=std::min(min_weight,w);
                    }
                  weight_error=std::max(weight_error,std::abs(sum-1.));
                }
              const double denominator=scale==0.?1.:scale;
              summary << name << ',' << change/denominator << ',' << stats.maximum_mean_error << ',' << order_error/denominator << ',' << basis_error/denominator << ',' << weight_error << ',' << min_weight << '\n';
              AssertThrow(stats.maximum_mean_error<1e-12 && order_error/denominator<1e-12 && basis_error/denominator<1e-11 && weight_error<1e-14,ExcMessage("Local tensor limiter invariant failed."));
              if (name=="zero" || name=="constant" || name=="piecewise_constant" || name=="affine" || name=="q2")
                AssertThrow(change/denominator<1e-12,ExcMessage("Exact polynomial/constant changed."));
              coefficients("local-input-"+name,incoming);
              coefficients("local-output-"+name,result);
              indicators("local-indicators-"+name,stats);
              if (name=="nonproportional" || name=="orientation")
                {
                  auto kx=*limiter;
                  kx.detector=Limiter::Detector::kxrcf;
                  kx.threshold=1.;
                  Stats original_kx,rotated_kx;
                  Components original_output,rotated_output;
                  model.limit_dg_transport(kx,incoming,original_output,zero,0.,original_kx);
                  model.limit_dg_transport(kx,rotated_input,rotated_output,zero,0.,rotated_kx);
                  double indicator_error=0.;
                  for (unsigned int k=0; k<original_kx.last.size(); ++k)
                    {
                      indicator_error=std::max(indicator_error,std::abs(original_kx.last[k].kxrcf-rotated_kx.last[k].kxrcf));
                      AssertThrow(original_kx.last[k].limited==rotated_kx.last[k].limited,ExcMessage("KXRCF mask basis covariance failed."));
                    }
                  AssertThrow(indicator_error<1e-11,ExcMessage("KXRCF tensor indicator covariance failed."));
                  indicators("local-kx-"+name,original_kx);
                  indicators("local-kx-rotated-"+name,rotated_kx);
                }
              if (name=="mixed_xy")
                {
                  auto kx=*limiter;
                  kx.detector=Limiter::Detector::kxrcf;
                  kx.threshold=1.;
                  Stats kx_stats;
                  model.limit_dg_transport(kx,incoming,result,zero,0.,kx_stats);
                  indicators("local-kx-mixed_xy",kx_stats);
                }
            }
          // Mixed inflow/outflow on every vertical face: use pointwise native Q2 v.
          const auto mixed_velocity=velocity([](const Point<dim> &p)
          {
            Tensor<1,dim> v;
            v[0]=p[1]-.4375;
            return v;
          });
          const auto mixed_op=model.make_dg_transport(mixed_velocity,4,false);
          auto mixed=model.make_dg_limiter(*mixed_op,mixed_velocity);
          mixed->detector=Limiter::Detector::kxrcf;
          const auto constant=nodal(op,[](const auto &,const auto &,double,double)
          {
            return basis(1.);
          });
          Components result;
          Stats boundary_stats;
          model.limit_dg_transport(*mixed,constant,result,zero,0.,boundary_stats);
          indicators("boundary-indicators",boundary_stats);
          auto face_data=output("inflow-points.csv");
          face_data << "cell,x,y,weight,boundary\n";
          for (unsigned int c=0; c<mixed->cells.size(); ++c)
            for (const auto &p:mixed->cells[c].inflow)
              face_data << c << ',' << p.point[0] << ',' << p.point[1] << ',' << p.weight << ',' << p.neighbor.empty() << '\n';
          auto lifecycle=output("limited-lifecycle.csv");
          lifecycle << "detector,repeat_error,late_source_rejected,incoming_unchanged\n";
          const Field smooth=[](const Point<dim> &p,double)
          {
            return basis(pulse(p[0],p[1]));
          };
          const auto input=model.project_dg_transport_field(op,smooth,0.,8);
          const auto saved=input->trial_stress;
          for (const auto detector:
          {
            Limiter::Detector::fu_shu,Limiter::Detector::kxrcf
          })
          {
            limiter->detector=detector;
            limiter->threshold=detector==Limiter::Detector::fu_shu?.1:1.;
            Stats a_stats,b_stats,failed_stats;
            const auto a=model.advance_dg_transport(op,*input,smooth,zero,0.,.001,limiter.get(),&a_stats);
            const auto b=model.advance_dg_transport(op,*input,smooth,zero,0.,.001,limiter.get(),&b_stats);
            bool rejected=false,unchanged=true;
            try
              {
                model.advance_dg_transport(op,*input,smooth,[](const Point<dim> &,double t)
                {
                  return t>0.?basis(std::numeric_limits<double>::quiet_NaN()):TensorField();
                },0.,.001,limiter.get(),&failed_stats);
              }
            catch (const std::exception &)
              {
                rejected=true;
              }
            for (unsigned int c=0; c<saved.size(); ++c) unchanged &= saved[c]==input->trial_stress[c];
            const double repeat=difference(model,*a,*b);
            lifecycle << (detector==Limiter::Detector::fu_shu?"fu_shu":"kxrcf") << ',' << repeat << ',' << rejected << ',' << unchanged << '\n';
            AssertThrow(repeat==0. && rejected && unchanged,ExcMessage("Limited trial lifecycle failed."));
          }
        }
        void snapshot(Model &model,const std::string &name,const Transport &op,const BV &v,const Components &incoming,const Field &boundary,double t)
        {
          coefficients("snapshot-"+name,incoming);
          auto limiter=model.make_dg_limiter(op,v);
          Components out;
          Stats stats;
          model.limit_dg_transport(*limiter,incoming,out,boundary,t,stats);
          indicators("snapshot-indicators-"+name,stats);
        }
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("PF4 limiter checks");
          prm.declare_entry("Calibration only","false",Patterns::Bool(),"Run local and saved-snapshot screening only.");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("PF4 limiter checks");
          calibration_only=prm.get_bool("Calibration only");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &model=Plugins::get_plugin_as_type<Model>(*sim.material_model);
          AssertThrow(dim==2 && model.uses_dg_trial_stress_history() && this->get_timestep_number()==0,ExcMessage("Fresh PF4 DG fixture required."));
          model.prepare_reconstructed_fault_mechanical_solve();
          const auto anchor=Access::dg_history_values(model);
          const BV phase_anchor(model.accepted_fe_strain_state());
          const auto Z_anchor=model.fe_strain_state->Z,I_anchor=model.fe_strain_state->I;
          const auto first=model.fe_strain_state->cells.begin()->second;
          nx=std::lround(1/(first->vertex(1)[0]-first->vertex(0)[0]));
          ny=std::lround(1/(first->vertex(2)[1]-first->vertex(0)[1]));
          auto constants=output("constants.csv");
          constants << "nx,ny,relaxation_rate,time\n" << nx << ',' << ny << ",0," << final_time << '\n';
          metrics=output("metrics.csv");
          metrics << "case,nx,ny,time,dt,steps,error_L2,reference_L2,norm_L2,max_error,min_xx,max_xx,initial_error\n";
          auto costs=output("costs.csv");
          costs << "case,steps,visits,limited,mean_flagged,safeguard_only,detector_seconds,reconstruction_seconds,transport_seconds,max_mean_error\n";
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          for (const std::string name:
          {"pulse_translation","signed_jump"
          })
          {
            const auto v=velocity([name](const Point<dim> &)
            {
              Tensor<1,dim> v;
              v[0]=.4;
              if (name=="pulse_translation") v[1]=.2;
              return v;
            });
            const auto op=model.make_dg_transport(v,4,false);
            if (name=="pulse_translation")
              {
                export_geometry(*op);
                if (calibration_only && nx==4) local_checks(model,*op,v);
              }
            const unsigned int steps=std::ceil(final_time*op->inverse_advection_time/.025);
            const Field exact=[name](const Point<dim> &p,double t)
            {
              if (name=="pulse_translation") return basis(pulse(p[0]-.4*t,p[1]-.2*t));
              const double x=p[0]-.4*t;
              return basis(x>=.25 && x<.5?1.:(x>=.5 && x<.75?-1.:0.));
            };
            const auto initial=model.project_dg_transport_field(*op,exact,0.,8);
            coefficients("initial-"+name,components(*initial));
            for (const std::string path:
            {"unlimited","kxrcf","fu_shu"
            })
            {
              if (calibration_only && path!="unlimited") continue;
              auto limiter=model.make_dg_limiter(*op,v);
              limiter->detector=path=="kxrcf"?Limiter::Detector::kxrcf:Limiter::Detector::fu_shu;
              limiter->threshold=path=="kxrcf"?1.:.1;
              Stats stats;
              auto state=initial;
              const auto begin=std::chrono::steady_clock::now();
              for (unsigned int step=0; step<steps; ++step)
                state=model.advance_dg_transport(*op,*state,exact,zero,step*final_time/steps,final_time/steps,path=="unlimited"?nullptr:limiter.get(),&stats);
              const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
              record(model,name+"-"+path,*state,exact,steps);
              costs << name+'-'+path << ',' << steps << ',' << stats.visits << ',' << stats.limited << ',' << stats.mean_flagged << ',' << stats.safeguard_only << ',' << stats.detector_seconds << ',' << stats.reconstruction_seconds << ',' << elapsed << ',' << stats.maximum_mean_error << '\n';
              if (calibration_only)
                {
                  snapshot(model,name+"-initial",*op,v,components(*initial),exact,0.);
                  snapshot(model,name+"-final",*op,v,components(*state),exact,final_time);
                }
            }
          }
          const bool unchanged=anchor==Access::dg_history_values(model) && Z_anchor==model.fe_strain_state->Z && I_anchor==model.fe_strain_state->I;
          auto accepted=output("accepted.csv");
          accepted << "name,value\naccepted_unchanged," << unchanged << '\n';
          BV phase_change(phase_anchor);
          phase_change-=model.accepted_fe_strain_state();
          AssertThrow(unchanged && phase_change.l2_norm()==0.,ExcMessage("Limiter published accepted state."));
          return {"PF4 limiter checks","completed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF4 limiter checks","Bounded nonpublishing DGQ2 tensor limiter experiment.")
  }
}
