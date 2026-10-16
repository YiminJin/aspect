/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/phase_field.h>
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
        unsigned int time_divisor=1;
        bool local_only=false;
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
        static TensorField rotate(const TensorField &a)
        {
          const double c=std::cos(.37),s=std::sin(.37);
          TensorField b;
          b[0][0]=c*c*a[0][0]+s*s*a[1][1]-2*c*s*a[0][1];
          b[1][1]=s*s*a[0][0]+c*c*a[1][1]+2*c*s*a[0][1];
          b[0][1]=c*s*(a[0][0]-a[1][1])+(c*c-s*s)*a[0][1];
          return b;
        }
        double quadratic_norm(const Transport &op,const Components &a) const
        {
          const QGauss<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(),op.space->element,rule,update_values|update_JxW_values);
          double value=0.;
          for (const auto &cell:op.cells)
            {
              fe.reinit(op.space->cells.at(cell.id));
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  TensorField tensor;
                  for (unsigned int i=0; i<cell.indices.size(); ++i)
                    for (unsigned int c=0; c<a.size(); ++c)
                      tensor[TensorField::unrolled_to_component_indices(c)]+=fe.shape_value(i,q)*a[c][cell.indices[i]];
                  value+=fe.JxW(q)*full_norm_square(tensor);
                }
            }
          return value;
        }
        void local_checks(Model &model,const Transport &op,const BV &v)
        {
          auto limiter=model.make_dg_limiter(op,v);
          limiter->detector=Limiter::Detector::scaling;
          limiter->scaling_reference=basis(1.);
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          auto summary=output("scaling-local.csv");
          summary << "case,before,after,change,mean_error,order_error,basis_error,indicator_error,theta_min,theta_max,idempotence_error\n";
          for (const std::string name:
          {"zero","constant","piecewise_constant","affine","q2","overshoot","nonproportional"
          })
          {
            const auto input=nodal(op,[name](const Point<dim> &p,const Point<dim> &center,double dx,double dy)
            {
              if (name=="zero") return TensorField();
              if (name=="constant") return basis(.3);
              if (name=="piecewise_constant") return basis(center[0]<.5?1.:-1.);
              if (name=="affine") return basis(p[0]-.5);
              if (name=="q2") return basis(.1+.2*p[0]*p[0]*p[1]*p[1]);
              const double xi=(p[0]-center[0])/dx,eta=(p[1]-center[1])/dy;
              auto a=basis(.2+1.8*xi+2*xi*eta+4*xi*xi*eta*eta);
              if (name=="nonproportional")
                {
                  TensorField b;
                  b[0][0]=1000;
                  b[1][1]=-1000;
                  b-=.56*basis(1.);
                  a+=(eta+.2*xi*eta)*b;
                }
              return a;
            });
            Components result,reordered,rotated_input=input,rotated_output;
            Stats stats,reverse_stats,rotation_stats;
            stats.measure_norm_changes=true;
            model.limit_dg_transport(*limiter,input,result,zero,0.,stats);
            Components repeated;
            Stats repeat_stats;
            model.limit_dg_transport(*limiter,result,repeated,zero,0.,repeat_stats);
            auto reversed=*limiter;
            std::reverse(reversed.cells.begin(),reversed.cells.end());
            model.limit_dg_transport(reversed,input,reordered,zero,0.,reverse_stats);
            for (unsigned int i=0; i<input[0].size(); ++i)
              {
                TensorField a;
                for (unsigned int c=0; c<input.size(); ++c) a[TensorField::unrolled_to_component_indices(c)]=input[c][i];
                const auto b=rotate(a);
                for (unsigned int c=0; c<input.size(); ++c) rotated_input[c][i]=b[TensorField::unrolled_to_component_indices(c)];
              }
            auto rotated=*limiter;
            rotated.scaling_reference=rotate(limiter->scaling_reference);
            model.limit_dg_transport(rotated,rotated_input,rotated_output,zero,0.,rotation_stats);
            double order_error=0.,basis_error=0.,indicator_error=0.,theta_min=1.,theta_max=0.,idempotence_error=0.;
            for (unsigned int i=0; i<input[0].size(); ++i)
              {
                TensorField a;
                for (unsigned int c=0; c<input.size(); ++c) a[TensorField::unrolled_to_component_indices(c)]=result[c][i];
                const auto b=rotate(a);
                for (unsigned int c=0; c<input.size(); ++c)
                  {
                    order_error=std::max(order_error,std::abs(result[c][i]-reordered[c][i])/1000.);
                    idempotence_error=std::max(idempotence_error,std::abs(result[c][i]-repeated[c][i])/1000.);
                    basis_error=std::max(basis_error,std::abs(rotated_output[c][i]-b[TensorField::unrolled_to_component_indices(c)])/1000.);
                  }
              }
            for (unsigned int k=0; k<stats.last.size(); ++k)
              {
                const double theta=stats.last[k].theta;
                AssertThrow(std::isfinite(theta) && theta>=0. && theta<=1.,ExcMessage("Invalid scaling factor."));
                theta_min=std::min(theta_min,theta);
                theta_max=std::max(theta_max,theta);
                indicator_error=std::max(indicator_error,std::abs(theta-rotation_stats.last[k].theta));
              }
            const auto &n=stats.norm_changes.front();
            summary << name << ',' << n.before << ',' << n.after << ',' << n.change << ',' << stats.maximum_mean_error << ',' << order_error << ',' << basis_error << ',' << indicator_error << ',' << theta_min << ',' << theta_max << ',' << idempotence_error << '\n';
            AssertThrow(stats.maximum_mean_error<1e-12 && order_error<1e-12 && basis_error<1e-12 && indicator_error<1e-12 && idempotence_error<1e-12
                        && n.change<=128.*std::numeric_limits<double>::epsilon()*n.before,ExcMessage("Scaling invariant failed."));
            coefficients("scaling-input-"+name,input);
            coefficients("scaling-output-"+name,result);
            auto factors=output("scaling-theta-"+name+".csv");
            factors << "cell,theta\n";
            for (unsigned int k=0; k<stats.last.size(); ++k) factors << k << ',' << stats.last[k].theta << '\n';
          }
          const auto invalid=nodal(op,[](const auto &,const auto &,double,double)
          {
            return basis(1.1);
          });
          bool rejected=false;
          try
            {
              Components out;
              Stats stats;
              model.limit_dg_transport(*limiter,invalid,out,zero,0.,stats);
            }
          catch (const std::exception &)
            {
              rejected=true;
            }
          auto rejection=output("scaling-rejection.csv");
          rejection << "name,value\ninadmissible_mean_rejected," << rejected << '\n';
          AssertThrow(rejected,ExcMessage("Scaling accepted an inadmissible mean."));
        }
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("PF4 stabilization checks");
          prm.declare_entry("Time divisor","1",Patterns::Integer(1,4),"Nested diagnostic timestep divisors1/2/4.");
          prm.declare_entry("Local only","false",Patterns::Bool(),"Run scaling invariants without transport.");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("PF4 stabilization checks");
          time_divisor=prm.get_integer("Time divisor");
          local_only=prm.get_bool("Local only");
          prm.leave_subsection();
          prm.leave_subsection();
          AssertThrow(time_divisor==1 || time_divisor==2 || time_divisor==4,ExcMessage("Use nested divisors1/2/4 only."));
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
          const double width=this->get_phase_field_handler().get_length_scale();
          auto constants=output("constants.csv");
          constants << "nx,ny,relaxation_rate,time,time_divisor,width\n" << nx << ',' << ny << ",0," << final_time << ',' << time_divisor << ',' << width << '\n';
          metrics=output("metrics.csv");
          metrics << "case,nx,ny,time,dt,steps,error_L2,reference_L2,norm_L2,max_error,min_xx,max_xx,initial_error\n";
          auto costs=output("costs.csv"),physical=output("physical.csv");
          costs << "case,steps,visits,limited,mean_flagged,safeguard_only,detector_seconds,reconstruction_seconds,transport_seconds,max_mean_error,norm_measurement_seconds\n";
          physical << "case,base_step,time,norm2\n";
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          for (const std::string name:
          {"pulse_translation","signed_jump","finite_width"
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
                if (local_only)
                  {
                    local_checks(model,*op,v);
                    break;
                  }
              }
            const unsigned int base_steps=std::ceil(final_time*op->inverse_advection_time/.025),steps=base_steps*time_divisor;
            const double dt=final_time/steps;
            const Field exact=[name,width](const Point<dim> &p,double t)
            {
              if (name=="pulse_translation") return basis(pulse(p[0]-.4*t,p[1]-.2*t));
              const double x=p[0]-.4*t;
              if (name=="signed_jump") return basis(x>=.25 && x<.5?1.:(x>=.5 && x<.75?-1.:0.));
              const double a=width/(2.*std::atanh(.8));
              return basis(.5*std::tanh((x-.25)/a)-std::tanh((x-.5)/a)+.5*std::tanh((x-.75)/a));
            };
            const auto initial=model.project_dg_transport_field(*op,exact,0.,8);
            coefficients("initial-"+name,components(*initial));
            for (const std::string path:
            {"unlimited","kxrcf","fu_shu","scaling"
            })
            {
              if (name=="pulse_translation" && path!="unlimited") continue;
              auto limiter=model.make_dg_limiter(*op,v);
              limiter->detector=path=="kxrcf"?Limiter::Detector::kxrcf:(path=="scaling"?Limiter::Detector::scaling:Limiter::Detector::fu_shu);
              limiter->threshold=path=="kxrcf"?1.:.1;
              limiter->scaling_reference=basis(1.);
              Stats stats;
              stats.measure_norm_changes=true;
              stats.capture_first_call=nx==32 && time_divisor==1 && name!="pulse_translation";
              auto state=initial;
              const std::string label=name+'-'+path;
              physical << label << ",0,0," << quadratic_norm(*op,components(*state)) << '\n';
              double elapsed=0.;
              const std::vector<unsigned int> common= {(base_steps+3)/4,(base_steps+1)/2,(3*base_steps+3)/4,base_steps};
              for (unsigned int step=0; step<steps; ++step)
                {
                  const auto begin=std::chrono::steady_clock::now();
                  state=model.advance_dg_transport(*op,*state,exact,zero,step*dt,dt,path=="unlimited"?nullptr:limiter.get(),&stats);
                  elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                  if ((step+1)%time_divisor==0 && std::find(common.begin(),common.end(),(step+1)/time_divisor)!=common.end())
                    {
                      const unsigned int base_step=(step+1)/time_divisor;
                      physical << label << ',' << base_step << ',' << (step+1)*dt << ',' << quadratic_norm(*op,components(*state)) << '\n';
                      if (step+1<steps) coefficients("common-"+label+'-'+std::to_string(base_step),components(*state));
                    }
                }
              record(model,label,*state,exact,steps);
              costs << label << ',' << steps << ',' << stats.visits << ',' << stats.limited << ',' << stats.mean_flagged << ',' << stats.safeguard_only << ',' << stats.detector_seconds << ',' << stats.reconstruction_seconds << ',' << elapsed << ',' << stats.maximum_mean_error << ',' << stats.norm_measurement_seconds << '\n';
              if (path!="unlimited")
                {
                  auto calls=output("norm-calls-"+label+".csv");
                  calls << "call,stage_time,before,after,change,positive_cells,negative_cells,means_outside_envelope,maximum_mean_excess\n";
                  for (unsigned int k=0; k<stats.norm_changes.size(); ++k)
                    {
                      const auto &a=stats.norm_changes[k];
                      calls << k << ',' << a.time << ',' << a.before << ',' << a.after << ',' << a.change << ',' << a.positive_cells << ',' << a.negative_cells << ',' << a.means_outside_envelope << ',' << a.maximum_mean_excess << '\n';
                    }
                  if (stats.capture_first_call)
                    {
                      coefficients("call-input-"+label,stats.first_incoming);
                      coefficients("call-output-"+label,stats.first_outgoing);
                    }
                  if (nx==32 && time_divisor==4)
                    {
                      // Named map diagnostic: fixed physical time and detector
                      // velocity; no transport and no physical-step commit.
                      auto held=components(*state);
                      auto history=output("held-"+label+".csv");
                      history << "pass,norm2,change,limited\n0," << quadratic_norm(*op,held) << ",0,0\n";
                      for (unsigned int pass=1; pass<=12; ++pass)
                        {
                          Components next;
                          Stats held_stats;
                          held_stats.measure_norm_changes=true;
                          model.limit_dg_transport(*limiter,held,next,exact,final_time,held_stats);
                          held=std::move(next);
                          history << pass << ',' << held_stats.norm_changes.front().after << ',' << held_stats.norm_changes.front().change << ',' << held_stats.limited << '\n';
                          if (pass==1 || pass==12) coefficients("held-field-"+label+'-'+std::to_string(pass),held);
                        }
                    }
                }
            }
          }
          const bool unchanged=anchor==Access::dg_history_values(model) && Z_anchor==model.fe_strain_state->Z && I_anchor==model.fe_strain_state->I;
          auto accepted=output("accepted.csv");
          accepted << "name,value\naccepted_unchanged," << unchanged << '\n';
          BV phase_change(phase_anchor);
          phase_change-=model.accepted_fe_strain_state();
          AssertThrow(unchanged && phase_change.l2_norm()==0.,ExcMessage("Stabilization published accepted state."));
          return {"PF4 stabilization checks","completed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF4 stabilization checks","Bounded timestep, direct norm, scaling and finite-width diagnostics.")
  }
}
