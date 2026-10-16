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
        void operator_checks(Model &model)
        {
          const auto v=velocity([](const Point<dim> &p)
          {
            Tensor<1,dim> v;
            v[0]=.2+.1*p[0]*p[0]*p[1];
            v[1]=-.15+.05*p[0]*p[1]*p[1];
            return v;
          });
          const auto op=model.make_dg_transport(v,4,false);
          auto matrix=output("operator.csv"),loads=output("operator-loads.csv");
          matrix << "i,j,value,divergence_correction\n";
          for (unsigned int i=0; i<op->matrix.m(); ++i)
            for (auto e=op->matrix.begin(i); e!=op->matrix.end(i); ++e)
              matrix << i << ',' << e->column() << ',' << e->value() << ',' << op->divergence_correction.el(i,e->column()) << '\n';
          loads << "kind,x,y,i,weight\n";
          for (unsigned int kind=0; kind<2; ++kind)
            for (const auto &load:kind?op->inflow:op->source)
              for (unsigned int i=0; i<load.indices.size(); ++i)
                loads << kind << ',' << load.point[0] << ',' << load.point[1] << ',' << load.indices[i] << ',' << load.weights[i] << '\n';
          auto controls=output("lifecycle.csv");
          controls << "name,value\n";
          const Field constant=[](const Point<dim> &,double)
          {
            return basis(1.);
          };
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          const auto incoming=model.project_dg_transport_field(*op,constant,0.,8);
          const auto a=model.advance_dg_transport(*op,*incoming,constant,zero,0.,.001);
          const auto b=model.advance_dg_transport(*op,*incoming,constant,zero,0.,.001);
          controls << "constant_error," << difference(model,*a,*incoming) << "\nrepeat_error," << difference(model,*a,*b) << '\n';
          const auto snapshot=incoming->trial_stress;
          bool rejected=false;
          try
            {
              model.advance_dg_transport(*op,*incoming,constant,[](const Point<dim> &,double)
              {
                return basis(std::numeric_limits<double>::quiet_NaN());
              },0.,.001);
            }
          catch (const std::exception &)
            {
              rejected=true;
            }
          bool unchanged=true;
          for (unsigned int c=0; c<snapshot.size(); ++c) unchanged &= snapshot[c]==incoming->trial_stress[c];
          controls << "invalid_source_rejected," << rejected << "\nincoming_unchanged," << unchanged << '\n';
          AssertThrow(rejected && unchanged && difference(model,*a,*b)==0.,ExcMessage("PF4 trial lifecycle failed."));
          rejected=false;
          try
            {
              model.advance_dg_transport(*op,*incoming,constant,zero,0.,10.);
            }
          catch (const std::exception &)
            {
              rejected=true;
            }
          controls << "unsafe_dt_rejected," << rejected << '\n';
          AssertThrow(rejected,ExcMessage("Unsafe explicit diagnostic CFL was admitted."));
        }
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &model=Plugins::get_plugin_as_type<Model>(*sim.material_model);
          AssertThrow(dim==2 && model.uses_dg_trial_stress_history() && this->get_timestep_number()==0,
                      ExcMessage("PF4 requires a fresh serial DG fixture."));
          model.prepare_reconstructed_fault_mechanical_solve();
          const auto anchor=Access::dg_history_values(model);
          const BV phase_anchor(model.accepted_fe_strain_state());
          const auto Z_anchor=model.fe_strain_state->Z,I_anchor=model.fe_strain_state->I;
          const auto cell=model.fe_strain_state->cells.begin()->second;
          nx=std::lround(1/(cell->vertex(1)[0]-cell->vertex(0)[0]));
          ny=std::lround(1/(cell->vertex(2)[1]-cell->vertex(0)[1]));
          const double rate=model.fe_strain_state->G/model.fe_strain_state->eta;
          auto constants=output("constants.csv");
          constants << "nx,ny,relaxation_rate,time\n" << nx << ',' << ny << ',' << rate << ',' << final_time << '\n';
          metrics=output("metrics.csv");
          metrics << "case,nx,ny,time,dt,steps,error_L2,reference_L2,norm_L2,max_error,min_xx,max_xx,initial_error\n";
          const Field zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          const auto translate_velocity=velocity([](const Point<dim> &)
          {
            Tensor<1,dim> v;
            v[0]=.4;
            v[1]=.2;
            return v;
          });
          const auto op=model.make_dg_transport(translate_velocity,4,false);
          export_geometry(*op);
          const auto steps=static_cast<unsigned int>(std::ceil(final_time*op->inverse_advection_time/.025));
          const Field translated=[](const Point<dim> &p,double t)
          {
            return basis(pulse(p[0]-.4*t,p[1]-.2*t));
          };
          const auto pulse_initial=model.project_dg_transport_field(*op,translated,0.,8);
          {
            auto initial=output("initial-pulse.csv");
            initial << "i,xx,yy,xy\n";
            for (unsigned int i=0; i<pulse_initial->space->dofs.n_dofs(); ++i)
              initial << i << ',' << pulse_initial->trial_stress[0][i] << ',' << pulse_initial->trial_stress[1][i] << ',' << pulse_initial->trial_stress[2][i] << '\n';
          }
          const auto translated_final=evolve(model,*op,pulse_initial,translated,zero,steps);
          record(model,"pulse_translation",*translated_final,translated,steps);
          if (nx==32)
            record(model,"pulse_translation_half_dt",*evolve(model,*op,pulse_initial,translated,zero,2*steps),translated,2*steps);
          const auto diverging_velocity=velocity([](const Point<dim> &p)
          {
            Tensor<1,dim> v;
            v[0]=.2+.3*p[0];
            v[1]=-.1*(p[1]-.5);
            return v;
          });
          const auto divergent=model.make_dg_transport(diverging_velocity,4,false);
          const auto divergence_steps=static_cast<unsigned int>(std::ceil(final_time*divergent->inverse_advection_time/.025));
          const Field expanded=[](const Point<dim> &p,double t)
          {
            return basis(pulse((p[0]+2./3.)*std::exp(-.3*t)-2./3.,.5+(p[1]-.5)*std::exp(.1*t)));
          };
          const auto expanded_initial=model.project_dg_transport_field(*divergent,expanded,0.,8);
          record(model,"pulse_divergence",*evolve(model,*divergent,expanded_initial,expanded,zero,divergence_steps),expanded,divergence_steps);
          const Field constant=[](const Point<dim> &,double)
          {
            return basis(1.);
          };
          const auto constant_initial=model.project_dg_transport_field(*divergent,constant,0.,8);
          record(model,"constant_divergence",*evolve(model,*divergent,constant_initial,constant,zero,divergence_steps),constant,divergence_steps);
          const auto density=model.make_dg_transport(diverging_velocity,4,false);
          density->matrix.add(-1.,density->divergence_correction);
          const Field density_inflow=[](const Point<dim> &,double t)
          {
            return basis(std::exp(-.2*t));
          };
          record(model,"density_negative",*evolve(model,*density,constant_initial,density_inflow,zero,divergence_steps),constant,divergence_steps);
          const auto jump_velocity=velocity([](const Point<dim> &)
          {
            Tensor<1,dim> v;
            v[0]=.4;
            return v;
          });
          const auto jump_op=model.make_dg_transport(jump_velocity,4,false);
          const unsigned int jump_steps=std::ceil(final_time*jump_op->inverse_advection_time/.025);
          const Field jump=[](const Point<dim> &p,double t)
          {
            const double x=p[0]-.4*t;
            return basis(x>=.25 && x<.5?1.:(x>=.5 && x<.75?-1.:0.));
          };
          const auto jump_initial=model.project_dg_transport_field(*jump_op,jump,0.,8);
          record(model,"signed_jump",*evolve(model,*jump_op,jump_initial,jump,zero,jump_steps),jump,jump_steps);
          const auto normal_velocity=velocity([](const Point<dim> &)
          {
            Tensor<1,dim> v;
            v[1]=.25;
            return v;
          });
          const auto normal=model.make_dg_transport(normal_velocity,4,true);
          const unsigned int normal_steps=std::ceil(final_time*(normal->inverse_advection_time+rate)/.025);
          const Field normal_cohesive=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1]));
          };
          const Field normal_direct=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1]-.25*t));
          };
          const Field normal_source=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*.25*chi_prime(p[1]));
          };
          const auto normal_initial=model.project_dg_transport_field(*normal,normal_cohesive,0.,8);
          const auto normal_evolved=evolve(model,*normal,normal_initial,normal_cohesive,normal_source,normal_steps);
          record(model,"normal_cohesive",*normal_evolved,normal_cohesive,normal_steps);
          const auto normal_mature=evolve(model,*normal,normal_initial,normal_direct,zero,normal_steps);
          record(model,"normal_mature",*normal_mature,normal_direct,normal_steps);
          record(model,"normal_omitted",*normal_mature,normal_cohesive,normal_steps);
          const Field normal_residual=[normal_cohesive,normal_direct](const Point<dim> &p,double t)
          {
            return normal_cohesive(p,t)-normal_direct(p,t);
          };
          auto residual=evolve(model,*normal,model.project_dg_transport_field(*normal,zero,0.,8),normal_residual,normal_source,normal_steps);
          auto reconstructed=model.project_dg_transport_field(*normal,normal_cohesive,final_time,8);
          for (unsigned int c=0; c<reconstructed->trial_stress.size(); ++c) reconstructed->trial_stress[c]-=residual->trial_stress[c];
          record(model,"normal_residual",*reconstructed,normal_direct,normal_steps);
          const auto tangent_velocity=velocity([](const Point<dim> &)
          {
            Tensor<1,dim> v;
            v[0]=.35;
            return v;
          });
          const auto tangent=model.make_dg_transport(tangent_velocity,4,true);
          const unsigned int tangent_steps=std::ceil(final_time*(tangent->inverse_advection_time+rate)/.025);
          const Field tangent_cohesive=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1])*slip(p[0]-.1*t));
          };
          const Field tangent_direct=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1])*slip(p[0]-.35*t));
          };
          const Field tangent_source=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1])*.25*.25*2*numbers::PI*std::cos(2*numbers::PI*(p[0]-.1*t)));
          };
          const auto tangent_initial=model.project_dg_transport_field(*tangent,tangent_cohesive,0.,8);
          record(model,"tangent_cohesive",*evolve(model,*tangent,tangent_initial,tangent_cohesive,tangent_source,tangent_steps),tangent_cohesive,tangent_steps);
          const auto tangent_mature=evolve(model,*tangent,tangent_initial,tangent_direct,zero,tangent_steps);
          record(model,"tangent_mature",*tangent_mature,tangent_direct,tangent_steps);
          record(model,"tangent_omitted",*tangent_mature,tangent_cohesive,tangent_steps);
          const Field tangent_residual=[tangent_cohesive,tangent_direct](const Point<dim> &p,double t)
          {
            return tangent_cohesive(p,t)-tangent_direct(p,t);
          };
          residual=evolve(model,*tangent,model.project_dg_transport_field(*tangent,zero,0.,8),tangent_residual,tangent_source,tangent_steps);
          reconstructed=model.project_dg_transport_field(*tangent,tangent_cohesive,final_time,8);
          for (unsigned int c=0; c<reconstructed->trial_stress.size(); ++c) reconstructed->trial_stress[c]-=residual->trial_stress[c];
          record(model,"tangent_residual",*reconstructed,tangent_direct,tangent_steps);
          const auto moving_velocity=velocity([](const Point<dim> &)
          {
            Tensor<1,dim> v;
            v[0]=.2;
            v[1]=.25;
            return v;
          });
          const auto moving=model.make_dg_transport(moving_velocity,4,true);
          const unsigned int moving_steps=std::ceil(final_time*(moving->inverse_advection_time+rate)/.025);
          const Field comoving=[rate](const Point<dim> &p,double t)
          {
            return shear(-5000*std::exp(-rate*t)*chi(p[1]-.25*t)*slip(p[0]-.2*t));
          };
          const auto moving_initial=model.project_dg_transport_field(*moving,comoving,0.,8);
          record(model,"comoving",*evolve(model,*moving,moving_initial,comoving,zero,moving_steps),comoving,moving_steps);
          if (nx==4) operator_checks(model);
          if (nx==8)
            {
              auto temporal=output("temporal.csv");
              temporal << "steps,difference_from_1024,reference_L2_error\n";
              const auto reference=evolve(model,*op,pulse_initial,translated,zero,1024);
              for (const unsigned int n:
              {
                16u,32u,64u,128u,512u
              })
              {
                const auto current=evolve(model,*op,pulse_initial,translated,zero,n);
                temporal << n << ',' << difference(model,*current,*reference) << ",0\n";
                record(model,"temporal_"+std::to_string(n),*current,translated,n);
              }
              record(model,"temporal_reference",*reference,translated,1024);
              for (const unsigned int q:
              {
                8u,12u
              })
              {
                const auto rule=model.make_dg_transport(normal_velocity,q,true);
                record(model,"normal_quad"+std::to_string(q),*evolve(model,*rule,normal_initial,normal_cohesive,normal_source,normal_steps),normal_cohesive,normal_steps);
              }
              const auto frozen_velocity=velocity([](const Point<dim> &)
              {
                return Tensor<1,dim>();
              });
              const auto frozen=model.make_dg_transport(frozen_velocity,4,true);
              const Field frozen_exact=[rate](const Point<dim> &,double t)
              {
                return basis(std::exp(-rate*t));
              };
              const auto frozen_initial=model.project_dg_transport_field(*frozen,frozen_exact,0.,8);
              record(model,"frozen",*evolve(model,*frozen,frozen_initial,frozen_exact,zero,16),frozen_exact,16);
              const auto rotation_velocity=velocity([](const Point<dim> &p)
              {
                Tensor<1,dim> v;
                v[0]=-(p[1]-.5);
                v[1]=p[0]-.5;
                return v;
              });
              const auto rotation=model.make_dg_transport(rotation_velocity,4,false);
              const unsigned int rotation_steps=std::ceil(final_time*rotation->inverse_advection_time/.025);
              const auto rotation_initial=model.project_dg_transport_field(*rotation,constant,0.,8);
              record(model,"rotation_nonobjective",*evolve(model,*rotation,rotation_initial,constant,zero,rotation_steps),constant,rotation_steps);
              const Field rotated=[](const Point<dim> &,double t)
              {
                TensorField a;
                a[0][0]=300+700*std::cos(2*t)-700*std::sin(2*t);
                a[1][1]=600-a[0][0];
                a[0][1]=700*(std::sin(2*t)+std::cos(2*t));
                return a;
              };
              const Field rotation_source=[](const Point<dim> &,double t)
              {
                TensorField a;
                a[0][0]=-1400*(std::sin(2*t)+std::cos(2*t));
                a[1][1]=-a[0][0];
                a[0][1]=1400*(std::cos(2*t)-std::sin(2*t));
                return a;
              };
              record(model,"rotation_objective_forced",*evolve(model,*rotation,rotation_initial,rotated,rotation_source,rotation_steps),rotated,rotation_steps);
            }
          if (nx>=8)
            {
              const auto rotation_velocity=velocity([](const Point<dim> &p)
              {
                Tensor<1,dim> v;
                v[0]=-(p[1]-.5);
                v[1]=p[0]-.5;
                return v;
              });
              const auto rotation=model.make_dg_transport(rotation_velocity,4,false);
              const unsigned int rotation_steps=std::ceil(final_time*rotation->inverse_advection_time/.025);
              const auto rotated_profile=[](const Point<dim> &p,const double t)
              {
                return chi(.5-std::sin(t)*(p[0]-.5)+std::cos(t)*(p[1]-.5));
              };
              const Field nonobjective=[rotated_profile](const Point<dim> &p,const double t)
              {
                return shear(1000*rotated_profile(p,t));
              };
              const Field objective=[rotated_profile](const Point<dim> &p,const double t)
              {
                const double value=1000*rotated_profile(p,t);
                TensorField a;
                a[0][0]=-value*std::sin(2*t);
                a[1][1]=-a[0][0];
                a[0][1]=value*std::cos(2*t);
                return a;
              };
              const Field objective_source=[rotated_profile](const Point<dim> &p,const double t)
              {
                const double value=2000*rotated_profile(p,t);
                TensorField a;
                a[0][0]=-value*std::cos(2*t);
                a[1][1]=-a[0][0];
                a[0][1]=-value*std::sin(2*t);
                return a;
              };
              const auto initial=model.project_dg_transport_field(*rotation,nonobjective,0.,8);
              const auto nonobjective_final=evolve(model,*rotation,initial,nonobjective,zero,rotation_steps);
              record(model,"rotation_profile_nonobjective",*nonobjective_final,nonobjective,rotation_steps);
              record(model,"rotation_profile_omitted",*nonobjective_final,objective,rotation_steps);
              record(model,"rotation_profile_objective_forced",
                     *evolve(model,*rotation,initial,objective,objective_source,rotation_steps),objective,rotation_steps);
            }
          AssertThrow(Access::dg_history_values(model)==anchor && model.accepted_fe_strain_state()==phase_anchor
                      && model.fe_strain_state->Z==Z_anchor && model.fe_strain_state->I==I_anchor,
                      ExcMessage("PF4 transport trials changed accepted bulk/phase/surface history."));
          auto controls=output("accepted.csv");
          controls << "accepted_history_unchanged\n1\n";
          this->get_pcout()<<"   PF4 prescribed material transport completed without accepted-state publication or limiter."<<std::endl;
          return {"PF4 DG transport:","completed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF4 transport checks",
                                  "Prescribed DG material-transport and commutator controls; no coupled advancement or publication.")
  }
}
