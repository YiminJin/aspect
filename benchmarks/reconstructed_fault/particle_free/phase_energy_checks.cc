/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/particle/particle_domain.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include "../../../tests/phase_field_test_access.h"

#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace aspect
{
  namespace Postprocess
  {
    // Loaded separately from PF2's observer: reuse its existing test friendship,
    // not an additional production hook or a coupled time integrator.
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        using PhaseAccess=internal::PhaseFieldTestAccess<dim>;
        using BV=LinearAlgebra::BlockVector;
        unsigned int block;

        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+name);
          out << std::setprecision(17);
          return out;
        }

        void set_phase(BV &state,const std::function<double(const Point<dim> &)> &function) const
        {
          BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=state;
          const auto component=this->introspection().variable("phase_field").first_component_index;
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (const auto v:cell->vertex_indices())
                owned[cell->vertex_dof_index(v,component)]=function(cell->vertex(v));
          owned.compress(VectorOperation::insert);
          state=owned;
        }

        double assemble(Simulator<dim> &sim,const BV &state,const bool tangent) const
        {
          double energy=0.;
          AssertThrow(PhaseAccess::assemble(*sim.phase_field_handler,sim.system_matrix,sim.system_rhs,
                                            state,tangent,nullptr,&energy),ExcMessage("PF3 trial is undefined."));
          return energy;
        }

        void export_reference(Model &model,const BV &phase,const BV &d1,const BV &d2) const
        {
          const auto &s=*model.fe_strain_state;
          const auto component=this->introspection().variable("phase_field").first_component_index;
          const auto offset=phase.get_block_indices().block_start(block);
          std::map<types::global_dof_index,Point<dim>> nodes;
          auto cells=output("cells.csv"),tensor=output("tensor.csv");
          cells << "cell,x0,y0,dx,dy,i0,i1,i2,i3\n";
          tensor << "cell,r,s,xx,yy,xy\n";
          const QGaussLobatto<dim> samples(3);
          unsigned int c=0;
          for (const auto &entry:s.cells)
            {
              const auto &cell=entry.second;
              const auto stress=model.evaluate_dg_trial_tensor(*s.dg_history,entry.first,samples);
              cells << c << ',' << cell->vertex(0)[0] << ',' << cell->vertex(0)[1] << ','
                    << cell->vertex(1)[0]-cell->vertex(0)[0] << ',' << cell->vertex(2)[1]-cell->vertex(0)[1];
              for (const auto v:cell->vertex_indices())
                {
                  const auto i=cell->vertex_dof_index(v,component)-offset;
                  nodes[i]=cell->vertex(v);
                  cells << ',' << i;
                }
              cells << '\n';
              for (unsigned int q=0; q<samples.size(); ++q)
                tensor << c << ',' << samples.point(q)[0] << ',' << samples.point(q)[1] << ','
                       << stress[q][0][0] << ',' << stress[q][1][1] << ',' << stress[q][0][1] << '\n';
              ++c;
            }
          auto nodal=output("nodes.csv");
          nodal << "i,x,y,accepted,phase,d1,d2\n";
          for (const auto &node:nodes)
            nodal << node.first << ',' << node.second[0] << ',' << node.second[1] << ','
                  << s.displacement_history.block(block)[node.first] << ',' << phase.block(block)[node.first] << ','
                  << d1.block(block)[node.first] << ',' << d2.block(block)[node.first] << '\n';
          auto surface=output("surface.csv"),profiles=output("profiles.csv"),rule=output("rule.csv");
          surface << "i,x,Z,I_accepted\n";
          for (unsigned int i=0; i<s.Z.size(); ++i)
            surface << i << ',' << s.vertices[i][0] << ',' << s.Z[i] << ',' << s.I[i] << '\n';
          profiles << "segment,xi,weight\n";
          for (const auto &p:s.profiles) profiles << p.segment << ',' << p.xi << ',' << p.weight << '\n';
          rule << "r,s,weight\n";
          const auto quadrature=model.mechanical_quadrature();
          for (unsigned int q=0; q<quadrature.size(); ++q)
            rule << quadrature.point(q)[0] << ',' << quadrature.point(q)[1] << ',' << quadrature.weight(q) << '\n';
          auto domains=output("domains.csv");
          domains << "particle,volume,i,w,gx,gy\n";
          const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          const auto &vertex_dofs=PhaseAccess::vertex_dofs(this->get_phase_field_handler());
          for (const auto &particle:pm.get_particle_handler())
            {
              const auto domain=pm.get_particle_domain_handler().get_particle_domain(particle.get_local_index());
              for (unsigned int j=0; j<domain.n_relevant_vertices(); ++j)
                domains << particle.get_id() << ',' << domain.volume() << ','
                        << vertex_dofs[domain.relevant_vertex_index(j)]-offset << ','
                        << domain.weighting_function_value(j) << ','
                        << domain.weighting_function_gradient(j)[0] << ','
                        << domain.weighting_function_gradient(j)[1] << '\n';
            }
          const auto coefficient=PhaseAccess::coefficients(this->get_phase_field_handler(), {.5,.5},.3,0.);
          auto constants=output("constants.csv");
          constants << "G,m,A,ell\n" << s.G << ',' << s.m << ',' << coefficient.reaction << ','
                    << std::sqrt(coefficient.diffusion/(2*coefficient.reaction)) << '\n';
        }

        void quadrature_checks(Model &model,const BV &interior,const BV &sharp)
        {
          const auto original=model.fe_strain_state;
          const auto original_order=model.fe_bulk_normal_quadrature_points;
          auto table=output("quadrature.csv"),loads=output("quadrature-loads.csv");
          table << "profile,order,energy,bulk,cohesive,fracture,I_min,I_max\n";
          loads << "profile,order,i,load\n";
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          const auto offset=interior.get_block_indices().block_start(block);
          for (const unsigned int order:
          {
            3u,6u,12u,24u,48u,96u
          })
          {
            // A scratch integration geometry shares immutable DG/phase/Z.
            // It is restored before any native mechanical operation.
            model.fe_bulk_normal_quadrature_points=order;
            auto scratch=std::make_shared<typename Model::FEStrainState>(*original);
            scratch->bulk.clear();
            model.fe_strain_state=scratch;
            const auto quad=model.mechanical_quadrature();
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,
                             update_values|update_quadrature_points|update_JxW_values);
            std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
            for (const auto &entry:scratch->cells)
              {
                fe.reinit(entry.second);
                entry.second->get_dof_indices(ids);
                for (unsigned int q=0; q<quad.size(); ++q)
                  {
                    typename Model::FEStrainState::BulkPoint p;
                    p.cell=entry.first;
                    p.x=fe.quadrature_point(q);
                    p.weight=fe.JxW(q);
                    p.segment=std::min<unsigned int>(scratch->Z.size()-2,p.x[0]*(scratch->Z.size()-1));
                    p.xi=p.x[0]*(scratch->Z.size()-1)-p.segment;
                    for (unsigned int v=0; v<p.dofs.size(); ++v)
                      {
                        const auto j=this->get_fe().component_to_system_index(pc,v);
                        p.dofs[v]=ids[j]-offset;
                        p.shape[v]=fe.shape_value_component(j,q,pc);
                      }
                    scratch->bulk.push_back(p);
                  }
              }
            for (unsigned int profile=0; profile<2; ++profile)
              {
                const auto &phase=profile?sharp:interior;
                const auto elastic=model.evaluate_fe_phase_energy(phase,false);
                const auto fracture=this->get_phase_field_handler().evaluate_cell_fracture(phase,false);
                AssertThrow(elastic.admissible,ExcMessage("Quadrature profile is inadmissible."));
                const auto &I=elastic.normalization;
                table << profile << ',' << order << ',' << elastic.energy+fracture.energy << ','
                      << elastic.bulk << ',' << elastic.cohesive << ',' << fracture.energy << ','
                      << *std::min_element(I.begin(),I.end()) << ',' << *std::max_element(I.begin(),I.end()) << '\n';
                std::vector<double> load(scratch->Z.size(),0.);
                std::vector<SymmetricTensor<2,dim>> tensors;
                for (const auto &cell:scratch->cells)
                  {
                    const auto values=model.evaluate_dg_trial_tensor(*scratch->dg_history,cell.first,quad);
                    tensors.insert(tensors.end(),values.begin(),values.end());
                  }
                unsigned int point=0;
                for (const auto &p:scratch->bulk)
                  {
                    double phi=0.;
                    for (unsigned int v=0; v<p.dofs.size(); ++v) phi+=p.shape[v]*phase.block(block)[p.dofs[v]];
                    const double normalization=(1-p.xi)*I[p.segment]+p.xi*I[p.segment+1];
                    const double chi=scratch->m*phi*(1+phi)/std::pow(1-phi,2)/normalization;
                    const double Z=(1-p.xi)*scratch->Z[p.segment]+p.xi*scratch->Z[p.segment+1];
                    const double traction=chi*(tensors[point++][0][1]-Z*chi*scratch->shear_sense)*scratch->shear_sense;
                    load[p.segment]+=p.weight*(1-p.xi)*traction;
                    load[p.segment+1]+=p.weight*p.xi*traction;
                  }
                for (unsigned int i=0; i<load.size(); ++i) loads << profile << ',' << order << ',' << i << ',' << load[i] << '\n';
              }
          }
          model.fe_bulk_normal_quadrature_points=original_order;
          model.fe_strain_state=original;
        }

      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &model=Plugins::get_plugin_as_type<Model>(*sim.material_model);
          AssertThrow(dim==2 && model.uses_dg_trial_stress_history() && this->get_timestep_number()==0,
                      ExcMessage("PF3 requires the fresh restricted DG fixture."));
          block=this->introspection().variable("phase_field").block_index;
          const auto old_time=sim.time,old_dt=sim.time_step;
          model.prepare_reconstructed_fault_mechanical_solve();
          for (unsigned int step=1; step<=2; ++step)
            {
              sim.timestep_number=step;
              sim.time_step=.125;
              sim.time=.125*step;
              sim.old_solution=sim.solution;
              sim.current_linearization_point=sim.solution;
              sim.solve_reconstructed_fault_stokes();
            }
          const auto &s=*model.fe_strain_state;
          const BV anchor(model.accepted_fe_strain_state());
          const auto dg_anchor=Access::dg_history_values(model),Z_anchor=s.Z,I_anchor=s.I;
          const auto surface_anchor=Access::current_normalization_integrals(model);
          BV phase(sim.solution),d1(sim.solution),d2(sim.solution),sharp(sim.solution);
          set_phase(phase,[](const Point<dim> &x)
          {
            return (.18+.45*std::exp(-std::pow((x[1]-.5)/.18,2)))*(1+.12*std::cos(2*numbers::PI*x[0]));
          });
          set_phase(d1,[](const Point<dim> &x)
          {
            return .7*std::sin(2.3+4*x[0]-3*x[1])+.3*std::cos(7*x[1]);
          });
          set_phase(d2,[](const Point<dim> &x)
          {
            return std::cos(5*x[0]+2*x[1]);
          });
          set_phase(sharp,[](const Point<dim> &x)
          {
            return .05+.945*std::max(0.,1-std::abs(x[1]-.5)/.0625);
          });
          export_reference(model,phase,d1,d2);
          const double energy=assemble(sim,phase,true);
          const BV rhs(sim.system_rhs);
          const auto elastic=model.evaluate_fe_phase_energy(phase,true);
          const auto fracture=this->get_phase_field_handler().evaluate_cell_fracture(phase,true);
          auto summary=output("energy.csv");
          summary << "total,bulk,cohesive,fracture,cpdi_fracture\n";
          double cpdi=0.,arithmetic=0.;
          AssertThrow(this->get_phase_field_handler().evaluate_fe_fracture(sim.system_matrix,sim.system_rhs,phase,cpdi,arithmetic),ExcInternalError());
          summary << energy << ',' << elastic.bulk << ',' << elastic.cohesive << ',' << fracture.energy << ',' << cpdi << '\n';
          auto gradient=output("gradient.csv"),hessian=output("hessian.csv"),normal=output("normalization.csv");
          gradient << "i,total,elastic,fracture\n";
          for (unsigned int i=0; i<elastic.gradient.size(); ++i)
            {
              gradient << i << ',' << -rhs.block(block)[i] << ',' << elastic.gradient[i] << ',' << fracture.gradient[i] << '\n';
              for (unsigned int j=0; j<elastic.gradient.size(); ++j)
                hessian << (j?",":"") << elastic.hessian(i,j)+fracture.hessian(i,j);
              hessian << '\n';
            }
          const auto normalization=model.evaluate_fe_normalization(phase,true);
          normal << "i,I,d1,d2\n";
          for (unsigned int i=0; i<normalization.values.size(); ++i)
            {
              double a=0.,b=0.;
              for (unsigned int j=0; j<elastic.gradient.size(); ++j)
                {
                  a+=normalization.derivative(i,j)*d1.block(block)[j];
                  b+=normalization.derivative(i,j)*d2.block(block)[j];
                }
              normal << i << ',' << normalization.values[i] << ',' << a << ',' << b << '\n';
            }
          auto differences=output("differences.csv");
          differences << "direction,epsilon,energy_difference,gradient_action,hessian_error,hessian_scale\n";
          for (unsigned int direction=0; direction<2; ++direction)
            {
              const auto &d=direction?d2:d1;
              double derivative=0.;
              std::vector<double> action(elastic.gradient.size(),0.);
              for (unsigned int i=0; i<action.size(); ++i)
                {
                  derivative-=rhs.block(block)[i]*d.block(block)[i];
                  for (unsigned int j=0; j<action.size(); ++j)
                    action[i]+=(elastic.hessian(i,j)+fracture.hessian(i,j))*d.block(block)[j];
                }
              for (const double epsilon:
              {
                1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8
              })
              {
                BV plus(phase),minus(phase);
                plus.block(block).add(epsilon,d.block(block));
                minus.block(block).add(-epsilon,d.block(block));
                const double ep=assemble(sim,plus,false);
                const BV rp(sim.system_rhs);
                const double em=assemble(sim,minus,false);
                const BV rm(sim.system_rhs);
                double error=0.,scale=1.;
                for (unsigned int i=0; i<action.size(); ++i)
                  {
                    error=std::max(error,std::abs((-rp.block(block)[i]+rm.block(block)[i])/(2*epsilon)-action[i]));
                    scale=std::max(scale,std::abs(action[i]));
                  }
                differences << direction << ',' << epsilon << ',' << (ep-em)/(2*epsilon) << ','
                            << derivative << ',' << error << ',' << scale << '\n';
              }
            }
          // Inputs ignored by this operator may be invalid; restore them before
          // any operation that legitimately validates particle material history.
          auto &particles=sim.phase_field_handler->get_associated_particle_manager().get_particle_handler();
          std::vector<std::vector<double>> saved;
          for (auto &p:particles)
            {
              saved.emplace_back(p.get_properties().begin(),p.get_properties().end());
              for (auto &value:p.get_properties()) value=std::numeric_limits<double>::quiet_NaN();
            }
          const double poisoned_energy=assemble(sim,phase,true);
          bool particle_independent=poisoned_energy==energy;
          for (unsigned int i=0; i<elastic.gradient.size(); ++i)
            {
              particle_independent &= sim.system_rhs.block(block)[i]==rhs.block(block)[i];
              for (unsigned int j=0; j<elastic.gradient.size(); ++j)
                particle_independent &= sim.system_matrix.block(block,block).el(i,j)==elastic.hessian(i,j)+fracture.hessian(i,j);
            }
          unsigned int pi=0;
          for (auto &p:particles)
            {
              std::copy(saved[pi].begin(),saved[pi].end(),p.get_properties().begin());
              ++pi;
            }
          AssertThrow(particle_independent,ExcMessage("PF3 phase operator consumed particle material/history values."));
          auto controls=output("controls.csv");
          controls << "name,value\nparticle_properties_independent," << particle_independent << '\n';
          for (const double constant:
          {
            0.,1.,1.01,-.01,.999999
          })
          {
            BV invalid(phase);
            set_phase(invalid,[constant](const Point<dim> &)
            {
              return constant;
            });
            const auto e=model.evaluate_fe_phase_energy(invalid,false);
            controls << "constant_" << constant << "_admissible," << e.admissible << '\n';
            AssertThrow(e.admissible==(constant==.999999),ExcMessage("Incorrect localization admissibility."));
          }
          quadrature_checks(model,phase,sharp);
          BV bounded(sim.solution);
          const double incoming_energy=assemble(sim,bounded,false);
          const BV lower(bounded);
          PhaseAccess::bounded(*sim.phase_field_handler,sim.system_matrix,sim.system_rhs,bounded,&lower);
          const double final_energy=assemble(sim,bounded,true);
          double kkt=0.,growth=0.,healing=0.,redistribution_error=0.,redistribution=0.;
          unsigned int contacts=0;
          for (unsigned int i=0; i<bounded.block(block).size(); ++i)
            {
              const double delta=bounded.block(block)[i]-lower.block(block)[i];
              growth=std::max(growth,delta);
              healing=std::max(healing,-delta);
              const double r=sim.system_rhs.block(block)[i];
              const bool contact=delta==0.;
              contacts+=contact;
              kkt=std::max(kkt,std::abs(contact?std::max(r,0.):r));
            }
          const auto updated=model.evaluate_fe_normalization(bounded,false);
          auto bounded_hessian=output("bounded-hessian.csv");
          for (unsigned int i=0; i<bounded.block(block).size(); ++i)
            {
              for (unsigned int j=0; j<bounded.block(block).size(); ++j)
                bounded_hessian << (j?",":"") << sim.system_matrix.block(block,block).el(i,j);
              bounded_hessian << '\n';
            }
          auto accepted_trial=output("bounded-phase.csv");
          accepted_trial << "i,phase,lower,rhs\n";
          for (unsigned int i=0; i<bounded.block(block).size(); ++i)
            accepted_trial << i << ',' << bounded.block(block)[i] << ',' << lower.block(block)[i] << ',' << sim.system_rhs.block(block)[i] << '\n';
          for (const auto &p:s.bulk)
            {
              const auto before=model.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,p.cell,p.x);
              const auto after=model.evaluate_dg_history_stress(*s.dg_history,s.Z,updated.values,p.cell,p.x,nullptr,&bounded);
              double phi0=0.,phi1=0.;
              for (unsigned int v=0; v<p.dofs.size(); ++v)
                {
                  phi0+=p.shape[v]*lower.block(block)[p.dofs[v]];
                  phi1+=p.shape[v]*bounded.block(block)[p.dofs[v]];
                }
              const double I0=(1-p.xi)*s.I[p.segment]+p.xi*s.I[p.segment+1];
              const double I1=(1-p.xi)*updated.values[p.segment]+p.xi*updated.values[p.segment+1];
              const double z=(1-p.xi)*s.Z[p.segment]+p.xi*s.Z[p.segment+1];
              const double dc=s.m*(phi1*(1+phi1)/std::pow(1-phi1,2)/I1-phi0*(1+phi0)/std::pow(1-phi0,2)/I0);
              SymmetricTensor<2,dim> expected;
              expected[0][1]=-z*dc*s.shear_sense;
              redistribution_error=std::max(redistribution_error,(after-before-expected).norm());
              redistribution=std::max(redistribution,expected.norm());
            }
          controls << "bounded_incoming_energy," << incoming_energy << "\nbounded_final_energy," << final_energy
                   << "\nbounded_kkt," << kkt << "\nbounded_growth," << growth << "\nbounded_healing," << healing
                   << "\nlower_contacts," << contacts << "\nredistribution_error," << redistribution_error
                   << "\nredistribution_norm," << redistribution << '\n';
          BV repeated(lower);
          PhaseAccess::bounded(*sim.phase_field_handler,sim.system_matrix,sim.system_rhs,repeated,&lower);
          controls << "repeated_bounded_exact," << (repeated.block(block)==bounded.block(block)) << '\n';
          AssertThrow(repeated.block(block)==bounded.block(block),ExcMessage("Repeated phase trials changed the solution."));
          AssertThrow(growth>0. && healing==0. && final_energy<incoming_energy && redistribution>0.
                      && redistribution_error<=1.e-10*redistribution,ExcMessage("Bounded cohesive phase trial failed."));
          AssertThrow(model.accepted_fe_strain_state()==anchor && Access::dg_history_values(model)==dg_anchor
                      && s.Z==Z_anchor && s.I==I_anchor && Access::current_normalization_integrals(model)==surface_anchor,
                      ExcMessage("PF3 trials mutated accepted histories."));
          controls << "accepted_history_unchanged,1\n";
          sim.time=old_time;
          sim.time_step=old_dt;
          sim.timestep_number=0;
          this->get_pcout() << "   PF3 phase functional controls passed; no phase/history publication." << std::endl;
          return {"PF3 phase operator:","passed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF3 phase energy checks",
                                  "Fixed-surface, immutable-DG phase derivatives, quadrature and bounded noncommitting trials.")
  }
}
