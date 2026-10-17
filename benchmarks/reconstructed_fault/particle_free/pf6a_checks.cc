/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Observer of the actual native endpoint outer loop, not a phase-first driver.
#include <aspect/simulator.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include "../../../tests/phase_field_test_access.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cstring>
#include <chrono>

namespace aspect
{
  namespace Postprocess
  {
    template<int dim> class MomentCycle: public Interface<dim>,public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;
        using PhaseAccess=internal::PhaseFieldTestAccess<dim>;
        using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        std::string prefix,archive_pairs_file;
        unsigned int block=0,last_step=0;
        bool initialized=false,operators_done=false,recovery_done=false,diagnostics=true;
        bool publication_checks=false,publication_checked=false;
        bool batch_snapshots=false,verify_batch=false;
        std::vector<double> previous_phase,previous_Z,slip,attempt_snapshot;
        unsigned int attempt_step=numbers::invalid_unsigned_int;
        double attempt_time=0.,attempt_dt=0.;
        std::shared_ptr<typename Model::FEStrainState> incoming;
        std::vector<std::vector<double>> incoming_surface;
        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+prefix+name);
          out<<std::setprecision(17);
          return out;
        }
        Model &model() const
        {
          return const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        }
        Simulator<dim> &simulator() const
        {
          return const_cast<Simulator<dim>&>(this->get_simulator());
        }
        std::vector<double> persistent() const
        {
          const auto &m=model();
          auto v=m.dg_history_values();
          const auto &s=*m.fe_strain_state;
          v.insert(v.end(),s.Z.begin(),s.Z.end());
          v.insert(v.end(),s.I.begin(),s.I.end());
          for (unsigned int i=0; i<s.displacement_history.size(); ++i)v.push_back(s.displacement_history[i]);
          v.push_back(s.committed_step);
          v.push_back(s.accepted_publications);
          const auto &manager=this->get_reconstructed_fault_manager();
          for (const auto &f:manager.get_faults())for (unsigned int i=0; i<f.n_vertices(); ++i)
              for (double a:f.get_properties(i))v.push_back(a);
          v.insert(v.end(),manager.get_slip_rate(0).begin(),manager.get_slip_rate(0).end());
          v.insert(v.end(),manager.get_timestep_committed_slip_rate(0).begin(),manager.get_timestep_committed_slip_rate(0).end());
          for (const auto &p:this->get_particle_manager(0).get_particle_handler())
            {
              for (unsigned int d=0; d<dim; ++d)v.push_back(p.get_location()[d]);
              for (double a:p.get_properties())v.push_back(a);
            }
          return v;
        }
        static bool same(const std::vector<double> &a,const std::vector<double> &b)
        {
          return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;
        }
        void check(const std::string &name,bool passed) const
        {
          std::ofstream out(this->get_output_directory()+"gates.csv",std::ios::app);
          if (out.tellp()==0) out<<"gate,passed\n";
          out<<name<<','<<passed<<std::endl;
          AssertThrow(passed,ExcMessage("PF6a gate: "+name));
        }
        void directions(BV &d1,BV &d2) const
        {
          d1=0.;
          d2=0.;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            for (const auto v:cell->vertex_indices())
              {
                const auto p=cell->vertex(v);
                const auto i=cell->vertex_dof_index(v,pc);
                d1[i]=std::exp(-40*std::pow(p[1]-.5,2))*(1+.1*std::cos(2*numbers::PI*p[0]));
                d2[i]=std::exp(-40*std::pow(p[1]-.5,2))*(p[1]-.5+.05*std::sin(2*numbers::PI*p[0]));
              }
          d1.compress(VectorOperation::insert);
          d2.compress(VectorOperation::insert);
        }
        std::vector<double> values(const BV &v) const
        {
          std::vector<double> a(v.block(block).size());
          for (unsigned int i=0; i<a.size(); ++i)a[i]=v.block(block)[i];
          return a;
        }
        void assign_phase(BV &v,const std::vector<double> &p) const
        {
          for (unsigned int i=0; i<p.size(); ++i)v.block(block)[i]=p[i];
          v.compress(VectorOperation::insert);
        }
        std::vector<double> gradient(const BV &v,const typename Model::FEEndpointTrial &e) const
        {
          auto g=model().evaluate_fe_endpoint_energy(v,e).gradient;
          auto f=this->get_phase_field_handler().evaluate_cell_fracture(v,false);
          for (unsigned int i=0; i<g.size(); ++i)g[i]+=f.gradient[i];
          return g;
        }
        void export_reference(Model &model,const BV &phase,const BV &d1,const BV &d2,
                              const typename Model::FEEndpointTrial *endpoint=nullptr) const
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
              const auto stress=model.evaluate_dg_trial_tensor(endpoint ? *endpoint->dg_history : *s.dg_history,entry.first,samples);
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
            surface << i << ',' << s.vertices[i][0] << ',' << (endpoint ? endpoint->Z[i] : s.Z[i]) << ',' << s.I[i] << '\n';
          profiles << "segment,xi,weight\n";
          for (const auto &p:s.profiles) profiles << p.segment << ',' << p.xi << ',' << p.weight << '\n';
          rule << "r,s,weight\n";
          const auto quadrature=model.mechanical_quadrature();
          for (unsigned int q=0; q<quadrature.size(); ++q)
            rule << quadrature.point(q)[0] << ',' << quadrature.point(q)[1] << ',' << quadrature.weight(q) << '\n';
          const auto coefficient=PhaseAccess::coefficients(this->get_phase_field_handler(), {.5,.5},.3,0.);
          auto constants=output("constants.csv");
          constants << "G,m,A,ell\n" << s.G << ',' << s.m << ',' << coefficient.reaction << ','
                    << std::sqrt(coefficient.diffusion/(2*coefficient.reaction)) << '\n';
        }

        void operator_checks()
        {
          auto &sim=simulator();
          auto &m=model();
          const auto snapshot=persistent();
          const BV saved(sim.solution),linear(sim.current_linearization_point);
          BV d1(saved),d2(saved);
          directions(d1,d2);
          BV trial(saved);
          auto p=values(m.accepted_fe_strain_state());
          const auto d=values(d1),e=values(d2);
          for (unsigned int i=0; i<p.size(); ++i)p[i]+=.01*d[i];
          assign_phase(trial,p);
          sim.solution=trial;
          sim.current_linearization_point=trial;
          bool trial_checkpoint_rejected=false;
          try {std::ostringstream bytes;aspect::oarchive archive(bytes);m.serialize_dg_accepted_history(archive);}
          catch(const std::exception &) {trial_checkpoint_rejected=true;}
          check("trial checkpoint rejected",trial_checkpoint_rejected);

          ReconstructedFaultVector V;
          PhaseAccess::mechanical_trial(sim,V);
          const BV base(sim.solution);
          const auto endpoint=m.make_fe_endpoint_trial(base,V);
          const auto repeat=m.make_fe_endpoint_trial(base,V);
          double repeat_error=0.;
          for (unsigned int c=0; c<3; ++c)
            for (unsigned int i=0; i<endpoint.dg_history->trial_stress[c].size(); ++i)
              repeat_error=std::max(repeat_error,std::abs(endpoint.dg_history->trial_stress[c][i]-repeat.dg_history->trial_stress[c][i]));
          check("repeated endpoint candidates",repeat_error==0. && endpoint.Z==repeat.Z);
          prefix="operators/";
          std::filesystem::create_directory(this->get_output_directory()+prefix);
          export_reference(m,base,d1,d2,&endpoint);
          auto ref=output("native.csv");
          ref<<"i,gradient,I\n";
          const auto g=gradient(base,endpoint);
          const auto energy=m.evaluate_fe_endpoint_energy(base,endpoint);
          auto fracture=this->get_phase_field_handler().evaluate_cell_fracture(base,true);
          for (unsigned int i=0; i<g.size(); ++i)ref<<i<<','<<g[i]<<','<<(i<energy.normalization.size()?energy.normalization[i]:0.)<<'\n';
          auto totals=output("energy.csv");
          totals<<"bulk,cohesive,fracture\n"<<energy.bulk<<','<<energy.cohesive<<','<<fracture.energy<<'\n';
          auto report=output("derivatives.csv");
          report<<"kind,direction,epsilon,energy_error,gradient_error,action_scale\n";
          struct Field:Function<dim>
          {
            Field(unsigned int n):Function<dim>(n) {} double value(const Point<dim> &x,unsigned int c=0)const override
            {
              return c<2 ? .001*x[0]*(1-x[0])*x[1]*(1-x[1])*(c==0?1.:-.7):0.;
            }
          } function(this->get_fe().n_components());
          BV du(base);
          du=0.;
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),function,du,this->introspection().component_masks.velocities);
          ReconstructedFaultVector dv=V;
          for (unsigned int j=0; j<dv[0].size(); ++j)dv[0][j]=.001*(1+.1*std::cos(.3*j));
          auto actions=output("actions.csv");
          actions<<"kind,direction,i,value\n";
          const auto fixed=m.evaluate_fe_phase_energy_at(base,true,false,endpoint.strain,endpoint.Z,nullptr,endpoint.dg_history.get());
          for (unsigned int mode=0; mode<2; ++mode)
            {
              const auto direction=mode==0?d:e;
              auto analytic=m.evaluate_fe_endpoint_action(base,endpoint,direction,&du,&dv).gradient;
              for (unsigned int i=0; i<analytic.size(); ++i)for (unsigned int j=0; j<direction.size(); ++j)analytic[i]+=fracture.hessian(i,j)*direction[j];
              double action_scale=1.;
              for (double a:analytic)action_scale=std::max(action_scale,std::abs(a));
              double best=1.,best_reduced=1.,best_energy=1.;
              // This tangent is a mechanical-response action, not a fixed-history phase Hessian.
              auto reduced=PhaseAccess::endpoint_tangent(sim,base,V,endpoint.strain,endpoint.Z);
              Vector<double> vec(direction.size());
              for (unsigned int i=0; i<direction.size(); ++i)vec[i]=direction[i];
              Vector<double> reduced_action=reduced(vec);
              for (unsigned int i=0; i<vec.size(); ++i)for (unsigned int j=0; j<vec.size(); ++j)reduced_action[i]+=fracture.hessian(i,j)*direction[j];
              double reduced_scale=std::max(1.,reduced_action.linfty_norm());
              for (unsigned int i=0; i<vec.size(); ++i)
                {
                  double partial=0.;
                  for (unsigned int j=0; j<vec.size(); ++j)partial+=(fixed.hessian(i,j)+fracture.hessian(i,j))*direction[j];
                  actions<<"fixed,"<<mode<<','<<i<<','<<partial<<'\n';
                  actions<<"reduced,"<<mode<<','<<i<<','<<reduced_action[i]<<'\n';
                }
              for (double eps:
                   {
                     1e-3,1e-4,1e-5,1e-6
                   })
                {
                  std::vector<std::vector<double>> grads;
                  double ep_energy[2];
                  for (unsigned int side=0; side<2; ++side)
                    {
                      const double a=side==0?eps:-eps;
                      BV v(base);
                      v.add(a,du);
                      auto pp=p;
                      for (unsigned int i=0; i<pp.size(); ++i)pp[i]+=a*direction[i];
                      assign_phase(v,pp);
                      auto rates=V;
                      for (unsigned int j=0; j<rates[0].size(); ++j)rates[0][j]+=a*dv[0][j];
                      auto et=m.make_fe_endpoint_trial(v,rates);
                      grads.push_back(gradient(v,et));
                      // Independent phase-energy FD keeps the mechanical endpoint fixed.
                      auto fixed=endpoint;
                      fixed.strain.block(block)=v.block(block);
                      ep_energy[side]=m.evaluate_fe_endpoint_energy(v,fixed).energy+this->get_phase_field_handler().evaluate_cell_fracture(v,false).energy;
                    }
                  double err=0.,dot=0.;
                  for (unsigned int i=0; i<g.size(); ++i)
                    {
                      err=std::max(err,std::abs((grads[0][i]-grads[1][i])/(2*eps)-analytic[i])/action_scale);
                      dot+=g[i]*direction[i];
                    }
                  double energy_err=std::abs((ep_energy[0]-ep_energy[1])/(2*eps)-dot)/std::max(1.,std::abs(dot));
                  best=std::min(best,err);
                  best_energy=std::min(best_energy,energy_err);
                  report<<"combined,"<<mode<<','<<eps<<','<<energy_err<<','<<err<<','<<action_scale<<'\n';
                  grads.clear();
                  for (double a:
                       {
                         eps,-eps
                       })
                    {
                      BV v(base);
                      auto pp=p;
                      for (unsigned int i=0; i<pp.size(); ++i)pp[i]+=a*direction[i];
                      assign_phase(v,pp);
                      sim.solution=v;
                      sim.current_linearization_point=v;
                      ReconstructedFaultVector rates;
                      PhaseAccess::mechanical_trial(sim,rates);
                      auto et=m.make_fe_endpoint_trial(sim.solution,rates);
                      grads.push_back(gradient(sim.solution,et));
                    }
                  err=0.;
                  for (unsigned int i=0; i<g.size(); ++i)err=std::max(err,std::abs((grads[0][i]-grads[1][i])/(2*eps)-reduced_action[i])/reduced_scale);
                  best_reduced=std::min(best_reduced,err);
                  report<<"reduced,"<<mode<<','<<eps<<",0,"<<err<<','<<reduced_scale<<'\n';
                }
              check("consistent combined endpoint action "+std::to_string(mode),best<1e-6);
              check("fixed endpoint phase energy derivative "+std::to_string(mode),best_energy<1e-6);
              check("consistent reduced tangent "+std::to_string(mode),best_reduced<1e-6);
            }
          sim.solution=saved;
          sim.current_linearization_point=linear;
          m.prepare_reconstructed_fault_mechanical_solve();
          prefix="";
          check("operator checks do not publish",same(snapshot,persistent()));
        }
        void recovery()
        {
          auto &sim=simulator();
          const auto before=persistent();
          const BV s(sim.solution),l(sim.current_linearization_point),o(sim.old_solution),oo(sim.old_old_solution);
          const auto metadata=PhaseAccess::endpoint_bookkeeping(sim);
          bool caught=false;
          PhaseAccess::endpoint_failure_hook(sim,[] {throw std::runtime_error("PF6a controlled outer rejection");});
          try
            {
              PhaseAccess::complete_step(sim);
            }
          catch (const std::runtime_error &error)
            {
              caught=std::string(error.what())=="PF6a controlled outer rejection";
              if (!caught)throw;
            }
          PhaseAccess::endpoint_failure_hook(sim, {});
          check("actual outer prepublication rejection reached",caught);
          check("outer rejection restores persistent histories",same(before,persistent()));
          check("outer rejection restores solution clocks bookkeeping",sim.solution==s && sim.current_linearization_point==l && sim.old_solution==o && sim.old_old_solution==oo && PhaseAccess::endpoint_bookkeeping(sim)==metadata);
          // Also reject after solve_timestep returned a staged endpoint. This
          // exercises late validation, separately from the real manager cutback.
          PhaseAccess::complete_step(sim);
          check("complete endpoint remains unpublished before outer acceptance",sim.pending_dg_endpoint_publication && same(before,persistent()));
          auto publish=std::move(sim.pending_dg_endpoint_publication);
          sim.pending_dg_endpoint_publication={};
          sim.solution.block(block)[0]+=1e-8;
          caught=false;
          try {publish(true);}
          catch(const std::exception &e)
            {
              caught=std::string(e.what()).find("Stale PF6a endpoint publication context")!=std::string::npos;
              if(!caught)throw;
            }
          check("deferred publication rejects stale context",caught);
          check("deferred rejection restores persistent histories",same(before,persistent()));
          check("deferred rejection restores solution clocks bookkeeping",sim.solution==s && sim.current_linearization_point==l && sim.old_solution==o && sim.old_old_solution==oo && PhaseAccess::endpoint_bookkeeping(sim)==metadata);
        }
        void publication_control()
        {
          auto &sim=simulator();auto &m=model();const auto &s=*m.fe_strain_state;
          const auto before=persistent();
          const BV saved(sim.solution),linear(sim.current_linearization_point),old(sim.old_solution),older(sim.old_old_solution);
          const auto metadata=PhaseAccess::endpoint_bookkeeping(sim);
          const auto rng_state=[&]()
          {std::ostringstream out;{aspect::oarchive archive(out);archive<<this->get_particle_manager(0).get_random_number_state();}return out.str();};
          const auto rng_before=rng_state();
          const auto c=m.compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
          const auto tractions=[&]()
          {
            std::vector<std::vector<double>> q(1,std::vector<double>(s.Z.size()));
            for(unsigned int j=0;j<s.Z.size();++j)q[0][j]=(c.beta*s.Z[j]+c.eta_ve*this->get_reconstructed_fault_manager().get_slip_rate(0)[j])/m.current_normalization_integrals[0][j];
            return q;
          };
          // Separate analytic affine sticking control, never installed as a
          // physical solution or an accepted history.
          class Affine:public Function<dim>
          {
            public:
              Affine(unsigned int n):Function<dim>(n) {}
              double value(const Point<dim>&p,unsigned int component) const override
              {return component==0 ? .001*(p[1]-.5) : 0.;}
          } function(this->introspection().n_components);
          BV affine(saved);
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),function,affine,this->introspection().component_masks.velocities);
          auto update=m.prepare_fe_strain_commit(affine,tractions());
          const auto plain=m.make_dg_history_trial(*s.dg_history,affine,c);
          bool identical=true;
          for(unsigned int component=0;component<3;++component)
            for(const auto i:plain->trial_stress[component].locally_owned_elements())
              identical=identical && plain->trial_stress[component][i]==update->dg_history->trial_stress[component][i];
          check("source bound requested only for publication",plain->source_roundoff.empty() && update->dg_history->source_roundoff.size()==s.cells.size());
          check("source bound leaves every DG coefficient identical",identical);
          double affine_error=0.;
          for(unsigned int component=0;component<3;++component)
            for(const auto i:update->dg_history->trial_stress[component].locally_owned_elements())
              affine_error=std::max(affine_error,std::abs(update->dg_history->trial_stress[component][i]-(component==2 ? .001*c.eta_ve : 0.)));
          check("analytic affine sticking DG source",affine_error<1e-9);
          auto report=output("publication-controls.csv");
          report<<"probe,error,bound,ratio\n"<<"affine-source,"<<affine_error<<",1e-9,"<<affine_error/1e-9<<'\n';
          // Corrupt a separate staged history, keeping its velocity-derived
          // bound fixed. The production guard must reject the inconsistency.
          const auto particle=this->get_particle_manager(0).get_particle_handler().begin();
          const auto id=particle->get_surrounding_cell()->id();const auto point=particle->get_location();
          PhaseAccess::endpoint_failure_hook(sim,[&]
          {
            auto candidate=m.prepare_fe_strain_commit(sim.solution,tractions());
            double scale=0.;const auto expected=m.candidate_fe_strain_stress(candidate,id,point,&scale);
            const double bound=128*std::numeric_limits<double>::epsilon()*std::max(1.,scale)+m.dg_source_publication_allowance(*candidate->dg_history,id,point);
            // An independently inconsistent incoming tensor must also fail;
            // source bounds must not grow with the inconsistent tensor.
            for(unsigned int component=0;component<3;++component)
              {
                typename Model::DGHistory incoming_copy(*s.dg_history);
                incoming_copy.trial_stress[component].add(1.);
                auto different=m.make_dg_history_trial(incoming_copy,sim.solution,c);
                auto wrong=std::make_shared<typename Model::FEStrainUpdate>(*candidate);wrong->dg_history=different;
                const auto delta=m.candidate_fe_strain_stress(wrong,id,point)-expected;
                const double error=std::sqrt(std::pow(delta.norm(),2)+std::pow(trace(delta),2));
                bool rejected=false;
                try {m.validate_fe_strain_publication(error,bound);}
                catch(const std::exception &){rejected=true;}
                check("inconsistent incoming tensor component "+std::to_string(component),rejected);
                report<<"inconsistent-incoming-"<<component<<','<<error<<','<<bound<<','<<error/bound<<'\n';
              }
            candidate->dg_history->trial_stress[2].add(1.);
            const auto inconsistent=m.candidate_fe_strain_stress(candidate,id,point);
            const double error=(inconsistent-expected).norm();
            report<<"inconsistent-staged-history,"<<error<<','<<bound<<','<<error/bound<<std::endl;
            m.validate_fe_strain_publication(error,bound);
          });
          bool rejected=false;
          try {PhaseAccess::complete_step(sim);}
          catch(const std::exception &e)
            {
              rejected=std::string(e.what()).find("Accepted FE strain does not reproduce the native Maxwell publication.")!=std::string::npos;
              if(!rejected)throw;
            }
          PhaseAccess::endpoint_failure_hook(sim,{});
          check("operand allowance rejects inconsistent history in outer solve",rejected);
          check("publication rejection preserves persistent histories",same(before,persistent()));
          check("publication rejection preserves particle RNG",rng_state()==rng_before);
          check("publication rejection restores solution clocks bookkeeping",sim.solution==saved && sim.current_linearization_point==linear && sim.old_solution==old && sim.old_old_solution==older && PhaseAccess::endpoint_bookkeeping(sim)==metadata);
          PhaseAccess::complete_step(sim);
          check("successful retry stages without publication",sim.pending_dg_endpoint_publication && same(before,persistent()));
          auto discard=std::move(sim.pending_dg_endpoint_publication);sim.pending_dg_endpoint_publication={};discard(false);
          check("outer rejected staged retry preserves histories",same(before,persistent()));
          check("outer rejected staged retry preserves particle RNG",rng_state()==rng_before);
          check("outer rejected staged retry restores bookkeeping",sim.solution==saved && sim.current_linearization_point==linear && PhaseAccess::endpoint_bookkeeping(sim)==metadata);
          publication_checked=true;
          // The actual outer loop now repeats the same trial, accepts and
          // publishes. The existing step/publication gates check exactly once.
        }
      public:
        static void declare_parameters(ParameterHandler &p)
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6a checks");
          p.declare_entry("Operator and rejection checks","true",Patterns::Bool());
          p.declare_entry("Publication roundoff checks","false",Patterns::Bool());
          p.declare_entry("Batch diagnostic stress evaluation","false",Patterns::Bool());
          p.declare_entry("Verify batched diagnostic stress","false",Patterns::Bool());
          p.declare_entry("Archive pairs file","",Patterns::FileName());
          p.leave_subsection();
          p.leave_subsection();
        }
        void parse_parameters(ParameterHandler &p) override
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6a checks");
          diagnostics=p.get_bool("Operator and rejection checks");
          publication_checks=p.get_bool("Publication roundoff checks");
          batch_snapshots=p.get_bool("Batch diagnostic stress evaluation");
          verify_batch=p.get_bool("Verify batched diagnostic stress");
          archive_pairs_file=p.get("Archive pairs file");
          p.leave_subsection();
          p.leave_subsection();
        }
        void initialize() override
        {
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
          {
            if (this->get_timestep_number()==0)return;
            auto &m=model();
            if(attempt_step==this->get_timestep_number())
              {
                check("native outer cutback preserves accepted histories",same(attempt_snapshot,persistent()));
                check("native outer cutback restores physical phase",values(this->get_solution())==values(m.accepted_fe_strain_state()));
                auto out=std::ofstream(this->get_output_directory()+"cutback.csv",std::ios::app);
                if(out.tellp()==0)out<<"step,rejected_time,rejected_dt,retry_time,retry_dt,publications\n";
                out<<std::setprecision(17)<<attempt_step<<','<<attempt_time<<','<<attempt_dt<<','<<this->get_time()<<','<<this->get_timestep()<<','<<m.fe_strain_state->accepted_publications<<'\n';
              }
            attempt_step=this->get_timestep_number();attempt_snapshot=persistent();
            attempt_time=this->get_time();attempt_dt=this->get_timestep();
            block=this->introspection().variable("phase_field").block_index;
            if(publication_checks && !publication_checked)publication_control();
            if (diagnostics && !operators_done && this->get_time()>=.25)
              {
                operator_checks();
                operators_done=true;
              }
            if (diagnostics && !recovery_done && this->get_time()>=.53125)
              {
                recovery();
                recovery_done=true;
              }
            incoming=std::make_shared<typename Model::FEStrainState>(*m.fe_strain_state);
            incoming_surface.clear();
            for (unsigned int j=0; j<this->get_reconstructed_fault_manager().get_fault(0).n_vertices(); ++j)
              {
                auto p=this->get_reconstructed_fault_manager().get_fault(0).get_properties(j);
                incoming_surface.emplace_back(p.begin(),p.end());
              }
          });
        }
        void save(std::map<std::string,std::string> &status) const override
        {
          std::ostringstream out;
          aspect::oarchive a(out);
          a &initialized &operators_done &recovery_done &last_step &previous_phase &previous_Z &slip;
          status["PF6a progress"]=out.str();
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          std::istringstream in(status.at("PF6a progress"));
          aspect::iarchive a(in);
          a &initialized &operators_done &recovery_done &last_step &previous_phase &previous_Z &slip;
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &m=model();
          auto &sim=simulator();
          const auto step=this->get_timestep_number();
          block=this->introspection().variable("phase_field").block_index;
          if (!archive_pairs_file.empty())
            {
              const auto before=persistent();
              std::ifstream pairs(archive_pairs_file);
              AssertThrow(bool(pairs),ExcMessage("Missing PF6a archive pairs."));
              std::string first,second;
              auto report=output("canonical-archives.csv");
              report<<"pair,raw_equal,raw_size_difference,canonical_equal\n";
              unsigned int pair=0;
              while (pairs>>first>>second)
                {
                  const auto canonical=[&](const std::string &name)
                  {
                    std::ifstream file(name,std::ios::binary);
                    std::string bytes((std::istreambuf_iterator<char>(file)),{});
                    std::istringstream in(bytes);
                    ReconstructedFaultManager<dim> copy(this->get_simulator());
                    {aspect::iarchive archive(in);archive>>copy;}
                    std::ostringstream out;
                    {aspect::oarchive archive(out);archive<<copy;}
                    return std::make_pair(bytes,out.str());
                  };
                  auto a=canonical(first),b=canonical(second);
                  report<<pair++<<','<<(a.first==b.first)<<','<<static_cast<long long>(b.first.size())-a.first.size()<<','<<(a.second==b.second)<<'\n';
                  check("complete native loaded manager archive",a.second==b.second);
                }
              check("archive comparison preserves current history",pair>0 && same(before,persistent()));
              return {"PF6a","complete native manager archives compared after identical native loading"};
            }
          if (initialized && step==last_step)return {"PF6a","restored accepted observer state"};
          const auto &s=*m.fe_strain_state;
          const auto &state=this->get_solution();
          const auto phase=values(state);
          if (!initialized)
            {
              std::ofstream out(this->get_output_directory()+"gates.csv");
              out<<"gate,passed\n";
              previous_phase=phase;
              previous_Z=s.Z;
              slip.assign(s.Z.size(),0.);
            }
          double growth=0.,healing=0.;
          for (unsigned int i=0; i<phase.size(); ++i)
            {
              growth=std::max(growth,phase[i]-previous_phase[i]);
              healing=std::max(healing,previous_phase[i]-phase[i]);
            }
          check("single publication step "+std::to_string(step),s.accepted_publications==step && s.committed_step==step);
          check("physical phase anchor step "+std::to_string(step),phase==values(s.displacement_history) && healing==0.);
          check("fixed geometry particles step "+std::to_string(step),(m.validate_fe_strain_restrictions(),true));
          const auto E=m.evaluate_fe_phase_energy(state,false);
          const auto F=this->get_phase_field_handler().evaluate_cell_fracture(state,false);
          const auto V=this->get_reconstructed_fault_manager().get_timestep_committed_slip_rate(0);
          double kkt=0.,zerror=0.;
          for (unsigned int i=0; i<phase.size(); ++i)
            {
              double g=E.gradient[i]+F.gradient[i];
              kkt=std::max(kkt,std::abs(phase[i]==previous_phase[i]?std::min(g,0.):g));
            }
          if (step)for (unsigned int j=0; j<s.Z.size(); ++j)
              {
                zerror=std::max(zerror,std::abs(s.Z[j]-std::exp(-this->get_timestep()/100.)*previous_Z[j]+1e8*std::expm1(-this->get_timestep()/100.)*V[j]));
                slip[j]+=this->get_timestep()*V[j];
              }
          check("surface law step "+std::to_string(step),zerror<1e-8);
          prefix="state-"+std::to_string(step)+"/";
          std::filesystem::create_directory(this->get_output_directory()+prefix);
          BV d1(state),d2(state);
          directions(d1,d2);
          export_reference(m,state,d1,d2);
          auto surf=output("accepted-surface.csv");
          surf<<"i,Z,I,V,slip\n";
          for (unsigned int j=0; j<s.Z.size(); ++j)surf<<j<<','<<s.Z[j]<<','<<s.I[j]<<','<<V[j]<<','<<slip[j]<<'\n';
          auto grad=output("gradient.csv");
          grad<<"i,gradient,lower\n";
          for (unsigned int i=0; i<phase.size(); ++i)grad<<i<<','<<E.gradient[i]+F.gradient[i]<<','<<previous_phase[i]<<'\n';
          const QGaussLobatto<dim> nodes(3);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),nodes,update_values|update_quadrature_points);
          auto velocity=output("velocity.csv");
          velocity<<"cell,r,s,ux,uy,p\n";
          unsigned int cellnum=0;
          for (const auto &entry:s.cells)
            {
              fe.reinit(entry.second);
              std::vector<Tensor<1,dim>> u(nodes.size());
              std::vector<double> p(nodes.size());
              fe[this->introspection().extractors.velocities].get_function_values(state,u);
              fe[this->introspection().extractors.pressure].get_function_values(state,p);
              for (unsigned int q=0; q<nodes.size(); ++q)velocity<<cellnum<<','<<nodes.point(q)[0]<<','<<nodes.point(q)[1]<<','<<u[q][0]<<','<<u[q][1]<<','<<p[q]<<'\n';
              ++cellnum;
            }
          auto points=output("physical.csv");
          points<<"cell,x,y,weight,phi,chi,xx,yy,xy\n";
          const auto quad=m.mechanical_quadrature();
          FEValues<dim> physical(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
          const bool compare_batch=batch_snapshots && verify_batch && (step==0 || step==1 || step==68 || step==72);
          const auto before_batch=compare_batch ? persistent() : std::vector<double>();
          double batch_seconds=0.,point_seconds=0.,batch_error=0.,batch_scale=1.;
          cellnum=0;
          for (const auto &entry:s.cells)
            {
              physical.reinit(entry.second);
              std::vector<double> phi(quad.size());
              physical[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(state,phi);
              std::vector<SymmetricTensor<2,dim>> tensor;
              if(batch_snapshots)
                {
                  const auto start=std::chrono::steady_clock::now();
                  tensor=m.evaluate_dg_trial_tensor(*s.dg_history,entry.first,quad);
                  batch_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                }
              for (unsigned int q=0; q<quad.size(); ++q)
                {
                  auto x=physical.quadrature_point(q);
                  unsigned int j=std::min<unsigned int>(s.I.size()-2,x[0]*(s.I.size()-1));
                  double xi=x[0]*(s.I.size()-1)-j,I=(1-xi)*s.I[j]+xi*s.I[j+1],h=s.m*phi[q]*(1+phi[q])/std::pow(1-phi[q],2),chi=h/I;
                  SymmetricTensor<2,dim> t;
                  if(batch_snapshots)
                    {
                      t=tensor[q];
                      const double z=(1-xi)*s.Z[j]+xi*s.Z[j+1];
                      t[0][1]-=(2*z*h/I)*(.5*s.shear_sense);
                    }
                  if(!batch_snapshots || compare_batch)
                    {
                      const auto start=std::chrono::steady_clock::now();
                      const auto point=m.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,entry.first,x);
                      point_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                      if(!batch_snapshots)t=point;
                      else
                        {
                          const auto delta=t-point;
                          batch_error=std::max(batch_error,std::sqrt(std::pow(delta.norm(),2)+std::pow(trace(delta),2)));
                          batch_scale=std::max(batch_scale,std::sqrt(std::pow(point.norm(),2)+std::pow(trace(point),2)));
                        }
                    }
                  points<<cellnum<<','<<x[0]<<','<<x[1]<<','<<physical.JxW(q)<<','<<phi[q]<<','<<chi<<','<<t[0][0]<<','<<t[1][1]<<','<<t[0][1]<<'\n';
                }++cellnum;
            }
          {
            std::ofstream report(this->get_output_directory()+"stress-evaluation.csv",step ? std::ios::app : std::ios::out);
            report<<std::setprecision(17);
            if(report.tellp()==0)report<<"step,batch,compared,batch_seconds,point_seconds,error_Pa,scale_Pa,scaled_error,points\n";
            report<<step<<','<<batch_snapshots<<','<<compare_batch<<','<<batch_seconds<<','<<point_seconds<<','<<batch_error<<','<<batch_scale<<','<<batch_error/batch_scale<<','<<s.cells.size()*quad.size()<<'\n';
          }
          if(compare_batch)
            {
              check("batched stress agrees with point evaluator step "+std::to_string(step),batch_error/batch_scale<1e-9);
              check("batched diagnostics preserve accepted state step "+std::to_string(step),same(before_batch,persistent()));
            }
          if (step && incoming)
            {
              const auto stored=persistent();
              const auto final=m.fe_strain_state;
              const BV saved_linear(sim.current_linearization_point);
              auto &fault=sim.reconstructed_fault_manager->get_fault(0);
              std::vector<std::vector<double>> saved_surface;
              for (unsigned int j=0; j<fault.n_vertices(); ++j)
                {
                  auto a=fault.get_properties(j);
                  saved_surface.emplace_back(a.begin(),a.end());
                  std::copy(incoming_surface[j].begin(),incoming_surface[j].end(),a.begin());
                }
              m.fe_strain_state=incoming;
              m.prepare_reconstructed_fault_mechanical_solve();
              const auto native=sim.reconstructed_fault_surface_system->evaluate_surface_residual(state, {V});
              const auto defect=PhaseAccess::endpoint_defects(sim,state, {V});
              auto loads=output("native-loads.csv");
              loads<<"i,shear,normal,cohesive,friction,residual\n";
              for (unsigned int j=0; j<V.size(); ++j)loads<<j<<','<<native.shear_traction[0][j]<<','<<native.normal_traction[0][j]<<','<<native.cohesive_traction[0][j]<<','<<native.friction_traction[0][j]<<','<<native.values[0][j]<<'\n';
              auto residual=output("mechanical.csv");
              residual<<"bulk_norm,surface_rms\n"<<defect.first<<','<<defect.second<<'\n';
              m.fe_strain_state=final;
              for (unsigned int j=0; j<fault.n_vertices(); ++j)std::copy(saved_surface[j].begin(),saved_surface[j].end(),fault.get_properties(j).begin());
              sim.current_linearization_point=saved_linear;
              m.prepare_reconstructed_fault_mechanical_solve();
              check("incoming-anchor residual probe does not publish step "+std::to_string(step),same(stored,persistent()));
            }
          const auto stored=persistent();
          {std::ofstream out(this->get_output_directory()+prefix+"persistent.bin",std::ios::binary);out.write(reinterpret_cast<const char*>(stored.data()),stored.size()*sizeof(double));}
          {std::ofstream out(this->get_output_directory()+prefix+"rng.bin",std::ios::binary);aspect::oarchive archive(out);archive<<this->get_particle_manager(0).get_random_number_state();}
          {std::ofstream out(this->get_output_directory()+prefix+"manager.bin",std::ios::binary);aspect::oarchive archive(out);archive<<this->get_reconstructed_fault_manager();}
          auto shared=output("shared.csv");shared<<"i,anchor,solution,old,oldold\n";
          for(unsigned int i=0;i<state.size();++i)shared<<i<<','<<s.displacement_history[i]<<','<<state[i]<<','<<sim.old_solution[i]<<','<<sim.old_old_solution[i]<<'\n';
          auto clocks=output("clock.csv");clocks<<"time,dt,step,committed,publications\n"<<this->get_time()<<','<<this->get_timestep()<<','<<step<<','<<s.committed_step<<','<<s.accepted_publications<<'\n';
          prefix="";
          auto metrics=std::ofstream(this->get_output_directory()+"metrics.csv",step?std::ios::app:std::ios::out);
          metrics<<std::setprecision(17);
          if (metrics.tellp()==0)metrics<<"step,time,dt,phase_max,growth,healing,KKT,bulk,cohesive,fracture,Zmax,Zerror,publications\n";
          metrics<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<*std::max_element(phase.begin(),phase.end())<<','<<growth<<','<<healing<<','<<kkt<<','<<E.bulk<<','<<E.cohesive<<','<<F.energy<<','<<*std::max_element(s.Z.begin(),s.Z.end())<<','<<zerror<<','<<s.accepted_publications<<'\n';
          previous_phase=phase;
          previous_Z=s.Z;
          initialized=true;
          last_step=step;
          return {"PF6a","actual native endpoint phase/mechanics; one accepted DG publication"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF6a cohesive endpoint","Bounded serial fixed-reference-surface DG cohesive phase/mechanics verification.")
  }
}

namespace aspect { namespace BoundaryVelocity {
  template<int dim> class PF6aIntervalRamp: public Interface<dim>,public SimulatorAccess<dim>
  {
  public:
    Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
    {
      Tensor<1,dim> velocity;
      velocity[0]=.256*std::max(0.,this->get_time()-.5*this->get_timestep())*(x[1]-.5);
      return velocity;
    }
  };
  ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(PF6aIntervalRamp,"PF6a interval ramp",
    "Fixed verification ramp with its exact interval mean, including native cutbacks.")
}}
