/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include "transport_method_fields.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class MomentCycle : public Interface<dim>,public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;
        using Update=typename Model::FEStrainUpdate;
        using Transport=typename Model::DGTransport;
        unsigned int steps=16,completed=0;
        bool preloaded=false;
        double start=.1,duration=.2;
        std::ofstream gates;
        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+name);
          out<<std::setprecision(17);
          return out;
        }
        void check(const std::string &name,const bool passed)
        {
          gates<<name<<','<<passed<<std::endl;
          AssertThrow(passed,ExcMessage("PF5 gate failed: "+name));
        }
        BV velocity(const unsigned int kind) const
        {
          struct Field : Function<dim>
          {
            Field(const unsigned int n,const unsigned int k):Function<dim>(n),kind(k) {}
            unsigned int kind;
            double value(const Point<dim> &p,const unsigned int c) const override
            {
              if (c>=2 || kind==0) return 0.;
              if (kind==1) return PF4Methods::velocity_source(p[0],p[1])[c];
              return c==0?.4*p[0]*(1-p[0])*(1-2*p[1]):-.4*(1-2*p[0])*p[1]*(1-p[1]);
            }
          } function(this->get_fe().n_components(),kind);
          BV v(this->get_solution());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),function,v,this->introspection().component_masks.velocities);
          return v;
        }
        std::vector<double> persistent(Model &m) const
        {
          auto values=m.dg_history_values();
          const auto &s=*m.fe_strain_state;
          values.insert(values.end(),s.Z.begin(),s.Z.end());
          values.insert(values.end(),s.I.begin(),s.I.end());
          for (unsigned int i=0; i<s.displacement_history.size(); ++i) values.push_back(s.displacement_history[i]);
          values.push_back(s.committed_step);
          values.push_back(s.accepted_publications);
          const auto &manager=this->get_reconstructed_fault_manager();
          for (const auto &fault:manager.get_faults())
            for (unsigned int j=0; j<fault.n_vertices(); ++j)
              for (const auto value:fault.get_properties(j)) values.push_back(value);
          values.insert(values.end(),manager.get_slip_rate(0).begin(),manager.get_slip_rate(0).end());
          values.insert(values.end(),manager.get_timestep_committed_slip_rate(0).begin(),manager.get_timestep_committed_slip_rate(0).end());
          for (const auto &particle:this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler())
            for (const auto value:particle.get_properties()) values.push_back(value);
          return values;
        }
        static bool unchanged(const std::vector<double> &a,const std::vector<double> &b)
        {
          // Registry-only provenance properties intentionally contain NaNs.
          return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;
        }
        bool equal(const std::shared_ptr<Update> &a,const std::shared_ptr<Update> &b) const
        {
          if (a->Z!=b->Z || a->I!=b->I || a->dg_cohesive!=b->dg_cohesive) return false;
          for (unsigned int c=0; c<a->dg_history->trial_stress.size(); ++c)
            for (unsigned int i=0; i<a->dg_history->space->dofs.n_dofs(); ++i)
              if (a->dg_history->trial_stress[c][i]!=b->dg_history->trial_stress[c][i]) return false;
          return true;
        }
        void field(Model &m,const std::string &name) const
        {
          auto out=output("field-"+name+".csv");
          out<<"i,xx,yy,xy\n";
          const auto &h=*m.fe_strain_state->dg_history;
          for (unsigned int i=0; i<h.space->dofs.n_dofs(); ++i)
            out<<i<<','<<h.trial_stress[0][i]<<','<<h.trial_stress[1][i]<<','<<h.trial_stress[2][i]<<'\n';
        }
        void geometry(Model &m,const Transport &op) const
        {
          auto constants=output("constants.csv"),basis=output("basis.csv"),cells=output("cells.csv"),matrix=output("operator.csv");
          auto first=op.space->cells.begin()->second;
          const unsigned int nx=std::lround(1./(first->vertex(1)[0]-first->vertex(0)[0]));
          constants<<"nx,ny,relaxation_rate,time,start,steps,G,eta,m\n"<<nx<<','<<2*nx<<','<<m.fe_strain_state->G/m.fe_strain_state->eta<<','<<duration<<','<<start<<','<<steps<<','<<m.fe_strain_state->G<<','<<m.fe_strain_state->eta<<','<<m.fe_strain_state->m<<'\n';
          basis<<"i,r,s\n";
          const auto &support=op.space->element.get_unit_support_points();
          for (unsigned int i=0; i<9; ++i) basis<<i<<','<<support[i][0]<<','<<support[i][1]<<'\n';
          cells<<"cell,x0,y0,dx,dy,i0,i1,i2,i3,i4,i5,i6,i7,i8\n";
          unsigned int k=0;
          for (const auto &cell:op.cells)
            {
              const auto it=op.space->cells.at(cell.id);
              cells<<k++<<','<<it->vertex(0)[0]<<','<<it->vertex(0)[1]<<','<<it->vertex(1)[0]-it->vertex(0)[0]<<','<<it->vertex(2)[1]-it->vertex(0)[1];
              for (const auto i:cell.indices) cells<<','<<i;
              cells<<'\n';
            }
          matrix<<"i,j,value\n";
          for (unsigned int i=0; i<op.matrix.m(); ++i)
            for (auto p=op.matrix.begin(i); p!=op.matrix.end(i); ++p)
              if (p->value()!=0.) matrix<<i<<','<<p->column()<<','<<p->value()<<'\n';
        }
        void record(Model &m,const std::string &label) const
        {
          field(m,label);
          std::ofstream binary(this->get_output_directory()+"persistent-"+label+".bin",std::ios::binary);
          {
            aspect::oarchive archive(binary);
            archive<<persistent(m);
          }
          const auto &s=*m.fe_strain_state;
          const auto &manager=this->get_reconstructed_fault_manager();
          // Native load canonicalizes an omitted default shear sense to +1.
          // Preserve raw bytes and compare the complete persistent manager
          // after that same native load operation, without changing the model.
          std::ostringstream raw_manager;
          {aspect::oarchive archive(raw_manager); archive<<manager;}
          std::ofstream raw(this->get_output_directory()+"manager-raw-"+label+".bin",std::ios::binary);
          raw<<raw_manager.str();
          ReconstructedFaultManager<dim> copy(this->get_simulator());
          std::istringstream incoming_manager(raw_manager.str());
          {aspect::iarchive archive(incoming_manager); archive>>copy;}
          std::ofstream manager_state(this->get_output_directory()+"manager-"+label+".bin",std::ios::binary);
          {aspect::oarchive archive(manager_state); archive<<copy;}
          auto surface=output("surface-"+label+".csv"),properties=output("properties-"+label+".csv"),shared=output("shared-"+label+".csv");
          surface<<"node,Z,I,elastic_slip,cohesive,V,Vcommitted\n";
          const auto cp=manager.get_property_information()[m.fault_property_indices.cohesive_traction].position;
          for (unsigned int j=0; j<s.Z.size(); ++j)
            surface<<j<<','<<s.Z[j]<<','<<s.I[j]<<','<<s.Z[j]/s.G<<','<<manager.get_fault(0).get_properties(j)[cp]<<','<<manager.get_slip_rate(0)[j]<<','<<manager.get_timestep_committed_slip_rate(0)[j]<<'\n';
          properties<<"node";
          const auto nprops=manager.get_fault(0).get_properties(0).size();
          for (unsigned int p=0; p<nprops; ++p) properties<<",p"<<p;
          properties<<'\n';
          for (unsigned int j=0; j<s.Z.size(); ++j)
            {
              properties<<j;
              for (const auto p:manager.get_fault(0).get_properties(j)) properties<<','<<p;
              properties<<'\n';
            }
          shared<<"i,anchor,solution,old,oldold\n";
          const auto &sim=this->get_simulator();
          for (unsigned int i=0; i<s.displacement_history.size(); ++i)
            shared<<i<<','<<s.displacement_history[i]<<','<<sim.solution[i]<<','<<sim.old_solution[i]<<','<<sim.old_old_solution[i]<<'\n';
          auto clock=output("clock-"+label+".csv");
          clock<<"time,dt,old_dt,step,committed_step,publications,completed\n"<<this->get_time()<<','<<this->get_timestep()<<','<<this->get_old_timestep()<<','<<this->get_timestep_number()<<','<<s.committed_step<<','<<s.accepted_publications<<','<<completed<<'\n';
          auto particles=output("particles-"+label+".csv");
          auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
          bool header=false;
          for (const auto &p:pm.get_particle_handler())
            {
              if (!header)
                {
                  particles<<"id,x,y";
                  for (unsigned int j=0; j<p.get_properties().size(); ++j) particles<<",p"<<j;
                  particles<<'\n';
                  header=true;
                }
              particles<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1];
              for (const auto value:p.get_properties()) particles<<','<<value;
              particles<<'\n';
            }
          std::ofstream rng(this->get_output_directory()+"rng-"+label+".bin",std::ios::binary);
          {
            aspect::oarchive archive(rng);
            archive<<pm.get_random_number_state();
          }
          auto stresses=output("stress-"+label+".csv"),loads=output("loads-"+label+".csv");
          stresses<<"cell,x,y,JxW,phi,chi,segment,N0,N1,xx,yy,xy\n";
          const auto rule=m.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points|update_JxW_values);
          std::vector<double> phase(rule.size()),shear(s.Z.size()),normal(s.Z.size());
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          auto &mutable_manager=this->get_reconstructed_fault_manager();
          mutable_manager.prepare_stokes_qp_projection_cache();
          unsigned int k=0;
          for (const auto &entry:s.dg_history->space->cells)
            {
              const auto cell=s.cells.at(entry.first);
              fe.reinit(cell);
              fe[FEValuesExtractors::Scalar(pc)].get_function_values(s.displacement_history,phase);
              const auto &associations=mutable_manager.get_stokes_qp_fault_associations(entry.first,rule,fe.get_quadrature_points());
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  const auto stress=m.retained_history_stress(entry.first,fe.quadrature_point(q), {});
                  const auto &a=associations[q];
                  AssertThrow(a.active,ExcMessage("PF5 fixture requires the native through-going full-normal map."));
                  const double I=a.shape_0*s.I[a.segment_index]+a.shape_1*s.I[a.segment_index+1];
                  const double chi=s.m*phase[q]*(1+phase[q])/std::pow(1-phase[q],2)/I;
                  stresses<<k<<','<<fe.quadrature_point(q)[0]<<','<<fe.quadrature_point(q)[1]<<','<<fe.JxW(q)<<','<<phase[q]<<','<<chi<<','<<a.segment_index<<','<<a.shape_0<<','<<a.shape_1<<','<<stress[0][0]<<','<<stress[1][1]<<','<<stress[0][1]<<'\n';
                  const auto S=manager.get_shear_sense(0)*symmetrize(outer_product(a.tangent,a.normal));
                  const auto N=symmetrize(outer_product(a.normal,a.normal));
                  for (unsigned int i=0; i<2; ++i)
                    {
                      const double w=fe.JxW(q)*chi*(i?a.shape_1:a.shape_0);
                      shear[a.segment_index+i]+=w*(stress*S);
                      normal[a.segment_index+i]-=w*(stress*N);
                    }
                }
              ++k;
            }
          loads<<"node,shear,normal\n";
          for (unsigned int j=0; j<shear.size(); ++j) loads<<j<<','<<shear[j]<<','<<normal[j]<<'\n';
        }
      public:
        static void declare_parameters(ParameterHandler &p)
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF5 checks");
          p.declare_entry("Steps","16",Patterns::Integer(8,128));
          p.leave_subsection();
          p.leave_subsection();
        }
        void parse_parameters(ParameterHandler &p) override
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF5 checks");
          steps=p.get_integer("Steps");
          p.leave_subsection();
          p.leave_subsection();
        }
        void save(std::map<std::string,std::string> &status) const override
        {
          status["PF5 prescribed progress"]=std::to_string(steps)+" "+std::to_string(completed)+" "+std::to_string(preloaded);
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          std::istringstream in(status.at("PF5 prescribed progress"));
          unsigned int stored;
          in>>stored>>completed>>preloaded;
          AssertThrow(stored==steps && preloaded && completed<steps,ExcMessage("PF5 observer restart progress mismatch."));
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &m=Plugins::get_plugin_as_type<Model>(*sim.material_model);
          gates=output("gates.csv");
          gates<<"gate,passed\n";
          check("PF5 selected",m.dg_accepted_history);
          auto &manager=this->get_reconstructed_fault_manager();
          if (!preloaded)
            {
              m.prepare_reconstructed_fault_mechanical_solve();
              manager.begin_slip_rate_nonlinear_solve();
              manager.begin_slip_rate_trial();
              manager.set_slip_rate_trial_values({std::vector<double>(m.fe_strain_state->Z.size(),.001)});
              manager.accept_slip_rate_trial();
              manager.validate_slip_rate_nonlinear_commit();
              manager.commit_slip_rate_nonlinear_solve();
              sim.timestep_number=1;
              sim.time_step=start;
              sim.time=start;
              const auto zero=m.make_dg_transport(velocity(0),4,false);
              auto preload=m.prepare_dg_accepted_step(*zero,velocity(1),0.);
              m.commit_dg_accepted_step(preload);
              preloaded=true;
              const auto v=velocity(2);
              sim.solution.block(this->introspection().block_indices.velocities)=v.block(this->introspection().block_indices.velocities);
              sim.old_solution=sim.solution;
              sim.old_old_solution=sim.solution;
              record(m,"initial");
            }
          else record(m,"restored");
          const auto v=velocity(2),zero_v=velocity(0);
          const auto op=m.make_dg_transport(v,4,false),zero_op=m.make_dg_transport(zero_v,4,false);
          geometry(m,*op);
          const double dt=duration/steps;
          if (completed==0)
            {
              // Native forcing/relaxation controls from the same nonzero anchor.
              const auto anchor=persistent(m);
              sim.timestep_number=2;
              sim.old_time_step=start;
              sim.time_step=dt;
              sim.time=start+dt;
              auto pure=m.prepare_dg_accepted_step(*zero_op,zero_v,start);
              auto native=m.prepare_dg_accepted_step(*zero_op,v,start);
              const auto c=Model::compute_maxwell_coefficients(m.fe_strain_state->eta,m.fe_strain_state->G,dt);
              const auto full=m.make_dg_history_trial(v,c);
              double relax_error=0.,source_error=0.;
              auto control=output("contributions.csv");
              control<<"relaxation_max_error,native_forcing_max_error\n";
              for (unsigned int comp=0; comp<3; ++comp)
                for (unsigned int i=0; i<op->space->dofs.n_dofs(); ++i)
                  {
                    relax_error=std::max(relax_error,std::abs(pure->dg_history->trial_stress[comp][i]-c.beta*m.fe_strain_state->dg_history->trial_stress[comp][i]));
                    source_error=std::max(source_error,std::abs(native->dg_history->trial_stress[comp][i]-full->trial_stress[comp][i]));
                  }
              control<<relax_error<<','<<source_error<<'\n';
              check("relaxation and source once",relax_error<1.e-10 && source_error<1.e-9);
              check("component controls do not publish",unchanged(anchor,persistent(m)));
            }
          for (unsigned int j=completed; j<steps; ++j)
            {
              const double t=start+j*dt;
              sim.timestep_number=j+2;
              sim.old_time_step=j?dt:start;
              sim.time_step=dt;
              sim.time=t+dt;
              const auto anchor=persistent(m);
              auto trial=m.prepare_dg_accepted_step(*op,v,t),repeat=m.prepare_dg_accepted_step(*op,v,t);
              check("repeated trial "+std::to_string(j),equal(trial,repeat) && unchanged(anchor,persistent(m)));
              if (j==2)
                {
                  sim.time_step=2*dt;
                  sim.time=t+2*dt;
                  auto rejected=m.prepare_dg_accepted_step(*op,v,t);
                  rejected.reset();
                  sim.time_step=dt;
                  sim.time=t+dt;
                  check("discarded larger interval",unchanged(anchor,persistent(m)));
                  check("retry identical",equal(trial,m.prepare_dg_accepted_step(*op,v,t)));
                  auto invalid=m.prepare_dg_accepted_step(*op,v,t);
                  invalid->dg_history->trial_stress[0][0]=std::numeric_limits<double>::quiet_NaN();
                  bool failed=false;
                  try
                    {
                      m.commit_dg_accepted_step(invalid);
                    }
                  catch (const std::exception &)
                    {
                      failed=true;
                    }
                  check("prepublication failure rollback",failed && unchanged(anchor,persistent(m)));
                  const auto wrong=m.make_dg_transport(v,4,true);
                  failed=false;
                  try
                    {
                      m.prepare_dg_accepted_step(*wrong,v,t);
                    }
                  catch (const std::exception &)
                    {
                      failed=true;
                    }
                  check("double relaxation rejected",failed && unchanged(anchor,persistent(m)));
                  manager.begin_slip_rate_nonlinear_solve();
                  manager.begin_slip_rate_trial();
                  manager.set_slip_rate_trial_values({std::vector<double>(m.fe_strain_state->Z.size(),.002)});
                  failed=false;
                  try
                    {
                      m.commit_dg_accepted_step(trial);
                    }
                  catch (const std::exception &)
                    {
                      failed=true;
                    }
                  manager.rollback_slip_rate_nonlinear_solve();
                  check("changed rate rejection and rollback",failed && unchanged(anchor,persistent(m)));
                }
              m.commit_dg_accepted_step(trial);
              const auto committed=persistent(m);
              bool duplicate=false;
              try
                {
                  m.commit_dg_accepted_step(repeat);
                }
              catch (const std::exception &)
                {
                  duplicate=true;
                }
              check("single publication "+std::to_string(j),duplicate && unchanged(committed,persistent(m)) && m.fe_strain_state->accepted_publications==j+2);
              completed=j+1;
              if (completed==steps/2 && !this->get_parameters().resume_computation)
                {
                  record(m,"mid");
                  sim.create_snapshot();
                  check("checkpoint does not mutate",unchanged(committed,persistent(m)));
                }
            }
          record(m,"final");
          check("nonzero Z",m.fe_strain_state->Z[0]>0.);
          return {"PF5 accepted history","completed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF5 accepted history","Bounded prescribed native accepted DG history and filesystem restart checks.")
  }
}
