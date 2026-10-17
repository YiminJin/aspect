/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Separate identical-field evaluation and frozen-history native mechanics.
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include "../../../tests/phase_field_test_access.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace aspect
{
  namespace Postprocess
  {
    template<int dim> class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;
        using Access=internal::PhaseFieldTestAccess<dim>;
        struct Data
        {
          unsigned int nx=0,ny=0,ns=0;
          std::array<std::vector<double>,3> phase,R,T,Z;
          static std::array<double,3> basis(double x)
          {
            return {{2*(x-.5) *(x-1),4*x *(1-x),2*x *(x-.5)}};
          }
          double phi(unsigned int k,const Point<dim> &x) const
          {
            const unsigned int i=std::min(nx-1,unsigned(x[0]*nx)),j=std::min(ny-1,unsigned(x[1]*ny));
            const double r=x[0]*nx-i,s=x[1]*ny-j;
            return (1-s)*((1-r)*phase[k][j*(nx+1)+i]+r*phase[k][j*(nx+1)+i+1])
                   +s*((1-r)*phase[k][(j+1)*(nx+1)+i]+r*phase[k][(j+1)*(nx+1)+i+1]);
          }
          double displacement(unsigned int k,const Point<dim> &x,unsigned int component) const
          {
            const unsigned int i=std::min(nx-1,unsigned(x[0]*nx)),j=std::min(ny-1,unsigned(x[1]*ny));
            const auto r=basis(x[0]*nx-i),s=basis(x[1]*ny-j);
            double value=0.;
            for (unsigned int a=0; a<3; ++a)for (unsigned int b=0; b<3; ++b)
                value+=r[a]*s[b]*R[k][2*((2*j+b)*(2*nx+1)+2*i+a)+component];
            return value;
          }
          double tensor(unsigned int k,const Point<dim> &x,const Point<dim> &center,unsigned int component) const
          {
            // Select the source polynomial by the target cell interior, preserving
            // the correct one-sided trace at a discontinuous source-cell boundary.
            const unsigned int i=std::min(nx-1,unsigned(center[0]*nx)),j=std::min(ny-1,unsigned(center[1]*ny));
            const auto r=basis(x[0]*nx-i),s=basis(x[1]*ny-j);
            double value=0.;
            for (unsigned int a=0; a<3; ++a)for (unsigned int b=0; b<3; ++b)
                value+=r[a]*s[b]*T[k][3*(9*(j*nx+i)+3*a+b)+component];
            return value;
          }
          void read(const std::string &path)
          {
            std::ifstream file(path);
            file>>nx>>ny>>ns;
            AssertThrow(file && nx==2 && ny==32 && ns==9,ExcMessage("Unexpected common spatial fixture."));
            for (unsigned int k=0; k<3; ++k)
              {
                phase[k].resize((nx+1)*(ny+1));
                R[k].resize((2*nx+1)*(2*ny+1)*2);
                T[k].resize(nx*ny*27);
                Z[k].resize(ns);
                for (auto *field:
                     {
                       &phase[k],&R[k],&T[k],&Z[k]
                     })for (double &v:*field)file>>v;
              }
            AssertThrow(bool(file),ExcMessage("Incomplete common physical fields."));
          }
        } data;
        std::string data_file,prefix;
        bool evolving=false,entry_installed=false;
        std::vector<double> previous_phase,previous_Z;
        void evolving_output()
        {
          auto &m=model();
          auto &sim=simulator();
          const auto step=this->get_timestep_number();
          const auto pb=this->introspection().variable("phase_field").block_index;
          if (!entry_installed)
            {
              check("fresh imported entry only",step==0 && m.uses_dg_trial_stress_history());
              data.read(data_file);
              install(2);
              entry_installed=true;
              previous_phase.resize(sim.solution.block(pb).size());
              for (unsigned int i=0; i<previous_phase.size(); ++i)previous_phase[i]=sim.solution.block(pb)[i];
              previous_Z=m.fe_strain_state->Z;
            }
          const auto &state=*m.fe_strain_state;
          check("one native accepted publication step "+std::to_string(step),state.committed_step==step && state.accepted_publications==step);
          typename Model::FEEndpointTrial accepted;
          accepted.strain=state.displacement_history;
          accepted.Z=state.Z;
          accepted.dg_history=state.dg_history;
          const ReconstructedFaultVector rates= {sim.reconstructed_fault_manager->get_timestep_committed_slip_rate(0)};
          prefix="event-"+std::to_string(step)+"/";
          export_state(accepted,&rates);
          const auto E=m.evaluate_fe_endpoint_energy(sim.solution,accepted);
          const auto F=this->get_phase_field_handler().evaluate_cell_fracture(sim.solution,false);
          double healing=0.,growth=0.,kkt=0.,peak=0.,zerror=0.;
          for (unsigned int i=0; i<previous_phase.size(); ++i)
            {
              const double phi=sim.solution.block(pb)[i],g=E.gradient[i]+F.gradient[i];
              healing=std::max(healing,previous_phase[i]-phi);
              growth=std::max(growth,phi-previous_phase[i]);
              peak=std::max(peak,phi);
              kkt=std::max(kkt,std::abs(phi==previous_phase[i]?std::min(g,0.):g));
              previous_phase[i]=phi;
            }
          if (step)
            {
              const double beta=std::exp(-this->get_timestep()/100),kappa=-1e8*std::expm1(-this->get_timestep()/100);
              for (unsigned int j=0; j<state.Z.size(); ++j)zerror=std::max(zerror,std::abs(state.Z[j]-beta*previous_Z[j]-kappa*rates[0][j]));
              check("Maxwell surface recurrence step "+std::to_string(step),zerror<128*std::numeric_limits<double>::epsilon()*std::max(1.,state.Z[0]));
            }
          check("physical irreversibility step "+std::to_string(step),healing==0.);
          auto file=std::ofstream(this->get_output_directory()+"event-summary.csv",std::ios::app);
          if (file.tellp()==0)file<<"step,time,dt,phase_max,growth,healing,KKT,Zerror,publications,energy\n";
          file<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<peak<<','<<growth<<','<<healing<<','<<kkt<<','<<zerror<<','<<state.accepted_publications<<','<<E.energy+F.energy<<'\n';
          previous_Z=state.Z;
          const QGaussLobatto<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points);
          auto tensor=output("tensor-nodes.csv");
          tensor<<"cell,x0,y0,dx,dy,r,s,xx,yy,xy,ux,uy\n";
          unsigned int c=0;
          for (const auto &entry:state.cells)
            {
              const auto &cell=entry.second;
              fe.reinit(cell);
              const auto T=m.evaluate_dg_trial_tensor(*state.dg_history,entry.first,rule);
              std::vector<Tensor<1,dim>> u(rule.size());
              fe[this->introspection().extractors.velocities].get_function_values(sim.solution,u);
              for (unsigned int q=0; q<rule.size(); ++q)
                tensor<<c<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','<<cell->vertex(1)[0]-cell->vertex(0)[0]<<','<<cell->vertex(2)[1]-cell->vertex(0)[1]<<','<<rule.point(q)[0]<<','<<rule.point(q)[1]<<','<<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]<<','<<u[q][0]<<','<<u[q][1]<<'\n';
              ++c;
            }
        }
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
        void check(const std::string &name,bool value) const
        {
          std::ofstream out(this->get_output_directory()+"gates.csv",std::ios::app);
          out<<name<<','<<value<<'\n';
          AssertThrow(value,ExcMessage("PF6 spatial: "+name));
        }
        void install(unsigned int k)
        {
          auto &m=model();
          auto &sim=simulator();
          const auto pb=this->introspection().variable("phase_field").block_index;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          struct Field:Function<dim>
          {
            const Data &data;
            unsigned int k,pc;
            bool history;
            Field(unsigned int n,const Data &d,unsigned int k,unsigned int pc,bool h):Function<dim>(n),data(d),k(k),pc(pc),history(h) {}
            double value(const Point<dim> &x,unsigned int c) const override
            {
              return c==pc?data.phi(k,x):(history && c<2?data.displacement(k,x,c):0.);
            }
          } phase(this->get_fe().n_components(),data,k,pc,false),history(this->get_fe().n_components(),data,k,pc,true);
          BV p(sim.solution),r(sim.solution);
          p=0.;
          r=0.;
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),phase,p);
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),history,r);
          sim.solution=p;
          sim.current_linearization_point=p;
          auto state=std::make_shared<typename Model::FEStrainState>(*m.fe_strain_state);
          state->displacement_history=m.uses_dg_trial_stress_history()?p:r;
          state->Z=data.Z[k];
          state->stress_samples.clear();
          state->committed_step=0;
          state->accepted_publications=0;
          m.fe_strain_state=state;
          if (m.uses_dg_trial_stress_history())
            {
              auto tensor=std::make_shared<typename Model::DGHistory>(*state->dg_history);
              std::vector<types::global_dof_index> indices(9);
              const auto &support=tensor->space->element.get_unit_support_points();
              for (const auto &entry:tensor->space->cells)
                {
                  const auto &cell=entry.second;
                  cell->get_dof_indices(indices);
                  for (unsigned int q=0; q<9; ++q)
                    {
                      const auto x=this->get_mapping().transform_unit_to_real_cell(cell,support[q]);
                      for (unsigned int c=0; c<3; ++c)tensor->trial_stress[c][indices[q]]=data.tensor(k,x,cell->center(),c);
                    }
                }
              for (auto &component:tensor->trial_stress)component.compress(VectorOperation::insert);
              state->dg_history=tensor;
            }
          state->I=m.evaluate_fe_normalization(p,false).values;
          m.current_normalization_integrals= {state->I};
          std::vector<double> q=state->Z;
          for (unsigned int j=0; j<q.size(); ++j)q[j]/=state->I[j];
          m.commit_cohesive_state({q});
          m.invalidate_normalization_cache();
          m.prepare_reconstructed_fault_mechanical_solve();
          sim.old_solution.block(pb)=p.block(pb);
        }
        std::vector<SymmetricTensor<2,dim>> tensor_values(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                          const Quadrature<dim> &rule,const typename Model::FEEndpointTrial &endpoint) const
        {
          if (model().uses_dg_trial_stress_history())return model().evaluate_dg_trial_tensor(*endpoint.dg_history,cell->id(),rule);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_gradients);
          fe.reinit(cell);
          std::vector<SymmetricTensor<2,dim>> result(rule.size());
          fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(endpoint.strain,result);
          for (auto &value:result)value=2*model().fe_strain_state->G*Utilities::Tensors::consistent_deviator(value);
          return result;
        }
        void export_state(const typename Model::FEEndpointTrial &endpoint,const ReconstructedFaultVector *rates=nullptr)
        {
          auto &m=model();
          auto &sim=simulator();
          const auto &state=*m.fe_strain_state;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          const auto pb=this->introspection().variable("phase_field").block_index;
          const auto offset=sim.solution.get_block_indices().block_start(pb);
          const auto energy=m.evaluate_fe_endpoint_energy(sim.solution,endpoint);
          const auto &I=energy.normalization;
          const unsigned int ny=std::lround(1./(state.cells.begin()->second->vertex(2)[1]-state.cells.begin()->second->vertex(0)[1]));
          check(prefix+"admitted nested mesh",ny==32 || ny==64 || ny==128);
          std::filesystem::create_directory(this->get_output_directory()+prefix);
          auto field=output("field.csv"),jumps=output("traces.csv"),vel=output("velocity.csv");
          field<<"id,cell,x0,y0,dx,dy,x,y,weight,phi,chi,Txx,Tyy,Txy,Zchi,xx,yy,xy\n";
          jumps<<"cell,x,y,side,Txy,Zchi,tau_xy\n";
          vel<<"cell,x,y,ux,uy\n";
          const QGauss<1> xrule(3);
          const QIterated<1> yrule(QGauss<1>(24),128/ny);
          std::vector<Point<dim>> points;
          std::vector<double> weights;
          for (unsigned int j=0; j<yrule.size(); ++j)for (unsigned int i=0; i<xrule.size(); ++i)
              {
                points.emplace_back(xrule.point(i)[0],yrule.point(j)[0]);
                weights.push_back(xrule.weight(i)*yrule.weight(j));
              }
          const Quadrature<dim> rule(points,weights);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points|update_JxW_values);
          unsigned int ci=0;
          for (const auto &entry:state.cells)
            {
              const auto &cell=entry.second;
              fe.reinit(cell);
              std::vector<double> phi(rule.size());
              fe[FEValuesExtractors::Scalar(pc)].get_function_values(sim.solution,phi);
              const auto T=tensor_values(cell,rule,endpoint);
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  const auto x=fe.quadrature_point(q);
                  const unsigned int s=std::min<unsigned int>(I.size()-2,x[0]*(I.size()-1));
                  const double xi=x[0]*(I.size()-1)-s,ii=(1-xi)*I[s]+xi*I[s+1];
                  const double chi=state.m*phi[q]*(1+phi[q])/std::pow(1-phi[q],2)/ii;
                  const double z=(1-xi)*endpoint.Z[s]+xi*endpoint.Z[s+1];
                  const auto tau=m.evaluate_fe_endpoint_stress(endpoint,I,entry.first,x);
                  const unsigned int iy=std::min(127u,unsigned(x[1]*128));
                  const unsigned int local=unsigned(std::lround((q/3)%(24)));
                  const unsigned int id=(iy*24+local)*6+unsigned(x[0]*2)*3+q%3;
                  field<<id<<','<<ci<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','
                       <<cell->vertex(1)[0]-cell->vertex(0)[0]<<','<<cell->vertex(2)[1]-cell->vertex(0)[1]<<','
                       <<x[0]<<','<<x[1]<<','<<fe.JxW(q)<<','<<phi[q]<<','<<chi<<','<<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]<<','<<z *chi<<','
                       <<tau[0][0]<<','<<tau[1][1]<<','<<tau[0][1]<<'\n';
                }
              std::vector<Point<dim>> ends;
              for (unsigned int side=0; side<2; ++side)for (unsigned int i=0; i<3; ++i)ends.emplace_back(xrule.point(i)[0],double(side));
              const Quadrature<dim> edge(ends);
              FEValues<dim> ef(this->get_mapping(),this->get_fe(),edge,update_values|update_quadrature_points);
              ef.reinit(cell);
              phi.resize(edge.size());
              ef[FEValuesExtractors::Scalar(pc)].get_function_values(sim.solution,phi);
              const auto et=tensor_values(cell,edge,endpoint);
              for (unsigned int q=0; q<edge.size(); ++q)
                {
                  const auto x=ef.quadrature_point(q);
                  const unsigned int s=std::min<unsigned int>(I.size()-2,x[0]*(I.size()-1));
                  const double xi=x[0]*(I.size()-1)-s,z=(1-xi)*endpoint.Z[s]+xi*endpoint.Z[s+1];
                  const double zchi=z*state.m*phi[q]*(1+phi[q])/std::pow(1-phi[q],2)/((1-xi)*I[s]+xi*I[s+1]);
                  jumps<<ci<<','<<x[0]<<','<<x[1]<<','<<q/3<<','<<et[q][0][1]<<','<<zchi<<','<<et[q][0][1]-zchi<<'\n';
                }
              const QGaussLobatto<dim> nodes(3);
              FEValues<dim> vf(this->get_mapping(),this->get_fe(),nodes,update_values|update_quadrature_points);
              vf.reinit(cell);
              std::vector<Tensor<1,dim>> u(nodes.size());
              vf[this->introspection().extractors.velocities].get_function_values(sim.solution,u);
              for (unsigned int q=0; q<nodes.size(); ++q)vel<<ci<<','<<vf.quadrature_point(q)[0]<<','<<vf.quadrature_point(q)[1]<<','<<u[q][0]<<','<<u[q][1]<<'\n';
              ++ci;
            }
          // Common ordinary-FE fracture functional on the actual mechanical rule.
          const auto coefficients=Access::coefficients(this->get_phase_field_handler(), {.5,.5},.3,0.);
          double fracture=0.;
          std::vector<double> gf(energy.gradient.size(),0.);
          FEValues<dim> ff(this->get_mapping(),this->get_fe(),m.mechanical_quadrature(),update_values|update_gradients|update_JxW_values);
          std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
          std::map<types::global_dof_index,Point<dim>> nodes;
          for (const auto &entry:state.cells)
            {
              const auto &cell=entry.second;
              ff.reinit(cell);
              cell->get_dof_indices(indices);
              std::vector<double> phi(ff.n_quadrature_points);
              std::vector<Tensor<1,dim>> gradient(phi.size());
              ff[FEValuesExtractors::Scalar(pc)].get_function_values(sim.solution,phi);
              ff[FEValuesExtractors::Scalar(pc)].get_function_gradients(sim.solution,gradient);
              for (unsigned int q=0; q<phi.size(); ++q)
                {
                  fracture+=ff.JxW(q)*(coefficients.reaction*phi[q]+.5*coefficients.diffusion*gradient[q].norm_square());
                  for (const auto v:cell->vertex_indices())
                    {
                      const auto i=this->get_fe().component_to_system_index(pc,v);
                      const auto global=indices[i]-offset;
                      gf[global]+=ff.JxW(q)*(coefficients.reaction*ff.shape_value_component(i,q,pc)+coefficients.diffusion*(gradient[q]*ff.shape_grad_component(i,q,pc)));
                    }
                }
              for (const auto v:cell->vertex_indices())nodes[cell->vertex_dof_index(v,pc)-offset]=cell->vertex(v);
            }
          std::vector<double> gc(gf.size(),0.);
          double cpdi=0.,allowance=0.;
          if (!m.uses_dg_trial_stress_history())
            {
              check(prefix+"actual CPDI fracture admitted",this->get_phase_field_handler().evaluate_fe_fracture(sim.system_matrix,sim.system_rhs,sim.solution,cpdi,allowance));
              for (unsigned int i=0; i<gc.size(); ++i)gc[i]=-sim.system_rhs.block(pb)[i];
            }
          auto grad=output("gradient.csv");
          grad<<"i,x,y,phi,elastic,FE_fracture,CPDI_fracture\n";
          for (const auto &node:nodes)grad<<node.first<<','<<node.second[0]<<','<<node.second[1]<<','<<sim.solution.block(pb)[node.first]<<','<<energy.gradient[node.first]<<','<<gf[node.first]<<','<<gc[node.first]<<'\n';
          auto totals=output("energy.csv");
          totals<<"bulk,cohesive,FE_fracture,CPDI_fracture,CPDI_available\n"<<energy.bulk<<','<<energy.cohesive<<','<<fracture<<','<<cpdi<<','<<(!m.uses_dg_trial_stress_history())<<'\n';
          auto surface=output("surface.csv");
          surface<<"i,x,I,Z,V\n";
          for (unsigned int j=0; j<I.size(); ++j)surface<<j<<','<<state.vertices[j][0]<<','<<I[j]<<','<<endpoint.Z[j]<<','<<(rates?(*rates)[0][j]:0.)<<'\n';
        }
      public:
        static void declare_parameters(ParameterHandler &p)
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6a spatial");
          p.declare_entry("Common fields file","",Patterns::FileName());
          p.declare_entry("Evolving imported entry","false",Patterns::Bool());
          p.leave_subsection();
          p.leave_subsection();
        }
        void parse_parameters(ParameterHandler &p) override
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6a spatial");
          data_file=p.get("Common fields file");
          evolving=p.get_bool("Evolving imported entry");
          p.leave_subsection();
          p.leave_subsection();
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &m=model();
          auto &sim=simulator();
          if (evolving)
            {
              evolving_output();
              return {"PF6a spatial","native coupled common-entry event"};
            }
          check("initial observer only",this->get_timestep_number()==0);
          data.read(data_file);
          const auto original=m.fe_strain_state;
          const auto original_I=m.current_normalization_integrals;
          auto &fault=sim.reconstructed_fault_manager->get_fault(0);
          std::vector<std::vector<double>> original_properties;
          for (unsigned int j=0; j<fault.n_vertices(); ++j)
            {
              const auto p=fault.get_properties(j);
              original_properties.emplace_back(p.begin(),p.end());
            }
          const BV solution(sim.solution),linear(sim.current_linearization_point),old(sim.old_solution);
          auto clock=Access::diagnostic_endpoint_clock(sim,.5625,1./128);
          install(0);
          typename Model::FEEndpointTrial identical;
          identical.strain=m.fe_strain_state->displacement_history;
          identical.Z=m.fe_strain_state->Z;
          identical.dg_history=m.fe_strain_state->dg_history;
          prefix="identical/";
          export_state(identical);
          install(2);
          identical.strain=m.fe_strain_state->displacement_history;
          identical.Z=m.fe_strain_state->Z;
          identical.dg_history=m.fe_strain_state->dg_history;
          prefix="entry_identical/";
          export_state(identical);
          install(1);
          const auto Z=m.fe_strain_state->Z;
          const BV anchor(m.fe_strain_state->displacement_history);
          const auto tensor=m.uses_dg_trial_stress_history()?m.dg_history_values():std::vector<double>();
          ReconstructedFaultVector rates;
          Access::mechanical_trial(sim,rates);
          check("phase frozen during native mechanical resolution",sim.solution.block(this->introspection().variable("phase_field").block_index)==anchor.block(this->introspection().variable("phase_field").block_index));
          const auto defects=Access::endpoint_defects(sim,sim.solution,rates);
          prefix="resolved/";
          const auto endpoint=m.make_fe_endpoint_trial(sim.solution,rates);
          export_state(endpoint,&rates);
          auto residual=output("residual.csv");
          residual<<"bulk_norm,surface_RMS\n"<<defects.first<<','<<defects.second<<'\n';
          check("incoming histories unchanged",m.fe_strain_state->Z==Z && m.fe_strain_state->displacement_history==anchor && (!m.uses_dg_trial_stress_history() || m.dg_history_values()==tensor));
          check("no diagnostic publication",m.fe_strain_state->accepted_publications==0 && m.fe_strain_state->committed_step==0);
          m.fe_strain_state=original;
          clock();
          sim.solution=solution;
          sim.current_linearization_point=linear;
          sim.old_solution=old;
          for (unsigned int j=0; j<fault.n_vertices(); ++j)
            {
              auto p=fault.get_properties(j);
              std::copy(original_properties[j].begin(),original_properties[j].end(),p.begin());
            }
          m.current_normalization_integrals=original_I;
          m.invalidate_normalization_cache();
          m.prepare_reconstructed_fault_mechanical_solve();
          sim.compute_current_constraints();
          return {"PF6a spatial","identical physical fields and independently resolved frozen mechanics"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF6a spatial","Bounded serial frozen-field and mechanical-resolution diagnostics.")
  }
}

namespace aspect
{
  namespace BoundaryVelocity
  {
    template<int dim> class PF6aSpatialRamp:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
        {
          Tensor<1,dim> u;
          if (this->get_timestep_number())u[0]=.256*(this->get_time()-.5*this->get_timestep())*(x[1]-.5);
          return u;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(PF6aSpatialRamp,"PF6a spatial interval ramp","Common-entry benchmark loading; exact interval mean and unloaded startup.")
  }
}
