/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Bounded native fixed-reference transfer and outer-lifecycle observer.
#include <aspect/simulator.h>
#include <aspect/mesh_refinement/interface.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include "../../../tests/phase_field_test_access.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <chrono>

namespace aspect
{
  namespace
  {
    int transfer_direction=0;
  }
  namespace MeshRefinement
  {
    template<int dim> class TransferMesh:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        void execute(Vector<float> &indicator) const override
        {
          indicator=0.;
        }
        void tag_additional_cells() const override
        {
          if (!transfer_direction) return;
          for (const auto &cell:this->get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->clear_refine_flag();
                cell->clear_coarsen_flag();
                const auto x=(transfer_direction<0 && cell->level()>0 ? cell->parent()->center() : cell->center());
                const bool patch=x[0]>.5 && x[0]<.75 && x[1]>.375 && x[1]<.4375;
                if (std::abs(transfer_direction)==2 && (!patch || (transfer_direction<0 && cell->level()!=3)))continue;
                if (transfer_direction>0)cell->set_refine_flag();
                else if (cell->level()>0)cell->set_coarsen_flag();
              }
        }
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(TransferMesh,"PF6 transfer mesh","Explicit uniform flags for the bounded transfer observer.")
  }
  namespace Postprocess
  {
    template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;
        using Access=internal::PhaseFieldTestAccess<dim>;
        using Records=std::map<std::string,std::vector<double>>;
        using Particles=std::map<types::particle_index,Point<dim>>;
        bool initialized=false,recovery_done=false,transfer_checks=true,snapshot_quadrature=false,local_checks=false,deep_uniform=true;
        unsigned int last_step=0;
        Particles initial_particles;
        Model &model() const
        {
          return const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));
        }
        Simulator<dim> &sim() const
        {
          return const_cast<Simulator<dim> &>(this->get_simulator());
        }
        bool root() const
        {
          return Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0;
        }
        void check(const std::string &name,bool passed) const
        {
          const bool global=Utilities::MPI::min(passed ? 1u:0u,this->get_mpi_communicator())==1;
          if (root())
            {
              std::ofstream f(this->get_output_directory()+"transfer-gates.csv",std::ios::app);
              if (f.tellp()==0)f<<"gate,passed\n";
              f<<name<<','<<global<<std::endl;
            }
          AssertThrow(global,ExcMessage("PF6 transfer gate: "+name));
        }
        void measure(const std::string &name,double error,double scale=1.) const
        {
          error=Utilities::MPI::max(error,this->get_mpi_communicator());
          scale=Utilities::MPI::max(scale,this->get_mpi_communicator());
          if (root())
            {
              std::ofstream f(this->get_output_directory()+"transfer-errors.csv",std::ios::app);
              f<<std::setprecision(17);
              if (f.tellp()==0)f<<"quantity,error,scale,relative\n";
              f<<name<<','<<error<<','<<scale<<','<<error/scale<<'\n';
            }
        }
        Particles particles(const std::string &label) const
        {
          Particles local,global;
          const bool domains=this->get_particle_manager(0).particle_domains_requested();
          if(model().uses_homogeneous_mesh_data())
            {
              check(label+" no particle domains configured or constructed",!domains && !this->get_particle_manager(0).particle_domain_handler);
              const auto &associations=this->get_reconstructed_fault_manager().get_locally_owned_particle_fault_associations();
              bool points_only=true;
              for(const auto &entry:associations)
                points_only=points_only && !std::isfinite(entry.particle_domain_volume) && entry.quadrature.empty();
              check(label+" geometric point associations have no domain measure",points_only);
              if(root())
                {
                  std::ofstream f(this->get_output_directory()+label+"-domain-free.csv");
                  f<<"domains_requested,domain_handler_present,point_associations_only,particles\n"
                   <<domains<<','<<bool(this->get_particle_manager(0).particle_domain_handler)<<','<<points_only<<','
                   <<this->get_particle_manager(0).get_particle_handler().n_global_particles()<<'\n';
                }
            }
          std::ostringstream text;
          text<<std::setprecision(17);
          for (const auto &p:this->get_particle_manager(0).get_particle_handler())
            {
              local.emplace(p.get_id(),p.get_location());
              text<<p.get_id()<<','<<Utilities::MPI::this_mpi_process(this->get_mpi_communicator())<<','<<p.get_surrounding_cell()->id().to_string();
              for (unsigned int d=0; d<dim; ++d)text<<','<<p.get_location()[d];
              for (unsigned int d=0; d<dim; ++d)text<<','<<p.get_reference_location()[d];
              if(domains)
                {
                  const auto domain=this->get_particle_manager(0).get_particle_domain_handler().get_particle_domain(p.get_local_index());
                  std::array<double,4> box={{1.,1.,0.,0.}};
                  for(const auto &x:domain.vertices())for(unsigned int d=0;d<2;++d)
                    {box[d]=std::min(box[d],x[d]);box[d+2]=std::max(box[d+2],x[d]);}
                  text<<','<<domain.volume();
                  for(double x:box)text<<','<<x;
                }
              for (double value:p.get_properties())text<<','<<value;
              text<<'\n';
            }
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
            for (const auto &p:part)check_local_insert(global,p);
          const auto rows=Utilities::MPI::all_gather(this->get_mpi_communicator(),text.str());
          if (root())
            {
              std::ofstream f(this->get_output_directory()+label+"-particles.csv");
              f<<"id,rank,CellId,x,y,r,s";
              if(domains)f<<",domain_volume,xmin,ymin,xmax,ymax";
              for (unsigned int i=0; i<this->get_particle_manager(0).get_particle_handler().n_properties_per_particle(); ++i)f<<",property"<<i;
              f<<'\n';
              for (const auto &row:rows)f<<row;
            }
          check(label+" unique particle owners",global.size()==this->get_particle_manager(0).get_particle_handler().n_global_particles());
          return global;
        }
        static void check_local_insert(Particles &map,const typename Particles::value_type &p)
        {
          AssertThrow(map.emplace(p).second,ExcMessage("Duplicate particle ID owner."));
        }
        Records coefficients() const
        {
          auto &m=model();
          Records local,global;
          const auto &h=*m.fe_strain_state->dg_history;
          std::vector<types::global_dof_index> ids(9);
          for (const auto &entry:h.space->cells)if (entry.second->is_locally_owned())
              {
                entry.second->get_dof_indices(ids);
                auto &v=local[entry.first.to_string()];
                for (const auto &c:h.trial_stress)for (auto i:ids)v.push_back(c[i]);
              }
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
            for (const auto &p:part)AssertThrow(global.emplace(p).second,ExcMessage("Duplicate DG CellId owner."));
          return global;
        }
        void assign(const Records &records)
        {
          auto &h=*model().fe_strain_state->dg_history;
          std::vector<types::global_dof_index> ids(9);
          for (auto &c:h.trial_stress)
            {
              LinearAlgebra::Vector owned(h.space->dofs.locally_owned_dofs(),this->get_mpi_communicator());
              const unsigned int component=&c-&h.trial_stress[0];
              for (const auto &entry:h.space->cells)if (entry.second->is_locally_owned())
                  {
                    entry.second->get_dof_indices(ids);
                    for (unsigned int i=0; i<9; ++i)owned[ids[i]]=records.at(entry.first.to_string())[9*component+i];
                  }
              owned.compress(VectorOperation::insert);
              c=owned;
            }
          h.source_roundoff.clear();
        }
        std::vector<double> persistent() const
        {
          auto &m=model();
          auto v=m.dg_history_values();
          const auto &s=*m.fe_strain_state;
          v.insert(v.end(),s.Z.begin(),s.Z.end());
          v.insert(v.end(),s.I.begin(),s.I.end());
          for (auto i:s.displacement_history.locally_owned_elements())v.push_back(s.displacement_history[i]);
          v.push_back(s.committed_step);
          v.push_back(s.accepted_publications);
          const auto &manager=this->get_reconstructed_fault_manager();
          for (const auto &f:manager.get_faults())for (unsigned int j=0; j<f.n_vertices(); ++j)for (double x:f.get_properties(j))v.push_back(x);
          for (double x:manager.get_slip_rate(0))v.push_back(x);
          for (double x:manager.get_timestep_committed_slip_rate(0))v.push_back(x);
          for (const auto &p:this->get_particle_manager(0).get_particle_handler())
            {
              v.push_back(p.get_id());
              for (unsigned int d=0; d<dim; ++d)v.push_back(p.get_location()[d]);
              for (double x:p.get_properties())v.push_back(x);
            }
          return v;
        }
        static bool same(const std::vector<double> &a,const std::vector<double> &b)
        {
          return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;
        }
        std::string rng() const
        {
          std::ostringstream out;
          {
            aspect::oarchive ar(out);
            ar<<this->get_particle_manager(0).get_random_number_state();
          }
          return out.str();
        }
        Records physical() const
        {
          auto &m=model();
          const auto &s=*m.fe_strain_state;
          Records local,global;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          // Common points independent of the current mesh. None lies on a child
          // face for the two uniform meshes in this fixture.
          for (unsigned int ix=0; ix<8; ++ix)for (unsigned int iy=0; iy<128; ++iy)
              {
                Point<dim> x;
                x[0]=(ix+.5)/8;
                x[1]=(iy+.5)/128;
                for (const auto &entry:s.cells)
                  {
                    const auto cell=entry.second;
                    bool inside=true;
                    for (unsigned int d=0; d<dim; ++d)inside=inside && x[d]>cell->vertex(0)[d] && x[d]<cell->vertex(1u<<d)[d];
                    if (!inside)continue;
                    const auto u=this->get_mapping().transform_real_to_unit_cell(cell,x);
                    FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({u}),update_values);
                    fe.reinit(cell);
                    std::vector<double> phi(1),lower(1);
                    const FEValuesExtractors::Scalar p(pc);
                    fe[p].get_function_values(this->get_solution(),phi);
                    fe[p].get_function_values(s.displacement_history,lower);
                    const auto T=m.evaluate_dg_trial_tensor(*s.dg_history,entry.first,Quadrature<dim>({u}))[0];
                    const auto tau=m.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,entry.first,x);
                    local[std::to_string(ix)+"-"+std::to_string(iy)]= {x[0],x[1],phi[0],lower[0],T[0][0],T[1][1],T[0][1],tau[0][0],tau[1][1],tau[0][1]};
                  }
              }
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
            for (const auto &p:part)AssertThrow(global.emplace(p).second,ExcMessage("Duplicate physical point owner."));
          check("common point ownership",global.size()==1024);
          return global;
        }
        void export_state(const std::string &label) const
        {
          const auto h=coefficients(),p=physical();
          Records local_cells,global_cells;
          for (const auto &entry:model().fe_strain_state->cells)
            local_cells[entry.first.to_string()]= {entry.second->vertex(0)[0],entry.second->vertex(0)[1],entry.second->vertex(1)[0],entry.second->vertex(2)[1]};
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local_cells))global_cells.insert(part.begin(),part.end());
          if (root())
            {
              std::ofstream geometry(this->get_output_directory()+label+"-cells.csv");
              geometry<<std::setprecision(17)<<"CellId,x0,y0,x1,y1\n";
              for (const auto &r:global_cells)
                {
                  geometry<<r.first;
                  for (double v:r.second)geometry<<','<<v;
                  geometry<<'\n';
                }
              std::ofstream nodes(this->get_output_directory()+"dg-support.csv");
              nodes<<std::setprecision(17)<<"i,r,s\n";
              const auto &unit=model().fe_strain_state->dg_history->space->element.get_unit_support_points();
              for (unsigned int i=0; i<unit.size(); ++i)nodes<<i<<','<<unit[i][0]<<','<<unit[i][1]<<'\n';
              std::ofstream f(this->get_output_directory()+label+"-tensor.csv");
              f<<std::setprecision(17)<<"CellId";
              for (unsigned int i=0; i<27; ++i)f<<",c"<<i;
              f<<'\n';
              for (const auto &r:h)
                {
                  f<<r.first;
                  for (double x:r.second)f<<','<<x;
                  f<<'\n';
                }
              std::ofstream g(this->get_output_directory()+label+"-physical.csv");
              g<<std::setprecision(17)<<"key,x,y,phase,lower,Txx,Tyy,Txy,xx,yy,xy\n";
              for (const auto &r:p)
                {
                  g<<r.first;
                  for (double x:r.second)g<<','<<x;
                  g<<'\n';
                }
              const auto &s=*model().fe_strain_state;
              std::ofstream a(this->get_output_directory()+label+"-surface.csv");
              a<<std::setprecision(17)<<"i,Z,I,V,committed_V";
              const auto &manager=this->get_reconstructed_fault_manager();
              const auto &fault=manager.get_fault(0);
              for (unsigned int i=0; i<fault.get_properties(0).size(); ++i)a<<",property"<<i;
              a<<'\n';
              for (unsigned int j=0; j<s.Z.size(); ++j)
                {
                  a<<j<<','<<s.Z[j]<<','<<s.I[j]<<','<<manager.get_slip_rate(0)[j]<<','<<manager.get_timestep_committed_slip_rate(0)[j];
                  for (double x:fault.get_properties(j))a<<','<<x;
                  a<<'\n';
                }
            }
          auto &m=model();
          const auto &s=*m.fe_strain_state;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          Records nodal,all_nodal;
          for (const auto &entry:s.cells)for (const auto v:entry.second->vertex_indices())
              {
                const auto x=entry.second->vertex(v);
                const auto i=entry.second->vertex_dof_index(v,pc);
                std::ostringstream key;
                key<<std::setprecision(17)<<x[0]<<","<<x[1];
                nodal[key.str()]= {this->get_solution()[i],s.displacement_history[i],this->get_old_solution()[i],this->get_old_old_solution()[i]};
              }
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),nodal))all_nodal.insert(part.begin(),part.end());
          // Same physical weak functional before/after remapping, without
          // applying a Maxwell step or feeding samples into mechanics.
          std::vector<double> shear(s.Z.size(),0.),cohesive(s.Z.size(),0.);
          const auto quad=m.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_JxW_values|update_quadrature_points);
          for (const auto &entry:s.cells)
            {
              fe.reinit(entry.second);
              std::vector<double> phi(quad.size());
              fe[FEValuesExtractors::Scalar(pc)].get_function_values(s.displacement_history,phi);
              for (unsigned int q=0; q<quad.size(); ++q)
                {
                  const auto x=fe.quadrature_point(q);
                  const auto j=std::min<unsigned int>(s.I.size()-2,x[0]*(s.I.size()-1));
                  const double xi=x[0]*(s.I.size()-1)-j;
                  const double I=(1-xi)*s.I[j]+xi*s.I[j+1],chi=s.m*phi[q]*(1+phi[q])/std::pow(1-phi[q],2)/I;
                  const auto stress=m.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,entry.first,x);
                  shear[j]+=fe.JxW(q)*chi*s.shear_sense*stress[0][1]*(1-xi);
                  shear[j+1]+=fe.JxW(q)*chi*s.shear_sense*stress[0][1]*xi;
                }
            }
          aspect::internal::endpoint_sum(shear,this->get_mpi_communicator());
          const QGauss<1> line(12);
          const unsigned int ns=s.Z.size()-1;
          for (unsigned int j=0; j<ns; ++j)for (unsigned int q=0; q<line.size(); ++q)
              {
                const double xi=line.point(q)[0],value=((1-xi)*s.Z[j]+xi*s.Z[j+1])/((1-xi)*s.I[j]+xi*s.I[j+1]);
                cohesive[j]+=line.weight(q)/ns*(1-xi)*value;
                cohesive[j+1]+=line.weight(q)/ns*xi*value;
              }
          if (root())
            {
              std::ofstream a(this->get_output_directory()+label+"-phase.csv");
              a<<std::setprecision(17)<<"x,y,phase,lower,old,older\n";
              for (const auto &r:all_nodal)
                {
                  a<<r.first;
                  for (double x:r.second)a<<','<<x;
                  a<<'\n';
                }
              std::ofstream b(this->get_output_directory()+label+"-weak-loads.csv");
              b<<std::setprecision(17)<<"i,shear,cohesive\n";
              for (unsigned int j=0; j<shear.size(); ++j)b<<j<<','<<shear[j]<<','<<cohesive[j]<<'\n';
            }
          {
            std::ofstream random(this->get_output_directory()+label+"-rng-rank-"+Utilities::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".txt");
            random<<rng();
          }
          std::ostringstream native_rows;
          native_rows<<std::setprecision(17);
          const Quadrature<dim> native_nodes(s.dg_history->space->element.get_unit_support_points());
          FEValues<dim> native_fe(this->get_mapping(),this->get_fe(),native_nodes,update_values);
          for (const auto &entry:s.cells)
            {
              native_fe.reinit(entry.second);
              const unsigned int nc=this->get_fe().n_components();
              std::vector<Vector<double>> current(native_nodes.size(),Vector<double>(nc)),old=current,older=current,anchor=current;
              native_fe.get_function_values(this->get_solution(),current);
              native_fe.get_function_values(this->get_old_solution(),old);
              native_fe.get_function_values(this->get_old_old_solution(),older);
              native_fe.get_function_values(s.displacement_history,anchor);
              for (unsigned int q=0; q<native_nodes.size(); ++q)for (unsigned int c=0; c<nc; ++c)
                  native_rows<<entry.first.to_string()<<','<<q<<','<<c<<','<<current[q][c]<<','<<old[q][c]<<','<<older[q][c]<<','<<anchor[q][c]<<'\n';
            }
          const auto native_parts=Utilities::MPI::all_gather(this->get_mpi_communicator(),native_rows.str());
          if (root())
            {
              std::ofstream f(this->get_output_directory()+label+"-native-fields.csv");
              f<<"CellId,node,component,solution,old,older,anchor\n";
              for (const auto &r:native_parts)f<<r;
            }
          const auto E=model().evaluate_fe_phase_energy(this->get_solution(),false);
          const auto F=this->get_phase_field_handler().evaluate_cell_fracture(this->get_solution(),true);
          const unsigned int phase_block=this->introspection().variable("phase_field").block_index;
          const auto phase_offset=s.displacement_history.get_block_indices().block_start(phase_block);
          std::map<types::global_dof_index,Point<dim>> local_phase_nodes,phase_nodes;
          for (const auto &entry:s.cells)for (const auto v:entry.second->vertex_indices())local_phase_nodes[entry.second->vertex_dof_index(v,pc)-phase_offset]=entry.second->vertex(v);
          for (const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local_phase_nodes))phase_nodes.insert(part.begin(),part.end());
          Vector<double> direction(F.gradient.size()),ones(direction),action(direction),constant_action(direction);
          ones=1.;
          for (const auto &entry:phase_nodes)direction[entry.first]=std::exp(-17*(std::pow(entry.second[1]-.5,2)+.3*std::pow(entry.second[0]-.4,2)));
          F.hessian.vmult(action,direction);
          F.hessian.vmult(constant_action,ones);
          if (root())
            {
              std::ofstream f(this->get_output_directory()+label+"-fracture-operator.csv");
              f<<std::setprecision(17)<<"x,y,gradient,roundoff,Hd,Hones\n";
              for (const auto &entry:phase_nodes)f<<entry.second[0]<<','<<entry.second[1]<<','<<F.gradient[entry.first]<<','<<F.roundoff_scale[entry.first]<<','<<action[entry.first]<<','<<constant_action[entry.first]<<'\n';
            }
          if (root())
            {
              std::ofstream f(this->get_output_directory()+label+"-energy.csv");
              f<<std::setprecision(17)<<"bulk,cohesive,fracture,step,publications\n"<<E.bulk<<','<<E.cohesive<<','<<F.energy<<','<<model().fe_strain_state->committed_step<<','<<model().fe_strain_state->accepted_publications<<'\n';
            }
        }
        void compare(const std::string &label,const Records &before,const Records &after,double bound) const
        {
          double error=0.,scale=1.;
          bool keys=before.size()==after.size();
          for (const auto &r:before)
            {
              if (!after.count(r.first))
                {
                  keys=false;
                  continue;
                }
              for (unsigned int i=0; i<r.second.size(); ++i)
                {
                  scale=std::max(scale,std::abs(r.second[i]));
                  error=std::max(error,std::abs(r.second[i]-after.at(r.first)[i]));
                }
            }
          measure(label,error,scale);
          check(label,keys && error/scale<bound);
        }
        void mesh(int direction)
        {
          const auto before=this->get_triangulation().n_global_active_cells();
          const auto start=std::chrono::steady_clock::now();
          transfer_direction=direction;
          sim().refine_mesh(4);
          transfer_direction=0;
          sim().compute_current_constraints();
          // The start_timestep signal precedes the normal matrix rebuild. A
          // diagnostic solve from that signal must complete native setup first.
          if (sim().rebuild_sparsity_and_matrices)
            {
              sim().setup_system_matrix(this->introspection().index_sets.system_partitioning);
              sim().setup_system_preconditioner(this->introspection().index_sets.system_partitioning);
              sim().rebuild_sparsity_and_matrices=false;
            }
          sim().rebuild_stokes_matrix=true;
          sim().rebuild_stokes_preconditioner=true;
          const double seconds=Utilities::MPI::max(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),this->get_mpi_communicator());
          if (root())
            {
              std::ofstream f(this->get_output_directory()+"transfer-cost.csv",std::ios::app);
              if (f.tellp()==0)f<<"direction,before,after,seconds\n";
              f<<std::setprecision(17)<<direction<<','<<before<<','<<this->get_triangulation().n_global_active_cells()<<','<<seconds<<'\n';
            }
        }
        void roundtrip(const std::string &label,const int direction=1)
        {
          const auto before=physical(),tensor=coefficients();
          const auto Z=model().fe_strain_state->Z,I=model().fe_strain_state->I;
          const auto publications=model().fe_strain_state->accepted_publications;
          const auto random=rng();
          const auto time=this->get_time(),dt=this->get_timestep();
          const auto step=this->get_timestep_number();
          export_state(label+"-before");
          mesh(direction);
          export_state(label+"-refined");
          compare(label+" refinement",before,physical(),1e-11);
          check(label+" refined scratch empty",model().fe_strain_state->dg_history->source_roundoff.empty());
          check(label+" refined fixed particle IDs coordinates",particles(label+"-refined")==initial_particles);
          mesh(-direction);
          export_state(label+"-coarsened");
          compare(label+" roundtrip physical",before,physical(),1e-11);
          compare(label+" roundtrip coefficients",tensor,coefficients(),1e-11);
          check(label+" physical clock unchanged",time==this->get_time() && dt==this->get_timestep() && step==this->get_timestep_number());
          check(label+" Z I and publication unchanged",model().fe_strain_state->Z==Z && model().fe_strain_state->I==I && model().fe_strain_state->accepted_publications==publications);
          check(label+" pure remap RNG unchanged",random==rng());
          check(label+" coarsened fixed particle IDs coordinates",particles(label+"-coarsened")==initial_particles);
        }
        void local_operators()
        {
          const FE_DGQ<dim> fe(2);
          const QGauss<dim> quadrature(5);
          const unsigned int n=fe.n_dofs_per_cell();
          const auto &nodes=fe.get_unit_support_points();
          double prolong_error=0.,restrict_error=0.,matrix_error=0.,mean_error=0.;
          FullMatrix<double> mass(n,n);
          for (unsigned int q=0; q<quadrature.size(); ++q)for (unsigned int i=0; i<n; ++i)for (unsigned int j=0; j<n; ++j)mass(i,j)+=quadrature.weight(q)*fe.shape_value(i,quadrature.point(q))*fe.shape_value(j,quadrature.point(q));
          mass.gauss_jordan();
          for (unsigned int child=0; child<GeometryInfo<dim>::max_children_per_cell; ++child)
            {
              FullMatrix<double> cross(n,n),independent(n,n);
              for (unsigned int q=0; q<quadrature.size(); ++q)
                {
                  const auto x=GeometryInfo<dim>::child_to_cell_coordinates(quadrature.point(q),child);
                  for (unsigned int i=0; i<n; ++i)for (unsigned int j=0; j<n; ++j)cross(i,j)+=quadrature.weight(q)/GeometryInfo<dim>::max_children_per_cell*fe.shape_value(i,x)*fe.shape_value(j,quadrature.point(q));
                }
              mass.mmult(independent,cross);
              const auto &R=fe.get_restriction_matrix(child);
              for (unsigned int i=0; i<n; ++i)for (unsigned int j=0; j<n; ++j)matrix_error=std::max(matrix_error,std::abs(R(i,j)-independent(i,j)));
            }
          for (unsigned int a=0; a<3; ++a)for (unsigned int b=0; b<3; ++b)
              {
                Vector<double> original(n),sum(n),fine(n),back(n);
                for (unsigned int i=0; i<n; ++i)original[i]=std::pow(nodes[i][0],a)*std::pow(nodes[i][1],b);
                for (unsigned int child=0; child<4; ++child)
                  {
                    fe.get_prolongation_matrix(child).vmult(fine,original);
                    for (unsigned int i=0; i<n; ++i)
                      {
                        auto x=GeometryInfo<dim>::child_to_cell_coordinates(nodes[i],child);
                        prolong_error=std::max(prolong_error,std::abs(fine[i]-std::pow(x[0],a)*std::pow(x[1],b)));
                      }
                    fe.get_restriction_matrix(child).vmult(back,fine);
                    sum+=back;
                  }
                for (unsigned int i=0; i<n; ++i)restrict_error=std::max(restrict_error,std::abs(original[i]-sum[i]));
              }
          // Genuinely discontinuous fine data: coarsening must preserve its
          // integral, but cannot preserve its norm or all information.
          Vector<double> coarse(n);
          double fine_mean=0.,fine_norm=0.;
          for (unsigned int child=0; child<4; ++child)
            {
              Vector<double> v(n),r(n);
              for (unsigned int i=0; i<n; ++i)v[i]=(child+1)*(1+nodes[i][0]*nodes[i][1]+.3*std::pow(nodes[i][0]*nodes[i][1],2));
              fe.get_restriction_matrix(child).vmult(r,v);
              coarse+=r;
              for (unsigned int q=0; q<quadrature.size(); ++q)
                {
                  double value=0.;
                  for (unsigned int i=0; i<n; ++i)value+=v[i]*fe.shape_value(i,quadrature.point(q));
                  fine_mean+=quadrature.weight(q)*value/4;
                  fine_norm+=quadrature.weight(q)*value*value/4;
                }
            }
          double coarse_mean=0.,coarse_norm=0.,l2=0.;
          for (unsigned int q=0; q<quadrature.size(); ++q)
            {
              double v=0.;
              for (unsigned int i=0; i<n; ++i)v+=coarse[i]*fe.shape_value(i,quadrature.point(q));
              coarse_mean+=quadrature.weight(q)*v;
              coarse_norm+=quadrature.weight(q)*v*v;
            }
          for (unsigned int child=0; child<4; ++child)for (unsigned int q=0; q<quadrature.size(); ++q)
              {
                const auto x=GeometryInfo<dim>::child_to_cell_coordinates(quadrature.point(q),child);
                double v=0.;
                for (unsigned int i=0; i<n; ++i)v+=coarse[i]*fe.shape_value(i,x);
                const auto z=quadrature.point(q);
                const double f=(child+1)*(1+z[0]*z[1]+.3*std::pow(z[0]*z[1],2));
                l2+=quadrature.weight(q)*std::pow(v-f,2)/4;
              }
          mean_error=std::abs(fine_mean-coarse_mean);
          measure("all nine monomials refinement",prolong_error);
          measure("all nine monomials roundtrip",restrict_error);
          measure("independent Gauss5 mass restriction",matrix_error);
          measure("lossy coarsening mean",mean_error);
          measure("lossy coarsening L2",std::sqrt(l2));
          measure("lossy coarsening quadratic norm decrease",fine_norm-coarse_norm);
          measure("L2 orthogonality identity",std::abs(fine_norm-coarse_norm-l2));
          check("independent all-mode transfer operators",prolong_error<1e-12 && restrict_error<1e-12 && matrix_error<1e-12 && mean_error<1e-12 && fine_norm>coarse_norm && std::abs(fine_norm-coarse_norm-l2)<1e-12);
        }
        void trace_control(const std::string &label="matching-cell-trace")
        {
          auto &m=model();
          const auto before=persistent();
          class Velocity:public Function<dim>
          {
              const unsigned int uc;
            public:
              Velocity(unsigned int n,unsigned int u):Function<dim>(n),uc(u) {}
              double value(const Point<dim> &x,unsigned int c=0) const override
              {
                return c==uc ? x[1]*x[1]+std::max(0.,x[1]-.5) : 0.;
              }
          } velocity(this->get_fe().n_components(),this->introspection().component_indices.velocities[0]);
          BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=this->get_solution();
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),velocity,owned,this->introspection().component_masks.velocities);
          BV bulk(this->get_solution());
          bulk=owned;
          const auto c=m.compute_maxwell_coefficients(m.fe_strain_state->eta,m.fe_strain_state->G,.01);
          const auto projected=m.make_dg_history_trial(*m.fe_strain_state->dg_history,bulk,c,true);
          std::vector<Point<dim>> points;
          std::vector<CellId> ids;
          for (const auto &entry:m.fe_strain_state->cells)
            if (entry.second->vertex(0)[1]==.5 || entry.second->vertex(2)[1]==.5)
              {
                auto x=entry.second->center();
                x[1]=.5;
                points.push_back(x);
                ids.push_back(entry.first);
              }
          Utilities::MPI::RemotePointEvaluation<dim> cache;
          cache.reinit(points,this->get_triangulation(),this->get_mapping());
          const auto averages=VectorTools::point_gradients<dim>(cache,this->get_dof_handler(),bulk,VectorTools::EvaluationFlags::avg,this->introspection().component_indices.velocities[0]);
          double error=0.,jump=0.;
          unsigned int negative=0;
          std::ostringstream rows;
          rows<<std::setprecision(17);
          for (unsigned int i=0; i<points.size(); ++i)
            {
              const auto cell=m.fe_strain_state->cells.at(ids[i]);
              const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,points[i]);
              FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_gradients);
              fe.reinit(cell);
              std::vector<Tensor<2,dim>> gradient(1);
              fe[this->introspection().extractors.velocities].get_function_gradients(bulk,gradient);
              const auto incoming=m.evaluate_dg_trial_tensor(*m.fe_strain_state->dg_history,ids[i],Quadrature<dim>({unit}))[0];
              const auto exact=c.beta*incoming+2*c.eta_ve*Utilities::Tensors::consistent_deviator(symmetrize(gradient[0]));
              const auto avg=c.beta*incoming+2*c.eta_ve*Utilities::Tensors::consistent_deviator(symmetrize(averages[i]));
              const auto T=m.evaluate_dg_trial_tensor(*projected,ids[i],Quadrature<dim>({unit}))[0];
              const double bound=m.dg_source_publication_allowance(*projected,ids[i],points[i])+128*std::numeric_limits<double>::epsilon()*std::max(1.,2*T.norm());
              const auto d=T-exact,wrong=T-avg;
              const double e=std::sqrt(std::pow(d.norm(),2)+std::pow(trace(d),2)),w=std::sqrt(std::pow(wrong.norm(),2)+std::pow(trace(wrong),2));
              error=std::max(error,e);
              jump=std::max(jump,w);
              m.validate_fe_strain_publication(e,bound);
              try
                {
                  m.validate_fe_strain_publication(w,bound);
                }
              catch (const std::exception &)
                {
                  ++negative;
                }
              rows<<ids[i].to_string()<<','<<points[i][0]<<','<<points[i][1]<<','<<gradient[0][0][1]<<','<<averages[i][0][1]<<','<<T[0][1]<<','<<e<<','<<w<<','<<bound<<'\n';
            }
          const auto all=Utilities::MPI::all_gather(this->get_mpi_communicator(),rows.str());
          if (root())
            {
              std::ofstream f(this->get_output_directory()+label+".csv");
              f<<"CellId,x,y,one_sided_gradient,averaged_gradient,DGxy,matching_error,averaged_error,bound\n";
              for (const auto &row:all)f<<row;
            }
          measure("matching-cell native source error",error);
          measure("averaged trace inconsistency",jump);
          const auto total=Utilities::MPI::sum(negative,this->get_mpi_communicator());
          check("averaged trace negative controls rejected",negative==points.size() && total>0);
          check("trace control unpublished",same(before,persistent()));
        }
        void inadmissible_coarsening(const int direction=1)
        {
          const auto progress=[this](const std::string &label)
          {
            std::ofstream f(this->get_output_directory()+"coarsening-progress-rank-"+Utilities::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".txt",std::ios::app);
            f<<label<<std::endl;
          };
          auto &s=sim();
          auto &m=model();
          progress("before refine");
          mesh(direction);
          progress("after refine");
          const auto saved=s.solution,linear=s.current_linearization_point,old=s.old_solution,older=s.old_old_solution;
          progress("saved native vectors");
          const auto anchor=m.fe_strain_state->displacement_history;
          const auto block=this->introspection().variable("phase_field").block_index;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          progress("saved anchor");
          BV perturbed(s.solution);
          LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          progress("allocated owned");
          owned=s.solution;
          progress("copied owned");
          for (const auto &entry:m.fe_strain_state->cells)for (const auto v:entry.second->vertex_indices())
              if (entry.second->vertex(v)[0]==(direction==1?.25:.625) && entry.second->vertex(v)[1]==(direction==1?.5:.40625))
                {
                  const auto i=entry.second->vertex_dof_index(v,pc);
                  if (owned.locally_owned_elements().is_element(i))owned[i]=static_cast<double>(owned[i])+.01;
                }
          progress("before compress");
          owned.compress(VectorOperation::insert);
          progress("before ghost copy");
          perturbed=owned;
          progress("after ghost copy");
          s.solution=perturbed;
          progress("copied solution");
          m.fe_strain_state->displacement_history.block(block)=perturbed.block(block);
          progress("copied anchor phase");
          const auto before=persistent();
          const auto n=this->get_triangulation().n_global_active_cells();
          bool rejected=false;
          progress("before negative coarsen");
          try
            {
              mesh(-direction);
            }
          catch (const std::exception &e)
            {
              rejected=std::string(e.what()).find("nonrepresentable physical phase/lower obstacle")!=std::string::npos;
              if (!rejected)throw;
            }
          progress("after negative coarsen");
          transfer_direction=0;
          check("nonrepresentable phase obstacle rejected before mesh mutation",rejected && n==this->get_triangulation().n_global_active_cells() && same(before,persistent()) && !m.dg_transfer);
          for (const auto &cell:this->get_triangulation().active_cell_iterators())if (cell->is_locally_owned())cell->clear_coarsen_flag();
          s.solution=saved;
          s.current_linearization_point=linear;
          s.old_solution=old;
          s.old_old_solution=older;
          m.fe_strain_state->displacement_history=anchor;
          mesh(-direction);
        }
        void lossy_coarsening()
        {
          const auto accepted=coefficients();
          mesh(1);
          auto fine=coefficients();
          const auto &h=*model().fe_strain_state->dg_history;
          for (const auto &entry:h.space->cells)if (entry.second->is_locally_owned())
              for (unsigned int c=0; c<3; ++c)for (unsigned int i=0; i<9; ++i)
                  fine[entry.first.to_string()][9*c+i]=std::sin((c+1.)*(i+1.)+5*entry.second->vertex(0)[0]+7*entry.second->vertex(0)[1]);
          assign(fine);
          export_state("lossy-fine");
          mesh(-1);
          export_state("lossy-coarse");
          check("lossy tensor coarsening does not publish",model().fe_strain_state->accepted_publications==0);
          assign(accepted);
        }
        void startup()
        {
          local_operators();
          trace_control();
          inadmissible_coarsening();
          const auto accepted=coefficients();
          Records manufactured=accepted;
          const auto &h=*model().fe_strain_state->dg_history;
          for (const auto &entry:h.space->cells)if (entry.second->is_locally_owned())
              for (unsigned int c=0; c<3; ++c)for (unsigned int i=0; i<9; ++i)
                  {
                    const auto x=this->get_mapping().transform_unit_to_real_cell(entry.second,h.space->element.get_unit_support_points()[i]);
                    double v=0.;
                    for (unsigned int a=0; a<3; ++a)for (unsigned int b=0; b<3; ++b)v+=(1+c+3*a+5*b)*((c+a+b)%2?-1.:1.)*std::pow(x[0],a)*std::pow(x[1],b);
                    manufactured[entry.first.to_string()][9*c+i]=v;
                  }
          assign(manufactured);
          roundtrip("global-complete-Q2");
          assign(accepted);
          lossy_coarsening();
          // Keep this mesh for the next real physical step: original cell-center
          // particles now lie on child faces and exercise actual publication.
          mesh(1);
        }
        void normalization_derivatives()
        {
          auto &m=model();
          const auto before=persistent();
          const auto &s=*m.fe_strain_state;
          const auto b=this->introspection().variable("phase_field").block_index;
          const auto pc=this->introspection().variable("phase_field").first_component_index;
          const auto offset=this->get_solution().get_block_indices().block_start(b);
          const auto phase=aspect::internal::endpoint_values(s.displacement_history.block(b),this->get_mpi_communicator());
          std::vector<double> local(phase.size()),direction(phase.size());
          for(const auto &entry:s.cells)for(const auto v:entry.second->vertex_indices())
            {
              const auto i=entry.second->vertex_dof_index(v,pc)-offset;
              if(s.displacement_history.block(b).locally_owned_elements().is_element(i))
                local[i]=phase[i]*(1-phase[i])*(1+.2*entry.second->vertex(v)[0]);
            }
          Utilities::MPI::sum(local,this->get_mpi_communicator(),direction);
          const auto base=m.evaluate_fe_normalization(s.displacement_history,true,&direction);
          constexpr double epsilon=1e-5;
          BV positive(s.displacement_history),negative(s.displacement_history);
          BV owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=s.displacement_history;
          for(const auto i:owned.block(b).locally_owned_elements())owned.block(b)[i]+=epsilon*direction[i];
          positive=owned;
          owned=s.displacement_history;
          for(const auto i:owned.block(b).locally_owned_elements())owned.block(b)[i]-=epsilon*direction[i];
          negative=owned;
          const auto plus=m.evaluate_fe_normalization(positive,true),minus=m.evaluate_fe_normalization(negative,true);
          check("normalization derivative trial admissible",base.admissible && plus.admissible && minus.admissible);
          double first_error=0.,first_scale=1.,second_error=0.,second_scale=1.;
          std::ostringstream text;
          text<<std::setprecision(17)<<"i,I,dI,d2I,fd_dI\n";
          for(unsigned int i=0;i<base.values.size();++i)
            {
              std::vector<double> force(base.values.size()),dforce(force),action(phase.size());
              force[i]=1.;
              m.add_fe_normalization_action_pullback(base,direction,force,dforce,action);
              double second=0.;
              for(unsigned int j=0;j<phase.size();++j)
                {
                  second+=direction[j]*action[j];
                  const double fd=(plus.derivative(i,j)-minus.derivative(i,j))/(2*epsilon);
                  second_error=std::max(second_error,std::abs(action[j]-fd));
                  second_scale=std::max(second_scale,std::abs(action[j]));
                }
              const double fd=(plus.values[i]-minus.values[i])/(2*epsilon);
              first_error=std::max(first_error,std::abs(base.action[i]-fd));
              first_scale=std::max(first_scale,std::abs(base.action[i]));
              text<<i<<','<<base.values[i]<<','<<base.action[i]<<','<<second<<','<<fd<<'\n';
            }
          measure("normalization directional derivative FD",first_error,first_scale);
          measure("normalization Hessian action FD",second_error,second_scale);
          check("normalization first and second derivatives retained",first_error/first_scale<1e-7 && second_error/second_scale<1e-7);
          check("normalization derivative diagnostics do not publish",same(before,persistent()));
          if(root())
            {
              std::ofstream f(this->get_output_directory()+"normalization-operator.csv");
              f<<text.str();
            }
        }
        void rejection()
        {
          auto &s=sim();
          const auto before=persistent();
          const auto random=rng();
          const auto bulk=s.solution,linear=s.current_linearization_point,old=s.old_solution,older=s.old_old_solution;
          const auto metadata=Access::endpoint_bookkeeping(s);
          Access::endpoint_failure_hook(s,[this]()
          {
            if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())+1==Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()))AssertThrow(false,ExcMessage("selected transfer rejection on last owner"));
          });
          bool rejected=false;
          try
            {
              Access::complete_step(s);
            }
          catch (const std::exception &e)
            {
              rejected=std::string(e.what()).find("selected transfer rejection")!=std::string::npos;
              if (!rejected)throw;
            }
          Access::endpoint_failure_hook(s, {});
          check("actual outer last-owner rejection",rejected);
          check("rejection persistent unchanged",same(before,persistent()));
          check("rejection RNG unchanged",random==rng());
          check("rejection vectors clocks bookkeeping unchanged",s.solution==bulk && s.current_linearization_point==linear && s.old_solution==old && s.old_old_solution==older && metadata==Access::endpoint_bookkeeping(s));
          Access::complete_step(s);
          check("retry staged and unpublished",bool(s.pending_dg_endpoint_publication) && same(before,persistent()));
          const auto cells=this->get_triangulation().n_global_active_cells();
          bool blocked=false;
          try
            {
              mesh(1);
            }
          catch (const std::exception &e)
            {
              blocked=std::string(e.what()).find("unpublished endpoint")!=std::string::npos;
              if (!blocked)throw;
            }
          transfer_direction=0;
          check("pending endpoint rejects transfer",blocked && cells==this->get_triangulation().n_global_active_cells() && same(before,persistent()));
          auto discard=std::move(s.pending_dg_endpoint_publication);
          s.pending_dg_endpoint_publication= {};
          discard(false);
          check("discard retry persistent unchanged",same(before,persistent()) && random==rng());
          check("discard retry bookkeeping unchanged",metadata==Access::endpoint_bookkeeping(s));
          Access::complete_step(s);
          const auto &particle=*this->get_particle_manager(0).get_particle_handler().begin();
          const auto id=particle.get_surrounding_cell()->id();
          const auto cell=model().fe_strain_state->evaluation_cells.at(id);
          const bool last=Utilities::MPI::this_mpi_process(this->get_mpi_communicator())+1==Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
          if (last)model().fe_strain_state->evaluation_cells.erase(id);
          auto publish=std::move(s.pending_dg_endpoint_publication);
          s.pending_dg_endpoint_publication= {};
          bool source_rejected=false;
          try
            {
              publish(true);
            }
          catch (const std::exception &e)
            {
              source_rejected=std::string(e.what()).find("DG matching-cell sampling failed")!=std::string::npos;
              if (!source_rejected)throw;
            }
          if (last)model().fe_strain_state->evaluation_cells.emplace(id,cell);
          check("terminal matching-cell error reaches all ranks",source_rejected);
          check("terminal source rejection preserves histories RNG",same(before,persistent()) && random==rng());
          check("terminal source rejection restores clocks vectors bookkeeping",s.solution==bulk && s.current_linearization_point==linear && s.old_solution==old && s.old_old_solution==older && metadata==Access::endpoint_bookkeeping(s));
          recovery_done=true;
        }
        void quadrature_snapshot()
        {
          auto &m=model();
          const auto saved=m.fe_strain_state;
          const auto before=persistent();
          const auto random=rng();
          const auto original=m.fe_bulk_normal_quadrature_points;
          for (const unsigned int q:
          {
            6u,12u,24u,32u
          })
          {
            m.fe_bulk_normal_quadrature_points=q;
            m.fe_strain_state.reset();
            m.prepare_fe_strain_history(false,true);
            auto &s=*m.fe_strain_state;
            s.dg_history=saved->dg_history;
            s.Z=saved->Z;
            s.I=saved->I;
            s.initial_fracture=saved->initial_fracture;
            s.committed_step=saved->committed_step;
            s.accepted_publications=saved->accepted_publications;
            export_state("frozen-G"+std::to_string(q));
              check("frozen quadrature fixed particles G"+std::to_string(q),particles("frozen-G"+std::to_string(q))==initial_particles);
            check("frozen physical quadrature preserves accepted state G"+std::to_string(q),same(before,persistent()) && random==rng());
          }
          m.fe_bulk_normal_quadrature_points=original;
          m.fe_strain_state=saved;
          m.invalidate_normalization_cache();
        }
        void hanging_constraints(const std::string &label) const
        {
          const auto &s=sim();
          double error=0.;
          unsigned int count=0;
          const auto offset=s.solution.get_block_indices().block_start(this->introspection().variable("phase_field").block_index);
          const auto phase_size=model().fe_strain_state->displacement_history.block(this->introspection().variable("phase_field").block_index).size();
          const std::array<const BV *,4> fields= {{&s.solution,&s.old_solution,&s.old_old_solution,&model().fe_strain_state->displacement_history}};
          std::array<std::vector<double>,4> values;
          for (unsigned int k=0; k<fields.size(); ++k)
            {
              values[k].resize(fields[k]->size());
              for (const auto i:fields[k]->locally_owned_elements())values[k][i]=(*fields[k])[i];
              aspect::internal::endpoint_sum(values[k],this->get_mpi_communicator());
            }
          for (const auto &line:s.constraints.get_lines())
            if (!line.entries.empty())
              {
                if (line.index>=offset && line.index<offset+phase_size)++count;
                for (const auto &v:values)
                  {
                    double expected=line.inhomogeneity;
                    for (const auto &entry:line.entries)expected+=entry.second*v[entry.first];
                    error=std::max(error,std::abs(v[line.index]-expected));
                  }
              }
          count=Utilities::MPI::max(count,this->get_mpi_communicator());
          measure(label+" native hanging constraints",error);
          check(label+" actual phase hanging nodes",count>0 && error<1e-10);
        }
        void local_transfer_checks()
        {
          const auto no_op_before=persistent();
          const auto no_op_random=rng();
          const auto no_op_cells=this->get_triangulation().n_global_active_cells();
          for(unsigned int i=0;i<2;++i)
            {
              mesh(2);
              check("smoothed local no-op preserves accepted state",this->get_triangulation().n_global_active_cells()==no_op_cells
                    && !model().dg_transfer && same(no_op_before,persistent()) && no_op_random==rng());
            }
          // Permit the explicit isolated diagnostic cell while retaining 2:1
          // vertex balance. Physical steps used the production smoothing flags.
          sim().triangulation.set_mesh_smoothing(Triangulation<dim>::limit_level_difference_at_vertices);
          // Capture the first local repartition before the production cache
          // consumes domains. Raw old slots are read only inside the known
          // pretransfer index range. New slots outside that range are NaN.
          const auto &pm=this->get_particle_manager(0);
          const auto capacity=pm.get_particle_handler().get_max_local_particle_index();
          const auto generation=pm.get_particle_domain_handler().geometry_version();
          std::map<types::particle_index,types::particle_index> old_slots;
          for (const auto &p:pm.get_particle_handler())old_slots[p.get_local_index()]=p.get_id();
          const auto dump_domains=[this,capacity,generation,old_slots](const std::string &label)
          {
            const auto &manager=this->get_particle_manager(0);
            const auto &domains=manager.get_particle_domain_handler();
            std::ofstream f(this->get_output_directory()+label+"-rank-"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
            f<<std::setprecision(17)<<"id,index,x,y,CellId,generation,old_capacity,previous_slot_id,raw_slot_volume,raw_slot_vertices,xmin,ymin,xmax,ymax\n";
            for (const auto &p:manager.get_particle_handler())
              {
                const auto i=p.get_local_index();
                const bool in_range=i<(domains.geometry_version()!=generation?manager.get_particle_handler().get_max_local_particle_index():capacity);
                const double volume=in_range?domains.get_particle_domain(i).volume():std::numeric_limits<double>::quiet_NaN();
                f<<p.get_id()<<','<<i<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','<<p.get_surrounding_cell()->id().to_string()<<','<<domains.geometry_version()<<','<<capacity<<',';
                if (old_slots.count(i))f<<old_slots.at(i);
                f<<','<<volume<<','<<(in_range?domains.get_particle_domain(i).vertices().size():0);
                std::array<double,4> bounds;
                bounds.fill(std::numeric_limits<double>::quiet_NaN());
                if(in_range && !domains.get_particle_domain(i).vertices().empty())
                  {
                    const auto &vertices=domains.get_particle_domain(i).vertices();
                    bounds={{vertices[0][0],vertices[0][1],vertices[0][0],vertices[0][1]}};
                    for(const auto &x:vertices)for(unsigned int d=0;d<2;++d)
                        {bounds[d]=std::min(bounds[d],x[d]);bounds[d+2]=std::max(bounds[d+2],x[d]);}
                  }
                for(double x:bounds)f<<','<<x;
                f<<'\n';
              }
          };
          dump_domains("first-local-before");
          bool captured=false;
          boost::signals2::scoped_connection capture(this->get_signals().post_refinement_load_user_data.connect([&](typename parallel::distributed::Triangulation<dim> &)
          {
            if (!captured)
              {
                dump_domains("first-local-after");
                captured=true;
                check("local first-remap particle domains rebuilt",pm.get_particle_domain_handler().geometry_version()>generation);
              }
          }));
          const auto accepted=coefficients();
          auto manufactured=accepted;
          const auto &h=*model().fe_strain_state->dg_history;
          for (const auto &entry:h.space->cells)if (entry.second->is_locally_owned())
              for (unsigned int c=0; c<3; ++c)for (unsigned int i=0; i<9; ++i)
                  {
                    const auto x=this->get_mapping().transform_unit_to_real_cell(entry.second,h.space->element.get_unit_support_points()[i]);
                    double v=0.;
                    for (unsigned int a=0; a<3; ++a)for (unsigned int b=0; b<3; ++b)v+=(1+c+3*a+5*b)*((c+a+b)%2?-1.:1.)*std::pow(x[0],a)*std::pow(x[1],b);
                    manufactured[entry.first.to_string()][9*c+i]=v;
                  }
          assign(manufactured);
          roundtrip("local-complete-Q2",2);
          assign(accepted);
          roundtrip("local-accepted",2);
          inadmissible_coarsening(2);
          const auto before=physical();
          const auto Z=model().fe_strain_state->Z,I=model().fe_strain_state->I;
          const auto count=model().fe_strain_state->accepted_publications;
          mesh(2);
          compare("local held physical state",before,physical(),1e-11);
          hanging_constraints("local-held");
          trace_control("local-matching-cell-trace");
          const auto persistent_before=persistent();
          const auto random=rng();
          bool blocked=false;
          try
            {
              sim().solve_reconstructed_fault_endpoint_phase(false);
            }
          catch (const std::exception &e)
            {
              blocked=std::string(e.what()).find("coupled hanging-node")!=std::string::npos
                      && std::string(e.what()).find("remain unqualified")!=std::string::npos;
              if (!blocked)throw;
            }
          check("local coupled phase evolution remains guarded",blocked && same(persistent_before,persistent()) && random==rng());
          check("local held fixed surface and publication",Z==model().fe_strain_state->Z && I==model().fe_strain_state->I && count==model().fe_strain_state->accepted_publications);
          check("local held fixed particle IDs coordinates",particles("local-held")==initial_particles);
          export_state("local-held");
        }
      public:
        static void declare_parameters(ParameterHandler &p)
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6 transfer checks");
          p.declare_entry("Transfer checks","true",Patterns::Bool());
          p.declare_entry("Snapshot quadrature control","false",Patterns::Bool());
          p.declare_entry("Local transfer checks","false",Patterns::Bool());
          p.declare_entry("Deep uniform roundtrip","true",Patterns::Bool());
          p.leave_subsection();
          p.leave_subsection();
        }
        void parse_parameters(ParameterHandler &p) override
        {
          p.enter_subsection("Postprocess");
          p.enter_subsection("PF6 transfer checks");
          transfer_checks=p.get_bool("Transfer checks");
          snapshot_quadrature=p.get_bool("Snapshot quadrature control");
          local_checks=p.get_bool("Local transfer checks");
          deep_uniform=p.get_bool("Deep uniform roundtrip");
          p.leave_subsection();
          p.leave_subsection();
        }
        void initialize() override
        {
          this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim> &,double &)
          {
            if (snapshot_quadrature)quadrature_snapshot();
            if(model().uses_homogeneous_mesh_data())
              {
                particles("mesh-restored");
                export_state("mesh-restored");
              }
            if (local_checks)
              {
                sim().triangulation.set_mesh_smoothing(Triangulation<dim>::limit_level_difference_at_vertices);
                hanging_constraints("local-restored");
                check("local restored fixed IDs coordinates",particles("local-restored")==initial_particles);
                export_state("local-restored");
              }
          });
          this->get_signals().post_checkpoint.connect([this](const std::string &)
          {
            if(model().uses_homogeneous_mesh_data())
              {
                particles("mesh-checkpoint");
                export_state("mesh-checkpoint");
              }
            if (local_checks)
              {
                hanging_constraints("local-checkpoint");
                particles("local-checkpoint");
                export_state("local-checkpoint");
              }
          });
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &)
          {
            if (!this->get_timestep_number())return;
            AssertThrow(!((snapshot_quadrature || local_checks) && this->get_parameters().resume_computation),
                        ExcMessage("Restore-only local/quadrature control must terminate before another physical timestep."));
            const auto label="before-step-"+std::to_string(this->get_timestep_number());
            check(label+" particles fixed",particles(label)==initial_particles);
            export_state(label);
            if (!recovery_done)rejection();
          });
        }
        void save(std::map<std::string,std::string> &status) const override
        {
          std::ostringstream out;
          {
            aspect::oarchive a(out);
            a &initialized &recovery_done &last_step &initial_particles;
          }
          status["PF6 transfer observer"]=out.str();
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          std::istringstream in(status.at("PF6 transfer observer"));
          aspect::iarchive a(in);
          a &initialized &recovery_done &last_step &initial_particles;
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto step=this->get_timestep_number();
          if (initialized && step==last_step)return {"PF6 transfer","restored observer"};
          if (!initialized)
            {
              check("effective RK2 freeze environment present",std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION")!=nullptr);
              if (root())
                {
                  std::ofstream f(this->get_output_directory()+"particle-freeze.txt");
                  f<<"integrator=RK2\nenvironment_present=1\nvalue="<<std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION")<<"\neffective_advection_dt=0\nconstitutive_dt=unchanged\n";
                }
              initial_particles=particles("initial");
              export_state("prescribed-initial");
              if (transfer_checks)startup();
              initialized=true;
            }
          check("accepted publication "+std::to_string(step),model().fe_strain_state->committed_step==step && model().fe_strain_state->accepted_publications==step);
          check("accepted fixed IDs coordinates "+std::to_string(step),particles("state-"+std::to_string(step))==initial_particles);
          export_state("state-"+std::to_string(step));
          if(step==1 && transfer_checks)normalization_derivatives();
          if (step==1 && transfer_checks && deep_uniform)roundtrip("accepted-step-1");
          if (step==2 && local_checks)local_transfer_checks();
          last_step=step;
          return {"PF6 transfer","fixed-reference diagnostics complete"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF6 fixed reference transfer","Bounded native accepted-history transfer and ownership checks.")
  }
}
