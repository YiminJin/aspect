/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/particle/manager.h>
#include <aspect/plugins.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>

namespace aspect
{
  namespace
  {
    struct TimelineStep
    {
      std::set<CellId> cells;
      std::map<types::particle_index,std::array<double,3>> inputs;
      std::map<types::particle_index,ReconstructedFaultManager<2>::ParticleFaultAssociation> associations;
      std::ofstream particles, nodes, qp;
    };
    std::map<unsigned int,TimelineStep> timeline;
    std::mutex timeline_mutex;

    std::ofstream open_csv(const std::string &path, const std::string &header)
    {
      std::ofstream file(path);
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file << std::setprecision(17) << header << '\n';
      return file;
    }

    template <int dim>
    class HistoryObserver : public Assemblers::Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &) const override
        {
          // This observer is additive by zero. It samples the actual first
          // cell assembly, not a post-solve reconstruction or a material update.
          std::lock_guard<std::mutex> lock(timeline_mutex);
          AssertThrow(dim==2,ExcNotImplemented());
          AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("This bounded timeline observer is one-rank only."));
          const auto &scratch=dynamic_cast<const internal::Assembly::Scratch::StokesSystem<dim> &>(base);
          const auto &values=scratch.finite_element_values;
          const auto tria_cell=values.get_cell();
          const typename DoFHandler<dim>::active_cell_iterator cell(
            &this->get_triangulation(),tria_cell->level(),tria_cell->index(),&this->get_dof_handler());
          const auto &pm=this->get_particle_manager(0);
          const unsigned int position=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          const unsigned int k=this->get_timestep_number();
          auto &state=timeline[k];
          const bool first=state.cells.insert(cell->id()).second;
          if (!state.particles.is_open())
            {
              if constexpr (dim==2)
                for (const auto &a : this->get_reconstructed_fault_manager().get_locally_owned_particle_fault_associations())
                  state.associations.emplace(a.particle_id,a);
              const std::string suffix="_"+std::to_string(k)+".csv";
              state.particles=open_csv(this->get_output_directory()+"first_assembly_particles"+suffix,
                "step,time,dt,id,x,y,cell,xx,yy,xy,active,fault,segment,xi");
              state.nodes=open_csv(this->get_output_directory()+"first_assembly_nodes"+suffix,
                "cell,x,y,component,published,working");
              state.qp=open_csv(this->get_output_directory()+"first_assembly_qp"+suffix,
                "cell,q,x,y,weight,xx,yy,xy");
            }
          for (const auto &p : pm.get_particle_handler().particles_in_cell(cell))
            {
              const auto properties=p.get_properties();
              const std::array<double,3> stress={{properties[position],properties[position+1],properties[position+2]}};
              if (first)
                {
                  state.inputs.emplace(p.get_id(),stress);
                  const auto &a=state.associations.at(p.get_id());
                  state.particles << k << ',' << this->get_time() << ',' << this->get_timestep() << ','
                    << p.get_id() << ',' << p.get_location()[0] << ',' << p.get_location()[1] << ',' << cell->id().to_string();
                  for (double v : stress) state.particles << ',' << v;
                  state.particles << ',' << a.active << ',' << a.fault_index << ',' << a.segment_index << ',' << a.xi << '\n';
                }
              else
                AssertThrow(state.inputs.at(p.get_id())==stress,
                            ExcMessage("Particle stress changed during mechanics before commit."));
            }
          if (!first) return;
          const auto &intro=this->introspection();
          const auto &fe=this->get_fe();
          const auto &working=this->get_current_linearization_point();
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          cell->get_dof_indices(indices);
          const auto &unit=fe.get_unit_support_points();
          for (unsigned int i=0; i<indices.size(); ++i)
            for (unsigned int c=0; c<3; ++c)
              if (fe.system_to_component_index(i).first==intro.component_indices.compositional_fields[c])
                {
                  const auto p=this->get_mapping().transform_unit_to_real_cell(cell,unit[i]);
                  state.nodes << cell->id().to_string() << ',' << p[0] << ',' << p[1] << ',' << c << ','
                              << this->get_solution()[indices[i]] << ',' << working[indices[i]] << '\n';
                }
          std::array<std::vector<double>,3> stress;
          for (unsigned int c=0; c<3; ++c)
            {
              stress[c].resize(values.n_quadrature_points);
              values[intro.extractors.compositional_fields[c]].get_function_values(working,stress[c]);
            }
          for (unsigned int q=0; q<values.n_quadrature_points; ++q)
            state.qp << cell->id().to_string() << ',' << q << ',' << values.quadrature_point(q)[0] << ','
                     << values.quadrature_point(q)[1] << ',' << values.JxW(q) << ','
                     << stress[0][q] << ',' << stress[1][q] << ',' << stress[2][q] << '\n';
        }
    };

    template <int dim>
    void connect_timeline(SimulatorSignals<dim> &signals)
    {
      signals.set_assemblers.connect([](const SimulatorAccess<dim> &, Assemblers::Manager<dim> &assemblers)
      {
        assemblers.stokes_system.push_back(std::make_unique<HistoryObserver<dim>>());
      });
      signals.post_restore_particles.connect([](Particle::Manager<dim> &pm)
      {
        const unsigned int k=pm.get_timestep_number();
        auto out=open_csv(pm.get_output_directory()+"pre_advection_"+std::to_string(k)+".csv",
                          "step,time,dt,id,x,y,xx,yy,xy");
        const unsigned int position=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
        for (const auto &p : pm.get_particle_handler())
          {
            out << k << ',' << pm.get_time() << ',' << pm.get_timestep() << ',' << p.get_id()
                << ',' << p.get_location()[0] << ',' << p.get_location()[1];
            for (unsigned int c=0; c<3; ++c) out << ',' << p.get_properties()[position+c];
            out << '\n';
          }
      });
    }
    ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_timeline<2>,connect_timeline<3>)
  }

  namespace Postprocess
  {
    template <int dim>
    class HistoryTimeline : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const unsigned int k=this->get_timestep_number();
          auto &state=timeline.at(k);
          state.particles.flush(); state.nodes.flush(); state.qp.flush();
          const auto &pm=this->get_particle_manager(0);
          const unsigned int position=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          auto out=open_csv(this->get_output_directory()+"committed_"+std::to_string(k)+".csv",
                            "step,time,dt,id,x,y,old_xx,old_yy,old_xy,new_xx,new_yy,new_xy");
          for (const auto &p : pm.get_particle_handler())
            {
              out << k << ',' << this->get_time() << ',' << this->get_timestep() << ',' << p.get_id()
                  << ',' << p.get_location()[0] << ',' << p.get_location()[1];
              for (double v : state.inputs.at(p.get_id())) out << ',' << v;
              for (unsigned int c=0; c<3; ++c) out << ',' << p.get_properties()[position+c];
              out << '\n';
            }
          return {"Stress timeline:","first assembly and terminal particle history recorded without reevaluation"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(HistoryTimeline,"stress timeline","Observe stress-history inputs and publication.")
  }
}
