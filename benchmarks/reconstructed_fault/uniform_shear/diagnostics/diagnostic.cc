/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <aspect/phase_field.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/postprocess/reconstructed_faults.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace
  {
    bool initial_state_exported = false;

    template <int dim>
    class PhaseOutput : public DataPostprocessorScalar<dim>
    {
      public:
        explicit PhaseOutput(const unsigned int component)
          : DataPostprocessorScalar<dim>("phi", update_values), component(component) {}
        void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input,
                                   std::vector<Vector<double>> &output) const override
        {
          for (unsigned int q=0; q<output.size(); ++q)
            output[q][0] = input.solution_values[q][component];
        }
      private:
        const unsigned int component;
    };

    template <int dim>
    void export_state(const SimulatorAccess<dim> &sim, const std::string &stage)
    {
      AssertThrow(dim == 2 && Utilities::MPI::n_mpi_processes(sim.get_mpi_communicator()) == 1,
                  ExcMessage("The K1 initialization diagnostic is 2-D and one-rank only."));
      auto file = [&](const std::string &name, const std::string &header)
      {
        std::ofstream out(sim.get_output_directory()+stage+"_"+name);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << std::setprecision(17) << header;
        return out;
      };
      const unsigned int component = sim.introspection().variable("phase_field").first_component_index;
      const auto &vertices = sim.get_triangulation().get_vertices();
      std::map<unsigned int, types::global_dof_index> vertex_dofs;
      auto cells = file("cells.csv", "v0,v1,v2,v3\n");
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            for (const unsigned int v : cell->vertex_indices())
              vertex_dofs[cell->vertex_index(v)] = cell->vertex_dof_index(v, component);
            cells << cell->vertex_index(0) << ',' << cell->vertex_index(1) << ','
                  << cell->vertex_index(2) << ',' << cell->vertex_index(3) << '\n';
          }
      auto nodes = file("nodes.csv", "vertex,dof,x,y,phi\n");
      for (const auto &entry : vertex_dofs)
        nodes << entry.first << ',' << entry.second << ',' << vertices[entry.first][0] << ','
              << vertices[entry.first][1] << ',' << sim.get_solution()[entry.second] << '\n';

      // Native mesh patches with only phi: mechanical fields here have not
      // been solved and must not be mistaken for accepted initial mechanics.
      PhaseOutput<dim> phase(component);
      DataOut<dim> data;
      data.attach_dof_handler(sim.get_dof_handler());
      data.add_data_vector(sim.get_solution(), phase);
      data.build_patches(sim.get_mapping(), 1);
      auto mesh = file("bulk.vtu", "");
      data.write_vtu(mesh);

      const auto &pm = sim.get_phase_field_handler().get_associated_particle_manager();
      const unsigned int H_position = pm.get_property_manager().get_data_info()
                                      .get_position_by_field_name("crack_driving_force");
      auto particles = file("particles.csv", "id,x,y,H,volume\n");
      auto stencil = file("cpdi.csv", "id,vertex,dof,w,gx,gy\n");
      for (const auto &particle : pm.get_particle_handler())
        {
          const auto domain = pm.get_particle_domain_handler().get_particle_domain(particle.get_local_index());
          particles << particle.get_id() << ',' << particle.get_location()[0] << ','
                    << particle.get_location()[1] << ',' << particle.get_properties()[H_position]
                    << ',' << domain.volume() << '\n';
          for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
            {
              const auto vertex = domain.relevant_vertex_index(i);
              const auto gradient = domain.weighting_function_gradient(i);
              stencil << particle.get_id() << ',' << vertex << ',' << vertex_dofs.at(vertex) << ','
                      << domain.weighting_function_value(i) << ',' << gradient[0] << ',' << gradient[1] << '\n';
            }
        }
    }

    template <int dim>
    void before_phase(const SimulatorAccess<dim> &sim)
    {
      if (sim.get_timestep_number() == 0)
        {
          export_state(sim, "before_phase");
          initial_state_exported = true;
        }
    }

    template <int dim>
    void inspect_constraints(const SimulatorAccess<dim> &sim, AffineConstraints<double> &constraints)
    {
      if (!initial_state_exported)
        return;
      std::ofstream out(sim.get_output_directory()+"phase_constraints.csv");
      out << std::setprecision(17) << "dof,master,weight,inhomogeneity\n";
      for (const auto &line : constraints.get_lines())
        {
          if (line.entries.empty())
            out << line.index << ",-1,0," << line.inhomogeneity << '\n';
          for (const auto &entry : line.entries)
            out << line.index << ',' << entry.first << ',' << entry.second << ',' << line.inhomogeneity << '\n';
        }
    }

    template <int dim>
    void before_mechanics(const SimulatorAccess<dim> &sim, Assemblers::Manager<dim> &)
    {
      if (!initial_state_exported || sim.get_reconstructed_fault_manager().get_faults().empty())
        return;
      export_state(sim, "pre_mechanics");
      const auto &manager = sim.get_reconstructed_fault_manager();
      // Reuse the production geometry/property exporter, but do not label
      // V_min as an accepted slip solution: mechanics has not run.
      const Postprocess::internal::ReconstructedFaultOutput<dim> faults(
        manager.get_faults(), manager.get_property_information());
      std::ofstream out(sim.get_output_directory()+"pre_mechanics_fault.vtu");
      faults.write_vtu(out, sim.get_time(), sim.get_timestep_number());
      out.close();
      sim.get_pcout() << "K1_INITIALIZATION_DIAGNOSTIC_COMPLETE: exported before first mechanical residual." << std::endl;
      // Deliberately enter the existing exception/rollback path after files
      // are closed. No solver equation, iterate or acceptance gate is changed.
      AssertThrow(false, ExcMessage("K1_INITIALIZATION_DIAGNOSTIC_COMPLETE"));
    }
  }

  template <int dim>
  void connect_k1_diagnostic(SimulatorSignals<dim> &signals)
  {
    signals.start_timestep.connect(&before_phase<dim>);
    signals.post_constraints_creation.connect(&inspect_constraints<dim>);
    signals.set_assemblers.connect(&before_mechanics<dim>);
  }
  namespace k1_diagnostic_registration
  {
    ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_k1_diagnostic<2>, connect_k1_diagnostic<3>)
  }
}
