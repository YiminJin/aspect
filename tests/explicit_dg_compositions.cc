/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/prescribed_solution/interface.h>
#include <aspect/material_model/simple.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class DGCoefficients : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto rank=std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
          const auto step=this->get_timestep_number();
          std::ofstream clocks(this->get_output_directory()+"publication-"+rank,step==0 ? std::ios::trunc : std::ios::app);
          clocks << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep() << '\n';
          std::ofstream file(this->get_output_directory()+"coefficients-"+std::to_string(step)+"-"+rank);
          file << std::setprecision(17) << this->get_time() << ' ' << this->get_timestep() << '\n';
          const auto &intro=this->introspection();
          const auto &system=this->get_fe();
          std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                for (unsigned int c=0; c<intro.n_compositional_fields; ++c)
                  {
                    const auto &base=system.base_element(intro.base_elements.compositional_fields[c]);
                    file << cell->id() << ' ' << intro.name_for_compositional_index(c) << ' ' << base.get_name() << ' '
                         << cell->vertex(0) << ' ' << cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
                    for (unsigned int i=0; i<base.n_dofs_per_cell(); ++i)
                      file << ' ' << this->get_solution()[indices[system.component_to_system_index(intro.component_indices.compositional_fields[c],i)]];
                    file << '\n';
                  }
              }
          return {"DG coefficients","written"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(DGCoefficients,"DG coefficient samples","Export owned DG coefficients by field and cell.")
  }

  namespace PrescribedSolution
  {
    template <int dim>
    class Companion : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void constrain_solution(const typename DoFHandler<dim>::active_cell_iterator &,
                                const std::vector<Point<dim>> &positions,
                                const std::vector<unsigned int> &components,
                                std::vector<bool> &constrained,
                                std::vector<double> &values) override
        {
          const auto &intro = this->introspection();
          const auto component = intro.component_indices.compositional_fields[intro.compositional_index_for_name("companion")];
          for (unsigned int i=0; i<positions.size(); ++i)
            if (components[i] == component && positions[i][0] < .51)
              {
                constrained[i] = true;
                values[i] = .25 + positions[i][0];
              }
        }
    };
    ASPECT_REGISTER_PRESCRIBED_SOLUTION(Companion, "prescribed companion", "Prescribe the left companion values.")
  }

  namespace MaterialModel
  {
    template <int dim>
    class TracerReaction : public Simple<dim>
    {
      public:
        void evaluate(const MaterialModelInputs<dim> &in,MaterialModelOutputs<dim> &out) const override
        {
          Simple<dim>::evaluate(in,out);
          const auto c=this->introspection().compositional_index_for_name("tracer");
          for (auto &row : out.reaction_terms)
            row[c]=this->get_timestep()>0. ? .01 : 0.;
        }
    };
    ASPECT_REGISTER_MATERIAL_MODEL(TracerReaction,"tracer reaction","Add a whole-step increment to the named tracer.")
  }
}
