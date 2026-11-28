/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field/midpoint_fracture.h>
#include <fstream>

namespace aspect::Postprocess
{
  /** Export the local rows used by the growing phase solve. The checker
   * reconstructs both test and trial traces independently on physical boxes. */
  template <int dim>
  class GrowingFractureRows : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        if constexpr (dim == 2)
          {
            const auto &model = this->get_parameters().surface_energy_model;
            const auto field = this->introspection().compositional_index_for_name(model.composition);
            const auto component = this->introspection().component_indices.compositional_fields[field];
            const auto block = this->introspection().block_indices.compositional_fields[field];
            const auto offset = this->get_solution().get_block_indices().block_start(block);
            const PhaseField::FractureRows rows(*this,field);
            const auto &fe = this->get_fe();
            const auto rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            const auto path = this->get_output_directory() + "fracture-rows-"
                              + Utilities::int_to_string(this->get_timestep_number(),4)
                              + "-r" + std::to_string(rank) + ".txt";
            std::ofstream out(path);
            out << std::setprecision(17);
            out << "scale " << model.length_scale << ' ' << this->get_parameters().phase_discontinuous_penalty << '\n';
            std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
            for (const auto &cell : this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto box = cell->bounding_box().get_boundary_points();
                  out << "cell " << cell->id() << ' ' << box.first << ' ' << box.second;
                  cell->get_dof_indices(indices);
                  for (unsigned int i=0; i<9; ++i)
                    out << ' ' << indices[fe.component_to_system_index(component,i)]-offset;
                  out << '\n';
                }
            for (const auto &[i, mass] : rows.mass)
              out << "mass " << i << ' ' << mass << '\n';
            for (const auto &[ij, value] : rows.stiffness)
              if (value != 0.)
                out << "stiffness " << ij.first << ' ' << ij.second << ' ' << value << '\n';
            out.close();
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(!out.fail()),this->get_mpi_communicator()),
                        ExcMessage("Could not write the fracture row diagnostic."));
          }
        return {"",""};
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(GrowingFractureRows, "growing fracture rows",
                                "Export the owned AT1 mass and stiffness rows on a small Cartesian mesh.")
}
