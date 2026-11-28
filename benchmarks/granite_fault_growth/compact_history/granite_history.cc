/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/postprocess/interface.h>

#include <fstream>

namespace aspect::Postprocess
{
  /** Record accepted strip histories and the spatial Bernstein phase for the
   * Cartesian granite specimen. A missing cell in a complete shard has nine
   * exactly zero coefficients. No native stress or solver data is copied. */
  template <int dim>
  class GraniteHistory : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        AssertThrow(dim == 2, ExcMessage("Granite history requires the two-dimensional strip specimen."));
        if constexpr (dim == 2)
          {
            const auto &fault = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> &>(this->get_material_model());
            const auto &material = fault.strip_history();
            const auto &history = material.history();
            const auto tag = Utilities::int_to_string(history.step, 4);
            const auto &comm = this->get_mpi_communicator();
            const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
            const std::string path = this->get_output_directory();
            const auto &fe = this->get_fe();
            const auto component = this->introspection().component_indices.compositional_fields[material.phase_field()];
            std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
            AssertThrow(fe.base_element(fe.component_to_base_index(component).first).n_dofs_per_cell() == 9,
                        ExcMessage("Granite history requires nine phase coefficients per cell."));

            std::ofstream phase(path+"granite-phase-"+tag+"-r"+std::to_string(rank)+".csv");
            phase << std::setprecision(17) << "x0,y0,hx,hy,c0,c1,c2,c3,c4,c5,c6,c7,c8\n";
            for (const auto &cell : this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(dofs);
                  std::array<double,9> coefficients;
                  for (unsigned int j=0; j<coefficients.size(); ++j)
                    coefficients[j] = this->get_solution()[dofs[fe.component_to_system_index(component,j)]];
                  if (std::any_of(coefficients.begin(), coefficients.end(), [](const double value)
                  {
                    return value != 0.;
                  }))
                  {
                    phase << cell->vertex(0)[0] << ',' << cell->vertex(0)[1] << ','
                          << cell->extent_in_direction(0) << ',' << cell->extent_in_direction(1);
                    for (const double value : coefficients)
                      phase << ',' << value;
                    phase << '\n';
                  }
                }
            phase.close();
            bool ok = !phase.fail();

            if (rank == 0)
              {
                std::ofstream strips(path+"granite-strips-"+tag+".csv");
                strips << std::setprecision(17) << "time_s,section,xi,formed,V,slip\n";
                const auto &edges = material.sections().edges;
                for (unsigned int k=0; k<history.sections.size(); ++k)
                  {
                    const auto &strip = history.sections[k];
                    strips << history.time << ',' << k << ',' << (edges[k]+edges[k+1])/2 << ','
                           << strip.formed << ',' << strip.rate << ',' << strip.slip << '\n';
                  }
                strips.close();
                ok = ok && !strips.fail();
              }
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(ok),comm),
                        ExcMessage("Unable to write the compact granite history."));
          }
        return {"Granite history", "accepted strip rates, slips and phase"};
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(GraniteHistory, "compact granite history",
                                "Record all signed strip rates and slips and nonzero phase patches on the granite grid.")
}
