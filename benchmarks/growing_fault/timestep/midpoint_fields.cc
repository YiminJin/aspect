/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect::Postprocess
{
  /** Use the ordinary FE API in both executables. Record physical cell
   * corners and the local basis index, independent of MPI DoF numbering. */
  template <int dim>
  class MidpointFields : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        const auto name=this->get_output_directory()+"mesh-"+Utilities::int_to_string(this->get_timestep_number(),4)
                        +"-r"+std::to_string(rank)+".csv";
        std::ofstream out(name);
        out<<std::setprecision(17)<<"cell,x0,y0,x1,y1,component,local,value\n";
        const auto &fe=this->get_fe();
        std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(ids);
              const auto &lo=cell->vertex(0), &hi=cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
              for (unsigned int i=0; i<ids.size(); ++i)
                {
                  const auto [c,j]=fe.system_to_component_index(i);
                  out<<cell->id()<<','<<lo[0]<<','<<lo[1]<<','<<hi[0]<<','<<hi[1]<<','<<c<<','<<j<<','
                     <<this->get_solution()[ids[i]]<<'\n';
                }
            }
        return {"Midpoint mesh fields",name};
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(MidpointFields,"midpoint mesh fields","Export cell geometry and native FE coefficients.")
}
