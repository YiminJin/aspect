#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
  template <int dim>
  class EndpointDump : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      auto &manager=this->get_reconstructed_fault_manager();
      const auto &intro=this->introspection();
      const auto &quad=intro.quadratures.velocities;
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points);
      std::vector<double> phi(quad.size());
      std::ofstream out(this->get_output_directory()+"all_sources_rank"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
      out<<std::setprecision(17)<<"cell,qp,x,y,phi,active,segment,xi,tx,ty\n";
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())
        if(cell->is_locally_owned())
          {
            fe.reinit(cell);
            fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(this->get_solution(),phi);
            const auto &a=manager.get_stokes_qp_fault_associations(cell->id(),quad,fe.get_quadrature_points());
            for (unsigned int q=0;q<quad.size();++q)
              out<<cell->id().to_string()<<','<<q<<','<<a[q].position[0]<<','<<a[q].position[1]<<','<<phi[q]<<','<<a[q].active<<','<<a[q].segment_index<<','<<a[q].xi<<','<<a[q].tangent[0]<<','<<a[q].tangent[1]<<'\n';
          }
      return {};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(EndpointDump,"endpoint association dump","Read-only full-quadrature endpoint investigation.")
}}
