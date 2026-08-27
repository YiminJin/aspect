/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/postprocess/reconstructed_faults.h>
#include <aspect/utilities.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    namespace internal
    {
      template <int dim>
      ReconstructedFaultOutput<dim>::ReconstructedFaultOutput(
        const std::vector<ReconstructedFault<dim>> &faults)
      {
        for (unsigned int fault_id = 0; fault_id < faults.size(); ++fault_id)
          {
            const ReconstructedFault<dim> &fault = faults[fault_id];
            const unsigned int first_point = points.size();
            for (unsigned int vertex_id = 0; vertex_id < fault.n_vertices(); ++vertex_id)
              {
                points.push_back(fault.vertex(vertex_id));
                point_fault_ids.push_back(fault_id);
                vertex_ids.push_back(vertex_id);
              }
            for (unsigned int cell_id = 0; cell_id < fault.n_cells(); ++cell_id)
              {
                cells.push_back({{first_point + cell_id, first_point + cell_id + 1}});
                cell_fault_ids.push_back(fault_id);
                cell_ids.push_back(cell_id);
              }
          }
      }


      template <int dim>
      void
      ReconstructedFaultOutput<dim>::write_vtu(
        std::ostream &output,
        const double time,
        const unsigned int timestep_number) const
      {
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
               << "  <UnstructuredGrid>\n"
               << "    <FieldData>\n"
               << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
               << time << "</DataArray>\n"
               << "      <DataArray type=\"UInt32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">"
               << timestep_number << "</DataArray>\n"
               << "    </FieldData>\n"
               << "    <Piece NumberOfPoints=\"" << points.size()
               << "\" NumberOfCells=\"" << cells.size() << "\">\n"
               << "      <Points>\n"
               << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
        for (const Point<dim> &point : points)
          output << point[0] << ' ' << point[1] << ' '
                 << (dim == 3 ? point[2] : 0.0) << ' ';
        output << "\n        </DataArray>\n"
               << "      </Points>\n"
               << "      <Cells>\n"
               << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n          ";
        for (const auto &cell : cells)
          output << cell[0] << ' ' << cell[1] << ' ';
        output << "\n        </DataArray>\n"
               << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ";
        for (unsigned int cell = 0; cell < cells.size(); ++cell)
          output << 2 * (cell + 1) << ' ';
        output << "\n        </DataArray>\n"
               << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ";
        for (unsigned int cell = 0; cell < cells.size(); ++cell)
          output << "3 "; // VTK_LINE
        output << "\n        </DataArray>\n"
               << "      </Cells>\n";

        // Future generic properties can be attached locally in these two XML
        // sections without changing geometry construction.
        const auto write_identifier_array = [&output](const std::string &name,
                                                      const std::vector<unsigned int> &values)
        {
          output << "        <DataArray type=\"UInt32\" Name=\"" << name
                 << "\" format=\"ascii\">\n          ";
          for (const unsigned int value : values)
            output << value << ' ';
          output << "\n        </DataArray>\n";
        };

        output << "      <PointData>\n";
        write_identifier_array("fault_id", point_fault_ids);
        write_identifier_array("vertex_id", vertex_ids);
        output << "      </PointData>\n"
               << "      <CellData>\n";
        write_identifier_array("fault_id", cell_fault_ids);
        write_identifier_array("cell_id", cell_ids);
        output << "      </CellData>\n";

        output << "    </Piece>\n"
               << "  </UnstructuredGrid>\n"
               << "</VTKFile>\n";
      }
    }


    template <int dim>
    void
    ReconstructedFaults<dim>::initialize()
    {
      AssertThrow(this->get_parameters().reconstruct_faults,
                  ExcMessage("The `reconstructed faults' postprocessor requires "
                             "`Formulation/Reconstruct faults from phase field'."));
      Utilities::create_directory(this->get_output_directory() + "reconstructed_faults",
                                  this->get_mpi_communicator(), true);
    }


    template <int dim>
    std::pair<std::string,std::string>
    ReconstructedFaults<dim>::execute(TableHandler &)
    {
      const auto &faults = this->get_reconstructed_fault_manager().get_faults();
      if (faults.empty())
        return {"Writing reconstructed faults:", "no reconstructed geometry available"};

      const std::string filename = "reconstructed_faults/reconstructed_faults-"
                                   + Utilities::int_to_string(this->get_timestep_number(), 5)
                                   + ".vtu";
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          const internal::ReconstructedFaultOutput<dim> data_out(faults);
          std::ofstream output(this->get_output_directory() + filename);
          AssertThrow(output, ExcMessage("Could not open reconstructed-fault output file <"
                                         + this->get_output_directory() + filename + ">."));
          const double output_time = this->convert_output_to_years()
                                     ? this->get_time() / year_in_seconds
                                     : this->get_time();
          data_out.write_vtu(output, output_time, this->get_timestep_number());

          times_and_vtu_file_names.emplace_back(output_time, filename);
          std::ofstream pvd(this->get_output_directory() + "reconstructed_faults.pvd");
          DataOutBase::write_pvd_record(pvd, times_and_vtu_file_names);
        }

      return {"Writing reconstructed faults:", filename};
    }


  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace internal
    {
#define INSTANTIATE(dim) \
      template class ReconstructedFaultOutput<dim>;

      ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
    }

    ASPECT_REGISTER_POSTPROCESSOR(ReconstructedFaults,
                                  "reconstructed faults",
                                  "Write reconstructed sharp-fault line geometry and built-in "
                                  "fault, vertex, and cell identifiers in VTU format.")
  }
}
