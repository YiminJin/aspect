/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include <aspect/postprocess/reconstructed_faults.h>
#include <aspect/utilities.h>
#include <aspect/simulator_signals.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iomanip>
#include <set>

namespace aspect
{
  namespace Postprocess
  {
    namespace internal
    {
      // Display aliases do not change the property registry or checkpoint layout.
      std::string property_output_name(std::string name)
      {
        const std::map<std::string,std::string> aliases = {
          {"phase field fault state", "slip_state"},
          {"phase field fault cohesive traction", "cohesive_traction"},
          {"phase field fault previous I h", "previous_I_h"},
          {"cumulative_signed_slip_m", "cumulative_slip"}
        };
        const auto alias = aliases.find(name);
        if (alias != aliases.end())
          return alias->second;
        const std::string chemical_prefix = "phase field fault chemical composition ";
        if (name.compare(0, chemical_prefix.size(), chemical_prefix) == 0)
          name = "composition_" + name.substr(chemical_prefix.size());
        std::replace(name.begin(), name.end(), ' ', '_');
        return name;
      }

      template <int dim>
      ReconstructedFaultOutput<dim>::ReconstructedFaultOutput(
        const std::vector<ReconstructedFault<dim>> &faults,
        const std::vector<typename ReconstructedFaultManager<dim>::PropertyInformation>
        &property_information,
        const std::vector<std::vector<double>> *fault_slip_rates,
        const std::vector<std::string> &excluded_properties)
      {
        if (fault_slip_rates != nullptr)
          Assert(fault_slip_rates->size() == faults.size(), ExcInternalError());
        std::vector<unsigned int> selected_properties;
        std::set<std::string> output_names = {"fault_id", "slip_rate"};
        for (unsigned int i = 0; i < property_information.size(); ++i)
          {
            const auto &property = property_information[i];
            if (std::find(excluded_properties.begin(), excluded_properties.end(), property.name)
                != excluded_properties.end())
              continue;
            const auto name = property_output_name(property.name);
            AssertThrow(output_names.insert(name).second,
                        ExcMessage("Reconstructed-fault output name collision: " + name));
            properties.push_back({name, property.n_components, {}});
            selected_properties.push_back(i);
          }

        for (unsigned int fault_id = 0; fault_id < faults.size(); ++fault_id)
          {
            const ReconstructedFault<dim> &fault = faults[fault_id];
            const unsigned int first_point = points.size();
            for (unsigned int vertex_id = 0; vertex_id < fault.n_vertices(); ++vertex_id)
              {
                points.push_back(fault.vertex(vertex_id));
                point_fault_ids.push_back(fault_id);
                if (fault_slip_rates != nullptr)
                  {
                    Assert((*fault_slip_rates)[fault_id].size() == fault.n_vertices(),
                           ExcInternalError());
                    const double value = (*fault_slip_rates)[fault_id][vertex_id];
                    AssertThrow(std::isfinite(value) && value >= 0.0,
                                ExcMessage("A reconstructed-fault output slip rate must be "
                                           "finite and nonnegative."));
                    slip_rates.push_back(value);
                  }
              }
            for (unsigned int cell_id = 0; cell_id < fault.n_cells(); ++cell_id)
              {
                cells.push_back({{first_point + cell_id, first_point + cell_id + 1}});
                cell_fault_ids.push_back(fault_id);
              }

            for (unsigned int property_index = 0;
                 property_index < properties.size(); ++property_index)
              for (unsigned int vertex_index = 0;
                   vertex_index < fault.n_vertices(); ++vertex_index)
                {
                  const ArrayView<const double> values = fault.get_properties(vertex_index);
                  const auto &information = property_information[selected_properties[property_index]];
                  AssertThrow(information.position + information.n_components <= values.size(),
                              ExcMessage("A reconstructed fault does not use the manager's "
                                         "property layout."));
                  properties[property_index].values.insert(
                    properties[property_index].values.end(),
                    values.begin() + information.position,
                    values.begin() + information.position + information.n_components);
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

        const auto write_identifier_array = [&output](const std::string &name,
                                                      const std::vector<unsigned int> &values)
        {
          output << "        <DataArray type=\"UInt32\" Name=\"" << name
                 << "\" format=\"ascii\">\n          ";
          for (const unsigned int value : values)
            output << value << ' ';
          output << "\n        </DataArray>\n";
        };

        const auto escape_xml_attribute = [](const std::string &text)
        {
          std::string escaped;
          for (const char character : text)
            switch (character)
              {
                case '&': escaped += "&amp;"; break;
                case '<': escaped += "&lt;"; break;
                case '>': escaped += "&gt;"; break;
                case '\"': escaped += "&quot;"; break;
                case '\'': escaped += "&apos;"; break;
                default: escaped += character;
              }
          return escaped;
        };

        output << "      <PointData>\n";
        write_identifier_array("fault_id", point_fault_ids);
        if (!slip_rates.empty())
          {
            output << "        <DataArray type=\"Float64\" Name=\"slip_rate\" "
                   << "NumberOfComponents=\"1\" format=\"ascii\">\n          ";
            for (const double value : slip_rates)
              output << value << ' ';
            output << "\n        </DataArray>\n";
          }
        for (const PropertyOutput &property : properties)
          {
            output << "        <DataArray type=\"Float64\" Name=\""
                   << escape_xml_attribute(property.name)
                   << "\" NumberOfComponents=\"" << property.n_components
                   << "\" format=\"ascii\">\n          ";
            for (const double value : property.values)
              output << value << ' ';
            output << "\n        </DataArray>\n";
          }
        output << "      </PointData>\n"
               << "      <CellData>\n";
        write_identifier_array("fault_id", cell_fault_ids);
        output << "      </CellData>\n";

        output << "    </Piece>\n"
               << "  </UnstructuredGrid>\n"
               << "</VTKFile>\n";
      }
    }

    template <int dim>
    void ReconstructedFaults<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Reconstructed faults");
      prm.declare_entry("Excluded properties", "", Patterns::List(Patterns::Anything()),
                        "Registered vertex property names to omit from visualization only. "
                        "Their stored values and checkpoint representation are unchanged.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    template <int dim>
    void ReconstructedFaults<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      prm.enter_subsection("Reconstructed faults");
      excluded_properties = Utilities::split_string_list(prm.get("Excluded properties"));
      prm.leave_subsection();
      prm.leave_subsection();
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
    void ReconstructedFaults<dim>::save(std::map<std::string,std::string> &status) const
    {
      std::ostringstream text;
      { aspect::oarchive archive(text); archive << times_and_vtu_file_names; }
      status["Reconstructed fault output"]=text.str();
    }

    template <int dim>
    void ReconstructedFaults<dim>::load(const std::map<std::string,std::string> &status)
    {
      const auto entry=status.find("Reconstructed fault output");
      if (entry==status.end()) return;
      std::istringstream text(entry->second);
      aspect::iarchive archive(text); archive >> times_and_vtu_file_names;
    }

    template <int dim>
    std::pair<std::string,std::string>
    ReconstructedFaults<dim>::execute(TableHandler &)
    {
      if (!this->get_signals().allow_native_output.empty()
          && !*this->get_signals().allow_native_output("reconstructed faults"))
        return {"", ""};
      const auto &fault_manager = this->get_reconstructed_fault_manager();
      const auto &faults = fault_manager.get_faults();
      if (faults.empty())
        return {"Writing reconstructed faults:", "no reconstructed geometry available"};

      const std::string filename = "reconstructed_faults/reconstructed_faults-"
                                   + Utilities::int_to_string(this->get_timestep_number(), 5)
                                   + ".vtu";
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          std::vector<std::vector<double>> timestep_committed_slip_rates;
          if (fault_manager.slip_rates_are_initialized())
            {
              timestep_committed_slip_rates.reserve(faults.size());
              for (unsigned int fault = 0; fault < faults.size(); ++fault)
                timestep_committed_slip_rates.push_back(
                  fault_manager.get_timestep_committed_slip_rate(fault));
            }
          const internal::ReconstructedFaultOutput<dim> data_out(
            faults,
            fault_manager.get_property_information(),
            fault_manager.slip_rates_are_initialized()
            ? &timestep_committed_slip_rates
            : nullptr,
            excluded_properties);
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
                                  "Write reconstructed sharp-fault line geometry, built-in "
                                  "identifiers, and registered generic vertex properties in "
                                  "VTU format.")
  }
}
