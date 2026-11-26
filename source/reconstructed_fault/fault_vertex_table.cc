/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/reconstructed_fault/fault_vertex_table.h>
#include <aspect/utilities.h>

#include <cmath>
#include <set>
#include <sstream>

namespace aspect
{
  template <int dim>
  FaultVertexTable<dim> FaultVertexTable<dim>::read(const std::string &filename, const MPI_Comm communicator)
  {
    FaultVertexTable table;
    const auto resolved_filename = Utilities::expand_ASPECT_SOURCE_DIR(filename);
    std::istringstream input(Utilities::read_and_distribute_file_content(resolved_filename, communicator));
    std::vector<std::string> column_names;
    std::string line;
    while (std::getline(input, line))
      {
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        std::vector<std::string> entries;
        for (std::string entry; row >> entry;)
          entries.push_back(entry);
        if (entries.empty())
          continue;

        if (column_names.empty())
          {
            column_names = std::move(entries);
            AssertThrow(column_names.size() >= 3 &&
                        column_names[0] == "id" && column_names[1] == "x_m" && column_names[2] == "y_m",
                        ExcMessage("Expected vertex header id x_m y_m; anonymous donor core-phase triples are ambiguous."));
            std::set<std::string> optional_columns;
            for (unsigned int column = 3; column < column_names.size(); ++column)
              {
                const std::string &name = column_names[column];
                const bool is_age = name == "initial_age_s";
                AssertThrow((is_age || name == "background_shear_Pa") && optional_columns.insert(name).second,
                            ExcMessage("Unknown or duplicate vertex column " + name));
              }
            continue;
          }

        AssertThrow(entries.size() == column_names.size() && entries[0] == std::to_string(table.positions.size()),
                    ExcMessage("Vertex rows require consecutive material IDs from zero and one value per named column."));
        Point<dim> position;
        for (unsigned int column = 1; column < column_names.size(); ++column)
          {
            std::istringstream number(entries[column]);
            double value;
            AssertThrow((number >> value) && number.peek() == std::char_traits<char>::eof() && std::isfinite(value),
                        ExcMessage("Invalid vertex value in " + column_names[column]));
            if (column < 3)
              position[column-1] = value;
            else if (column_names[column] == "initial_age_s")
              table.initial_ages.push_back(value);
            else
              table.background_shear_tractions.push_back(value);
          }
        table.positions.push_back(position);
      }
    AssertThrow(table.positions.size() >= 2,
                ExcMessage("A material trace requires at least two ordered vertices."));
    return table;
  }

#define INSTANTIATE(dim) template struct FaultVertexTable<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
