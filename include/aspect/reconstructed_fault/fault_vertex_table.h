/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_reconstructed_fault_fault_vertex_table_h
#define _aspect_reconstructed_fault_fault_vertex_table_h

#include <aspect/global.h>
#include <deal.II/base/point.h>

#include <string>
#include <vector>

namespace aspect
{
  /**
   * Initial data for the ordered vertices of a prescribed two-dimensional
   * polyline. Row i describes material ID i, starting at zero. The optional
   * columns are empty when omitted from the table.
   */
  template <int dim>
  struct FaultVertexTable
  {
    /** Read the named ASCII table collectively. Rank zero reads the file
     * after expanding $ASPECT_SOURCE_DIR, and all ranks parse the same rows.
     * Blank lines and comments beginning with '#' are ignored. */
    static FaultVertexTable read(const std::string &filename, const MPI_Comm communicator);

    std::vector<Point<dim>> positions;               ///< Coordinates in metres.
    std::vector<double> initial_ages;                ///< Optional initial ages in seconds.
    std::vector<double> background_shear_tractions;  ///< Optional persistent shear loads in Pa.
  };
}

#endif
