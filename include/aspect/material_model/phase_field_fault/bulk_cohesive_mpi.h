/* Communication for the bounded fixed-mesh bulk cohesive solver. */
#ifndef ASPECT_BULK_COHESIVE_MPI_H
#define ASPECT_BULK_COHESIVE_MPI_H
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <deal.II/lac/affine_constraints.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      inline Vector<double> global_values(const LinearAlgebra::BlockVector &v,
                                          const MPI_Comm communicator)
      {
        Vector<double> result(v.size());
        for (unsigned int b = 0; b < v.n_blocks(); ++b)
          {
            const auto block = aspect::internal::endpoint_values(v.block(b), communicator);
            std::copy(block.begin(), block.end(), result.begin() + v.get_block_indices().block_start(b));
          }
        return result;
      }

      inline AffineConstraints<double> global_constraints(const AffineConstraints<double> &local,
                                                          const MPI_Comm communicator)
      {
        using Line = AffineConstraints<double>::ConstraintLine;
        const std::vector<Line> lines(local.get_lines().begin(), local.get_lines().end());
        std::map<types::global_dof_index, Line> unique;
        for (const auto &part : Utilities::MPI::all_gather(communicator, lines))
          for (const auto &line : part)
            {
              const auto inserted = unique.emplace(line.index, line);
              AssertThrow(inserted.second || (inserted.first->second.entries == line.entries
                                              && inserted.first->second.inhomogeneity == line.inhomogeneity),
                          ExcMessage("Inconsistent distributed bulk constraints."));
            }
        AffineConstraints<double> result;
        for (const auto &entry : unique)
          {
            result.add_line(entry.first);
            result.add_entries(entry.first, entry.second.entries);
            result.set_inhomogeneity(entry.first, entry.second.inhomogeneity);
          }
        result.close();
        return result;
      }

      inline void sum(Vector<double> &v, const MPI_Comm communicator)
      {
        if (Utilities::MPI::n_mpi_processes(communicator) == 1)
          return;
        std::vector<double> values(v.begin(), v.end());
        aspect::internal::endpoint_sum(values, communicator);
        std::copy(values.begin(), values.end(), v.begin());
      }
    }
  }
}
#endif
