/* Communication helpers for the restricted native endpoint verification path. */
#ifndef ASPECT_RECONSTRUCTED_FAULT_ENDPOINT_MPI_H
#define ASPECT_RECONSTRUCTED_FAULT_ENDPOINT_MPI_H

#include <aspect/global.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector.h>

namespace aspect
{
  namespace internal
  {
    /** Replicate a phase search vector, counting only its uniquely owned entries.
     * FE histories remain distributed; these vectors do not publish history. */
    inline dealii::Vector<double> endpoint_values(const LinearAlgebra::Vector &source,
                                                 const MPI_Comm communicator)
    {
      std::vector<double> local(source.size(),0.),global(source.size());
      for (const auto i:source.locally_owned_elements())local[i]=source[i];
      dealii::Utilities::MPI::sum(local,communicator,global);
      dealii::Vector<double> result(global.size());
      std::copy(global.begin(),global.end(),result.begin());
      return result;
    }

    inline void endpoint_assign(LinearAlgebra::Vector &target,
                                const dealii::Vector<double> &source)
    {
      AssertDimension(target.size(),source.size());
      for (const auto i:target.locally_owned_elements())target[i]=source[i];
      target.compress(dealii::VectorOperation::insert);
    }

    inline void endpoint_sum(std::vector<double> &values,const MPI_Comm communicator)
    {
      const auto local=values;
      dealii::Utilities::MPI::sum(local,communicator,values);
    }

    /** Preserve extended local accumulation through the collective reduction. */
    inline double endpoint_energy_sum(const long double local,const MPI_Comm communicator)
    {
      long double global=0.;
      const int error=MPI_Allreduce(&local,&global,1,MPI_LONG_DOUBLE,MPI_SUM,communicator);
      AssertThrow(error==MPI_SUCCESS,dealii::ExcMessage("Endpoint energy reduction failed."));
      return static_cast<double>(global);
    }
  }
}
#endif
