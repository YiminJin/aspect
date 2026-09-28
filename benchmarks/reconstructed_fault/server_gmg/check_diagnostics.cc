// Check failure reporting with empty ranks and a bad entry on just one rank.
#include "fault_gmg_diagnostics.h"
#include <deal.II/base/utilities.h>
#include <sstream>
#include <iostream>

int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  dealii::deal_II_exceptions::disable_abort_on_exception();
  const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::IndexSet owned(1);
  if (rank == 0) owned.add_index(0);
  dealii::LinearAlgebra::distributed::Vector<double> v(owned,MPI_COMM_WORLD);
  std::ostringstream out;
  v = 1.;
  aspect::internal::FaultGMGDiagnostics::vector_summary("positive",v,out,true);
  for (const double value : {-1., std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    {
      if (rank == 0) v.local_element(0) = value;
      bool caught = false;
      try
        {
          aspect::internal::FaultGMGDiagnostics::vector_summary("injected",v,out,true);
        }
      catch (const dealii::ExceptionBase &) { caught = true; }
      AssertThrow(caught, dealii::ExcMessage("Bad diagonal was not rejected on every rank."));
    }
  if (rank == 0) std::cout << "GMG diagnostic failure checks passed\n";
}
