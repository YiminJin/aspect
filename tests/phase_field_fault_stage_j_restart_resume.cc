#include "phase_field_fault_stage_j_restart.cc"
#include <filesystem>

namespace
{
  int copy_stage_j_checkpoint()
  {
    unsigned int success=1;
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::error_code error;
        std::filesystem::create_directories("output-phase_field_fault_stage_j_restart_resume", error);
        if (!error)
          std::filesystem::copy("output-phase_field_fault_stage_j_restart_create/restart",
                                "output-phase_field_fault_stage_j_restart_resume/restart",
                                std::filesystem::copy_options::recursive
                                | std::filesystem::copy_options::overwrite_existing, error);
        success = !error;
      }
    success = dealii::Utilities::MPI::broadcast(MPI_COMM_WORLD, success, 0);
    AssertThrow(success, dealii::ExcMessage("Could not copy the Stage-J test checkpoint."));
    return 0;
  }
  const int checkpoint_ready = copy_stage_j_checkpoint();
}
