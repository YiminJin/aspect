// Test-only observer; load after the BP3 library, never in production.
#include "../plugin/runtime.h"
#include <aspect/postprocess/interface.h>
#include <malloc.h>
#include <fstream>
#include <iomanip>
namespace aspect { namespace Postprocess {
template<int dim> class BP3MapMeasurement : public Interface<dim>,public SimulatorAccess<dim>
{
public:
  std::pair<std::string,std::string> execute(TableHandler &) override
  {
    if(this->get_timestep_number()!=0) return {};
    const auto &map=BP3Benchmark::work_initial_H;
    const auto before=mallinfo2().uordblks;
    const auto copy=map;
    const auto allocated=mallinfo2().uordblks-before;
    std::ostringstream stream;
    {aspect::oarchive archive(stream);archive<<map;}
    if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
      {
        std::ofstream out(this->get_output_directory()+"audit_map_size.csv");
        out<<"entries,map_object_bytes,copy_heap_allocation_bytes,serialized_map_bytes,ranks\n"
           <<copy.size()<<','<<sizeof(map)<<','<<allocated<<','<<stream.str().size()<<','
           <<Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())<<'\n';
      }
    return {};
  }
};
ASPECT_REGISTER_POSTPROCESSOR(BP3MapMeasurement,"BP3 map measurement","Test-only replicated audit map size measurement.")
}}
