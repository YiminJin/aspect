#include <aspect/simulator.h>
#include <aspect/particle/manager.h>
#include <aspect/postprocess/interface.h>
#include <fstream>
#include <iomanip>
namespace aspect { namespace Postprocess {
  template<int dim> class OrdinaryAudit : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      AssertThrow(!this->get_parameters().enable_phase_field && !this->get_parameters().reconstruct_faults,
                  ExcMessage("Ordinary fixture enabled a fault/phase path."));
      AssertThrow(this->get_signals().allow_native_output.empty(), ExcMessage("Unexpected output gate."));
      const auto &manager=this->get_particle_manager(0);
      const auto tag=std::to_string(this->get_timestep_number())+"-"+
                     std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()));
      std::ofstream out(this->get_output_directory()+"ordinary-"+tag+".txt");
      out.exceptions(std::ios::failbit|std::ios::badbit);
      out<<std::hexfloat<<this->get_time()<<' '<<this->get_timestep()<<'\n';
      for (const auto &particle:manager.get_particle_handler())
        {
          out<<particle.get_id();
          for(unsigned int d=0;d<dim;++d) out<<' '<<particle.get_location()[d];
          for(const auto value:particle.get_properties()) out<<' '<<value;
          out<<'\n';
        }
      out<<manager.get_random_number_state()<<'\n';
      LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,
                                      this->get_mpi_communicator());
      owned=this->get_solution();
      for(const auto i:owned.locally_owned_elements()) out<<i<<' '<<owned[i]<<'\n';
      return {"Ordinary feature-disabled audit:","verified"};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(OrdinaryAudit,"r7 ordinary audit","Record ordinary particle/RNG/bulk restart state.")
}}
