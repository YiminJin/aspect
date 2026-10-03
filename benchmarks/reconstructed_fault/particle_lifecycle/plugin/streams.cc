#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/particle/manager.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
  template <int dim>
  class ParticleStreams : public Interface<dim>, public SimulatorAccess<dim>
  {
    void capture(const std::string &stage) const
    {
      const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      for (unsigned int m=0; m<this->n_particle_managers(); ++m)
        {
          const auto &pm=this->get_particle_manager(m);
          std::ofstream out(this->get_output_directory()+"streams-"+stage+"-"+
                            std::to_string(this->get_timestep_number())+"-manager"+
                            std::to_string(m)+"-rank"+std::to_string(rank)+".txt");
          out << pm.get_random_number_state() << '\n'
              << pm.get_particle_handler().get_next_free_particle_index() << '\n';
          std::map<types::particle_index,std::vector<double>> data;
          for (const auto &p : pm.get_particle_handler())
            {
              auto &values=data[p.get_id()];
              for (unsigned int d=0; d<dim; ++d) values.push_back(p.get_location()[d]);
              for (double v : p.get_properties()) values.push_back(v);
            }
          out << std::hexfloat;
          for (const auto &p : data)
            {
              out << p.first;
              for (double v : p.second) out << ' ' << v;
              out << '\n';
            }
        }
    }
    public:
    void initialize() override
    {
      this->get_signals().pre_checkpoint_store_user_data.connect([this](auto &){capture("checkpoint");});
      this->get_signals().post_resume_load_user_data.connect([this](auto &){capture("resume");});
    }
    std::pair<std::string,std::string> execute(TableHandler &) override
    {capture("accepted"); return {"Particle streams:","recorded"};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(ParticleStreams,"particle streams","Record every manager/rank stream and particle state for checkpoint regression.")
}}
