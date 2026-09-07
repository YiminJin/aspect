#include "phase_field_frozen_history.cc"
#include <aspect/particle/particle_domain.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyParticleDomainRegeneration : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &pm = this->get_phase_field_handler().get_associated_particle_manager();
          auto &domains = const_cast<Particle::ParticleDomainHandler<dim> &>(
            pm.get_particle_domain_handler());
          const auto collect = [&]()
          {
            std::vector<double> values;
            for (const auto &particle : pm.get_particle_handler())
              {
                const auto domain = domains.get_particle_domain(particle.get_local_index());
                values.push_back(domain.volume());
                for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                  {
                    values.push_back(domain.relevant_vertex_index(i));
                    values.push_back(domain.weighting_function_value(i));
                    for (unsigned int d=0; d<dim; ++d)
                      values.push_back(domain.weighting_function_gradient(i)[d]);
                  }
              }
            return values;
          };
          const auto before = collect();
          // At this boundary all particle ghosts have been exchanged. Rebuilding
          // the cache, as restart does, must preserve the same discrete CPDI data.
          domains.generate_particle_domains();
          const auto after = collect();
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(before.size() == after.size()),
                                          this->get_mpi_communicator()) == 1,
                      ExcMessage("Rebuilding particle domains changed CPDI support."));
          double error = 0.0;
          for (unsigned int i=0; i<before.size(); ++i)
            error = std::max(error, std::abs(after[i]-before[i])
                                    /std::max(1.0,std::abs(before[i])));
          error = Utilities::MPI::max(error, this->get_mpi_communicator());
          this->get_pcout() << "Particle-domain regeneration relative error: " << error << std::endl;
          AssertThrow(error < 1.e-11,
                      ExcMessage("Advection and restart build different CPDI operators for identical particles."));
          return {"Particle-domain regeneration:", "verified"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyParticleDomainRegeneration,
      "verify particle domain regeneration", "Check cache identity after production particle advection.")
  }
}
