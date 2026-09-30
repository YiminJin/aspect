// Reuse the existing lifecycle assertions; this observer only records counters.
#include "../../../../tests/phase_field_fault_ih.cc"
#include "../../../../tests/phase_field_fault_ih_cache.cc"

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class NormalizationCacheAudit : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
          const auto [valid, hits, integrations, requests] =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::normalization_cache_status(model);
          std::ofstream out(this->get_output_directory()+"cache_counts_"
                            +std::to_string(this->get_timestep_number())+"_rank"
                            +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out << "valid,hits,integrations,last_requests\n"
              << valid << ',' << hits << ',' << integrations << ',' << requests << '\n';
          return {"Normalization cache audit:", "recorded"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(NormalizationCacheAudit, "normalization cache audit",
                                  "Read existing per-rank normalization counters without preparing values.")
  }
}
