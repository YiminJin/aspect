#ifndef ASPECT_BENCHMARK_BP3_CONFIGURATION_H
#define ASPECT_BENCHMARK_BP3_CONFIGURATION_H
#include <deal.II/base/parameter_handler.h>
#include <string>
namespace BP3
{
  // Parameter values only: no physical history, output schedule, or cache ownership.
  struct Configuration
  {
    std::string model_identity;
    std::string resolved_settings;
  };
  Configuration read_configuration(const dealii::ParameterHandler &prm);
}
#endif
