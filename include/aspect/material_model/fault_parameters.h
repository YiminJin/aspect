/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_fault_parameters_h
#define _aspect_material_model_fault_parameters_h

#include <deal.II/base/parameter_handler.h>

namespace aspect::MaterialModel::FaultParameters
{
  /** Declare a material or resistance coefficient in its owning subsection.
   * Keep track of explicit assignments, including a supplied default value,
   * so the two fault routes may retain different defaults for omitted entries. */
  void declare_entry(dealii::ParameterHandler &, const std::string &name,
                     const std::string &value, const dealii::Patterns::PatternBase &pattern,
                     const std::string &documentation);

  /** Observe an existing declaration made by an upstream owner (the Simple
   * viscosity or the parsed function for mature exterior stress). */
  void observe_entry(dealii::ParameterHandler &, const std::string &name,
                     const dealii::Patterns::PatternBase &pattern);

  /** After the registered resistance laws have declared both material
   * subsections, observe any coefficients added by external declare_entry
   * callbacks. They use the ordinary deal.II plugin interface. */
  void observe_external_coefficients(dealii::ParameterHandler &);

  /** Resolve explicit inputs for the active fault before reading material or
   * law parameters. Omitted entries retain that route's declared defaults. */
  void resolve(dealii::ParameterHandler &, bool growing);
}
#endif
