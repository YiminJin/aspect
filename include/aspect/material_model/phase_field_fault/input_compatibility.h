/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_phase_field_fault_input_compatibility_h
#define _aspect_material_model_phase_field_fault_input_compatibility_h

#include <deal.II/base/parameter_handler.h>

namespace aspect::MaterialModel::FaultInputCompatibility
{
  /** Formulations that consume a material input, independent of its spelling. */
  enum class Use
  {
    mature, growing, both
  };

  /** Declare a material or resistance coefficient in its owning subsection.
   * Keep track of explicit assignments, including a supplied default value,
   * so the two fault routes may retain different defaults for omitted entries. */
  void declare_entry(dealii::ParameterHandler            &prm,
                     const std::string                   &name,
                     const std::string                   &value,
                     const dealii::Patterns::PatternBase &pattern,
                     const std::string                   &documentation);

  /** Declare an input restricted to the indicated mechanics. An explicitly
   * supplied default also requires that formulation. */
  void declare_entry(dealii::ParameterHandler            &prm,
                     Use                                  use,
                     const std::string                   &name,
                     const std::string                   &value,
                     const dealii::Patterns::PatternBase &pattern,
                     const std::string                   &documentation);

  /** Observe an existing declaration made by an upstream owner (the Simple
   * viscosity or the parsed function for mature exterior stress). */
  void observe_entry(dealii::ParameterHandler            &prm,
                     const std::string                   &name,
                     const dealii::Patterns::PatternBase &pattern);

  void observe_entry(dealii::ParameterHandler            &prm,
                     Use                                  use,
                     const std::string                   &name,
                     const dealii::Patterns::PatternBase &pattern);

  /** Install the cross-subsection alias for Simple viscosity after ASPECT has
   * declared every registered material's parameters and before parsing. */
  void observe_simple_viscosity(dealii::ParameterHandler &prm);

  /** After the registered resistance laws have declared both material
   * subsections, observe the coefficients added by their ordinary deal.II
   * callbacks and expose growing-only coefficients under the public name. */
  void observe_external_coefficients(dealii::ParameterHandler &prm);

  /** Expand the pressure-source spelling "model default" for the selected
   * formulation before comparing aliases or reading the friction model. */
  std::string normal_stress_source(const std::string &choice, bool growing);

  /** Resolve explicit inputs for the active fault before reading material or
   * law parameters. Omitted entries retain that route's declared defaults.
   * Historical strip registrations ignore Simple viscosity. */
  void resolve(dealii::ParameterHandler &prm,
               bool                      growing,
               bool                      historical_strip_inputs);
}
#endif
