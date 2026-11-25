/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_fault_parameters_h
#define _aspect_material_model_fault_parameters_h

#include <deal.II/base/parameter_handler.h>

namespace aspect::MaterialModel::FaultParameters
{
  /** Attach explicit-input observers once all material plugins have declared
   * their settings. The two fault declarations identify shared coefficients
   * (including external laws) and controls belonging to only one route. */
  template <int dim>
  void observe_material_inputs(dealii::ParameterHandler &);

  /** Was the named shared input explicitly assigned in this material
   * subsection, including an assignment equal to its declared default? */
  bool explicitly_set(const dealii::ParameterHandler &, const std::string &subsection,
                      const std::string &name);

  /** Resolve explicit inputs for the active fault before reading material or
   * law parameters. Omitted entries retain that route's declared defaults. */
  void resolve(dealii::ParameterHandler &, bool growing);
}
#endif
