/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_friction_normal_stress_h
#define _aspect_friction_normal_stress_h

#include <aspect/simulator_access.h>

namespace aspect::MaterialModel::Rheology
{
  /** Choice of the positive normal compression (Pa) supplied to fault friction.
   * The mature law historically resolves the local normal traction; the growing
   * strip law historically prescribes one constant. Both can instead use the
   * simulator's reference pressure at a physical position on the fault. */
  class FrictionNormalStress
  {
    public:
      enum class Source
      {
        resolved_traction, prescribed_constant, adiabatic_pressure
      };

      static void declare_parameters(ParameterHandler &prm);
      /** Resolve the model default once during material parsing. Local. */
      static FrictionNormalStress parse(const ParameterHandler &prm, Source model_default);
      bool uses_adiabat() const;

      /** Identify the static reference pressure for a material checkpoint.
       * Called after initialization. Local; equivalent pressure tables can
       * be compared independently on each rank. Empty for legacy closures. */
      template <int dim>
      std::string checkpoint_profile(const SimulatorAccess<dim> &sim) const;

    private:
      Source source = Source::resolved_traction;
      std::string adiabatic_model, function_definition;
      unsigned int table_size = 0;
      bool time_dependent_table = false;
  };
}
#endif
