#include "phase_field_fault_ih.cc"
#include "phase_field_fault_surface_system.cc"

#include <aspect/adiabatic_conditions/interface.h>

namespace aspect
{
  namespace AdiabaticConditions
  {
    template <int dim>
    class UninitializedFaultPressure : public Interface<dim>
    {
      public:
        void initialize() override
        {}

        bool is_initialized() const override
        {
          return false;
        }

        double temperature(const Point<dim> &) const override
        {
          return 293.0;
        }

        double pressure(const Point<dim> &) const override
        {
          return 2.e6;
        }

        double density(const Point<dim> &) const override
        {
          return 3300.0;
        }

        double density_derivative(const Point<dim> &) const override
        {
          return 0.0;
        }
    };



    ASPECT_REGISTER_ADIABATIC_CONDITIONS_MODEL(
      UninitializedFaultPressure,
      "uninitialized fault pressure test",
      "An intentionally uninitialized adiabatic model used only to verify "
      "the reconstructed-fault friction configuration diagnostic.")
  }
}
