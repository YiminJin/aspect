#ifndef _aspect_tests_phase_field_test_access_h
#define _aspect_tests_phase_field_test_access_h

#include <aspect/phase_field.h>

namespace aspect
{
  namespace internal
  {
    template <int dim>
    class PhaseFieldTestAccess
    {
      public:
        static unsigned int solve(const PhaseFieldHandler<dim> &handler,
                                  const LinearAlgebra::BlockSparseMatrix &matrix,
                                  const LinearAlgebra::BlockVector &rhs,
                                  LinearAlgebra::BlockVector &update)
        {
          return handler.solve_phase_field_system(matrix,rhs,update);
        }

        static bool assemble(const PhaseFieldHandler<dim> &handler,
                             LinearAlgebra::BlockSparseMatrix &matrix,
                             LinearAlgebra::BlockVector &rhs,
                             const LinearAlgebra::BlockVector &state,
                             const bool assemble_jacobian = true,
                             double *roundoff_allowance = nullptr)
        {
          return handler.assemble_phase_field_system(matrix, rhs, state, assemble_jacobian,roundoff_allowance);
        }

        static const std::vector<types::global_dof_index> &
        vertex_dofs(const PhaseFieldHandler<dim> &handler)
        {
          return handler.vertex_to_dof_indices;
        }

        // Diagnostic decomposition of the single-material production residual.
        // The complete assembly remains the independent consistency check.
        static std::pair<double,double>
        single_material_coefficients(const PhaseFieldHandler<dim> &handler,
                                     const double phi, const double H)
        {
          AssertThrow(handler.degradation_functions.size() == 1,
                      ExcMessage("This phase diagnostic requires one material."));
          const double energy = handler.critical_energy_densities[0];
          const double ell = handler.geometric_function->get_length_scale();
          return {2*energy*ell*ell,
                  H*handler.degradation_functions[0]->first_derivative(phi)
                  +energy*handler.geometric_function->first_derivative(phi)};
        }
    };
  }
}
#endif
