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
        static bool assemble(const PhaseFieldHandler<dim> &handler,
                             LinearAlgebra::BlockSparseMatrix &matrix,
                             LinearAlgebra::BlockVector &rhs,
                             const LinearAlgebra::BlockVector &state)
        {
          return handler.assemble_phase_field_system(matrix, rhs, state, true);
        }

        static const std::vector<types::global_dof_index> &
        vertex_dofs(const PhaseFieldHandler<dim> &handler)
        {
          return handler.vertex_to_dof_indices;
        }
    };
  }
}
#endif
