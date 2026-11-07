/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_PHASE_FIELD_ASSEMBLERS_H
#define ASPECT_PHASE_FIELD_ASSEMBLERS_H

#include <aspect/simulator/assemblers/interface.h>

namespace aspect
{
  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        /** Borrowed scalar FE data at the frozen accepted velocity. No material
         * prefactor, physical-time mass, or implicit BDF history belongs here. */
        template <int dim>
        struct PhaseTransport : ScratchBase<dim>
        {
          PhaseTransport(const FEValues<dim> &values,
                         const std::vector<Tensor<1,dim>> &velocity,
                         const std::vector<double> &divergence)
            : values(values), velocity(velocity), divergence(divergence) {}
          const FEValues<dim> &values;
          const std::vector<Tensor<1,dim>> &velocity;
          const std::vector<double> &divergence;
        };
      }
      namespace CopyData
      {
        template <int dim>
        struct PhaseTransport : CopyDataBase<dim>
        {
          explicit PhaseTransport(FullMatrix<double> &matrix) : matrix(matrix) {}
          FullMatrix<double> &matrix;
        };
      }
    }
  }

  namespace Assemblers
  {
    /** Spatial material-advection cell kernel, independent of time integration.
     * (w,phi_dot) = (grad w.v,phi) + (w div v,phi) - <w,v.n phi_up>.
     * The explicit operator and registered assembler use this same kernel. */
    template <int dim>
    void assemble_phase_transport_cell(const FEValues<dim> &fe,
                                       const std::vector<Tensor<1,dim>> &velocity,
                                       const std::vector<double> &divergence,
                                       FullMatrix<double> &matrix)
    {
      for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < matrix.m(); ++i)
          for (unsigned int j = 0; j < matrix.n(); ++j)
            matrix(i,j) += fe.JxW(q) * fe.shape_value(j,q)
                           * (velocity[q] * fe.shape_grad(i,q) + divergence[q] * fe.shape_value(i,q));
    }

    /** Phase-specific spatial assembler owned by the ordinary assembler manager.
     * SSPRK stages apply local mass inverses separately; equilibrium uses SIPG. */
    template <int dim>
    class PhaseTransport : public Interface<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &copy_base) const override
        {
          const auto &scratch = dynamic_cast<internal::Assembly::Scratch::PhaseTransport<dim>&>(scratch_base);
          auto &copy = dynamic_cast<internal::Assembly::CopyData::PhaseTransport<dim>&>(copy_base);
          assemble_phase_transport_cell(scratch.values, scratch.velocity, scratch.divergence, copy.matrix);
        }
    };
  }
}
#endif
