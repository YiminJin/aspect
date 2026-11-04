/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef _aspect_reconstructed_fault_objective_stress_h
#define _aspect_reconstructed_fault_objective_stress_h

#include <deal.II/base/symmetric_tensor.h>
#include <cmath>

namespace aspect
{
  namespace ReconstructedFaultUtilities
  {
    /** Frozen total-material Jaumann rotation, with L_ij = d v_i / d x_j.
     * The input is the in-plane block of a spatial, full-3D deviatoric tensor;
     * zz = -xx-yy is unchanged by this plane-strain rotation. This operation
     * contains neither relaxation nor forcing, transport, or a slip-spin split.
     */
    inline dealii::SymmetricTensor<2,2>
    rotate_plane_strain_stress(const dealii::SymmetricTensor<2,2> &stress,
                              const dealii::Tensor<2,2> &velocity_gradient,
                              const double dt)
    {
      const double angle = .5*dt*(velocity_gradient[1][0]-velocity_gradient[0][1]);
      AssertThrow(std::isfinite(angle), dealii::ExcMessage("Nonfinite Jaumann rotation angle."));
      // Retain the exact legacy operands in the zero-spin limit.
      if (angle == 0.)
        return stress;
      dealii::Tensor<2,2> rotation;
      rotation[0][0] = rotation[1][1] = std::cos(angle);
      rotation[1][0] = std::sin(angle);
      rotation[0][1] = -rotation[1][0];
      return dealii::symmetrize(rotation*dealii::Tensor<2,2>(stress)*dealii::transpose(rotation));
    }
    /** Stable finite-rotation change R A R^T-A. Avoid subtracting two large
     * history tensors when the accepted spin angle is very small. */
    inline dealii::SymmetricTensor<2,2>
    rotate_plane_strain_stress_increment(const dealii::SymmetricTensor<2,2> &stress,
                                        const dealii::Tensor<2,2> &velocity_gradient,
                                        const double dt)
    {
      const double angle=.5*dt*(velocity_gradient[1][0]-velocity_gradient[0][1]);
      AssertThrow(std::isfinite(angle),dealii::ExcMessage("Nonfinite Jaumann rotation angle."));
      const double sine=std::sin(angle),cosine=std::cos(angle);
      const double dc=-2*sine*sine,ss=2*sine*cosine;
      const double difference=.5*(stress[0][0]-stress[1][1]);
      dealii::SymmetricTensor<2,2> change;
      change[0][0]=dc*difference-ss*stress[0][1];
      change[1][1]=-change[0][0];
      change[0][1]=ss*difference+dc*stress[0][1];
      return change;
    }
  }
}
#endif
