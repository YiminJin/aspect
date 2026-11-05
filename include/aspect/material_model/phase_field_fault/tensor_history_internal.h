/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_FAULT_TENSOR_HISTORY_INTERNAL_H
#define ASPECT_FAULT_TENSOR_HISTORY_INTERNAL_H
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_field.h>
#include <aspect/material_model/history/tensor_advection.h>
namespace aspect
{
  namespace MaterialModel
  {
    /**
     * Material interpretation of independently discretized tensor fields.
     * In direct mode primary is physical deviatoric stress tau (Pa), in the
     * spatial frame. In separated mode primary is auxiliary T and residual is
     * auxiliary R (both Pa); mechanics reconstructs T-R-B. Neither R nor mature
     * surface memory Z is a cohesive elastic history. Plane strain completes
     * zz=-(xx+yy). Candidate copies share the space but copy primary coefficients;
     * residual pointers retain their existing shared, immutable-parent lifetime.
     * Only the existing checkpoint adapter serializes accepted coefficients.
     */
    template <int dim>
    struct PhaseFieldFault<dim>::DGHistory
    {
      /** Numerical space alias retained for material/checker construction. */
      using Space = typename TensorField::Space;
      /** Constitutive meaning of primary; the default remains separated. */
      enum class Representation
      {
        separated_trial, physical_stress
      };
      /** Representation identity validated at the checkpoint boundary. */
      Representation representation = Representation::separated_trial;
      /** Physical tau or auxiliary T, never both in one history. */
      TensorField primary;
      /** Auxiliary R for separated objective history; null in direct mode. */
      std::shared_ptr<TensorField> residual;
      /** Arithmetic bound for the native-source projection of one candidate. */
      struct SourceRoundoff
      {
        /** Componentwise projection forward-error bound, Pa. */
        SymmetricTensor<2, dim> projection;
        /** Absolute native derivative accumulation used to scale that bound. */
        Tensor<2, dim> gradient_scale;
      };
      /** Candidate-only arithmetic diagnostics; never serialized. */
      std::map<CellId, SourceRoundoff> source_roundoff;
    };
  }
}
#endif
