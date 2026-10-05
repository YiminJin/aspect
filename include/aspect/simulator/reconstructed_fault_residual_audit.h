/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef aspect_reconstructed_fault_residual_audit_h
#define aspect_reconstructed_fault_residual_audit_h

namespace aspect
{
  namespace internal
  {
    // Diagnostic assembly channels only. The solver changes this between
    // synchronous WorkStream calls; cell workers only read it. Shadow results
    // never enter the nonlinear acceptance criterion.
    enum class FaultResidualAuditChannel { normal, unknowns, frozen };
    extern FaultResidualAuditChannel fault_residual_audit_channel;
  }
}
#endif
