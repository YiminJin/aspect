/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef aspect_cohesive_bulk_history_h
#define aspect_cohesive_bulk_history_h

#include <aspect/simulator_access.h>
#include <deal.II/fe/fe_dgq.h>
#include <array>
#include <map>

namespace aspect
{
  namespace CohesiveInterface
  {
    /** Fixed-mesh, constant-coefficient Maxwell memory for the exact-jump path.
     *
     * T = T_DG + 2G dev3(sym grad d_R) - Pi_3[2G dev3(sym grad d_R)].
     * Pi_3 is the same complete DGQ2 Gauss3 projection in both occurrences.
     * The native remainder is retained, not discarded on non-affine cells.
     * Analytic jump-lifting memory belongs to the surface owner and is added
     * separately. No transport, mesh transfer, objective rate or phase law.
     */
    template <int dim> class BulkHistory : public SimulatorAccess<dim>
    {
    public:
      using Tensor = SymmetricTensor<2, 3>;
      using Coefficients = std::array<double, 27>; // xx, yy, xy; all nine Q2 modes
      using CellRecords = std::map<std::string, std::vector<double>>;
      using Cell = typename DoFHandler<dim>::active_cell_iterator;

      BulkHistory (const Simulator<dim> &, double G, double eta);

      /** Fresh physical stress prescription, projected once into DGQ2.
       * Nonzero native memory is optional and contributes only its exact
       * projection remainder. Never use this operation to reset accepted
       * history or to convert an existing checkpoint. */
      void initialize (const Function<dim> &physical_tensor,
                       const LinearAlgebra::BlockVector *native_memory = nullptr);
      /** Fresh full compatible native memory plus an independent DG tensor.
       * Explicit conversion path; callers must check physical stress/storage. */
      void initialize_native (const LinearAlgebra::BlockVector &,
                              const Function<dim> &independent_tensor);

      std::vector<Tensor> evaluate (const Cell &, const Quadrature<dim> &) const;
      std::shared_ptr<BulkHistory> advance (const LinearAlgebra::BlockVector &du, double dt) const;

      /** Independent point recurrence check. Returns absolute Pa and scaled
       * full-tensor discrepancies; throws with CellId/physical/unit location
       * at the retained 1e-12 relative allowance. */
      std::pair<double, double> check_advance (const BulkHistory &trial,
                                               const LinearAlgebra::BlockVector &du, double dt,
                                               const Quadrature<dim> &) const;
      double storage (const Quadrature<dim> &) const;

      /** Physical CellId records use the accepted-DG checkpoint convention:
       * one uniquely owned contribution, checked global cell coverage. */
      CellRecords pack () const;
      void restore (const CellRecords &);

      const std::map<CellId, Coefficients> &
      dg_coefficients () const
      {
        return tensor;
      }
      const LinearAlgebra::BlockVector &
      native_memory () const
      {
        return native;
      }
      double
      shear_modulus () const
      {
        return G;
      }
      double
      viscosity () const
      {
        return eta;
      }

    private:
      static Tensor dev3 (const SymmetricTensor<2, dim> &);
      std::map<CellId, Coefficients> project_native (const LinearAlgebra::BlockVector &) const;
      void check_layout () const;
      double G, eta;
      FE_DGQ<dim> element;
      std::map<CellId, Cell> cells;
      std::map<CellId, std::vector<Point<dim>>> vertices;
      std::map<CellId, Coefficients> tensor, projection;
      LinearAlgebra::BlockVector native;
    };
  }
}
#endif
