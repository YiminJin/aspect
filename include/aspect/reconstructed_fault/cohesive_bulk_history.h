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
    class Geometry;
    /** Fixed-mesh, constant-coefficient Maxwell memory for the exact-jump path.
     *
     * T = T_DG + 2G dev3(sym grad d_R) - Pi_3[2G dev3(sym grad d_R)].
     * Pi_3 is the same complete DGQ2 Gauss3 projection in both occurrences.
     * The native remainder is retained, not discarded on non-affine cells.
     * Analytic jump-lifting memory belongs to the surface owner and is added
     * separately. Optional reaction anchors rotate the complete physical
     * tensor, retaining this native remainder. No material transport, mesh
     * transfer or independently selected phase law is supplied by this owner.
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
      std::vector<Tensor> evaluate_sided(const Cell &,const Quadrature<dim> &,int side) const;
      std::shared_ptr<BulkHistory> advance (const LinearAlgebra::BlockVector &du, double dt) const;
      /** Complete physical-stress rotation before the BE reaction/source.
       * The physical callback includes the native remainder AND lifting memory.
       * Retain native memory exactly and project only the finite rotation
       * increment into the independent tensor. This is a discrete projected
       * Jaumann step, not a claim of exact nonpolynomial closure. */
      using PhysicalField=std::function<std::vector<Tensor>(const Cell &,const Quadrature<dim> &)>;
      using GradientField=std::function<std::vector<dealii::Tensor<2,dim>>(const Cell &,const Quadrature<dim> &)>;
      std::shared_ptr<BulkHistory> rotated_anchor(const PhysicalField &,
        const GradientField &,double dt,unsigned int quadrature_order,
        const std::function<Quadrature<dim>(const Cell &)> &quadrature={}) const;
      /** Piece/side Q2 history for the discontinuous regular lifting field.
       * Only formed material geometry defines this space. Native FE remainder
       * stays exact; temporary candidate tails are never history partitions. */
      std::shared_ptr<BulkHistory> cut_rotated_anchor(const PhysicalField &,
        const GradientField &,double dt,unsigned int quadrature_order,const Geometry &) const;
      /** Cell-interior trace on exact cell faces in the cut representation. */
      Point<dim> trace_point(const Cell &,const Point<dim> &unit) const;

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
      struct CutRegion
      {
        std::array<double,4> bounds;
        Coefficients coefficients;
      };
      unsigned int cut_region(const Point<dim> &,int side=0) const;
      Tensor cut_value(const CellId &,const Point<dim> &,const Point<dim> &,int side=0) const;
      Coefficients restrict_polynomial(const Coefficients &,const std::array<double,4> &) const;
      void pack_cut(CellRecords &) const;
      void restore_cut(const CellRecords &);
      static Tensor dev3 (const SymmetricTensor<2, dim> &);
      std::map<CellId, Coefficients> project_native (const LinearAlgebra::BlockVector &) const;
      void check_layout () const;
      double G, eta;
      FE_DGQ<dim> element;
      std::map<CellId, Cell> cells;
      std::map<CellId, std::vector<Point<dim>>> vertices;
      std::map<CellId, Coefficients> tensor, projection;
      // In cut mode tensor is a derived global-DG projection retained for the
      // existing diagnostics; cut_tensor is the independent physical history.
      std::shared_ptr<const Geometry> cut_geometry;
      std::map<CellId,std::map<unsigned int,CutRegion>> cut_tensor;
      LinearAlgebra::BlockVector native;
    };
  }
}
#endif
