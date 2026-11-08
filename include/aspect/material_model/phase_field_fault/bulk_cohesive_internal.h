/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_BULK_COHESIVE_INTERNAL_H
#define ASPECT_MATERIAL_MODEL_BULK_COHESIVE_INTERNAL_H
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_field.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <aspect/material_model/phase_field_fault/bulk_tip.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_gradient.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Accepted fixed-mesh bulk-cohesive owner. Separate from mature T/R/tau
     * and surface cohesive histories; point-law temporaries never publish here. */
    template <int dim>
    struct PhaseFieldFault<dim>::BulkCohesiveState
    {
      /** Native quadrature-point physical history; tensor and scalar flow rules
       * use distinct entries. No fault-surface normalization is stored here. */
      struct LocalHistory
      {
        BulkCohesive::Tensor stress {}; ///< Direct physical stress only in the bulk-tip mode.
        double a = 0.; ///< Elastic engineering shear.
        double signed_shear = 0.; ///< Signed inelastic shear.
        double absolute_shear = 0.; ///< Accumulated absolute inelastic shear.
        double q = 0.; ///< Cohesive stress (Pa).
        double rate = 0.; ///< Inelastic shear rate (1/s).
        BulkCohesive::Tensor A {}; ///< Elastic strain tensor.
        BulkCohesive::Tensor K {}; ///< Inelastic strain tensor.
        BulkCohesive::Tensor Q {}; ///< Cohesive stress tensor (Pa).
        BulkCohesive::Tensor W {}; ///< Inelastic strain rate (1/s).
        double L = 0.;
      };
      BulkFlowRule flow_rule = BulkFlowRule::scalar;
      bool direct_tip() const { return flow_rule==BulkFlowRule::bulk_tip_sections; }
      /** Fixed-direction control: actual x-Gauss sections span the entire box.
       * The age is physical everywhere from initialization, including ahead of
       * the initial damage. Geometry observations never create/reset this state. */
      struct Section
      {
        double coordinate=0., weight=0., age=0., slip=0., rate=0., force=0.;
      };
      std::vector<Section> sections;
      std::map<CellId,std::vector<unsigned int>> section_ids;
      std::map<CellId,std::vector<double>> normal_weights;
      bool tensor_flow() const
      {
        return flow_rule != BulkFlowRule::scalar;
      }
      std::string checkpoint_signature() const
      {
        if (direct_tip()) return "bulk tip physical tau A DGQ2 sections BE v1";
        if (flow_rule == BulkFlowRule::coulomb_fixed_phase)
          return "bulk cohesive closed Coulomb fixed phase cell history v1";
        if (flow_rule == BulkFlowRule::fixed_tensor)
          return "bulk cohesive fixed tensor cell history v1";
        if (flow_rule == BulkFlowRule::radial_tensor)
          return "bulk cohesive radial tensor cell history v1";
        return "bulk cohesive cell history v3";
      }

      /** Common diagnostics retain the scalar response interface. In tensor
       * mode scalar a/k/q/w are signed xy projections for diagnostics ONLY;
       * constitutive actions, storage and persistent history use tensor_point. */
      struct PointResponse : BulkCohesive::Response
      {
        BulkCohesive::TensorReturn::Response tensor_point;
        BulkCohesive::Tip::Response tip_point;
        // Allocated only for the experimental Coulomb mode.
        std::shared_ptr<const BulkCohesive::CoulombReturn::Response> coulomb_point;
      };
      /** Independent numerical trial tensor, with interpretation owned here. */
      std::shared_ptr<TensorField> tensor;
      /** Same native FE velocity-memory representation as the accepted FE path,
       * without that path's surface/normalization owner. Only the velocity block
       * is nonzero: A+ = beta (A_n + du). The projection is a derived cache. */
      LinearAlgebra::BlockVector native_history;
      std::shared_ptr<TensorField> native_projection;
      std::map<CellId, typename DoFHandler<dim>::active_cell_iterator> native_cells;
      /** Rebuild the derived tensor projection after accepted native publication. */
      void rebuild_native_projection(const PhaseFieldFault<dim> &model);
      /** Sample the native trial tensor on readable cells, without publication. */
      std::vector<SymmetricTensor<2, dim>> evaluate_tensor(
        const PhaseFieldFault<dim> &model, const CellId &cell_id, const Quadrature<dim> &quadrature) const;
      /** Compare sampled tensor with reference; return maximum point mismatch. */
      double check_tensor(const PhaseFieldFault<dim> &model, const CellId &cell_id, const Quadrature<dim> &quadrature,
                          const std::vector<BulkCohesive::Tensor> &reference) const;
      std::map<CellId, std::vector<LocalHistory>> history;
      Vector<double> phase;
      std::vector<Point<dim>> mesh;
      BulkCohesive::Parameters parameters;
      unsigned int quadrature = 0; ///< Native Gauss order; part of persistence identity.
      unsigned int step = 0; ///< Accepted step number.
      std::uint64_t publications = 0;
      double time = 0.; ///< Accepted endpoint time (s).
      // Only the fixed-phase Coulomb mode uses these physical parameters.
      double background_pressure = 0., coulomb_mu = 0.;
      double slip_length = BulkCohesive::SlipGradient::experimental_length();
      std::shared_ptr<const BulkCohesive::SlipGradient::Space> slip_space;
      std::vector<double> material_record() const
      {
        std::vector<double> values {parameters.G, parameters.eta, parameters.m, parameters.resistance};
        if (flow_rule == BulkFlowRule::coulomb_fixed_phase)
          {
            values.push_back(background_pressure);
            values.push_back(coulomb_mu);
            values.push_back(0.);
          }
        if (slip_length > 0.)
          values.push_back(slip_length);
        return values;
      }
    };

    /** Explicit persistence packet. Field order/signatures remain checkpoint ABI;
     * derived FE projections and observations are reconstructed after loading. */
    template <int dim>
    struct PhaseFieldFault<dim>::BulkCohesiveArchive
    {
      std::string signature = "bulk cohesive BE native remainder v2";
      unsigned int quadrature = 0; ///< Native Gauss order; part of persistence identity.
      unsigned int step = 0; ///< Accepted step number.
      std::uint64_t publications = 0;
      double time = 0.; ///< Accepted endpoint time (s).
      std::vector<double> material, phase, tensor, local, native;
      std::vector<std::string> cells;
      std::vector<Point<dim>> mesh;
    };
  }
}
#endif
