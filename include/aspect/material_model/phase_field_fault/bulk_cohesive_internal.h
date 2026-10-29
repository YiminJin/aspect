/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_MATERIAL_MODEL_BULK_COHESIVE_INTERNAL_H
#define ASPECT_MATERIAL_MODEL_BULK_COHESIVE_INTERNAL_H
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_gradient.h>

namespace aspect { namespace MaterialModel
{
  template<int dim> struct PhaseFieldFault<dim>::BulkCohesiveState
  {
    struct LocalHistory
    {
      double a=0., signed_shear=0., absolute_shear=0., q=0., rate=0.;
      BulkCohesive::Tensor A{},K{},Q{},W{};
      double L=0.;
    };
    BulkFlowRule flow_rule=BulkFlowRule::scalar;
    bool tensor_flow() const { return flow_rule!=BulkFlowRule::scalar; }
    std::string checkpoint_signature() const
    {
      if(flow_rule==BulkFlowRule::coulomb_fixed_phase)return "bulk cohesive closed Coulomb fixed phase cell history v1";
      if(flow_rule==BulkFlowRule::fixed_tensor)return "bulk cohesive fixed tensor cell history v1";
      if(flow_rule==BulkFlowRule::radial_tensor)return "bulk cohesive radial tensor cell history v1";
      return "bulk cohesive cell history v3";
    }

    /** Common diagnostics retain the scalar response interface. In tensor
     * mode scalar a/k/q/w are signed xy projections for diagnostics ONLY;
     * constitutive actions, storage and persistent history use tensor_point. */
    struct PointResponse : BulkCohesive::Response
    {
      BulkCohesive::TensorReturn::Response tensor_point;
      // Allocated only for the experimental Coulomb mode.
      std::shared_ptr<const BulkCohesive::CoulombReturn::Response> coulomb_point;
    };
    std::shared_ptr<DGHistory> tensor;
    // Same native FE velocity-memory representation as the accepted FE path,
    // without that path's surface/normalization owner. Only the velocity block
    // is nonzero: A+ = beta (A_n + du). The projection is a derived cache.
    LinearAlgebra::BlockVector native_history;
    std::shared_ptr<DGHistory> native_projection;
    std::map<CellId,typename DoFHandler<dim>::active_cell_iterator> native_cells;
    void rebuild_native_projection(const PhaseFieldFault<dim> &);
    std::vector<SymmetricTensor<2,dim>> evaluate_tensor(
      const PhaseFieldFault<dim> &,const CellId &,const Quadrature<dim> &) const;
    double check_tensor(const PhaseFieldFault<dim> &,const CellId &,const Quadrature<dim> &,
                        const std::vector<BulkCohesive::Tensor> &) const;
    std::map<CellId,std::vector<LocalHistory>> history;
    Vector<double> phase;
    std::vector<Point<dim>> mesh;
    BulkCohesive::Parameters parameters;
    unsigned int quadrature=0,step=0;
    std::uint64_t publications=0;
    double time=0.;
    // Only the fixed-phase Coulomb mode uses these physical parameters.
    double background_pressure=0.,coulomb_mu=0.;
    double slip_length=BulkCohesive::SlipGradient::experimental_length();
    std::shared_ptr<const BulkCohesive::SlipGradient::Space> slip_space;
    std::vector<double> material_record() const
    {
      std::vector<double> values{parameters.G,parameters.eta,parameters.m,parameters.resistance};
      if(flow_rule==BulkFlowRule::coulomb_fixed_phase)
        {values.push_back(background_pressure);values.push_back(coulomb_mu);values.push_back(0.);}
      if(slip_length>0.)values.push_back(slip_length);
      return values;
    }
  };

  template<int dim> struct PhaseFieldFault<dim>::BulkCohesiveArchive
  {
    std::string signature="bulk cohesive BE native remainder v2";
    unsigned int quadrature=0,step=0;
    std::uint64_t publications=0;
    double time=0.;
    std::vector<double> material,phase,tensor,local,native;
    std::vector<std::string> cells;
    std::vector<Point<dim>> mesh;
  };
}}
#endif
