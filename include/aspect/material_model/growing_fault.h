/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_growing_fault_h
#define _aspect_material_model_growing_fault_h
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/growing_fault/section_response.h>
#include <aspect/reconstructed_fault/material_sections.h>

namespace aspect::GrowingFault
{
  /** Accepted or trial strip and point state at one physical clock. */
  struct History
  {
    unsigned int step=0;
    double time=0.;
    std::vector<SectionHistory> sections;
    std::map<CellId,std::vector<PointHistory>> cells;
  };

  /** Full-step energies and work per unit out-of-plane thickness (J/m).
   * Prescribed-velocity work includes both wall and interior reactions. */
  struct Balance
  {
    double bulk=0.,cohesive=0.,fracture=0.,prescribed_velocity_work=0.,bulk_loss=0.,cohesive_loss=0.,friction_work=0.;
    double compliance_remainder=0.,phase_work=0.,storage_change=0.,adjoint_defect=0.,mechanical=0.,kkt=0.;
    unsigned int iterations=0;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &bulk &cohesive &fracture &prescribed_velocity_work &bulk_loss &cohesive_loss &friction_work
      &compliance_remainder &phase_work &storage_change &adjoint_defect &mechanical &kkt &iterations;
    }
  };

  /** Material owner for a fixed-reference, single-forward-front midpoint
   * solve. The simulator owns the velocity, pressure and Bernstein phase. The
   * material owns point tensors and strip history. A fresh attempt borrows
   * accepted state. Only the post-accept callback publishes its returned pair. */
  template <int dim>
  class Material : public MaterialModel::Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &);
      void parse_parameters(ParameterHandler &) override;
      void initialize() override;
      bool is_compressible() const override
      {
        return false;
      }
      void evaluate(const MaterialModel::MaterialModelInputs<dim> &,MaterialModel::MaterialModelOutputs<dim> &) const override;
      void save(std::map<std::string,std::string> &) const override;
      void load(const std::map<std::string,std::string> &) override;

      /** Collective single attempt. Returns the stage velocity and pressure
       * and candidate endpoint phase in the ordinary solution vector. The
       * simulator uses the returned status for its pre-acceptance completion signal. */
      SolverControl solve_midpoint_fields(LinearAlgebra::BlockVector &solution);
      const History &history() const
      {
        return accepted;
      }
      const MaterialSections &sections() const
      {
        return *geometry;
      }
      /** False at initialization and for legacy checkpoints without work data. */
      bool has_accepted_balance() const
      {
        return balance_available;
      }
      const Balance &balance() const
      {
        AssertThrow(balance_available,ExcMessage("No accepted balance is available for this growing fault checkpoint or initial state."));
        return last_balance;
      }
      unsigned int phase_field() const
      {
        return field;
      }
      const CohesiveMaxwell &maxwell() const
      {
        return point_law;
      }
      const MaterialModel::Rheology::FaultResistance &friction() const
      {
        return *law;
      }
      const Point<2> &seed_tip() const
      {
        return anchor;
      }
    protected:
      /** Select a constitutive law in the Growing fault parameter subsection. */
      virtual std::unique_ptr<MaterialModel::Rheology::FaultResistance> make_resistance(ParameterHandler &) const;
    private:
      unsigned int order,field;
      double degrees;
      Point<2> anchor;
      CohesiveMaxwell point_law;
      std::unique_ptr<MaterialModel::Rheology::FaultResistance> law;
      std::unique_ptr<MaterialSections> geometry;
      History accepted;
      std::unique_ptr<History> trial;
      Balance last_balance,trial_balance;
      bool balance_available=false;
      void initialize_native_history();
      /** Deferred until ASPECT restores the mesh and reconstructs native rules. */
      std::string restored;
      bool legacy_checkpoint=false;
      std::vector<double> checkpoint_parameters() const;
  };
}
#endif
