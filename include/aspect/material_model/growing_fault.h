/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_growing_fault_h
#define _aspect_material_model_growing_fault_h
#include <aspect/material_model/interface.h>
#include <aspect/material_model/rheology/friction_normal_stress.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/rheology/cohesive_maxwell.h>
#include <aspect/reconstructed_fault/material_sections.h>
#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect::GrowingFault
{
  /** A stage is the midpoint of a physical step. Tensors retain zz in plane
   * strain. This state belongs to one native point on the fixed material mesh. */
  struct PointHistory : MaxwellState
  {
    double signed_shear=0., absolute_shear=0.;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &stress &cohesive_strain &signed_shear &absolute_shear;
    }
  };

  /** Replicated, ordered material-strip history. Slip is signed and in metres. */
  struct SectionHistory
  {
    double slip = 0.;       //!< Accumulated signed slip (m).
    double angle = 0.;      //!< Plane angle relative to the seed tangent (rad).
    double rate = 0.;       //!< Signed midpoint rate of the accepted interval (m/s).
    double resistance = 0.; //!< Signed resistance at that rate and stage age (Pa).
    bool formed=false;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &slip &angle &rate &resistance &formed;
    }
  };

  /** Pressures (Pa) at which the signed strip roots are solved. Spatial
   * prescriptions also retain their physical stations for endpoint output. */
  struct SectionPressures
  {
    std::vector<double> values;
    std::vector<Point<2>> stations;
  };

  /** Accepted or trial endpoint at one physical clock. Every rank has all
   * strip records and pressure stations. Each rank owns the native points in
   * its cells, ordered by the material quadrature rule. */
  struct History
  {
    unsigned int step=0;
    double time=0.;
    std::vector<SectionHistory> sections;
    SectionPressures friction_pressures;
    std::map<CellId,std::vector<PointHistory>> cells;
  };

  /** Full-step energies and work per unit out-of-plane thickness (J/m).
   * Prescribed-velocity work includes both wall and interior reactions. */
  struct Balance
  {
    double bulk = 0., cohesive = 0., fracture = 0.;
    double prescribed_velocity_work = 0.;
    double bulk_loss = 0., cohesive_loss = 0., friction_work = 0.;
    double compliance_remainder = 0., phase_work = 0., storage_change = 0.;
    double adjoint_defect = 0.;   //!< Native friction work minus signed strip work (J/m).
    double mechanical = 0., kkt = 0.; //!< Scaled mechanical and phase convergence measures.
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
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm) override;
      void initialize() override;
      bool is_compressible() const override
      {
        return false;
      }
      void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                    MaterialModel::MaterialModelOutputs<dim>      &out) const override;
      void save(std::map<std::string,std::string> &status) const override;
      void load(const std::map<std::string,std::string> &status) override;

      /** Discard any rejected trial. Seed owned native points on the first call. */
      void begin_attempt();

      /** Hold this converged endpoint until the simulator accepts or repeats the step. */
      void stage_candidate(History &&candidate, const Balance &ledger);
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
      /** Freeze the per-strip compression (Pa) from the accepted graph for one attempt. */
      SectionPressures section_pressures() const;
      unsigned int phase_field() const
      {
        return phase_field_index;
      }
      const CohesiveMaxwell &maxwell() const
      {
        return point_law;
      }
      const MaterialModel::Rheology::FaultResistance &friction() const
      {
        return *resistance_law;
      }
      const Point<2> &seed_tip() const
      {
        return seed_tip_position;
      }
    private:
      unsigned int quadrature_order;
      unsigned int phase_field_index;
      double seed_angle_degrees;
      double prescribed_compression; //!< Constant compression in Pa when that source is selected.
      MaterialModel::Rheology::FrictionNormalStress normal_stress;
      Point<2> seed_tip_position;
      CohesiveMaxwell point_law;
      std::unique_ptr<MaterialModel::Rheology::FaultResistance> resistance_law;
      std::unique_ptr<MaterialSections> geometry;
      History accepted;
      std::unique_ptr<History> trial;
      Balance last_balance,trial_balance;
      bool balance_available=false;
      void initialize_native_history();
      /** Deferred until ASPECT restores the mesh and reconstructs native rules. */
      std::string checkpoint_bytes;
      bool restore_v1=false;
      std::vector<double> checkpoint_parameters() const;
  };
}
#endif
