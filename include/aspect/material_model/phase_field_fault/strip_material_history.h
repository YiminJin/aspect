/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_phase_field_fault_strip_material_history_h
#define _aspect_material_model_phase_field_fault_strip_material_history_h
#include <aspect/material_model/rheology/friction_normal_stress.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/rheology/cohesive_maxwell.h>
#include <aspect/reconstructed_fault/material_sections.h>
#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect::GrowingFault
{
  /** A stage is the midpoint of a physical step. Tensors retain zz in plane
   * strain. This state belongs to one native point on the current material mesh. */
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

  /**
   * Material state for the growing fault. Native tensor records
   * follow the owned (CellId, quadrature point) pairs. Ordered strip records
   * and their pressure stations are replicated on every rank. The selected
   * signed resistance law owns any additional strip state, including RSF age.
   *
   * The midpoint solver uses the same accepted parent for every assembly and
   * retry. It stages one endpoint, including its work ledger. The material
   * promotes that endpoint only after the simulator accepts the step.
   */
  template <int dim>
  class StripMaterialHistory
  {
    public:
      StripMaterialHistory(const SimulatorAccess<dim> &simulator,
                           ParameterHandler           &prm,
                           unsigned int                phase_field_index);

      /** Declare geometry, Maxwell and resistance inputs for the strip formulation. */
      static void declare_parameters(ParameterHandler &prm);

      /**
       * Build the native rules on the current mesh at startup or after mesh
       * restoration. On restart install saved native tensors, replicated strip
       * records and law-owned state before the pre-solve observers. Collective.
       */
      void initialize_native_history();

      /** Before mesh refinement, attach accepted point history to its owning
       * cells for migration by p4est. A rejected trial is disposable. Collective. */
      void prepare_history_transfer(parallel::distributed::Triangulation<dim> &mesh);

      /** Rebuild cell rules and recover accepted records on the new owners
       * after FE prolongation. Physical time and strip history stay fixed. Collective. */
      void finish_history_transfer(parallel::distributed::Triangulation<dim> &mesh);

      /** Maximum physical station width (m), or zero for the legacy uniform grid. Local. */
      double target_strip_width() const
      {
        return requested_strip_width;
      }

      /** Unit seed tangent available before native rule construction. Local. */
      Tensor<1,2> seed_tangent() const
      {
        return MaterialSections::seed_tangent(seed_angle_degrees);
      }

      /** Whether the material-strip rules and accepted point history exist on this mesh. Local. */
      bool has_native_history() const
      {
        return geometry != nullptr;
      }

      /** Publish the staged endpoint and advance law-owned state by the accepted duration (s). */
      void accept_endpoint(double duration);

      /** Discard a rejected trial. On first entry collectively build the owned native rules. */
      void begin_attempt();

      /** Keep a converged endpoint and its full-interval work (J/m) until the repeat decision. */
      void stage_candidate(History &&candidate, const Balance &ledger);

      /** Replicated section endpoints awaiting publication, or null on initialization/failure. */
      const std::vector<SectionHistory> *staged_sections() const
      {
        return trial ? &trial->sections : nullptr;
      }

      /** Collectively gather accepted native records and write their checkpoint on rank zero. */
      void save(std::map<std::string,std::string> &status) const;

      /** Read the checkpoint metadata. The mesh-restoration callback rebinds the native points. */
      void load(const std::map<std::string,std::string> &status);

      /** Read the accepted endpoint once the native rules have been built. */
      const History &history() const
      {
        require_native_history();
        return accepted;
      }
      /**
       * Read the fixed material rules built in the first midpoint attempt or
       * in the checkpoint mesh-restoration callback. Local, including on MPI.
       */
      const MaterialSections &sections() const
      {
        require_native_history();
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
      /** Furthest seed-connected formed station with phase trace at least 0.05.
       * Both traces at a DG interface are considered. The caller supplies the
       * FE solution and planes at the same clock. Collective on the mesh. */
      Point<2> phase_supported_tip(const std::vector<double>           &angles,
                                   const std::vector<unsigned int>     &formed,
                                   const LinearAlgebra::BlockVector    &solution) const;
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
      void require_native_history() const
      {
        AssertThrow(geometry, ExcMessage("Growing strip rules and history are available after the first midpoint "
                                         "attempt begins or after checkpoint mesh restoration."));
      }

      /** The owning material supplies the fixed mesh, reference pressure and MPI communicator. */
      const SimulatorAccess<dim> &simulator;

      unsigned int quadrature_order;
      const unsigned int phase_field_index;
      double seed_angle_degrees;
      double requested_strip_width;
      unsigned int transfer_handle = numbers::invalid_unsigned_int;
      double prescribed_compression; //!< Constant compression in Pa when that source is selected.
      MaterialModel::Rheology::FrictionNormalStress normal_stress;
      Point<2> seed_tip_position;
      Stress initial_deviatoric_stress; //!< Uniform fresh-start native stress in Cartesian coordinates (Pa).
      CohesiveMaxwell point_law;
      std::unique_ptr<MaterialModel::Rheology::FaultResistance> resistance_law;
      std::unique_ptr<MaterialSections> geometry;
      History accepted;
      std::unique_ptr<History> trial;
      Balance last_balance,trial_balance;
      bool balance_available=false;
      /** Deferred until ASPECT restores the mesh and reconstructs native rules. */
      std::string checkpoint_bytes;
      bool restore_v1=false;
      std::vector<double> checkpoint_parameters() const;
  };
}
#endif
