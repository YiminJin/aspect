/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_growing_fault_midpoint_material_h
#define _aspect_material_model_growing_fault_midpoint_material_h
#include <aspect/material_model/growing_fault/section_response.h>
#include <aspect/reconstructed_fault/material_sections.h>
#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect::GrowingFault
{
  /** Replicated, ordered material-strip history. Slip is signed and in metres. */
  struct SectionHistory
  {
    double slip=0., angle=0., rate=0., resistance=0.;
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

  /** Accepted or trial strip and point state at one physical clock. */
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
    double bulk=0.,cohesive=0.,fracture=0.,prescribed_velocity_work=0.,bulk_loss=0.,cohesive_loss=0.,friction_work=0.;
    double compliance_remainder=0.,phase_work=0.,storage_change=0.,adjoint_defect=0.,mechanical=0.,kkt=0.;
    unsigned int iterations=0;
    template <class Archive> void serialize(Archive &ar, const unsigned int)
    {
      ar &bulk &cohesive &fracture &prescribed_velocity_work &bulk_loss &cohesive_loss &friction_work
      &compliance_remainder &phase_work &storage_change &adjoint_defect &mechanical &kkt &iterations;
    }
  };

  /** Operations needed by the plane-strain Maxwell/AT1 midpoint solve.
   * Every rank sees the same strip topology and constitutive parameters.
   * Native bulk state is owned by cell. Physical acceptance and restart are
   * the provider's responsibility. The provider never runs an FE solve. */
  class MidpointMaterial
  {
    public:
      virtual ~MidpointMaterial() = default;
      /** Sample the prescribed compression on the accepted geometry at attempt entry. Local. */
      virtual SectionPressures section_pressures() const = 0;
      virtual unsigned int phase_field() const = 0;
      virtual const CohesiveMaxwell &maxwell() const = 0;
      virtual const MaterialModel::Rheology::FaultResistance &friction() const = 0;
      virtual const MaterialSections &sections() const = 0;
      virtual const Point<2> &seed_tip() const = 0;
      virtual const History &history() const = 0;
      /** Collective attempt entry. Construct initial native history if needed
       * and discard a previous candidate. Preserve all accepted values. */
      virtual void begin_attempt() = 0;
      /** Stage the returned history pending the simulator's repeat decision. */
      virtual void stage_candidate(History &&,const Balance &) = 0;
  };
}
#endif
