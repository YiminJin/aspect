/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_growing_fault_h
#define _aspect_material_model_growing_fault_h
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/growing_fault/midpoint_material.h>

namespace aspect::GrowingFault
{
  /** Material owner for a fixed-reference, single-forward-front midpoint
   * solve. The simulator owns the velocity, pressure and Bernstein phase. The
   * material owns point tensors and strip history. A fresh attempt borrows
   * accepted state. Only the post-accept callback publishes its returned pair. */
  template <int dim>
  class Material : public MaterialModel::Interface<dim>, public SimulatorAccess<dim>, public MidpointMaterial
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

      void begin_attempt() override;
      void stage_candidate(History &&,const Balance &) override;
      const History &history() const override
      {
        return accepted;
      }
      const MaterialSections &sections() const override
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
      unsigned int phase_field() const override
      {
        return field;
      }
      const CohesiveMaxwell &maxwell() const override
      {
        return point_law;
      }
      const MaterialModel::Rheology::FaultResistance &friction() const override
      {
        return *law;
      }
      const Point<2> &seed_tip() const override
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
