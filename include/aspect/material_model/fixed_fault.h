/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_fixed_fault_h
#define _aspect_material_model_fixed_fault_h

#include <aspect/material_model/simple.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      /** Regularized positive-speed rate-and-state law. Mechanics uses the
       * incoming age. The exact constant-speed aging solution is applied at
       * publication. All arguments use SI units. */
      struct MatureFaultFriction
      {
        double reference_coefficient, direct_effect, evolution_effect;
        double reference_speed, slip_distance, damping;

        /** Resistance (Pa) and its derivative (Pa s/m) at fixed age. */
        std::pair<double,double> resistance(double speed, double age, double compression) const;

        /** Age in seconds after sliding at constant speed for dt seconds. */
        double advance_age(double speed, double age, double dt) const;
      };
    }

    /** A straight, vertical mature fault in homogeneous plane shear. The
     * tangent points up and the normal points left. The two ends of the
     * single Q1 segment share one constant history in this symmetry class.
     * Physical xx, yy and xy stress occupy static DGQ2 compositional fields.
     * The missing zz component is -xx-yy. */
    template <int dim>
    class FixedFault : public Simple<dim>
    {
      public:
        /** Accepted surface state in seconds, metres and metres per second. */
        struct Section
        {
          double age, slip, rate;
        };

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;
        void update() override;
        void evaluate(const MaterialModelInputs<dim> &in, MaterialModelOutputs<dim> &out) const override;
        void create_additional_named_outputs(MaterialModelOutputs<dim> &out) const override;

        /** Solve the section balance at the given bulk Picard iterate.
         * Only the rate used in the next Stokes assembly is replaced. */
        void prepare_stokes(const LinearAlgebra::BlockVector &iterate);

        /** Publish the converged section and DG stress after the repeat
         * decision. The zero-duration initialization publishes only a rate. */
        void accept_step(LinearAlgebra::BlockVector &solution);

        /** Physical history shared by both endpoints of the fixed section. */
        const Section &accepted_section() const;
        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;

      private:
        struct SectionMean
        {
          Tensor<2,dim> gradient;
          double pressure;
          SymmetricTensor<2,dim> stress;
        };

        /** Collectively integrate the bulk fields against the normalized
         * section profile. Check homogeneous shear, stress and phase on
         * physical parents and endpoints. */
        SectionMean section_mean(const LinearAlgebra::BlockVector &fields, bool check_symmetry) const;
        double section_rate(const SectionMean &mean) const;
        SymmetricTensor<2,dim> stress_offset(double rate) const;

        Rheology::MatureFaultFriction friction;
        double viscosity, modulus, background_shear, background_compression;
        double initial_interval, minimum_speed, width;
        /** Last published section, replicated on the simulator communicator. */
        Section accepted;
        // Maxwell viscosity (Pa s), assembly speed (m/s) and rotated,
        // relaxed parent stress (Pa) for the current attempt.
        double kappa, trial_rate;
        SymmetricTensor<2,dim> relaxed_parent;
    };
  }
}
#endif
