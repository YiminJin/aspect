/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_phase_field_fault_h
#define _aspect_material_model_phase_field_fault_h

#include <aspect/material_model/equation_of_state/linearized_incompressible.h>
#include <aspect/reconstructed_fault/stokes_coupling.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Single-lithology Maxwell material in plane strain with mature-fault
     * friction. The native static DG fields carry the accepted spatial stress
     * in Pa. Independent Q1 material vertices carry age, slip and rate. */
    template <int dim>
    class PhaseFieldFault : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        PhaseFieldFault();
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;
        void initialize() override;
        void update() override;
        bool is_compressible() const override;
        void evaluate(const MaterialModelInputs<dim> &in, MaterialModelOutputs<dim> &out) const override;
        void create_additional_named_outputs(MaterialModelOutputs<dim> &out) const override;
        /** Ordered geometry and last accepted physical state for diagnostics. */
        const std::vector<typename StationaryFault<dim>::Label> &labels() const;

        const FaultPhaseProfile<dim> &phase_profile() const
        {
          return profile;
        }
        /** Instantaneous physical endpoint before projection into DG history. */
        SymmetricTensor<2,dim> endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               const Point<dim> &x, double phase, const SymmetricTensor<2,dim> &strain) const;
      protected:
        /** FE operations shared by materials with the same Maxwell response. */
        FaultStokesCoupling<dim> coupling;
      private:
        friend class aspect::FaultStokesCoupling<dim>;
        using Stress = SymmetricTensor<2,dim>;
        Stress viscous_stress(const Stress &strain_rate) const;

        Stress relaxed_parent(const Stress &parent, const Tensor<2,dim> &accepted_gradient,
                              const Tensor<2,dim> &slip_gradient, double physical_dt, double relaxation) const;
        EquationOfState::LinearizedIncompressible<dim> eos;
        Rheology::RateAndStateFriction friction;
        StationaryFault<dim> fault;
        FaultPhaseProfile<dim> profile;
        double degradation_magnitude;
        std::array<unsigned int,3> stress_fields; // xx, yy, xy by role
        unsigned int phase_field;
        bool fit_trace, trace_initialized = false;
        double conductivity, viscosity, modulus, background_shear, background_compression;
        double initial_interval, minimum_speed, kappa;
    };
  }
}
#endif
