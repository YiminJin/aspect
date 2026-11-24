/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_phase_field_fault_h
#define _aspect_material_model_phase_field_fault_h

#include <aspect/material_model/equation_of_state/linearized_incompressible.h>
#include <aspect/reconstructed_fault/stokes_coupling.h>
#include <deal.II/base/parsed_function.h>

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
          return accepted->profile;
        }
        /** Stress used by Stokes at the current endpoint, in Pa. */
        SymmetricTensor<2,dim> endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               const Point<dim> &x, double phase, const SymmetricTensor<2,dim> &strain) const;
        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;
      protected:
        /** Eulerian Maxwell viscosity (Pa s) at the constitutive point. */
        virtual double bulk_viscosity(const Point<dim> &) const
        {
          return viscosity;
        }
        /** Exterior physical parent stress, with a private RK clock in seconds. */
        virtual Function<dim> &exterior_stress()
        {
          return *inflow_stress;
        }
        /** FE operations shared by materials with the same Maxwell response. */
        FaultStokesCoupling<dim> coupling;
      private:
        friend class aspect::FaultStokesCoupling<dim>;
        using Stress = SymmetricTensor<2,dim>;
        Stress viscous_stress(const Stress &strain_rate, const Point<dim> &x) const;
        double stiffness(const Point<dim> &x) const;
        double relaxation(const Point<dim> &x) const;
        void bind_candidate(const LinearAlgebra::BlockVector &fields);

        Stress rotate_parent(const Stress &parent, const Tensor<2,dim> &accepted_gradient,
                             const Tensor<2,dim> &slip_gradient, double physical_dt) const;
        EquationOfState::LinearizedIncompressible<dim> eos;
        Rheology::RateAndStateFriction friction;
        // A profile borrows its own curve. Swapping owners publishes both together.
        struct Section
        {
          StationaryFault<dim> fault;
          FaultPhaseProfile<dim> profile {fault};
        };
        std::unique_ptr<Section> accepted, candidate;
        /** Active geometry and transfer for the current Stokes evaluation. */
        Section &section()
        {
          return candidate ? *candidate : *accepted;
        }
        /** Both interior seed modes use the projected Maxwell response. */
        bool interior_seed = false;
        bool advect_vertices = false;
        double interval;
        double degradation_magnitude;
        std::array<unsigned int,3> stress_fields; // xx, yy, xy by role
        unsigned int phase_field;
        /** Exterior physical parent stress (Pa) at SSPRK stage time (seconds).
         * Present only when the accepted stress is materially advected. */
        std::unique_ptr<Functions::ParsedFunction<dim>> inflow_stress;
        bool fit_trace, trace_initialized = false;
        double conductivity, viscosity, modulus, background_shear, background_compression;
        double initial_interval, minimum_speed;
    };
  }
}
#endif
