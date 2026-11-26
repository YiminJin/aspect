/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_phase_field_fault_h
#define _aspect_material_model_phase_field_fault_h

#include <aspect/material_model/equation_of_state/linearized_incompressible.h>
#include <aspect/reconstructed_fault/stokes_coupling.h>
#include <aspect/material_model/rheology/friction_normal_stress.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace MaterialModel
  {
    /**
     * Single-lithology Maxwell material in plane strain with mature-fault
     * friction. Three static DG fields store the accepted spatial deviatoric
     * stress (xx, yy, xy in Pa). Slip and slip rate live on the ordered
     * vertices of a piecewise linear material trace. The resistance law owns
     * any additional constitutive history indexed by those vertices.
     *
     * @ingroup MaterialModels
     */
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
        void evaluate(const MaterialModelInputs<dim> &in,
                      MaterialModelOutputs<dim>      &out) const override;
        void create_additional_named_outputs(MaterialModelOutputs<dim> &out) const override;

        /** Return the accepted trace geometry, slip (m) and rate (m/s) in material ID order. */
        const std::vector<typename MaterialFaultTrace<dim>::Label> &labels() const;

        /** Return accepted ages (s) in material ID order, or an empty vector for a state-free law. */
        const std::vector<double> &label_ages() const;

        /** Return the accepted surface Maxwell memory (Pa m) in material ID order. */
        const std::vector<double> &label_memory() const
        {
          return maxwell_memory;
        }

        /** Return the normalized slip distribution associated with the accepted trace. */
        const FaultPhaseProfile<dim> &phase_profile() const
        {
          return accepted->profile;
        }

        /**
         * Evaluate the deviatoric endpoint stress (Pa) at a point in the
         * given cell using its phase and symmetric velocity gradient (1/s).
         * For a boundary-spanning trace this uses the local slip response
         * before projection onto the stress element for Stokes assembly.
         */
        SymmetricTensor<2,dim>
        endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                        const Point<dim> &position,
                        double phase,
                        const SymmetricTensor<2,dim> &strain_rate) const;

        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;

      protected:
        /**
         * Return positive-in-compression normal stress (Pa) for friction.
         * The adiabatic pressure is evaluated at the physical foot on the
         * trace. Resolved traction uses the bulk pressure and the normal
         * component of deviatoric stress at the section quadrature point.
         */
        double normal_compression(const Point<dim> &trace_position,
                                  double            pressure,
                                  double            normal_deviatoric_stress) const;

        /** Return the Eulerian Maxwell viscosity (Pa s) at the constitutive point. */
        virtual double bulk_viscosity(const Point<dim> &) const
        {
          return viscosity;
        }

        /** Return the exterior parent stress (xx, yy, xy in Pa) evaluated at SSPRK stage time (s). */
        virtual Function<dim> &exterior_stress()
        {
          return *inflow_stress;
        }

        /** Couples the local Maxwell response and the section balance to the Stokes equations. */
        FaultStokesCoupling<dim> coupling;

      private:
        friend class aspect::FaultStokesCoupling<dim>;
        using Stress = SymmetricTensor<2,dim>;

        /** Stress (Pa) due to the deviatoric strain rate over the current Maxwell interval. */
        Stress viscous_stress(const Stress     &strain_rate,
                              const Point<dim> &position) const;

        /** Effective viscosity (Pa s) multiplying twice the deviatoric strain rate in Stokes. */
        double effective_maxwell_viscosity(const Point<dim> &position) const;

        /** Dimensionless fraction of the parent stress retained over the Maxwell interval. */
        double stress_relaxation_factor(const Point<dim> &position) const;

        /**
         * On the first Stokes iterate of an interior-seed attempt, copy the
         * accepted trace, advect its vertices if requested, and normalize its
         * slip profile using the phase in the supplied fields.
         */
        void prepare_trial_section(const LinearAlgebra::BlockVector &fields);

        /**
         * Rotate the parent stress (Pa) over the physical timestep (s) using
         * the spin of the accepted velocity gradient minus the slip gradient.
         * At initialization the physical timestep is zero.
         */
        Stress rotate_parent_stress(const Stress        &parent_stress,
                                    const Tensor<2,dim> &velocity_gradient,
                                    const Tensor<2,dim> &slip_gradient,
                                    double               physical_dt) const;

        /** Publish the converged rates (m/s) in material ID order and advance history by dt (s). */
        void accept_section_rates(const std::vector<double> &rates,
                                  double                     dt);

        /** Return the checkpoint key for the selected resistance law's label records. */
        std::string label_checkpoint_key() const;

        /**
         * The phase profile borrows the trace on which it was normalized.
         * Moving the owning pointer publishes both together after acceptance.
         */
        struct Section
        {
          MaterialFaultTrace<dim> trace;
          FaultPhaseProfile<dim> profile {trace};
        };

        /** Use the trial section when present, otherwise the accepted section. */
        Section &active_section()
        {
          return candidate ? *candidate : *accepted;
        }

        EquationOfState::LinearizedIncompressible<dim> eos;
        std::unique_ptr<Rheology::FaultResistance> friction;
        /** Persistent background shear traction (Pa) in material ID order. */
        std::vector<double> label_background_shear;
        std::vector<double> maxwell_memory;
        std::unique_ptr<Section> accepted, candidate;

        /** Both fixed and advected interior seeds use the projected Maxwell response. */
        bool interior_seed = false;
        bool advect_vertices = false;

        /** Maxwell integration interval (s), including the nonzero initialization interval at step zero. */
        double interval;
        double degradation_magnitude;
        /** Compositional indices for xx, yy and xy, in tensor component order. */
        std::array<unsigned int,3> stress_fields;
        unsigned int phase_field;

        /** Inflow boundary data for runs that materially advect the accepted stress. */
        std::unique_ptr<Functions::ParsedFunction<dim>> inflow_stress;
        bool fit_trace;
        bool trace_initialized = false;
        double conductivity;           // Thermal conductivity (W/(m K)).
        double viscosity;              // Bulk Maxwell viscosity (Pa s).
        double modulus;                // Elastic shear modulus (Pa).
        double background_shear;       // Uniform fallback for the vertex shear tractions (Pa).
        double background_compression; // Additional compression for resolved traction (Pa).
        Rheology::FrictionNormalStress normal_stress;
        double initial_interval;       // Maxwell interval for the initial Stokes solve (s).
        double minimum_speed;          // Lower bound for the positive section slip rate (m/s).
    };
  }
}
#endif
