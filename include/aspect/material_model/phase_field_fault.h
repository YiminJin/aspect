/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_phase_field_fault_h
#define _aspect_material_model_phase_field_fault_h

#include <aspect/material_model/equation_of_state/linearized_incompressible.h>
#include <aspect/reconstructed_fault/stokes_coupling.h>
#include <aspect/simulator/solver/midpoint_fault.h>
#include <aspect/material_model/rheology/friction_normal_stress.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace GrowingFault
  {
    template <int dim> class StripMaterialHistory;
  }

  namespace MaterialModel
  {
    /**
     * Plane-strain phase-field fault material. The method of the configured
     * phase composition selects the mechanical formulation. A prescribed
     * trace stores positive slip speeds on ordered material vertices and
     * accepted spatial deviatoric stresses (xx, yy, xy in Pa) in DG fields.
     * Midpoint evolution of an AT1 phase instead couples signed rates on
     * material strips to tensors stored at native material quadrature points.
     * The selected resistance law owns its optional constitutive state.
     * Both formulations publish history after the simulator accepts the step.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class PhaseFieldFault : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        PhaseFieldFault();
        ~PhaseFieldFault() override;

        /** Whether the phase method selects signed strip rates with midpoint AT1 evolution. */
        bool uses_growing_strips() const
        {
          return strips != nullptr;
        }

        /**
         * Read the strip material after parameter parsing. Its mesh-dependent
         * rules and history are available from the first midpoint attempt on
         * a fresh run, or after mesh restoration on a resumed run.
         */
        const GrowingFault::StripMaterialHistory<dim> &strip_history() const;

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

        /** Replicated material-ID rates (m/s) for timestep selection before publication.
         * The completed vector is empty when the attempt has no converged candidate.
         * Reading either vector leaves the accepted state and the candidate intact. */
        struct RatePrediction
        {
          const Rheology::FaultResistance &law;
          std::vector<double> parent, completed;
        };
        RatePrediction timestep_rates() const;

        /** Return accepted ages (s) in material ID order, or an empty vector for a state-free law. */
        const std::vector<double> &label_ages() const;

        /** Return the accepted surface Maxwell memory (Pa m) in material ID order. */
        const std::vector<double> &label_memory() const
        {
          require_mature_trace();
          return maxwell_memory;
        }

        /** Return the normalized slip distribution associated with the accepted trace. */
        const FaultPhaseProfile<dim> &phase_profile() const
        {
          require_mature_trace();
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

        /**
         * Return the mature material's bulk Maxwell viscosity (Pa s) at the
         * given physical position. Requires a parsed prescribed-trace material.
         */
        virtual double bulk_viscosity(const Point<dim> &) const
        {
          require_mature_trace();
          return viscosity;
        }

        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;

      protected:
        /** Which strip input convention a material registration supports. */
        enum class StripInputs
        {
          trace_only, //!< An extension of mature trace mechanics, including a spatial viscosity override.
          current,    //!< The midpoint solver uses the configured homogeneous strip Maxwell response.
          historical  //!< Same strip mechanics, with the original growing input's ignored Simple viscosity.
        };

        /**
         * Parse the selected phase and configure its mechanics. A derived material
         * that implements the strip contract calls this from its parameter reader
         * with the input convention it supports. The inherited parameter reader
         * treats an independently registered material as a mature trace extension.
         */
        void parse_fault_parameters(ParameterHandler &prm, StripInputs strip_inputs);

        /**
         * Return positive-in-compression normal stress (Pa) for friction.
         * The adiabatic pressure is evaluated at the physical foot on the
         * trace. Resolved traction uses the bulk pressure and the normal
         * component of deviatoric stress at the section quadrature point.
         * Requires a parsed prescribed-trace material.
         */
        double normal_compression(const Point<dim> &trace_position,
                                  double            pressure,
                                  double            normal_deviatoric_stress) const;

        /**
         * Supply the mature exterior parent stress (xx, yy, xy in Pa).
         * Called with Advect physical stress enabled. The transport solver
         * sets the function time (s) before evaluating each SSPRK stage.
         */
        virtual Function<dim> &exterior_stress()
        {
          require_mature_trace();
          AssertThrow(inflow_stress, ExcMessage("Exterior parent stress requires Advect physical stress = true."));
          return *inflow_stress;
        }

        /** Couples the local Maxwell response and the section balance to the Stokes equations. */
        std::unique_ptr<FaultStokesCoupling<dim>> coupling;

      private:
        friend class aspect::FaultStokesCoupling<dim>;
        friend SolverControl GrowingFault::advance_midpoint(const SimulatorAccess<2> &,
                                                            MaterialModel::Interface<2> &,
                                                            LinearAlgebra::BlockVector &);

        /** Only the midpoint solver can stage a new strip endpoint. */
        std::unique_ptr<GrowingFault::StripMaterialHistory<dim>> strips;

        /** Configure the trace, its Maxwell response and RSF inputs after field selection. */
        void parse_trace_parameters(ParameterHandler &prm);

        void require_mature_trace() const
        {
          AssertThrow(accepted, ExcMessage("This operation requires mature trace mechanics."));
        }

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

        /** Collectively rebuild the derived normal integrals from the accepted trace and old FE phase. */
        void initialize_accepted_profile();

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
          explicit Section(const int slip_sense) : profile(trace, slip_sense) {}

          MaterialFaultTrace<dim> trace;
          FaultPhaseProfile<dim> profile;
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
        /** Ordered input positions (m), before any fit or material motion, for checkpoint identity. */
        std::vector<Point<dim>> prescribed_vertices;
        /** Initial material support radius (m). Zero denotes the full in-box normal. */
        double prescribed_half_width;
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
