/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/initial_topography_model/zero_topography.h>
#include <deal.II/physics/transformations.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      /** Accepted in-plane deviatoric stress components in Pa for visualization. */
      template <int dim>
      struct FaultPhysicalStress : NamedAdditionalMaterialOutputs<dim>
      {
        explicit FaultPhysicalStress(const unsigned int n)
          : NamedAdditionalMaterialOutputs<dim>({"fault_tau_xx","fault_tau_yy","fault_tau_xy"},n)
        {}
      };
    }

    template <int dim>
    void PhaseFieldFault<dim>::create_additional_named_outputs(MaterialModelOutputs<dim> &out) const
    {
      if (!out.template has_additional_output_object<FaultPhysicalStress<dim>>())
        out.additional_outputs.push_back(std::make_unique<FaultPhysicalStress<dim>>(out.n_evaluation_points()));
    }

    template <int dim>
    PhaseFieldFault<dim>::PhaseFieldFault() : profile(fault), coupling(*this) {}

    template <int dim>
    std::vector<double> PhaseFieldFault<dim>::section_rates(const LinearAlgebra::BlockVector &iterate) const
    {
      return coupling.solve_rates(iterate);
    }

    template <int dim>
    void PhaseFieldFault<dim>::initialize()
    {
      coupling.initialize_simulator(this->get_simulator());
      coupling.initialize();
    }

    template <int dim>
    void PhaseFieldFault<dim>::update()
    {
      const double dt = this->get_timestep_number() == 0 ? initial_interval : this->get_timestep();
      kappa = -viscosity*std::expm1(-modulus*dt/viscosity);
      if (!trace_initialized)
        {
          if (fit_trace)
            fault.fit(*this,phase_field,this->get_old_solution());
          profile.initialize(*this,phase_field,this->get_old_solution(),degradation_magnitude);
          trace_initialized = true;
        }
      coupling.prepare_parent(std::exp(-modulus*dt/viscosity));
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress PhaseFieldFault<dim>::viscous_stress(const Stress &strain_rate) const
    {
      return 2.*kappa*Utilities::Tensors::consistent_deviator(strain_rate);
    }

    template <int dim>
    SymmetricTensor<2,dim> PhaseFieldFault<dim>::endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                 const Point<dim> &x, const double phase, const Stress &strain) const
    {
      return viscous_stress(strain)+coupling.stress_offset(cell,x,phase);
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress
    PhaseFieldFault<dim>::relaxed_parent(const Stress &parent, const Tensor<2,dim> &accepted_gradient,
                                         const Tensor<2,dim> &slip_gradient, const double physical_dt, const double relaxation) const
    {
      if constexpr (dim == 2)
        {
          // Initialization relaxes the parent over its Maxwell interval but has
          // zero physical duration for slip, age and the accepted spin rotation.
          const auto gradient = accepted_gradient - slip_gradient;
          const double angle = .5*physical_dt*(gradient[1][0]-gradient[0][1]);
          const auto rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
          return relaxation*symmetrize(rotation*static_cast<Tensor<2,dim>>(parent)*transpose(rotation));
        }
      AssertThrow(false,ExcNotImplemented());
      return {};
    }

    template <int dim>
    bool PhaseFieldFault<dim>::is_compressible() const
    {
      return eos.is_compressible();
    }

    template <int dim>
    void PhaseFieldFault<dim>::evaluate(const MaterialModelInputs<dim> &in, MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> background(1);
      const auto offset = out.template get_additional_output_object<FaultStressOffset<dim>>();
      const auto physical = out.template get_additional_output_object<FaultPhysicalStress<dim>>();
      for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
        {
          eos.evaluate(in,q,background);
          out.densities[q] = background.densities[0];
          out.thermal_expansion_coefficients[q] = background.thermal_expansion_coefficients[0];
          out.specific_heat[q] = background.specific_heat_capacities[0];
          out.compressibilities[q] = background.compressibilities[0];
          out.entropy_derivative_pressure[q] = background.entropy_derivative_pressure[0];
          out.entropy_derivative_temperature[q] = background.entropy_derivative_temperature[0];
          out.thermal_conductivities[q] = conductivity;
          out.viscosities[q] = kappa;
          std::fill(out.reaction_terms[q].begin(),out.reaction_terms[q].end(),0.);
          if (physical)
            for (unsigned int c=0; c<stress_fields.size(); ++c)
              physical->output_values[c][q] = in.composition[q][stress_fields[c]];
          if (offset)
            offset->stress[q] = coupling.stress_offset(in.current_cell,in.position[q],in.composition[q][phase_field]);
        }
    }

    template <int dim>
    const std::vector<typename StationaryFault<dim>::Label> &PhaseFieldFault<dim>::labels() const
    {
      return fault.labels();
    }

    template <int dim>
    void PhaseFieldFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      prm.enter_subsection("Phase field fault");
      prm.declare_entry("Fault vertices","",Patterns::List(Patterns::List(Patterns::Double(),2,2),0,
                                                           Patterns::List::max_int_value,";"),"Ordered x,y label positions in metres separated by semicolons. "
                        "The default joins the ends of the left box boundary.");
      prm.declare_entry("Degradation magnitude","1",Patterns::Double(0.),"Dimensionless m in the mature AT1 degradation law with curvature one.");
      prm.declare_entry("Fit trace to phase","true",Patterns::Bool(),"Fit the ordered seed to the initialized FE ridge. Constant phase retains the seed.");
      prm.declare_entry("Profile half width","0",Patterns::Double(0.),"Normal association support in metres. Zero uses the full physical cross-section.");
      prm.declare_entry("Shear modulus","1e10",Patterns::Double(0.),"Elastic shear modulus in Pa.");
      prm.declare_entry("Initialization interval","1",Patterns::Double(0.),"Maxwell interval in seconds for the initial Stokes solve.");
      prm.declare_entry("Background shear traction","0",Patterns::Double(),"Initial surface shear loading in Pa, positive along the slip sense.");
      prm.declare_entry("Background compression","5e7",Patterns::Double(0.),"Compression in Pa added to the dynamic normal traction.");
      prm.declare_entry("Reference friction coefficient","0.6",Patterns::Double(0.),"Dimensionless reference friction coefficient.");
      prm.declare_entry("Direct effect","0.01",Patterns::Double(0.),"Dimensionless direct effect parameter.");
      prm.declare_entry("Evolution effect","0.015",Patterns::Double(0.),"Dimensionless evolution effect parameter.");
      prm.declare_entry("Reference speed","1e-6",Patterns::Double(0.),"Reference slip speed in m/s.");
      prm.declare_entry("Minimum speed","1e-20",Patterns::Double(0.),"Minimum admissible sliding speed in m/s.");
      prm.declare_entry("Slip distance","0.01",Patterns::Double(0.),"Characteristic slip distance in m.");
      prm.declare_entry("Radiation damping","0",Patterns::Double(0.),"Radiation damping coefficient in Pa s/m.");
      prm.declare_entry("Initial age","1",Patterns::Double(0.),"Initial section age in seconds.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    template <int dim>
    void PhaseFieldFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      // Some wrappers parse their base material without forwarding initialize().
      AssertThrow(&this->get_material_model() == this,
                  ExcMessage("Phase field fault must be the top-level material model so Stokes and "
                             "the fault use the same Maxwell coefficient."));
      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = NonlinearDependence::temperature;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
      prm.enter_subsection("Termination criteria");
      const bool checkpoint_on_termination = prm.get_bool("Checkpoint on termination");
      prm.leave_subsection();
      const auto &p = this->get_parameters();
      AssertThrow(prm.get("Resume computation") == "false" && p.resume_checkpoint_id == 0 && p.resume_time < 0. &&
                  p.checkpoint_steps == 0 && p.checkpoint_time_secs == 0 &&
                  p.additional_checkpoint_times.empty() && !checkpoint_on_termination,
                  ExcMessage("Phase field fault requires a fresh run with checkpointing disabled: label history is in memory."));
      prm.enter_subsection("Material model");
      prm.enter_subsection("Simple model");
      // The Simple model subsection supplies the single-lithology background
      // constants. Only EOS entry zero is consumed, irrespective of DG stress.
      eos.parse_parameters(prm,1);
      conductivity = prm.get_double("Thermal conductivity");
      viscosity = prm.get_double("Viscosity");
      AssertThrow(prm.get_double("Thermal viscosity exponent") == 0 && prm.get_double("Composition viscosity prefactor") == 1.,
                  ExcMessage("The stationary fault requires a constant bulk viscosity."));
      prm.leave_subsection();
      prm.enter_subsection("Phase field fault");
      modulus = prm.get_double("Shear modulus");
      initial_interval = prm.get_double("Initialization interval");
      background_shear = prm.get_double("Background shear traction");
      background_compression = prm.get_double("Background compression");
      friction = {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),
                  prm.get_double("Evolution effect"),prm.get_double("Reference speed"),
                  prm.get_double("Slip distance"),prm.get_double("Radiation damping")
                 };
      minimum_speed = prm.get_double("Minimum speed");
      const double age = prm.get_double("Initial age");
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      AssertThrow(Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(this->get_initial_topography_model()),
                  ExcMessage("The stationary fault requires the zero initial topography model "
                             "because its normalization uses the rectangular box width."));
      std::vector<Point<dim>> vertices;
      for (const auto &entry : Utilities::split_string_list(prm.get("Fault vertices"),';'))
        {
          const auto coordinates = Utilities::string_to_double(Utilities::split_string_list(entry));
          Point<dim> point;
          point[0] = coordinates.at(0);
          point[1] = coordinates.at(1);
          vertices.push_back(point);
        }
      if (vertices.empty())
        {
          vertices = {box.get_origin(),box.get_origin()};
          vertices.back()[1] += box.get_extents()[1];
        }
      degradation_magnitude = prm.get_double("Degradation magnitude");
      AssertThrow(degradation_magnitude>0.,ExcMessage("Degradation magnitude must be positive."));
      fit_trace = prm.get_bool("Fit trace to phase");
      const double support = prm.get_double("Profile half width");
      fault.initialize(box.get_origin(),box.get_extents(),vertices,age,friction.reference_speed,
                       support > 0. ? support : std::numeric_limits<double>::max());
      prm.leave_subsection();
      prm.leave_subsection();
      AssertThrow(std::min({viscosity,modulus,initial_interval,friction.direct_effect,minimum_speed,
                            friction.reference_speed,friction.slip_distance,age
                           }) > 0.,
                  ExcMessage("The fixed fault requires positive material, time and speed scales."));
      AssertThrow(friction.reference_speed >= minimum_speed,
                  ExcMessage("The reference speed must be at least the minimum admissible speed."));
      AssertThrow(dim == 2 && !p.enable_elasticity && p.nonlinear_solver == Parameters<dim>::NonlinearSolver::no_Advection_iterated_Stokes
                  && p.stokes_solver_type == Parameters<dim>::StokesSolverType::block_amg && p.stokes_velocity_degree == 2
                  && !p.run_postprocessors_on_nonlinear_iterations && !p.mesh_deformation_enabled && !p.include_melt_transport && p.pressure_normalization == "no"
                  && p.nonlinear_solver_failure_strategy != Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep,
                  ExcMessage("The fixed fault requires a fixed 2-D Q2 box with ASPECT elasticity disabled, block AMG, no pressure normalization, "
                             "no Advection, iterated Stokes, and convergence or retry."));
      AssertThrow(!p.enable_prescribed_dilation && (p.formulation_mass_conservation ==
                                                    Parameters<dim>::Formulation::MassConservation::incompressible || p.formulation_mass_conservation ==
                                                    Parameters<dim>::Formulation::MassConservation::ask_material_model),
                  ExcMessage("Stationary fault mechanics requires incompressible continuity without prescribed dilation."));
      AssertThrow(p.initial_adaptive_refinement == 0 && p.adaptive_refinement_interval == 0 &&
                  p.additional_refinement_times.empty(),
                  ExcMessage("The stationary FE phase profile requires a fixed Cartesian mesh."));
      const auto &intro = this->introspection();
      const std::array<std::string,3> names = {{"fault_stress_xx","fault_stress_yy","fault_stress_xy"}};
      AssertThrow(intro.n_compositional_fields == 4,ExcMessage("Expected three fault stress fields and phase_field."));
      for (unsigned int c=0; c<3; ++c)
        {
          stress_fields[c] = intro.compositional_index_for_name(names[c]);
          AssertThrow(p.composition_descriptions[stress_fields[c]].type == CompositionalFieldDescription::stress,
                      ExcMessage(names[c]+" must have compositional type stress."));
        }
      phase_field = intro.compositional_index_for_name("phase_field");
      AssertThrow(p.composition_descriptions[phase_field].type == CompositionalFieldDescription::generic,
                  ExcMessage("phase_field must have compositional type generic."));
      for (unsigned int c=0; c<intro.n_compositional_fields; ++c)
        AssertThrow(p.composition_degrees[c] == 2 && p.use_discontinuous_composition_discretization[c] &&
                    p.compositional_field_methods[c] == Parameters<dim>::AdvectionFieldMethod::static_field,
                    ExcMessage("Fault stress and phase require static DGQ2 fields."));

      kappa = -viscosity*std::expm1(-modulus*initial_interval/viscosity);

    }

    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,"phase field fault",
                                   "Stationary mature fault with independent Q1 labels spanning a Cartesian box. "
                                   "The static DG phase supplies the fitted ridge and its normalized slip distribution. "
                                   "The named additional outputs fault_tau_xx, fault_tau_yy and fault_tau_xy read accepted physical stress in Pa.")
  }
}
