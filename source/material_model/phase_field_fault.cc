/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault/input_compatibility.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/reconstructed_fault/fault_vertex_table.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/material_model/rheology/fault_resistance.h>
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
        explicit FaultPhysicalStress(const unsigned int n_evaluation_points)
          : NamedAdditionalMaterialOutputs<dim>({"fault_tau_xx", "fault_tau_yy", "fault_tau_xy"}, n_evaluation_points)
        {}
      };
    }

    template <int dim>
    void PhaseFieldFault<dim>::create_additional_named_outputs(MaterialModelOutputs<dim> &out) const
    {
      if (!uses_growing_strips() && !out.template has_additional_output_object<FaultPhysicalStress<dim>>())
        out.additional_outputs.push_back(std::make_unique<FaultPhysicalStress<dim>>(out.n_evaluation_points()));
    }

    template <int dim>
    PhaseFieldFault<dim>::PhaseFieldFault() = default;

    template <int dim>
    PhaseFieldFault<dim>::~PhaseFieldFault() = default;

    template <int dim>
    const GrowingFault::StripMaterialHistory<dim> &PhaseFieldFault<dim>::strip_history() const
    {
      AssertThrow(strips, ExcMessage("This operation requires growing strip mechanics."));
      return *strips;
    }

    template <int dim>
    typename PhaseFieldFault<dim>::RatePrediction PhaseFieldFault<dim>::timestep_rates() const
    {
      RatePrediction rates {strips ? strips->friction() :*friction, {}, {}};
      if (strips)
        {
          // All strips age for the full duration at their signed midpoint rate,
          // including unformed strips and strips first admitted in this attempt.
          for (const auto &strip : strips->history().sections)
            rates.parent.push_back(strip.rate);
          if (const auto *endpoint = strips->staged_sections())
            for (const auto &strip : *endpoint)
              rates.completed.push_back(strip.rate);
        }
      else
        {
          for (const auto &label : labels())
            rates.parent.push_back(label.rate);
          if (const auto *endpoint = coupling->converged_rates())
            rates.completed = *endpoint;
        }
      return rates;
    }

    template <int dim>
    void PhaseFieldFault<dim>::initialize()
    {
      if (strips)
        {
          this->get_pcout() << "   Growing fault midpoint solver: bordered matrix gathered on MPI rank 0, "
                            << "UMFPACK direct factorization (separate velocity and pressure blocks)." << std::endl;
          this->get_pcout() << "   Fault Maxwell material (growing strips): shear modulus = " << strips->maxwell().G
                            << " Pa, viscosity = " << strips->maxwell().eta << " Pa s." << std::endl;
          this->get_signals().post_resume_load_user_data.connect([this](parallel::distributed::Triangulation<dim> &)
          {
            strips->initialize_native_history();
          });
          this->get_signals().pre_refinement_store_user_data.connect([this](parallel::distributed::Triangulation<dim> &mesh)
          {
            strips->prepare_history_transfer(mesh);
          });
          this->get_signals().post_refinement_load_user_data.connect([this](parallel::distributed::Triangulation<dim> &mesh)
          {
            strips->finish_history_transfer(mesh);
          });
          this->get_signals().post_accept_timestep.connect([this](LinearAlgebra::BlockVector &)
          {
            strips->accept_endpoint(this->get_timestep());
          });
          return;
        }

      this->get_pcout() << "   Mature fault Stokes solver: ASPECT block AMG." << std::endl;
      coupling->initialize_simulator(this->get_simulator());
      coupling->initialize();
      coupling->connect_section_balance();
      this->get_signals().post_resume_load_user_data.connect([this](parallel::distributed::Triangulation<dim> &)
      {
        // load() has restored the accepted labels. ASPECT has now restored
        // their mesh and old solution, so a fitted seed must not be fitted again.
        if (!complete_contacts)
          initialize_accepted_profile();
        trace_initialized = true;
      });
    }

    template <int dim>
    void PhaseFieldFault<dim>::initialize_accepted_profile()
    {
      accepted->profile.initialize(*this, phase_field, this->get_old_solution(),
                                   degradation_magnitude, normal_integral_subdivisions, complete_contacts);
    }

    template <int dim>
    void PhaseFieldFault<dim>::update()
    {
      if (strips)
        return;

      interval = this->get_timestep_number() == 0 ? initial_interval : this->get_timestep();
      candidate.reset();
      if (!trace_initialized)
        {
          if (fit_trace)
            accepted->trace.fit_boundary_spanning_seed_to_phase_ridge(*this, phase_field, this->get_old_solution());
          if (!interior_seed || this->get_timestep_number() > 0)
            initialize_accepted_profile();
          trace_initialized = true;
        }
      coupling->prepare_parent();
    }

    template <int dim>
    void PhaseFieldFault<dim>::prepare_trial_section(const LinearAlgebra::BlockVector &fields)
    {
      if (!interior_seed || candidate)
        return;

      candidate = std::make_unique<Section>(accepted->profile.slip_sense);
      candidate->trace = accepted->trace;
      if (advect_vertices && this->get_timestep() > 0.)
        candidate->trace.advect_with_accepted_velocity(*this, this->get_timestep());
      candidate->profile.initialize(*this, phase_field, fields, degradation_magnitude, normal_integral_subdivisions);
    }

    template <int dim>
    double PhaseFieldFault<dim>::normal_compression(const Point<dim> &trace_position,
                                                    const double      pressure,
                                                    const double      normal_deviatoric_stress) const
    {
      require_mature_trace();
      if (normal_stress.uses_adiabat())
        return this->get_adiabatic_conditions().pressure(trace_position);
      return background_compression + pressure - normal_deviatoric_stress;
    }

    template <int dim>
    double PhaseFieldFault<dim>::effective_maxwell_viscosity(const Point<dim> &position) const
    {
      const double eta = bulk_viscosity(position);
      return -eta * std::expm1(-modulus * interval / eta);
    }

    template <int dim>
    double PhaseFieldFault<dim>::stress_relaxation_factor(const Point<dim> &position) const
    {
      return std::exp(-modulus * interval / bulk_viscosity(position));
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress
    PhaseFieldFault<dim>::viscous_stress(const Stress     &strain_rate,
                                         const Point<dim> &position) const
    {
      return 2. * effective_maxwell_viscosity(position) * Utilities::Tensors::consistent_deviator(strain_rate);
    }

    template <int dim>
    SymmetricTensor<2,dim>
    PhaseFieldFault<dim>::endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          const Point<dim> &position,
                                          const double phase,
                                          const Stress &strain_rate) const
    {
      require_mature_trace();
      return viscous_stress(strain_rate, position) + coupling->stress_offset(cell, position, phase);
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress
    PhaseFieldFault<dim>::rotate_parent_stress(const Stress        &parent_stress,
                                               const Tensor<2,dim> &velocity_gradient,
                                               const Tensor<2,dim> &slip_gradient,
                                               const double         physical_dt) const
    {
      if constexpr (dim == 2)
        {
          // Initialization relaxes the parent over its Maxwell interval but has
          // zero physical duration for slip, age and the accepted spin rotation.
          const auto gradient = velocity_gradient - slip_gradient;
          const double angle = .5 * physical_dt * (gradient[1][0] - gradient[0][1]);
          const auto rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
          return symmetrize(rotation * static_cast<Tensor<2,dim>>(parent_stress) * transpose(rotation));
        }
      AssertThrow(false, ExcNotImplemented());
      return {};
    }

    template <int dim>
    bool PhaseFieldFault<dim>::is_compressible() const
    {
      return !uses_growing_strips() && eos.is_compressible();
    }

    template <int dim>
    void PhaseFieldFault<dim>::evaluate(const MaterialModelInputs<dim> &in,
                                        MaterialModelOutputs<dim>      &out) const
    {
      if (strips)
        {
          // The midpoint solver assembles its native Maxwell response. Ordinary
          // material consumers see the bulk viscosity, unit density and heat
          // capacity, and zero thermal conductivity.
          for (unsigned int q = 0; q < in.n_evaluation_points(); ++q)
            {
              out.viscosities[q] = strips->maxwell().eta;
              out.densities[q] = 1.;
              out.specific_heat[q] = 1.;
              out.thermal_conductivities[q] = 0.;
              out.thermal_expansion_coefficients[q] = out.compressibilities[q] = 0.;
              out.entropy_derivative_pressure[q] = out.entropy_derivative_temperature[q] = 0.;
              std::fill(out.reaction_terms[q].begin(), out.reaction_terms[q].end(), 0.);
            }
          return;
        }

      EquationOfStateOutputs<dim> background(1);
      const auto offset = out.template get_additional_output_object<FaultStressOffset<dim>>();
      const auto physical = out.template get_additional_output_object<FaultPhysicalStress<dim>>();
      for (unsigned int q = 0; q < in.n_evaluation_points(); ++q)
        {
          eos.evaluate(in, q, background);
          out.densities[q] = background.densities[0];
          out.thermal_expansion_coefficients[q] = background.thermal_expansion_coefficients[0];
          out.specific_heat[q] = background.specific_heat_capacities[0];
          out.compressibilities[q] = background.compressibilities[0];
          out.entropy_derivative_pressure[q] = background.entropy_derivative_pressure[0];
          out.entropy_derivative_temperature[q] = background.entropy_derivative_temperature[0];
          out.thermal_conductivities[q] = conductivity;
          out.viscosities[q] = effective_maxwell_viscosity(in.position[q]);
          std::fill(out.reaction_terms[q].begin(), out.reaction_terms[q].end(), 0.);
          if (physical)
            for (unsigned int c = 0; c < stress_fields.size(); ++c)
              physical->output_values[c][q] = in.composition[q][stress_fields[c]];
          if (offset)
            offset->stress[q] = coupling->stokes_stress_offset(in.current_cell, in.position[q], in.composition[q][phase_field]);
        }
    }

    template <int dim>
    const std::vector<typename MaterialFaultTrace<dim>::Label> &PhaseFieldFault<dim>::labels() const
    {
      require_mature_trace();
      return accepted->trace.labels();
    }

    template <int dim>
    const std::vector<double> &PhaseFieldFault<dim>::label_ages() const
    {
      require_mature_trace();
      return friction->ages_in_seconds();
    }

    template <int dim>
    void PhaseFieldFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      using FaultInputCompatibility::Use;
      Rheology::FrictionNormalStress::declare_parameters(prm);
      GrowingFault::StripMaterialHistory<dim>::declare_parameters(prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field fault");
        {
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Advect physical stress", "false",
                                                 Patterns::Bool(),
                                                 "Transport the accepted DG stress with the accepted velocity before "
                                                 "rotation and relaxation. Exterior parent stress specifies xx; yy; xy "
                                                 "in Pa, at x,y in metres and t in seconds.");
          prm.enter_subsection("Exterior parent stress");
          {
            Functions::ParsedFunction<dim>::declare_parameters(prm, 3);
            const char *const function_parameters[] = {"Variable names", "Function constants", "Function expression"};
            for (const auto name : function_parameters)
              FaultInputCompatibility::observe_entry(prm, Use::mature, name, Patterns::Anything());
          }
          prm.leave_subsection();

          FaultInputCompatibility::declare_entry(prm, Use::mature, "Fault geometry", "automatic",
                                                 Patterns::Selection("automatic|boundary spanning|fixed interior|advected interior"),
                                                 "Boundary spanning uses the stationary section law. Both interior modes use "
                                                 "the prescribed material seed and the same Q8 projected Maxwell response. "
                                                 "Advected interior moves vertices with accepted velocity and requires "
                                                 "bounded phase and physical stress transport. Fixed interior holds the "
                                                 "vertices fixed. Automatic preserves earlier inputs by selecting advected "
                                                 "interior for the phase field composition role and boundary spanning for "
                                                 "the generic role.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Fault vertex file", "",
                                                 Patterns::FileName(),
                                                 "Ordered ASCII table: id x_m y_m, optionally initial_age_s and "
                                                 "background_shear_Pa. IDs start at zero. Row order sets the tangent; "
                                                 "shear persists through all timesteps.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Mature slip sense", "along tangent",
                                                 Patterns::Selection("along tangent|opposite tangent"),
                                                 "The tangent follows increasing vertex ID and its counterclockwise "
                                                 "normal points to the positive side. Positive slip speed moves this "
                                                 "side relative to the negative side along or opposite the tangent. "
                                                 "Background shear is measured in this slip sense. Accumulated slip "
                                                 "and aging use the positive speed in either sense.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Fault vertices", "",
                                                 Patterns::List(Patterns::List(Patterns::Double(), 2, 2),
                                                                0, Patterns::List::max_int_value, ";"),
                                                 "Ordered x,y label positions in metres separated by semicolons. "
                                                 "The default joins the ends of the left box boundary.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Degradation magnitude", "1",
                                                 Patterns::Double(0.),
                                                 "Scale of mature excess compliance and of its projected normal integral. "
                                                 "At fixed phase it cancels in the normalized slip weight.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Fit trace to phase", "true",
                                                 Patterns::Bool(),
                                                 "Fit a straight boundary-spanning seed to the initialized FE phase "
                                                 "ridge. Constant phase retains the seed. Set to false to use the "
                                                 "prescribed vertex positions, including curved traces.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Profile half width", "0",
                                                 Patterns::Double(0.),
                                                 "Normal association support in metres. Zero uses the full physical "
                                                 "cross-section.");

          FaultInputCompatibility::declare_entry(prm, "Viscosity", "5e24",
                                                 Patterns::Double(0.),
                                                 "Maxwell viscosity in Pa s. The default is 5e24 for a mature trace "
                                                 "and 1e8 for growing strips.");
          FaultInputCompatibility::declare_entry(prm, "Shear modulus", "1e10",
                                                 Patterns::Double(0.),
                                                 "Elastic shear modulus in Pa. With no explicit setting, mature traces "
                                                 "use 1e10 and growing strips use 1e6.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Initialization interval", "1",
                                                 Patterns::Double(0.),
                                                 "Maxwell interval in seconds for the initial Stokes solve.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Background shear traction", "0",
                                                 Patterns::Double(),
                                                 "Persistent shear loading in Pa along the slip sense; the per-vertex "
                                                 "column overrides this uniform value.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Background compression", "",
                                                 Patterns::List(Patterns::Double(0.), 0, 1),
                                                 "Compression in Pa added to resolved normal traction (default 5e7 Pa). "
                                                 "Leave empty for adiabatic pressure, which supplies the complete "
                                                 "compression.");

          FaultInputCompatibility::declare_entry(prm, Use::mature, "Normal integral subdivisions", "0", Patterns::Integer(0),
                                                 "Number of three-point panels on each material segment for projecting the normal integral. "
                                                 "Zero selects one for boundary-spanning traces and eight for interior traces.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Normal traction filter length", "0", Patterns::Double(0.),
                                                 "Helmholtz length in metres for the resolved normal traction perturbation "
                                                 "on a mature boundary-spanning trace. Zero uses the pointwise traction.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Complete prescribed contact profiles", "false", Patterns::Bool(),
                                                 "Continue the fixed initial phase outside the box for normalization and extend the "
                                                 "endpoint rate into each in-box terminal wedge. Requires a straight, unfitted trace "
                                                 "with finite support and an initial phase function valid on the exterior lattice.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Report section convergence", "false", Patterns::Bool(),
                                                 "Report the material identity with the largest relative rate change when the "
                                                 "mature section requires another Stokes Picard iteration.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Record section tractions", "false", Patterns::Bool(),
                                                 "Project accepted shear and normal tractions onto the material vertices for output.");

          FaultInputCompatibility::declare_entry(prm, Use::mature, "Minimum speed", "1e-20",
                                                 Patterns::Double(0.),
                                                 "Minimum admissible sliding speed in m/s.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Initial slip rate", "",
                                                 Patterns::List(Patterns::Double(0.), 0, 1),
                                                 "Positive initial material vertex speed in m/s, used before the first "
                                                 "section solve. An empty entry uses Reference speed for compatibility "
                                                 "with earlier trace inputs.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Initial label ages", "",
                                                 Patterns::List(Patterns::Double(0.)),
                                                 "Optional ordered initial material ages in seconds.");
          FaultInputCompatibility::declare_entry(prm, Use::mature, "Initial label shear tractions", "",
                                                 Patterns::List(Patterns::Double()),
                                                 "Optional persistent shear tractions in Pa, ordered by vertex.");

          Rheology::FaultResistance::declare_parameters(prm, Rheology::FaultResistance::Route::mature);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void PhaseFieldFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      const auto name = prm.get({"Material model"}, "Model name");
      if (name == "phase field fault")
        parse_fault_parameters(prm, StripInputs::current);
      else if (name == "growing fault")
        parse_fault_parameters(prm, StripInputs::historical);
      else
        parse_fault_parameters(prm, StripInputs::trace_only);
    }

    template <int dim>
    void PhaseFieldFault<dim>::parse_fault_parameters(ParameterHandler  &prm,
                                                      const StripInputs strip_inputs)
    {
      using AdvectionMethod = typename Parameters<dim>::AdvectionFieldMethod;
      const auto &parameters = this->get_parameters();
      const auto &methods = parameters.compositional_field_methods;
      phase_field = this->introspection().compositional_index_for_name(parameters.surface_energy_model.composition);
      const bool growing = methods[phase_field] == AdvectionMethod::midpoint_fault_phase;

      // A midpoint field on a different composition must be diagnosed before
      // the trace path tries to read its three stress fields. The historical
      // material name also required a midpoint solve on the configured phase.
      if (prm.get({"Material model"}, "Model name") == "growing fault" ||
          std::find(methods.begin(), methods.end(), AdvectionMethod::midpoint_fault_phase) != methods.end())
        AssertThrow(growing && &this->get_material_model() == this,
                    ExcMessage("Growing fault requires a top-level midpoint material and a midpoint fault phase."));

      // A wrapper can parse a base material without forwarding initialize().
      AssertThrow(&this->get_material_model() == this,
                  ExcMessage("Phase field fault must be the top-level material model so Stokes and "
                             "the fault use the same Maxwell coefficient."));
      AssertThrow(!growing || strip_inputs != StripInputs::trace_only,
                  ExcMessage("The " + prm.get({"Material model"}, "Model name") +
                             " selector requires mature trace mechanics. Its material registration "
                             "does not support midpoint strip mechanics."));
      FaultInputCompatibility::resolve(prm, growing, strip_inputs == StripInputs::historical);
      // Trace normalization and the pressure stations on the growing graph
      // both use the faces of the undeformed box.
      AssertThrow(Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(
                    this->get_initial_topography_model()),
                  ExcMessage("The fault material requires the zero initial topography model. "
                             "Set Geometry model / Initial topography model / Model name = zero topography."));
      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = growing ? NonlinearDependence::none : NonlinearDependence::temperature;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;

      if (growing)
        {
          parameters.surface_energy_model.validate();
          strips = std::make_unique<GrowingFault::StripMaterialHistory<dim>>(*this, prm, phase_field);
        }
      else
        {
          AssertThrow(dim == 2, ExcMessage("Reconstructed fault mechanics requires 2-D plane strain."));
          coupling = std::make_unique<FaultStokesCoupling<dim>>(*this);
          parse_trace_parameters(prm);
        }
    }

    template <int dim>
    void PhaseFieldFault<dim>::parse_trace_parameters(ParameterHandler &prm)
    {
      using AdvectionMethod = typename Parameters<dim>::AdvectionFieldMethod;
      using NonlinearSolver = typename Parameters<dim>::NonlinearSolver;
      using MassConservation = typename Parameters<dim>::Formulation::MassConservation;
      normal_stress = Rheology::FrictionNormalStress::parse(prm,
                                                            Rheology::FrictionNormalStress::Source::resolved_traction);

      const auto &parameters = this->get_parameters();
      const auto &introspection = this->introspection();
      const std::array<std::string,3> stress_field_names = {{"fault_stress_xx", "fault_stress_yy", "fault_stress_xy"}};
      AssertThrow(introspection.n_compositional_fields == 4,
                  ExcMessage("Expected three fault stress fields and one phase composition."));
      for (unsigned int c=0; c<stress_field_names.size(); ++c)
        {
          const auto &field_name = stress_field_names[c];
          stress_fields[c] = introspection.compositional_index_for_name(field_name);
          AssertThrow(stress_fields[c] != phase_field,
                      ExcMessage("Phase composition cannot be a mature fault stress component: " + field_name));
          AssertThrow(parameters.composition_descriptions[stress_fields[c]].type ==
                      CompositionalFieldDescription::stress,
                      ExcMessage(field_name + " must have compositional type stress."));
        }
      const bool transport_phase =
        parameters.compositional_field_methods[phase_field] == AdvectionMethod::explicit_dg_ssprk3;

      double initial_speed;
      bool has_phase_field_role;
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Simple model");
        {
          // The single-lithology background uses only entry zero of the
          // equation of state, irrespective of the number of DG stress fields.
          eos.parse_parameters(prm, 1);
          conductivity = prm.get_double("Thermal conductivity");
          AssertThrow(prm.get_double("Thermal viscosity exponent") == 0 &&
                      prm.get_double("Composition viscosity prefactor") == 1.,
                      ExcMessage("The base Maxwell viscosity requires zero thermal exponent and unit "
                                 "compositional prefactor."));
        }
        prm.leave_subsection();

        prm.enter_subsection("Phase field fault");
        {
          const std::string fault_geometry = prm.get("Fault geometry");
          accepted = std::make_unique<Section>(prm.get("Mature slip sense") == "along tangent" ? 1 : -1);
          // Older inputs selected the geometry through the role of the phase composition.
          has_phase_field_role =
            parameters.composition_descriptions[phase_field].type == CompositionalFieldDescription::phase_field;
          advect_vertices = fault_geometry == "advected interior" ||
                            (fault_geometry == "automatic" && has_phase_field_role);
          interior_seed = advect_vertices || fault_geometry == "fixed interior";
          AssertThrow(!advect_vertices || transport_phase,
                      ExcMessage("An advected interior seed requires bounded SSPRK3 transport of the "
                                 "selected phase composition."));
          AssertThrow(!transport_phase || interior_seed,
                      ExcMessage("Transported phase requires an interior fault seed."));

          if (prm.get_bool("Advect physical stress"))
            {
              AssertThrow(this->get_geometry_model().get_periodic_boundary_pairs().empty(),
                          ExcMessage("Advected physical stress requires a nonperiodic Cartesian box."));
              prm.enter_subsection("Exterior parent stress");
              {
                inflow_stress = std::make_unique<Functions::ParsedFunction<dim>>(3);
                inflow_stress->parse_parameters(prm);
              }
              prm.leave_subsection();
            }
          viscosity = prm.get_double("Viscosity");
          modulus = prm.get_double("Shear modulus");
          initial_interval = prm.get_double("Initialization interval");
          background_shear = prm.get_double("Background shear traction");
          AssertThrow(!normal_stress.uses_adiabat() || prm.get("Background compression").empty(),
                      ExcMessage("Adiabatic friction supplies the complete pressure; omit Background compression."));
          normal_integral_subdivisions = prm.get_integer("Normal integral subdivisions");
          if (normal_integral_subdivisions==0)
            normal_integral_subdivisions = interior_seed ? 8 : 1;
          normal_filter_length = prm.get_double("Normal traction filter length");
          report_section_convergence = prm.get_bool("Report section convergence");
          record_section_tractions = prm.get_bool("Record section tractions");
          AssertThrow(normal_filter_length == 0. || (!normal_stress.uses_adiabat() && !interior_seed),
                      ExcMessage("Normal traction filtering requires a boundary-spanning resolved-traction fault."));
          background_compression = prm.get("Background compression").empty() ?
                                   5e7 : prm.get_double("Background compression");
          initial_speed = prm.get("Initial slip rate").empty() ?
                          prm.get_double("Reference speed") : prm.get_double("Initial slip rate");
          friction = Rheology::FaultResistance::create(prm, Rheology::FaultResistance::Route::mature);
          minimum_speed = prm.get_double("Minimum speed");

          const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

          prescribed_vertices.clear();
          for (const auto &entry : Utilities::split_string_list(prm.get("Fault vertices"), ';'))
            {
              const auto coordinates = Utilities::string_to_double(Utilities::split_string_list(entry));
              Point<dim> vertex;
              vertex[0] = coordinates.at(0);
              vertex[1] = coordinates.at(1);
              prescribed_vertices.push_back(vertex);
            }
          auto initial_ages = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial label ages")));
          label_background_shear = Utilities::string_to_double(
                                     Utilities::split_string_list(prm.get("Initial label shear tractions")));
          const auto vertex_filename = prm.get("Fault vertex file");
          if (!vertex_filename.empty())
            {
              AssertThrow(prescribed_vertices.empty(), ExcMessage("Choose Fault vertex file or inline Fault vertices."));
              auto table = FaultVertexTable<dim>::read(vertex_filename, this->get_mpi_communicator());
              prescribed_vertices = std::move(table.positions);
              if (!table.initial_ages.empty())
                {
                  AssertThrow(initial_ages.empty(), ExcMessage("Duplicate inline and file source for initial_age_s"));
                  initial_ages = std::move(table.initial_ages);
                }
              if (!table.background_shear_tractions.empty())
                {
                  AssertThrow(label_background_shear.empty(),
                              ExcMessage("Duplicate inline and file source for background_shear_Pa"));
                  label_background_shear = std::move(table.background_shear_tractions);
                }
            }
          friction->set_initial_ages(std::move(initial_ages));
          if (prescribed_vertices.empty())
            {
              prescribed_vertices = {box.get_origin(), box.get_origin()};
              prescribed_vertices.back()[1] += box.get_extents()[1];
            }

          degradation_magnitude = prm.get_double("Degradation magnitude");
          AssertThrow(degradation_magnitude > 0., ExcMessage("Degradation magnitude must be positive."));
          fit_trace = prm.get_bool("Fit trace to phase");
          prescribed_half_width = prm.get_double("Profile half width");
          complete_contacts = prm.get_bool("Complete prescribed contact profiles");
          AssertThrow(!complete_contacts || (!interior_seed && !fit_trace && prescribed_half_width>0.),
                      ExcMessage("Contact completion requires a fixed boundary-spanning trace and finite support."));
          AssertThrow(!interior_seed || (!fit_trace && (!advect_vertices || inflow_stress)),
                      ExcMessage("Interior mechanics requires a prescribed seed with fitting disabled. "
                                 "Moving its vertices also requires physical stress transport."));
          accepted->trace.initialize(box.get_origin(),
                                     box.get_extents(),
                                     prescribed_vertices,
                                     initial_speed,
                                     prescribed_half_width > 0. ? prescribed_half_width : std::numeric_limits<double>::max(),
                                     interior_seed);
          const auto n_vertices = labels().size();
          friction->initialize(n_vertices);
          if (label_background_shear.empty())
            label_background_shear.assign(n_vertices, background_shear);
          AssertThrow(label_background_shear.size() == n_vertices,
                      ExcMessage("Material initial shear must match the number of fault vertices."));
          maxwell_memory.assign(n_vertices, 0.);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      AssertThrow(std::min({viscosity, modulus, initial_interval, minimum_speed, initial_speed}) > 0.,
                  ExcMessage("The fault material requires positive material, time and speed scales."));
      AssertThrow(initial_speed >= minimum_speed,
                  ExcMessage("The initial slip rate must be at least the minimum admissible speed."));

      const auto required_solver = transport_phase ? NonlinearSolver::single_Advection_iterated_Stokes :
                                   NonlinearSolver::no_Advection_iterated_Stokes;
      AssertThrow(dim == 2 &&
                  !parameters.enable_elasticity &&
                  parameters.nonlinear_solver == required_solver &&
                  parameters.stokes_solver_type == Parameters<dim>::StokesSolverType::block_amg &&
                  parameters.stokes_velocity_degree == 2 &&
                  !parameters.run_postprocessors_on_nonlinear_iterations &&
                  !parameters.mesh_deformation_enabled &&
                  !parameters.include_melt_transport &&
                  (normal_stress.uses_adiabat() || parameters.pressure_normalization == "no") &&
                  parameters.nonlinear_solver_failure_strategy !=
                  Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep,
                  ExcMessage("The fault material requires a fixed 2-D Q2 box with ASPECT elasticity disabled, "
                             "block AMG, no pressure normalization for resolved traction, a composition solve for "
                             "selected phase transport, iterated Stokes, and convergence or retry."));
      AssertThrow(!parameters.enable_prescribed_dilation &&
                  (parameters.formulation_mass_conservation == MassConservation::incompressible ||
                   parameters.formulation_mass_conservation == MassConservation::ask_material_model),
                  ExcMessage("Fault mechanics requires incompressible continuity without prescribed dilation."));
      AssertThrow(parameters.initial_adaptive_refinement == 0 &&
                  parameters.adaptive_refinement_interval == 0 &&
                  parameters.additional_refinement_times.empty(),
                  ExcMessage("The stationary FE phase profile requires a fixed Cartesian mesh."));

      // Both static roles use the same nodal DG basis. Only phase-role SSPRK3
      // supplies the bounded Bernstein representation for transported damage.
      AssertThrow(has_phase_field_role ||
                  (!transport_phase &&
                   parameters.composition_descriptions[phase_field].type == CompositionalFieldDescription::generic),
                  ExcMessage("Mature phase requires a generic or phase field composition for static DG, "
                             "or a phase field composition for bounded SSPRK3 transport."));
      for (unsigned int c=0; c<introspection.n_compositional_fields; ++c)
        {
          const auto required_method = transport_phase && c == phase_field ? AdvectionMethod::explicit_dg_ssprk3 :
                                       AdvectionMethod::static_field;
          AssertThrow(parameters.composition_degrees[c] == 2 &&
                      parameters.use_discontinuous_composition_discretization[c] &&
                      parameters.compositional_field_methods[c] == required_method,
                      ExcMessage("Fault stress requires static DGQ2. Phase requires static DGQ2 or bounded SSPRK3."));
        }

      interval = initial_interval;
    }

    template <int dim>
    void PhaseFieldFault<dim>::accept_section_rates(const std::vector<double> &rates,
                                                    const double               dt)
    {
      active_section().trace.accept_section_rates(rates, dt);
      for (unsigned int i = 0; i < rates.size(); ++i)
        {
          friction->accept_rate(i, rates[i], dt);
          if (candidate && dt > 0.)
            {
              const auto &position = candidate->trace.labels()[i].position;
              maxwell_memory[i] = stress_relaxation_factor(position) * maxwell_memory[i]
                                  + effective_maxwell_viscosity(position) * rates[i];
            }
        }
      if (candidate)
        accepted = std::move(candidate);
    }

    template <int dim>
    std::string PhaseFieldFault<dim>::label_checkpoint_key() const
    {
      // Preserve the original RSF archive key. All other laws use their registered identifier.
      if (friction->name() == "rate and state")
        return "Material fault labels v1";
      return "Material fault labels " + friction->name();
    }

    template <int dim>
    void PhaseFieldFault<dim>::save(std::map<std::string,std::string> &status) const
    {
      if (strips)
        {
          strips->save(status);
          return;
        }

      std::ostringstream records;
      records << std::setprecision(17) << labels().size() << '\n';
      // Keep the original per-label record order. RSF writes age between shear
      // and slip rate. A state-free law omits that column.
      const auto &resistance_state = friction->state();
      for (unsigned int i = 0; i < labels().size(); ++i)
        {
          const auto &label = labels()[i];
          records << label.position << ' ' << label.half_width << ' ' << label_background_shear[i] << ' ';
          if (!resistance_state.empty())
            records << resistance_state[i] << ' ';
          records << label.rate << ' ' << label.slip << ' ' << maxwell_memory[i] << '\n';
        }
      status[label_checkpoint_key()] = records.str();

      // Keep the prescribed seed separate from the fitted or advected
      // positions in the accepted records. File names carry no physical identity.
      std::ostringstream geometry;
      {
        aspect::oarchive archive(geometry);
        const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
        archive << interior_seed << advect_vertices << fit_trace << accepted->profile.slip_sense
                << box.get_origin() << box.get_extents() << prescribed_vertices << prescribed_half_width;
      }
      status["Mature fault trace v1"] = geometry.str();

      std::ostringstream signature;
      {
        aspect::oarchive archive(signature);
        archive << friction->name() << friction->parameters();
      }
      status["Mature fault resistance v1"] = signature.str();
      if (complete_contacts)
        {
          std::ostringstream bytes;
          {
            aspect::oarchive archive(bytes);
            archive << degradation_magnitude << normal_integral_subdivisions << accepted->profile.normal_integrals();
          }
          status["Mature completed normal integrals v1"] = bytes.str();
        }
      if (normal_filter_length > 0.)
        status["Mature normal filter length"] = Utilities::to_string(normal_filter_length,17);
      if (normal_stress.uses_adiabat())
        status["Mature friction pressure profile"] = normal_stress.checkpoint_profile(*this);
    }

    template <int dim>
    void PhaseFieldFault<dim>::load(const std::map<std::string,std::string> &status)
    {
      if (strips)
        {
          strips->load(status);
          return;
        }

      AssertThrow(!status.count("Native growing fault v1") && !status.count("Native growing fault v2"),
                  ExcMessage("Cannot resume a growing fault checkpoint with mature trace mechanics."));
      const auto geometry = status.find("Mature fault trace v1");
      if (geometry != status.end())
        {
          std::istringstream bytes(geometry->second);
          aspect::iarchive archive(bytes);
          bool interior, advected, fitted;
          int slip_sense;
          Point<dim> origin, extents;
          std::vector<Point<dim>> vertices;
          double half_width;
          archive >> interior >> advected >> fitted >> slip_sense >> origin >> extents >> vertices >> half_width;
          const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
          AssertThrow(slip_sense == accepted->profile.slip_sense,
                      ExcMessage("Mature fault slip sense differs from the checkpoint."));
          AssertThrow(interior == interior_seed && advected == advect_vertices && fitted == fit_trace,
                      ExcMessage("Mature fault geometry or fitting mode differs from the checkpoint."));
          AssertThrow(origin == box.get_origin() && extents == box.get_extents() && vertices == prescribed_vertices,
                      ExcMessage("Mature fault box or ordered prescribed vertices differ from the checkpoint."));
          AssertThrow(half_width == prescribed_half_width,
                      ExcMessage("Checkpoint normal support convention or initial half width differs from the configured trace."));
        }
      else
        {
          // Before trace metadata was introduced only interior faults with
          // the original shear sense could write physical label records.
          AssertThrow(interior_seed && accepted->profile.slip_sense == 1,
                      ExcMessage("A legacy mature fault checkpoint requires interior mechanics and the original slip sense."));
        }

      // Older archives identify the law only through its label-record key.
      // Check the additional physical signature before installing any history.
      const auto signature = status.find("Mature fault resistance v1");
      if (signature != status.end())
        {
          std::istringstream bytes(signature->second);
          aspect::iarchive archive(bytes);
          std::string law;
          std::vector<double> parameters;
          archive >> law >> parameters;
          AssertThrow(law == friction->name(),
                      ExcMessage("Mature fault resistance law differs from the checkpoint."));
          AssertThrow(parameters == friction->parameters(),
                      ExcMessage("Mature fault resistance parameters differ from the checkpoint."));
        }
      const auto completed = status.find("Mature completed normal integrals v1");
      AssertThrow(complete_contacts==(completed!=status.end()),
                  ExcMessage("Contact completion convention differs from the mature checkpoint."));
      if (complete_contacts)
        {
          double scale;
          std::vector<double> saved;
          std::istringstream bytes(completed->second);
          aspect::iarchive archive(bytes);
          unsigned int subdivisions;
          archive >> scale >> subdivisions >> saved;
          AssertThrow(subdivisions==normal_integral_subdivisions,
                      ExcMessage("Completed profile station quadrature differs from the checkpoint."));
          AssertThrow(scale==degradation_magnitude,ExcMessage("Completed profile scale differs from the checkpoint."));
          accepted->profile.restore_completed_integrals(scale,saved);
        }
      const auto filter = status.find("Mature normal filter length");
      AssertThrow((filter == status.end() && normal_filter_length == 0.) ||
                  (filter != status.end() && filter->second == Utilities::to_string(normal_filter_length,17)),
                  ExcMessage("Mature normal traction filter differs from the checkpoint."));
      const auto profile = status.find("Mature friction pressure profile");
      AssertThrow((profile != status.end()) == normal_stress.uses_adiabat() &&
                  (profile == status.end() || profile->second == normal_stress.checkpoint_profile(*this)),
                  ExcMessage("Mature fault friction pressure source or profile differs from the checkpoint."));
      const auto entry = status.find(label_checkpoint_key());
      AssertThrow(entry != status.end(),
                  ExcMessage("Missing mature fault labels in checkpoint."));

      std::istringstream records(entry->second);
      auto restored_labels = labels();
      unsigned int n_labels = 0;
      records >> n_labels;
      AssertThrow(n_labels == restored_labels.size(),
                  ExcMessage("Incompatible mature fault topology in checkpoint."));
      auto resistance_state = friction->state();
      std::vector<double> restored_shear(n_labels), restored_memory(n_labels);
      for (unsigned int i = 0; i < n_labels; ++i)
        {
          auto &label = restored_labels[i];
          records >> label.position >> label.half_width >> restored_shear[i];
          if (!resistance_state.empty())
            records >> resistance_state[i];
          records >> label.rate >> label.slip >> restored_memory[i];
        }
      AssertThrow(records, ExcMessage("Incomplete mature fault checkpoint."));
      if (geometry != status.end())
        AssertThrow(restored_shear == label_background_shear,
                    ExcMessage("Persistent vertex shear tractions differ from the checkpoint."));
      accepted->trace.replace_material_labels(std::move(restored_labels));
      label_background_shear = std::move(restored_shear);
      maxwell_memory = std::move(restored_memory);
      friction->restore_state(std::move(resistance_state));
    }

    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,"phase field fault",
                                   "Plane-strain fault mechanics. The midpoint fault phase method selects irreversible AT1 growth on material strips. "
                                   "Other supported field methods use an existing ordered trace with Q1 material labels. "
                                   "Static or transported DG phase supplies the normalized slip distribution. "
                                   "Mature named outputs fault_tau_xx, fault_tau_yy and fault_tau_xy read accepted stress in Pa; growing fault history exports native tensors.")

    namespace
    {
      // Both names construct the same class. The canonical registration above
      // declares both input subsections and installs their observation actions.
      template <int dim>
      void register_growing_fault_alias()
      {
        using Factory = aspect::internal::Plugins::RegisterHelper<Interface<dim>, PhaseFieldFault<dim>>;
        const auto no_additional_parameters = [](ParameterHandler &) {};
        register_material_model<dim>("growing fault",
                                     "Compatibility spelling for phase field fault with midpoint fault phase: plane-strain "
                                     "Maxwell response, signed strip resistance and irreversible AT1 evolution.",
                                     no_additional_parameters,
                                     &Factory::factory);
      }

      const auto growing_fault_registered = (register_growing_fault_alias<2>(), register_growing_fault_alias<3>(), true);
    }
  }
}
