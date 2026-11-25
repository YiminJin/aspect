/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/phase_field_fault.h>
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
      // The first row names columns so the donor's x,y,core-phase triples
      // cannot be mistaken for a material identity, position or age.
      template <int dim>
      void read_vertices(const std::string &text, std::vector<Point<dim>> &vertices,
                         std::vector<double> &ages,std::vector<double> &shear)
      {
        std::istringstream input(text);
        std::vector<std::string> header;
        std::string line;
        while (std::getline(input,line))
          {
            line=line.substr(0,line.find('#'));
            std::istringstream row(line);
            std::vector<std::string> tokens {std::istream_iterator<std::string>(row),{}};
            if (tokens.empty()) continue;
            if (header.empty())
              {
                header=std::move(tokens);
                AssertThrow(header.size()>=3 && header[0]=="id" && header[1]=="x_m" && header[2]=="y_m",
                            ExcMessage("Expected vertex header id x_m y_m; anonymous donor core-phase triples are ambiguous."));
                std::set<std::string> used;
                for (unsigned int j=3; j<header.size(); ++j)
                  {
                    const bool age=header[j]=="initial_age_s";
                    AssertThrow((age || header[j]=="background_shear_Pa") && used.insert(header[j]).second,
                                ExcMessage("Unknown or duplicate vertex column "+header[j]));
                    AssertThrow((age?ages:shear).empty(),ExcMessage("Duplicate inline and file source for "+header[j]));
                  }
                continue;
              }
            AssertThrow(tokens.size()==header.size() && tokens[0]==std::to_string(vertices.size()),
                        ExcMessage("Vertex rows require consecutive material IDs from zero and one value per named column."));
            Point<dim> point;
            for (unsigned int j=1; j<header.size(); ++j)
              {
                std::istringstream number(tokens[j]);
                double value;
                AssertThrow((number>>value) && number.peek()==std::char_traits<char>::eof() && std::isfinite(value),
                            ExcMessage("Invalid vertex value in "+header[j]));
                if (j<3) point[j-1]=value;
                else if (header[j]=="initial_age_s") ages.push_back(value);
                else shear.push_back(value);
              }
            vertices.push_back(point);
          }
        AssertThrow(vertices.size()>=2,ExcMessage("A material trace requires at least two ordered vertices."));
      }

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
    PhaseFieldFault<dim>::PhaseFieldFault() : coupling(*this), accepted(std::make_unique<Section>()) {}

    template <int dim>
    void PhaseFieldFault<dim>::initialize()
    {
      this->get_pcout() << "   Mature fault Stokes solver: ASPECT block AMG." << std::endl;
      coupling.initialize_simulator(this->get_simulator());
      coupling.initialize();
      coupling.connect_section_balance();
    }

    template <int dim>
    void PhaseFieldFault<dim>::update()
    {
      interval = this->get_timestep_number() == 0 ? initial_interval : this->get_timestep();
      candidate.reset();
      if (!trace_initialized)
        {
          if (fit_trace)
            accepted->trace.fit_boundary_spanning_seed_to_phase_ridge(*this,phase_field,this->get_old_solution());
          if (!interior_seed || this->get_timestep_number()>0)
            accepted->profile.initialize(*this,phase_field,this->get_old_solution(),degradation_magnitude,interior_seed ? 8 : 1);
          trace_initialized = true;
        }
      coupling.prepare_parent();
    }

    template <int dim>
    void PhaseFieldFault<dim>::bind_candidate(const LinearAlgebra::BlockVector &fields)
    {
      if (!interior_seed || candidate) return;
      candidate=std::make_unique<Section>();
      candidate->trace=accepted->trace;
      if (advect_vertices && this->get_timestep()>0.)
        candidate->trace.advect_with_accepted_velocity(*this,this->get_timestep());
      candidate->profile.initialize(*this,phase_field,fields,degradation_magnitude,8);
    }

    template <int dim>
    double PhaseFieldFault<dim>::normal_compression(const Point<dim> &foot, const double pressure,
                                                    const double normal_deviatoric_stress) const
    {
      if (normal_stress.uses_adiabat())
        return this->get_adiabatic_conditions().pressure(foot);
      return background_compression + pressure - normal_deviatoric_stress;
    }

    template <int dim>
    double PhaseFieldFault<dim>::stiffness(const Point<dim> &x) const
    {
      const double eta=bulk_viscosity(x);
      return -eta*std::expm1(-modulus*interval/eta);
    }

    template <int dim>
    double PhaseFieldFault<dim>::relaxation(const Point<dim> &x) const
    {
      return std::exp(-modulus*interval/bulk_viscosity(x));
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress PhaseFieldFault<dim>::viscous_stress(const Stress &strain_rate,const Point<dim> &x) const
    {
      return 2.*stiffness(x)*Utilities::Tensors::consistent_deviator(strain_rate);
    }

    template <int dim>
    SymmetricTensor<2,dim> PhaseFieldFault<dim>::endpoint_stress(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                 const Point<dim> &x, const double phase, const Stress &strain) const
    {
      return viscous_stress(strain,x)+coupling.stress_offset(cell,x,phase);
    }

    template <int dim>
    typename PhaseFieldFault<dim>::Stress
    PhaseFieldFault<dim>::rotate_parent(const Stress &parent, const Tensor<2,dim> &accepted_gradient,
                                        const Tensor<2,dim> &slip_gradient, const double physical_dt) const
    {
      if constexpr (dim == 2)
        {
          // Initialization relaxes the parent over its Maxwell interval but has
          // zero physical duration for slip, age and the accepted spin rotation.
          const auto gradient = accepted_gradient - slip_gradient;
          const double angle = .5*physical_dt*(gradient[1][0]-gradient[0][1]);
          const auto rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
          return symmetrize(rotation*static_cast<Tensor<2,dim>>(parent)*transpose(rotation));
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
          out.viscosities[q] = stiffness(in.position[q]);
          std::fill(out.reaction_terms[q].begin(),out.reaction_terms[q].end(),0.);
          if (physical)
            for (unsigned int c=0; c<stress_fields.size(); ++c)
              physical->output_values[c][q] = in.composition[q][stress_fields[c]];
          if (offset)
            offset->stress[q] = coupling.stress_offset(in.current_cell,in.position[q],in.composition[q][phase_field]);
        }
    }

    template <int dim>
    const std::vector<typename MaterialFaultTrace<dim>::Label> &PhaseFieldFault<dim>::labels() const
    {
      return accepted->trace.labels();
    }

    template <int dim>
    const std::vector<double> &PhaseFieldFault<dim>::label_ages() const
    {
      return friction->ages_in_seconds();
    }

    template <int dim>
    void PhaseFieldFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      Rheology::FrictionNormalStress::declare_parameters(prm);
      prm.enter_subsection("Material model");
      prm.enter_subsection("Phase field fault");
      prm.declare_entry("Advect physical stress","false",Patterns::Bool(),
                        "Transport the accepted DG stress with the accepted velocity before rotation and relaxation. Exterior parent stress specifies xx; yy; xy in Pa, at x,y in metres and t in seconds.");
      prm.enter_subsection("Exterior parent stress");
      Functions::ParsedFunction<dim>::declare_parameters(prm,3);
      prm.leave_subsection();
      prm.declare_entry("Fault geometry","automatic",
                        Patterns::Selection("automatic|boundary spanning|fixed interior|advected interior"),
                        "Boundary spanning uses the stationary section law. Both interior modes use the prescribed "
                        "material seed and the same Q8 projected Maxwell response. Advected interior moves vertices "
                        "with accepted velocity and requires bounded phase and physical stress transport. Fixed "
                        "interior holds the vertices fixed. Automatic preserves earlier inputs by selecting advected "
                        "interior for the phase field composition role and boundary spanning for the generic role.");
      prm.declare_entry("Fault vertex file","",Patterns::FileName(),
                        "Ordered ASCII table: id x_m y_m, optionally initial_age_s and background_shear_Pa. "
                        "IDs start at zero. Row order sets slip sense; shear persists through all timesteps.");
      prm.declare_entry("Fault vertices","",Patterns::List(Patterns::List(Patterns::Double(),2,2),0,
                                                           Patterns::List::max_int_value,";"),"Ordered x,y label positions in metres separated by semicolons. "
                        "The default joins the ends of the left box boundary.");
      prm.declare_entry("Degradation magnitude","1",Patterns::Double(0.),"Scale of mature excess compliance and of its projected normal integral. At fixed phase it cancels in the normalized slip weight.");
      prm.declare_entry("Fit trace to phase","true",Patterns::Bool(),"Fit the ordered seed to the initialized FE ridge. Constant phase retains the seed.");
      prm.declare_entry("Profile half width","0",Patterns::Double(0.),"Normal association support in metres. Zero uses the full physical cross-section.");
      prm.declare_entry("Shear modulus","1e10",Patterns::Double(0.),"Elastic shear modulus in Pa.");
      prm.declare_entry("Initialization interval","1",Patterns::Double(0.),"Maxwell interval in seconds for the initial Stokes solve.");
      prm.declare_entry("Background shear traction","0",Patterns::Double(),"Persistent shear loading in Pa along the slip sense; the per-vertex column overrides this uniform value.");
      prm.declare_entry("Background compression","",Patterns::List(Patterns::Double(0.),0,1),
                        "Compression in Pa added to resolved normal traction (default 5e7 Pa). "
                        "Leave empty for adiabatic pressure, which supplies the complete compression.");
      Rheology::FaultResistance::declare_parameters(prm,Rheology::FaultResistance::Route::mature);
      prm.declare_entry("Minimum speed","1e-20",Patterns::Double(0.),"Minimum admissible sliding speed in m/s.");
      prm.declare_entry("Initial label ages","",Patterns::List(Patterns::Double(0.)),"Optional ordered initial material ages in seconds.");
      prm.declare_entry("Initial label shear tractions","",Patterns::List(Patterns::Double()),"Optional persistent shear tractions in Pa, ordered by vertex.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    template <int dim>
    void PhaseFieldFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      AssertThrow(dim==2,ExcMessage("Reconstructed fault mechanics requires 2-D plane strain."));
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
      normal_stress = Rheology::FrictionNormalStress::parse(prm,Rheology::FrictionNormalStress::Source::resolved_traction);
      const auto &p = this->get_parameters();
      phase_field = this->introspection().compositional_index_for_name("phase_field");
      const bool transport_phase = p.compositional_field_methods[phase_field] == Parameters<dim>::AdvectionFieldMethod::explicit_dg_ssprk3;
      const bool fresh_run = prm.get("Resume computation") == "false" && p.resume_checkpoint_id == 0 && p.resume_time < 0. &&
                             p.checkpoint_steps == 0 && p.checkpoint_time_secs == 0 &&
                             p.additional_checkpoint_times.empty() && !checkpoint_on_termination;
      prm.enter_subsection("Material model");
      AssertThrow(prm.get({"Growing fault"},"Prescribed fault compression").empty() &&
                  prm.get_double({"Growing fault"},"Seed tangent angle")==0. &&
                  Utilities::string_to_double(Utilities::split_string_list(prm.get({"Growing fault"},"Forward seed tip")))==std::vector<double>({0.,0.}),
                  ExcMessage("Growing fault seed and compression settings require the midpoint fault phase method."));
      prm.enter_subsection("Simple model");
      // The Simple model subsection supplies the single-lithology background
      // constants. Only EOS entry zero is consumed, irrespective of DG stress.
      eos.parse_parameters(prm,1);
      conductivity = prm.get_double("Thermal conductivity");
      viscosity = prm.get_double("Viscosity");
      AssertThrow(prm.get_double("Thermal viscosity exponent") == 0 && prm.get_double("Composition viscosity prefactor") == 1.,
                  ExcMessage("The base Maxwell viscosity requires zero thermal exponent and unit compositional prefactor."));
      prm.leave_subsection();
      prm.enter_subsection("Phase field fault");
      const std::string geometry = prm.get("Fault geometry");
      // Before the geometry parameter existed, interior inputs required the
      // phase-field role and boundary-spanning inputs required a generic field.
      const bool phase_role = p.composition_descriptions[phase_field].type == CompositionalFieldDescription::phase_field;
      advect_vertices = geometry == "advected interior" || (geometry == "automatic" && phase_role);
      interior_seed = advect_vertices || geometry == "fixed interior";
      AssertThrow(!advect_vertices || transport_phase,
                  ExcMessage("An advected interior seed requires bounded SSPRK3 transport of phase_field."));
      AssertThrow(!transport_phase || interior_seed,
                  ExcMessage("Transported phase requires an interior fault seed."));
      AssertThrow(interior_seed || fresh_run,
                  ExcMessage("Phase field fault requires a fresh run with checkpointing disabled: label history is in memory."));
      if (prm.get_bool("Advect physical stress"))
        {
          AssertThrow(this->get_geometry_model().get_periodic_boundary_pairs().empty(),
                      ExcMessage("Advected physical stress requires a nonperiodic Cartesian box."));
          prm.enter_subsection("Exterior parent stress");
          inflow_stress = std::make_unique<Functions::ParsedFunction<dim>>(3);
          inflow_stress->parse_parameters(prm);
          prm.leave_subsection();
        }
      modulus = prm.get_double("Shear modulus");
      initial_interval = prm.get_double("Initialization interval");
      background_shear = prm.get_double("Background shear traction");
      AssertThrow(!normal_stress.uses_adiabat() || prm.get("Background compression").empty(),
                  ExcMessage("Adiabatic friction supplies the complete pressure; omit Background compression."));
      background_compression = prm.get("Background compression").empty() ? 5e7 : prm.get_double("Background compression");
      const double initial_speed=prm.get_double("Reference speed");
      friction=Rheology::FaultResistance::create(prm,Rheology::FaultResistance::Route::mature);
      minimum_speed=prm.get_double("Minimum speed");
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      AssertThrow(Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(this->get_initial_topography_model()),
                  ExcMessage("The fault material requires the zero initial topography model "
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
      auto ages=Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial label ages")));
      initial_shear=Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial label shear tractions")));
      const auto filename=Utilities::expand_ASPECT_SOURCE_DIR(prm.get("Fault vertex file"));
      if (!filename.empty())
        {
          AssertThrow(vertices.empty(),ExcMessage("Choose Fault vertex file or inline Fault vertices."));
          read_vertices(Utilities::read_and_distribute_file_content(filename,this->get_mpi_communicator()),vertices,ages,initial_shear);
        }
      friction->set_initial_ages(std::move(ages));
      if (vertices.empty())
        {
          vertices = {box.get_origin(),box.get_origin()};
          vertices.back()[1] += box.get_extents()[1];
        }
      degradation_magnitude = prm.get_double("Degradation magnitude");
      AssertThrow(degradation_magnitude>0.,ExcMessage("Degradation magnitude must be positive."));
      fit_trace = prm.get_bool("Fit trace to phase");
      const double support = prm.get_double("Profile half width");
      AssertThrow(!interior_seed || (!fit_trace && (!advect_vertices || inflow_stress)),
                  ExcMessage("Interior mechanics requires a prescribed seed with fitting disabled. Moving its vertices also requires physical stress transport."));
      accepted->trace.initialize(box.get_origin(),box.get_extents(),vertices,initial_speed,
                                 support > 0. ? support : std::numeric_limits<double>::max(),interior_seed);
      const auto count=labels().size();
      friction->initialize(count);
      if (initial_shear.empty()) initial_shear.assign(count,background_shear);
      AssertThrow(initial_shear.size()==count,ExcMessage("Material initial shear must match the number of fault vertices."));
      maxwell_memory.assign(count,0.);
      prm.leave_subsection();
      prm.leave_subsection();
      AssertThrow(std::min({viscosity,modulus,initial_interval,minimum_speed,initial_speed}) > 0.,
                  ExcMessage("The fault material requires positive material, time and speed scales."));
      AssertThrow(initial_speed >= minimum_speed,
                  ExcMessage("The reference speed must be at least the minimum admissible speed."));
      AssertThrow(dim == 2 && !p.enable_elasticity && p.nonlinear_solver == (transport_phase ? Parameters<dim>::NonlinearSolver::single_Advection_iterated_Stokes : Parameters<dim>::NonlinearSolver::no_Advection_iterated_Stokes)
                  && p.stokes_solver_type == Parameters<dim>::StokesSolverType::block_amg && p.stokes_velocity_degree == 2
                  && !p.run_postprocessors_on_nonlinear_iterations && !p.mesh_deformation_enabled && !p.include_melt_transport && (normal_stress.uses_adiabat() || p.pressure_normalization == "no")
                  && p.nonlinear_solver_failure_strategy != Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep,
                  ExcMessage("The fault material requires a fixed 2-D Q2 box with ASPECT elasticity disabled, block AMG, no pressure normalization for resolved traction, "
                             "a composition solve for selected phase transport, iterated Stokes, and convergence or retry."));
      AssertThrow(!p.enable_prescribed_dilation && (p.formulation_mass_conservation ==
                                                    Parameters<dim>::Formulation::MassConservation::incompressible || p.formulation_mass_conservation ==
                                                    Parameters<dim>::Formulation::MassConservation::ask_material_model),
                  ExcMessage("Fault mechanics requires incompressible continuity without prescribed dilation."));
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
      AssertThrow((interior_seed ? phase_role : p.composition_descriptions[phase_field].type == CompositionalFieldDescription::generic),
                  ExcMessage("Interior mechanics requires the phase field role. Boundary-spanning mechanics requires generic phase."));
      for (unsigned int c=0; c<intro.n_compositional_fields; ++c)
        AssertThrow(p.composition_degrees[c] == 2 && p.use_discontinuous_composition_discretization[c] &&
                    p.compositional_field_methods[c] == (transport_phase && c==phase_field ? Parameters<dim>::AdvectionFieldMethod::explicit_dg_ssprk3 : Parameters<dim>::AdvectionFieldMethod::static_field),
                    ExcMessage("Fault stress requires static DGQ2. Phase requires static DGQ2 or bounded SSPRK3."));

      interval = initial_interval;

    }

    template <int dim>
    void PhaseFieldFault<dim>::accept_section(const std::vector<double> &rates,const double dt)
    {
      section().trace.accept_section_rates(rates,dt);
      for (unsigned int i=0; i<rates.size(); ++i)
        {
          friction->accept_rate(i,rates[i],dt);
          if (candidate && dt>0.)
            {
              const auto &x=candidate->trace.labels()[i].position;
              maxwell_memory[i]=relaxation(x)*maxwell_memory[i]+stiffness(x)*rates[i];
            }
        }
      if (candidate) accepted=std::move(candidate);
    }

    template <int dim>
    std::string PhaseFieldFault<dim>::label_checkpoint_key() const
    {
      // Preserve the original RSF archive key. All other laws use their registered identifier.
      if (friction->name()=="rate and state")
        return "Material fault labels v1";
      return "Material fault labels "+friction->name();
    }

    template <int dim>
    void PhaseFieldFault<dim>::save(std::map<std::string,std::string> &status) const
    {
      if (!interior_seed) return;
      std::ostringstream out;
      out<<std::setprecision(17)<<labels().size()<<'\n';
      const auto &state=friction->state();
      for (unsigned int i=0; i<labels().size(); ++i)
        {
          const auto &p=labels()[i];
          out<<p.position<<' '<<p.half_width<<' '<<initial_shear[i]<<' ';
          if (!state.empty()) out<<state[i]<<' ';
          out<<p.rate<<' '<<p.slip<<' '<<maxwell_memory[i]<<'\n';
        }
      // Retain the repaired RSF wire format, including its per-label age.
      status[label_checkpoint_key()]=out.str();
      if (normal_stress.uses_adiabat())
        status["Mature friction pressure profile"] = normal_stress.checkpoint_profile(*this);
    }

    template <int dim>
    void PhaseFieldFault<dim>::load(const std::map<std::string,std::string> &status)
    {
      AssertThrow(!status.count("Native growing fault v1") && !status.count("Native growing fault v2"),
                  ExcMessage("Cannot resume a growing fault checkpoint with mature trace mechanics."));
      if (!interior_seed) return;
      const auto profile = status.find("Mature friction pressure profile");
      AssertThrow((profile!=status.end()) == normal_stress.uses_adiabat() &&
                  (profile==status.end() || profile->second==normal_stress.checkpoint_profile(*this)),
                  ExcMessage("Mature fault friction pressure source or profile differs from the checkpoint."));
      const auto entry=status.find(label_checkpoint_key());
      AssertThrow(entry!=status.end(),ExcMessage("Missing interior fault labels in checkpoint."));
      std::istringstream in(entry->second);
      auto restored=labels();
      unsigned int n=0;
      in>>n;
      AssertThrow(n==restored.size(),ExcMessage("Incompatible interior fault topology in checkpoint."));
      auto state=friction->state();
      std::vector<double> shear(n), memory(n);
      for (unsigned int i=0; i<n; ++i)
        {
          auto &p=restored[i];
          in>>p.position>>p.half_width>>shear[i];
          if (!state.empty()) in>>state[i];
          in>>p.rate>>p.slip>>memory[i];
        }
      AssertThrow(in,ExcMessage("Incomplete interior fault checkpoint."));
      accepted->trace.replace_material_labels(std::move(restored));
      initial_shear=std::move(shear);
      maxwell_memory=std::move(memory);
      friction->restore_state(std::move(state));
      trace_initialized=false;
    }

    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,"phase field fault",
                                   "Plane-strain fault mechanics. The midpoint fault phase method selects irreversible AT1 growth on material strips. "
                                   "Other supported field methods use an existing ordered trace with Q1 material labels. "
                                   "Static or transported DG phase supplies the normalized slip distribution. "
                                   "Mature named outputs fault_tau_xx, fault_tau_yy and fault_tau_xy read accepted stress in Pa; growing fault history exports native tensors.")
  }
}
