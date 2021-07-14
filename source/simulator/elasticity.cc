#include <aspect/elasticity.h>
#include <aspect/simulator.h>
#include <aspect/mesh_deformation/interface.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/vector_tools.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    ElasticInputs<dim>::ElasticInputs (const unsigned int n_points)
    :
      stress_values (n_points, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
      old_stress_values (n_points, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
      convection_terms (n_points, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
      velocity_gradients (n_points, numbers::signaling_nan<Tensor<2,dim> >())
    {}


    template <int dim>
    void
    ElasticInputs<dim>::fill (const LinearAlgebra::BlockVector &solution,
                              const LinearAlgebra::BlockVector &old_solution,
                              const LinearAlgebra::BlockVector &/*old_old_solution*/,
                              const FEValuesBase<dim>          &fe_values,
                              const Introspection<dim>         &introspection)
    {
      const unsigned int comp0_idx = introspection.variable("deviatoric stress").first_component_index;

      FEValuesExtractors::SymmetricTensor<2> stress_extractor(comp0_idx);
      fe_values[stress_extractor].get_function_values (solution, stress_values);
      fe_values[stress_extractor].get_function_values (old_solution, old_stress_values);

      fe_values[introspection.extractors.velocities].get_function_gradients (solution, velocity_gradients);

      std::vector<Tensor<1,dim> > component_gradients (stress_values.size());
      std::vector<Tensor<1,dim> > velocity_values (stress_values.size());
      fe_values[introspection.extractors.velocities].get_function_values (solution, velocity_values);

      for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      {
        FEValuesExtractors::Scalar component_extractor(comp0_idx+c);
        fe_values[component_extractor].get_function_gradients (solution, component_gradients);

        const TableIndices<2> indices = SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
        for (unsigned int q = 0; q < velocity_values.size(); ++q)
          convection_terms[q][indices] = velocity_values[q] * component_gradients[q];
      }
    }


    template <int dim>
    ElasticOutputs<dim>::ElasticOutputs (const unsigned int n_points)
      : elastic_shear_moduli (n_points, numbers::signaling_nan<double>())
    {}


    template <int dim>
    void ElasticOutputs<dim>::average (const MaterialAveraging::AveragingOperation operation,
                                       const FullMatrix<double> &projection_matrix,
                                       const FullMatrix<double> &expansion_matrix)
    {
      average_property (operation, projection_matrix, expansion_matrix,
                        elastic_shear_moduli);
    }
  }


  namespace Assemblers
  {
    template <int dim>
    void
    ElasticRHSTerm<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                                  internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      if (this->get_timestep_number() == 0)
        return;

      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim> &> (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim> &> (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int n_q_points = scratch.finite_element_values.n_quadrature_points;
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      const MaterialModel::ElasticInputs<dim> *elastic_inputs = 
        scratch.material_model_inputs.template get_additional_input<MaterialModel::ElasticInputs<dim> >();
      AssertThrow (elastic_inputs != nullptr, ExcInternalError());

      const MaterialModel::ElasticOutputs<dim> *elastic_outputs =
        scratch.material_model_outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >();
      AssertThrow (elastic_outputs != nullptr, ExcInternalError());

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Tensor<2,dim> vorticity = 0.5 * (elastic_inputs->velocity_gradients[q] -
                                         transpose(elastic_inputs->velocity_gradients[q]));
        SymmetricTensor<2,dim> rotation_term = symmetrize(-vorticity * elastic_inputs->stress_values[q] +
                                                          elastic_inputs->stress_values[q] * vorticity );

        const double viscosity = scratch.material_model_outputs.viscosities[q];
        const double elastic_shear_modulus = elastic_outputs->elastic_shear_moduli[q];
        const SymmetricTensor<2,dim> elastic_stress = (-elastic_inputs->old_stress_values[q] / this->get_timestep()
                                                        + elastic_inputs->convection_terms[q] + rotation_term )
                                                      * 
                                                      (viscosity / elastic_shear_modulus);

        for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
          {
            data.local_rhs(i_stokes) += ( elastic_stress *
                                          scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i, q) )
                                        * scratch.finite_element_values.JxW(q);
            ++i_stokes;
          }
          ++i;
        }
      }
    }


    template <int dim>
    void
    ElasticRHSTerm<dim>::
    create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      if (outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >() != nullptr)
        return;
      
      const unsigned int n_points = outputs.viscosities.size();
      outputs.additional_outputs.push_back(
        std::make_unique<MaterialModel::ElasticOutputs<dim> >(n_points));
    }


    template <int dim>
    void
    ElasticRHSBoundaryTerm<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                                          internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      if (this->get_timestep_number() == 0)
        return;

      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int n_q_points = scratch.face_finite_element_values.n_quadrature_points;
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      const MaterialModel::ElasticInputs<dim> *elastic_inputs = 
        scratch.material_model_inputs.template get_additional_input<MaterialModel::ElasticInputs<dim> >();
      AssertThrow (elastic_inputs != nullptr, ExcInternalError());

      const MaterialModel::ElasticOutputs<dim> *elastic_outputs =
        scratch.material_model_outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >();
      AssertThrow (elastic_outputs != nullptr, ExcInternalError());

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Tensor<2,dim> vorticity = 0.5 * (elastic_inputs->velocity_gradients[q] -
                                         transpose(elastic_inputs->velocity_gradients[q]));

        SymmetricTensor<2,dim> rotation_term = symmetrize(-vorticity * elastic_inputs->stress_values[q] +
                                                          elastic_inputs->stress_values[q] * vorticity );

        const double viscosity = scratch.material_model_outputs.viscosities[q];
        const double elastic_shear_modulus = elastic_outputs->elastic_shear_moduli[q];
        const SymmetricTensor<2,dim> elastic_stress = (-elastic_inputs->old_stress_values[q] / this->get_timestep()
                                                        + elastic_inputs->convection_terms[q] + rotation_term )
                                                      *
                                                      (viscosity / elastic_shear_modulus);

        for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
          {
            data.local_rhs(i_stokes) -= ( (elastic_stress * scratch.face_finite_element_values.normal_vector(q))
                                          * scratch.face_finite_element_values[introspection.extractors.velocities].value(i, q) )
                                        * scratch.face_finite_element_values.JxW(q);
            ++i_stokes;
          }
          ++i;
        }
      }
    }


    template <int dim>
    void
    ElasticRHSBoundaryTerm<dim>::
    create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      if (outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >() != nullptr)
        return;

      const unsigned int n_points = outputs.viscosities.size();
      outputs.additional_outputs.push_back(
        std::make_unique<MaterialModel::ElasticOutputs<dim> >(n_points));
    }
  }


  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        template <int dim>
        DeviatoricStressSystem<dim>::
        DeviatoricStressSystem (const FiniteElement<dim> &fe,
                                const Mapping<dim>       &mapping,
                                const Quadrature<dim>    &quadrature,
                                const Quadrature<dim-1>  &face_quadrature,
                                const UpdateFlags         update_flags,
                                const UpdateFlags         face_update_flags,
                                const unsigned int        stress_dofs_per_cell,
                                const unsigned int        n_compositional_fields)
        :
          ScratchBase<dim>(),

          fe_values (mapping, fe, quadrature, update_flags),
          face_fe_values (face_quadrature.size() > 0
                          ?
                          std_cxx14::make_unique<FEFaceValues<dim> >(mapping, fe, 
                                                                     face_quadrature, 
                                                                     face_update_flags)
                          :
                          nullptr),
          neighbor_face_fe_values (face_quadrature.size() > 0
                                   ?
                                   std_cxx14::make_unique<FEFaceValues<dim> >(mapping, fe,
                                                                              face_quadrature, 
                                                                              face_update_flags)
                                   :
                                   nullptr),
          subface_fe_values (face_quadrature.size() > 0
                             ?
                             std_cxx14::make_unique<FESubfaceValues<dim> >(mapping, fe, 
                                                                           face_quadrature, 
                                                                           face_update_flags)
                             :
                             nullptr),

          local_dof_indices (fe.dofs_per_cell, numbers::invalid_dof_index),
          neighbor_dof_indices ((face_quadrature.size() > 0 ? fe.dofs_per_cell : 0), 
                                numbers::invalid_dof_index),

          phi (stress_dofs_per_cell, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
          u_dot_grad_phi (stress_dofs_per_cell, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
          face_phi (stress_dofs_per_cell, numbers::signaling_nan<SymmetricTensor<2,dim> >()),
          neighbor_face_phi (stress_dofs_per_cell, numbers::signaling_nan<SymmetricTensor<2,dim> >()),

          old_stress_values (quadrature.size(), numbers::signaling_nan<SymmetricTensor<2,dim> >()),
          old_old_stress_values (quadrature.size(), numbers::signaling_nan<SymmetricTensor<2,dim> >()),
          velocity_values (quadrature.size(), numbers::signaling_nan<Tensor<1,dim> >()),
          velocity_gradients (quadrature.size(), numbers::signaling_nan<Tensor<2,dim> >()),
          mesh_velocity_values (quadrature.size(), numbers::signaling_nan<Tensor<1,dim> >()),
          face_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1,dim> >()),
          face_mesh_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1,dim> >()),

          material_model_inputs (quadrature.size(), n_compositional_fields),
          material_model_outputs (quadrature.size(), n_compositional_fields)
        {}


        template <int dim>
        DeviatoricStressSystem<dim>::
        DeviatoricStressSystem (const DeviatoricStressSystem &scratch)
        :
          ScratchBase<dim>(scratch),

          fe_values (scratch.fe_values.get_mapping(),
                     scratch.fe_values.get_fe(),
                     scratch.fe_values.get_quadrature(),
                     scratch.fe_values.get_update_flags()),
          face_fe_values (scratch.face_fe_values.get()
                          ?
                          std_cxx14::make_unique<FEFaceValues<dim> >(scratch.face_fe_values->get_mapping(),
                                                                     scratch.face_fe_values->get_fe(),
                                                                     scratch.face_fe_values->get_quadrature(),
                                                                     scratch.face_fe_values->get_update_flags())
                          :
                          nullptr),
          neighbor_face_fe_values (scratch.neighbor_face_fe_values.get()
                                   ?
                                   std_cxx14::make_unique<FEFaceValues<dim> >(scratch.neighbor_face_fe_values->get_mapping(),
                                                                              scratch.neighbor_face_fe_values->get_fe(),
                                                                              scratch.neighbor_face_fe_values->get_quadrature(),
                                                                              scratch.neighbor_face_fe_values->get_update_flags())
                                   :
                                   nullptr),
          subface_fe_values (scratch.subface_fe_values.get()
                             ?
                             std_cxx14::make_unique<FESubfaceValues<dim> >(scratch.subface_fe_values->get_mapping(),
                                                                           scratch.subface_fe_values->get_fe(),
                                                                           scratch.subface_fe_values->get_quadrature(),
                                                                           scratch.subface_fe_values->get_update_flags())
                             :
                             nullptr),

          local_dof_indices (scratch.local_dof_indices),
          neighbor_dof_indices (scratch.neighbor_dof_indices),

          phi (scratch.phi),
          u_dot_grad_phi (scratch.u_dot_grad_phi),
          face_phi (scratch.face_phi),
          neighbor_face_phi (scratch.neighbor_face_phi),

          old_stress_values (scratch.old_stress_values),
          old_old_stress_values (scratch.old_old_stress_values),
          velocity_values (scratch.velocity_values),
          velocity_gradients (scratch.velocity_gradients),
          mesh_velocity_values (scratch.mesh_velocity_values),
          face_velocity_values (scratch.face_velocity_values),
          face_mesh_velocity_values (scratch.face_mesh_velocity_values),

          material_model_inputs (scratch.material_model_inputs),
          material_model_outputs (scratch.material_model_outputs)
        {}
      }


      namespace CopyData
      {
        template <int dim>
        DeviatoricStressSystem<dim>::
        DeviatoricStressSystem (const unsigned int stress_dofs_per_cell,
                                const bool         discontinuous)
        :
          local_matrix (stress_dofs_per_cell,
                        stress_dofs_per_cell),
          local_matrices_int_ext ((discontinuous
                                   ?
                                   GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell
                                   :
                                   0),
                                  FullMatrix<double>(stress_dofs_per_cell, stress_dofs_per_cell)),
          local_matrices_ext_int ((discontinuous
                                   ?
                                   GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell
                                   :
                                   0),
                                  FullMatrix<double>(stress_dofs_per_cell, stress_dofs_per_cell)),
          local_matrices_ext_ext ((discontinuous
                                   ?
                                   GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell
                                   :
                                   0),
                                  FullMatrix<double>(stress_dofs_per_cell, stress_dofs_per_cell)),

          local_rhs (stress_dofs_per_cell),
          
          assembled_matrices ((discontinuous
                               ?
                               GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell
                               :
                               0), false),

          local_dof_indices (stress_dofs_per_cell),
          neighbor_dof_indices ((discontinuous
                                 ?
                                 GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell
                                 :
                                 0),
                                std::vector<types::global_dof_index>(stress_dofs_per_cell))
        {}


        template <int dim>
        DeviatoricStressSystem<dim>::
        DeviatoricStressSystem (const DeviatoricStressSystem &data)
        :
          local_matrix (data.local_matrix),
          local_matrices_int_ext (data.local_matrices_int_ext),
          local_matrices_ext_int (data.local_matrices_ext_int),
          local_matrices_ext_ext (data.local_matrices_ext_ext),
          local_rhs (data.local_rhs),

          assembled_matrices (data.assembled_matrices),

          local_dof_indices (data.local_dof_indices),
          neighbor_dof_indices (data.neighbor_dof_indices)
        {}
      }
    }
  }


  template <int dim>
  ElasticityHandler<dim>::Parameters::Parameters ()
    : boundary_stress_function (SymmetricTensor<2,dim>::n_independent_components)
  {}


  template <int dim>
  void
  ElasticityHandler<dim>::Parameters::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection("Elasticity");
    {
      prm.declare_entry("Elastic shear moduli", "5e10",
                        Patterns::List(Patterns::Double(0)),
                        "List of elastic shear moduli for each compositional field. "
                        "Units: $Pa$.");

      prm.declare_entry("Initial time step", "1000",
                        Patterns::Double(0),
                        "Viscoelastic models need a time step to discretize the time "
                        "derivative of stress in the constitutive equation. This parameter "
                        "determines the time step at the initial state, when function "
                        "compute_timestep() has not been called.");

      prm.enter_subsection("Discretization");
      {
        prm.declare_entry("Stress polynomial degree", "1",
                          Patterns::Integer(1),
                          "The polynomial degree to use for the stress components.");
        prm.declare_entry("Use discontinuous stress discretization", "false",
                          Patterns::Bool(),
                          "Whether to use a stress discretization that is discontinuous "
                          "as opposed to continuous. If set to true, then the "
                          "upwind discontinuous Galerkin method will be adopted to "
                          "stabilize the constitutive equation; otherwise, the streamline "
                          "upwind Petrov-Galerkin (SUPG) method will be adopted.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Solver parameters");
      {
        prm.declare_entry("Linear solver tolerance", "1e-12",
                          Patterns::Double(0,1),
                          "The relative tolerance up to which the linear system for "
                          "the stress system gets solved.");

        prm.declare_entry("GMRES solver restart length", "200",
                          Patterns::Integer(1),
                          "The number of iterations that define the GMRES solver "
                          "restart length. Increasing this parameter makes the "
                          "solver more robust and decreases the number of "
                          "iterations. Be aware that increasing this number "
                          "increases the memory usage of the linear solver, and "
                          "makes individual iterations more expensive.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Stress boundary conditions");
      {
        prm.declare_entry("Prescribed stress boundary indicators", "",
                          Patterns::Anything(),
                          "");

        Functions::ParsedFunction<dim>::declare_parameters (
          prm, SymmetricTensor<2,dim>::n_independent_components);
      }
      prm.leave_subsection();

      prm.enter_subsection("Stabilization");
      {
        prm.declare_entry("Use WENO limiter", "false",
                          Patterns::Bool(),
                          "If set to true then WENO limiters will be applied to suppress "
                          "numerical oscillations.");

        prm.declare_entry("WENO epsilon", "1e-6",
                          Patterns::Double(0),
                          "When calculating the nonlinear weights of WENO reconstructions, "
                          "the denominator is the square sum of the smoothness indicators, "
                          "which may be close to zero. Therefore, a small number is added to "
                          "the denominator to avoid division by zero. This parameter determines "
                          "the value of the small number.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }


  template <int dim>
  void
  ElasticityHandler<dim>::Parameters::parse_parameters (ParameterHandler &prm,
                                                        const bool convert_to_years)
  {
    unsigned int n_compositional_fields = numbers::invalid_unsigned_int;
    prm.enter_subsection("Compositional fields");
    {
      n_compositional_fields = prm.get_integer("Number of fields");
    }
    prm.leave_subsection();

    prm.enter_subsection("Elasticity");
    {
      elastic_shear_moduli = Utilities::possibly_extend_from_1_to_N (
        Utilities::string_to_double(Utilities::split_string_list(prm.get("Elastic shear moduli"))),
        n_compositional_fields + 1,
        "Elastic shear moduli");

      initial_time_step = prm.get_double("Initial time step");
      if (convert_to_years)
        initial_time_step *= year_in_seconds;

      prm.enter_subsection("Discretization");
      {
        stress_degree = prm.get_integer("Stress polynomial degree");
        use_discontinuous_stress_discretization = 
          prm.get_bool("Use discontinuous stress discretization");
      }
      prm.leave_subsection();

      prm.enter_subsection("Solver parameters");
      {
        linear_solver_tolerance = prm.get_double("Linear solver tolerance");
        gmres_restart_length = prm.get_integer("GMRES solver restart length");
      }
      prm.leave_subsection();

      prm.enter_subsection("Stabilization");
      {
        use_weno_limiter = prm.get_bool("Use WENO limiter");
        weno_epsilon     = prm.get_double("WENO epsilon");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }


  template <int dim>
  ElasticityHandler<dim>::ElasticityHandler (Simulator<dim> &simulator,
                                             ParameterHandler &prm)
  :
    sim (simulator)
  {
    parameters.parse_parameters (prm, sim.parameters.convert_to_years);

    sim.signals.edit_finite_element_variables.connect(
      std::bind(&ElasticityHandler<dim>::edit_finite_element_variables,
                std::ref(*this),
                std::placeholders::_1));

    sim.signals.set_assemblers.connect(
      std::bind(&ElasticityHandler<dim>::set_assemblers,
                std::ref(*this),
                std::placeholders::_1,
                std::placeholders::_2));
  }


  namespace
  {
    template <int dim>
    std::shared_ptr<FiniteElement<dim> >
    new_FE_Q_or_DGQ (const bool discontinuous,
                     const unsigned int degree)
    {
      if (discontinuous)
        return std::make_shared<FE_DGQ<dim> >(degree);
      else
        return std::make_shared<FE_Q<dim> >(degree);
    }
  }
  

  template <int dim>
  void
  ElasticityHandler<dim>::
  edit_finite_element_variables (std::vector<VariableDeclaration<dim> > &variables)
  {
    variables.push_back(VariableDeclaration<dim>(
                        "deviatoric stress",
                        new_FE_Q_or_DGQ<dim>(parameters.use_discontinuous_stress_discretization,
                                             parameters.stress_degree),
                        SymmetricTensor<2,dim>::n_independent_components,
                        1));
  }


  template <int dim>
  void
  ElasticityHandler<dim>::
  set_assemblers (const SimulatorAccess<dim> &,
                  Assemblers::Manager<dim> &assemblers) const
  {
    assemblers.stokes_system.push_back(std_cxx14::make_unique<Assemblers::ElasticRHSTerm<dim> >());
    assemblers.stokes_system_on_boundary_face.push_back(std_cxx14::make_unique<Assemblers::ElasticRHSBoundaryTerm<dim> >());

    assemblers.stokes_system_assembler_on_boundary_face_properties.need_face_finite_element_evaluation = true;
    assemblers.stokes_system_assembler_on_boundary_face_properties.needed_update_flags |= update_values | 
                                                                                          update_normal_vectors | 
                                                                                          update_JxW_values;
    if (parameters.use_discontinuous_stress_discretization)
      assemblers.stokes_system_assembler_on_boundary_face_properties.needed_update_flags |= update_gradients;
  }


  namespace 
  {
    template <int dim>
    class BoundaryStressFunction : public Function<dim>
    {
      public:
        BoundaryStressFunction (const unsigned int n_components,
                                const unsigned int begin_component,
                                const Function<dim> &function_object);

        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;

      private:
        const unsigned int begin_component;
        const unsigned int end_component;

        const Function<dim> *function_object;
    };


    template <int dim>
    BoundaryStressFunction<dim>::
    BoundaryStressFunction (const unsigned int n_components,
                            const unsigned int begin_component_,
                            const Function<dim> &function_object_)
      : Function<dim>(n_components)
      , begin_component (begin_component_)
      , end_component (begin_component_ + SymmetricTensor<2,dim>::n_independent_components)
      , function_object(&function_object_)
    {}


    template <int dim>
    double
    BoundaryStressFunction<dim>::value (const Point<dim> &p,
                                        const unsigned int component) const
    {
      Assert (component < this->n_components,
              ExcIndexRange (component, 0, this->n_components));

      if (component >= begin_component && component < end_component)
        return function_object->value(p, component - begin_component);
      else
        return 0;
    }


    template <int dim>
    void
    BoundaryStressFunction<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
    {
      AssertDimension(values.size(), this->n_components);

      values = 0;
      for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
        values[begin_component+c] = function_object->value(p, c);
    }
  }


  template <int dim>
  void
  ElasticityHandler<dim>::
  add_current_constraints (AffineConstraints<double> &constraints)
  {
    if (parameters.use_discontinuous_stress_discretization)
      return;

    const unsigned int comp0_idx = sim.introspection.variable("deviatoric stress").first_component_index;

    BoundaryStressFunction<dim> func (
      sim.introspection.n_components, comp0_idx, parameters.boundary_stress_function);

    for (auto p = parameters.prescribed_stress_boundary_indicators.begin();
         p != parameters.prescribed_stress_boundary_indicators.end(); ++p)
    {
      std::vector<bool> mask(sim.introspection.n_components, false);
      for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
        mask[comp0_idx + c] = true;

      VectorTools::interpolate_boundary_values(*sim.mapping,
                                               sim.dof_handler,
                                               *p,
                                               func,
                                               constraints,
                                               mask);
    }
  }


  template <int dim>
  void
  ElasticityHandler<dim>::initialize (ParameterHandler &prm)
  {
    prm.enter_subsection("Elasticity");
    {
      prm.enter_subsection("Stress boundary conditions");
      {
        const std::vector<std::string> x_boundary_stress_indicators
          = Utilities::split_string_list(prm.get("Prescribed stress boundary indicators"));

        parameters.prescribed_stress_boundary_indicators.clear();
        for (auto p = x_boundary_stress_indicators.begin(); p != x_boundary_stress_indicators.end(); ++p)
          parameters.prescribed_stress_boundary_indicators.insert(
            sim.geometry_model->translate_symbolic_boundary_name_to_id(*p));

        parameters.boundary_stress_function.parse_parameters(prm);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }


  template <int dim>
  void
  ElasticityHandler<dim>::
  local_assemble_deviatoric_stress_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           internal::Assembly::Scratch::DeviatoricStressSystem<dim> &scratch,
                                           internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
  {
    data.local_matrix = 0;
    data.local_rhs = 0;

    scratch.fe_values.reinit(cell);

    const unsigned int comp0_idx = sim.introspection.variable("deviatoric stress").first_component_index;

    const FEValuesExtractors::SymmetricTensor<2> stress_extractor(comp0_idx);
    const FEValuesExtractors::Vector mesh_velocity_extractor(0);
    std::vector<FEValuesExtractors::Scalar> component_extractors;
    for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      component_extractors.push_back(FEValuesExtractors::Scalar(comp0_idx+c));

    const unsigned int n_q_points = scratch.fe_values.n_quadrature_points;
    const unsigned int stress_dofs_per_cell = data.local_dof_indices.size();

    scratch.fe_values[stress_extractor].get_function_values (sim.old_solution,
                                                             scratch.old_stress_values);
    scratch.fe_values[stress_extractor].get_function_values (sim.old_old_solution,
                                                             scratch.old_old_stress_values);

    scratch.fe_values[sim.introspection.extractors.velocities].get_function_values (sim.current_linearization_point,
                                                                                    scratch.velocity_values);
    scratch.fe_values[sim.introspection.extractors.velocities].get_function_gradients (sim.current_linearization_point,
                                                                                       scratch.velocity_gradients);

    // Get the mesh velocity, as we need to subtract it off the convection velocity.
    if (sim.parameters.mesh_deformation_enabled)
      scratch.fe_values[mesh_velocity_extractor].get_function_values (sim.mesh_deformation->get_mesh_velocity(),
                                                                      scratch.mesh_velocity_values);

    // Compute material properties.
    sim.compute_material_model_input_values (sim.current_linearization_point,
                                             scratch.fe_values,
                                             cell,
                                             true,
                                             scratch.material_model_inputs);

    sim.material_model->create_additional_inputs (scratch.material_model_inputs);
    sim.material_model->fill_additional_material_model_inputs (scratch.material_model_inputs,
                                                               sim.current_linearization_point,
                                                               sim.old_solution,
                                                               sim.old_old_solution,
                                                               scratch.fe_values,
                                                               sim.introspection);

    if (scratch.material_model_outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >() == nullptr)
      scratch.material_model_outputs.additional_outputs.push_back (std::make_unique<MaterialModel::ElasticOutputs<dim> >(n_q_points));

    sim.material_model->evaluate (scratch.material_model_inputs,
                                  scratch.material_model_outputs);
    MaterialModel::MaterialAveraging::average (sim.parameters.material_averaging,
                                               cell,
                                               scratch.fe_values.get_quadrature(),
                                               scratch.fe_values.get_mapping(),
                                               scratch.material_model_outputs);

    cell->get_dof_indices (scratch.local_dof_indices);
    // Extract local dof indices corresponding to deviatoric stress.
    for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
    {
      const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
      if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
      {
        data.local_dof_indices[i_stress] = scratch.local_dof_indices[i];
        ++i_stress;
      }
      ++i;
    }

    const MaterialModel::ElasticOutputs<dim> *
    elastic_outputs = scratch.material_model_outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >();

    // calculate cell length h along the speed direction for SUPG scheme
    double h = 0;
    if (!parameters.use_discontinuous_stress_discretization)
    {
      Tensor<1,dim> l[dim];
      for (unsigned int d = 0; d < dim; ++d)
        l[d] = cell->face(d*2+1)->center() - cell->face(d*2)->center();

      double volume = 0;
      Tensor<1,dim> u_avg;
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        volume += scratch.fe_values.JxW(q);
        u_avg += scratch.velocity_values[q] * scratch.fe_values.JxW(q);
      }
      u_avg *= 1. / volume;
      const double u_norm = u_avg.norm();
      const Tensor<1,dim> e_u = u_avg / u_norm;

      for (unsigned int d = 0; d < dim; ++d)
        h += std::abs(l[d] * e_u);
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      Tensor<1,dim> current_u = scratch.velocity_values[q];
      // Subtract off the mesh velocity for ALE corrections if necessary.
      if (sim.parameters.mesh_deformation_enabled)
        current_u -= scratch.mesh_velocity_values[q];

      // Precompute the values of shape functions and their gradients.
      for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
      {
        const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
        if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
        {
          const unsigned int c = comp - comp0_idx;
          scratch.phi[i_stress] = scratch.fe_values[stress_extractor].value(i, q);

          scratch.u_dot_grad_phi[i_stress] = 0;
          Tensor<1,dim> grad_phi = scratch.fe_values[component_extractors[c]].gradient(i, q);
          scratch.u_dot_grad_phi[i_stress][SymmetricTensor<2,dim>::unrolled_to_component_indices(c)] = current_u * grad_phi;
          ++i_stress;
        }
        ++i;
      }

      const Tensor<2,dim> vorticity = 0.5 * (scratch.velocity_gradients[q] - transpose(scratch.velocity_gradients[q]));

      const double mu = elastic_outputs->elastic_shear_moduli[q];
      const double mu_over_eta_eff = mu / scratch.material_model_outputs.viscosities[q];

      // Do the actual assembly.
      for (unsigned int i = 0; i < stress_dofs_per_cell; ++i)
      {
        data.local_rhs(i) += ( ( scratch.old_stress_values[q] 
                                 + (2.0 * mu * sim.time_step) *
                                   scratch.material_model_inputs.strain_rate[q]
                               ) * (scratch.phi[i] + (h / scratch.velocity_values[q].norm()) * scratch.u_dot_grad_phi[i]) 
                             ) * scratch.fe_values.JxW(q);

        for (unsigned int j = 0; j < stress_dofs_per_cell; ++j)
        {
          data.local_matrix(i,j) += ( ( scratch.phi[j] * mu_over_eta_eff
                                        + scratch.u_dot_grad_phi[j]
                                        - symmetrize(vorticity * scratch.phi[j] - scratch.phi[j] * vorticity)
                                      ) * (scratch.phi[i] + (h / scratch.velocity_values[q].norm()) * scratch.u_dot_grad_phi[i])
                                    ) * sim.time_step * scratch.fe_values.JxW(q);
        }
      }
    }

    // Do assembly on cell faces.
    if (parameters.use_discontinuous_stress_discretization)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell; ++f)
      {
        data.local_matrices_ext_int[f] = 0;
        data.local_matrices_int_ext[f] = 0;
        data.local_matrices_ext_ext[f] = 0;
        data.assembled_matrices[f] = false;
      }

      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        if (!cell->at_boundary(face_no) || cell->has_periodic_neighbor(face_no))
          local_assemble_deviatoric_stress_system_on_interior_faces (cell, face_no, scratch, data);
        else
          local_assemble_deviatoric_stress_system_on_boundary_faces (cell, face_no, scratch, data);
      }
    }
  }


  template <int dim>
  void
  ElasticityHandler<dim>::
  local_assemble_deviatoric_stress_system_on_boundary_faces 
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   const unsigned int face_no,
   internal::Assembly::Scratch::DeviatoricStressSystem<dim> &scratch,
   internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
  {
    const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
    if (parameters.prescribed_stress_boundary_indicators.find(face->boundary_id())
        == parameters.prescribed_stress_boundary_indicators.end())
      return;

    scratch.face_fe_values->reinit (cell, face_no);

    const unsigned int comp0_idx = sim.introspection.variable("deviatoric stress").first_component_index;

    const FEValuesExtractors::SymmetricTensor<2> stress_extractor(comp0_idx);
    const FEValuesExtractors::Vector mesh_velocity_extractor(0);

    (*scratch.face_fe_values)[sim.introspection.extractors.velocities].get_function_values(sim.current_linearization_point,
                                                                                           scratch.face_velocity_values);
    // Get the mesh velocity, as we need to subtract it off the convection velocity.
    if (sim.parameters.mesh_deformation_enabled)
      (*scratch.face_fe_values)[mesh_velocity_extractor].get_function_values(sim.mesh_deformation->get_mesh_velocity(),
                                                                             scratch.face_mesh_velocity_values);

    const unsigned int stress_dofs_per_cell = data.local_dof_indices.size();
    const unsigned int n_q_points = scratch.face_fe_values->n_quadrature_points;

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Precompute the values of shape functions.
      for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
      {
        const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
        if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
        {
          scratch.face_phi[i_stress] = (*scratch.face_fe_values)[stress_extractor].value(i, q);
          ++i_stress;
        }
        ++i;
      }

      SymmetricTensor<2,dim> dirichlet_value;
      for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      {
        TableIndices<2> indices = SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
        dirichlet_value[indices] = parameters.boundary_stress_function.value (
          scratch.face_fe_values->quadrature_point(q), c);
      }

      Tensor<1,dim> current_u = scratch.face_velocity_values[q];
      // Subtract off the mesh velocity for ALE corrections if necessary.
      if (sim.parameters.mesh_deformation_enabled)
        current_u -= scratch.face_mesh_velocity_values[q];

      const double normal_velocity = current_u * scratch.face_fe_values->normal_vector(q);
      const bool inflow = (current_u * scratch.face_fe_values->normal_vector(q) < 0.);

      for (unsigned int i = 0; i < stress_dofs_per_cell; ++i)
      {
        data.local_rhs(i)
        -= (inflow
            ?
            normal_velocity
            * (dirichlet_value * scratch.face_phi[i])
            :
            0.)
           * scratch.face_fe_values->JxW(q);

        for (unsigned int j = 0; j < stress_dofs_per_cell; ++j)
        {
          data.local_matrix(i,j)
          -= (inflow
              ?
              normal_velocity
              * (scratch.face_phi[i] * scratch.face_phi[j])
              :
              0.)
             * scratch.face_fe_values->JxW(q);
        }
      }
    }
  }



  template <int dim>
  void
  ElasticityHandler<dim>::
  local_assemble_deviatoric_stress_system_on_interior_faces
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   const unsigned int face_no,
   internal::Assembly::Scratch::DeviatoricStressSystem<dim> &scratch,
   internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
  {
    scratch.face_fe_values->reinit (cell, face_no);

    const unsigned int comp0_idx = sim.introspection.variable("deviatoric stress").first_component_index;

    const FEValuesExtractors::SymmetricTensor<2> stress_extractor(comp0_idx);
    const FEValuesExtractors::Vector mesh_velocity_extractor(0);

    (*scratch.face_fe_values)[sim.introspection.extractors.velocities].get_function_values(sim.current_linearization_point,
                                                                                           scratch.face_velocity_values);
    // Get the mesh velocity, as we need to subtract it off the convection velocity.
    if (sim.parameters.mesh_deformation_enabled)
      (*scratch.face_fe_values)[mesh_velocity_extractor].get_function_values(sim.mesh_deformation->get_mesh_velocity(),
                                                                             scratch.face_mesh_velocity_values);


    const unsigned int stress_dofs_per_cell = data.local_dof_indices.size();
    const unsigned int n_q_points = scratch.face_fe_values->n_quadrature_points;

    const typename DoFHandler<dim>::cell_iterator
    neighbor = cell->neighbor_or_periodic_neighbor(face_no);

    // 'neighbor' defined above is NOT active_cell_iterator, so this includes cells that are refined.
    Assert (neighbor.state() == IteratorState::valid, ExcInternalError());
    const bool cell_has_periodic_neighbor = cell->has_periodic_neighbor(face_no);

    if (!neighbor->has_children())
    {
      if (neighbor->level() == cell->level() &&
          neighbor->is_active() &&
          ((neighbor->is_locally_owned() && cell->index() < neighbor->index())
           ||
           (!neighbor->is_locally_owned() && cell->subdomain_id() < neighbor->subdomain_id())))
      {
        Assert (cell->is_locally_owned(), ExcInternalError());
        // cell and neighbor are equal-sized, and cell has been chosen to assemble this face, 
        // so calculate from cell

        const unsigned int neighbor2 = 
          (cell->has_periodic_neighbor(face_no)
           ?
           // how does the periodic neighbor talk about this cell?
           cell->periodic_neighbor_of_periodic_neighbor(face_no)
           :
           // how does the neighbor talk about this cell?
           cell->neighbor_of_neighbor(face_no));

        // set up neighbor values
        scratch.neighbor_face_fe_values->reinit (neighbor, neighbor2);

        // Get all dof indices on the neighbor, then extract those
        // that correspond to the components we are interested in.
        neighbor->get_dof_indices (scratch.neighbor_dof_indices);
        for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
        {
          const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
          if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
          {
            data.neighbor_dof_indices[face_no * GeometryInfo<dim>::max_children_per_face][i_stress] = scratch.neighbor_dof_indices[i];
            ++i_stress;
          }
          ++i;
        }

        data.assembled_matrices[face_no * GeometryInfo<dim>::max_children_per_face] = true;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Precompute the values of shape functions.
          for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
          {
            const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
            if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
            {
              scratch.face_phi[i_stress]          = (*scratch.face_fe_values)[stress_extractor].value(i, q);
              scratch.neighbor_face_phi[i_stress] = (*scratch.neighbor_face_fe_values)[stress_extractor].value(i, q);
              ++i_stress;
            }
            ++i;
          }

          Tensor<1,dim> current_u = scratch.face_velocity_values[q];
          // Subtract off the mesh velocity for ALE corrections if necessary.
          if (sim.parameters.mesh_deformation_enabled)
            current_u -= scratch.face_mesh_velocity_values[q];

          const double normal_velocity = current_u * scratch.face_fe_values->normal_vector(q);

          // The discontinuous Galerkin method uses 2 types of jumps over edges:
          // undirected and directed jumps. Undirected jumps are dependent only
          // on the order of the numbering of cells. Directed jumps are dependent
          // on the direction of the flow. Thus the flow-dependent terms below are
          // only calculated if the face is an inflow face.
          const bool inflow = (current_u * scratch.face_fe_values->normal_vector(q) < 0.);

          for (unsigned int i = 0; i < stress_dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < stress_dofs_per_cell; ++j)
            {
              data.local_matrix(i,j)
              -= (inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.face_phi[i] * scratch.face_phi[j])
                  :
                  0.)
                 * scratch.face_fe_values->JxW(q);
              
              data.local_matrices_int_ext[face_no * GeometryInfo<dim>::max_children_per_face](i,j)
              += (inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.face_phi[i] * scratch.neighbor_face_phi[j])
                  :
                  0.)
                 * scratch.face_fe_values->JxW(q);

              data.local_matrices_ext_int[face_no * GeometryInfo<dim>::max_children_per_face](i,j)
              -= (!inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.neighbor_face_phi[i] * scratch.face_phi[j])
                  :
                  0.)
                 * scratch.face_fe_values->JxW(q);

              data.local_matrices_ext_ext[face_no * GeometryInfo<dim>::max_children_per_face](i,j)
              += (!inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.neighbor_face_phi[i] * scratch.neighbor_face_phi[j])
                  :
                  0.)
                 * scratch.face_fe_values->JxW(q);
            }
          }
        }
      }
      else
      {
        // neighbor is taking responsibility for assembly of this face, because
        // either (1) neighbor is coarser, or
        //        (2) neighbor is equally-sized and
        //           (a) neighbor is on a different subdomain, with lower subdomain_id(), or
        //           (b) neighbor is on the same subdomain and has lower index().
      }
    }
    // neighbor has children, so always assemble from here.
    else
    {
      const unsigned int neighbor2 = 
        (cell_has_periodic_neighbor
         ?
         cell->periodic_neighbor_face_no(face_no)
         :
         cell->neighbor_face_no(face_no));

      // Loop over subfaces. We know that the neighbor is finer, so we could loop over the subfaces of the current
      // face. But if we are at a periodic boundary, then the face of the current cell has no children, so instead use
      // the children of the periodic neighbor's corresponding face since we know that the later does indeed have
      // children (because we know that the neighbor is refined).
      typename DoFHandler<dim>::face_iterator neighbor_face = neighbor->face(neighbor2);
      for (unsigned int subface_no = 0; subface_no < neighbor_face->number_of_children(); ++subface_no)
      {
        const typename DoFHandler<dim>::active_cell_iterator neighbor_child
          = (cell_has_periodic_neighbor
             ?
             cell->periodic_neighbor_child_on_subface(face_no, subface_no)
             :
             cell->neighbor_child_on_subface(face_no, subface_no));

        // set up subface values
        scratch.subface_fe_values->reinit (cell, face_no, subface_no);
        scratch.neighbor_face_fe_values->reinit (neighbor_child, neighbor2);

        // subface->face
        (*scratch.subface_fe_values)[sim.introspection.extractors.velocities].get_function_values (sim.current_linearization_point,
                                                                                                   scratch.face_velocity_values);

        // Get the mesh velocity, as we need to subtract it off the convection velocity.
        if (sim.parameters.mesh_deformation_enabled)
          (*scratch.subface_fe_values)[mesh_velocity_extractor].get_function_values (sim.mesh_deformation->get_mesh_velocity(),
                                                                                     scratch.face_mesh_velocity_values);
        // Get all dof indices on the neighbor, then extract those
        // that correspond to the solution field we are interested in.
        neighbor_child->get_dof_indices (scratch.neighbor_dof_indices);
        for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
        {
          const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
          if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
          {
            data.neighbor_dof_indices[face_no * GeometryInfo<dim>::max_children_per_face + subface_no][i_stress] = scratch.neighbor_dof_indices[i];
            ++i_stress;
          }
          ++i;
        }

        data.assembled_matrices[face_no * GeometryInfo<dim>::max_children_per_face + subface_no] = true;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Precompute the values of shape functions.
          for (unsigned int i = 0, i_stress = 0; i_stress < stress_dofs_per_cell; /*increment at end of loop*/)
          {
            const unsigned int comp = sim.finite_element.system_to_component_index(i).first;
            if (comp >= comp0_idx && comp < comp0_idx + SymmetricTensor<2,dim>::n_independent_components)
            {
              scratch.face_phi[i_stress]          = (*scratch.subface_fe_values)[stress_extractor].value(i, q);
              scratch.neighbor_face_phi[i_stress] = (*scratch.neighbor_face_fe_values)[stress_extractor].value(i, q);
              ++i_stress;
            }
            ++i;
          }

          Tensor<1,dim> current_u = scratch.face_velocity_values[q];
          // Substract off the mesh velocity for ALE corrections if necessary.
          if (sim.parameters.mesh_deformation_enabled)
            current_u -= scratch.face_mesh_velocity_values[q];

          const double normal_velocity = current_u * scratch.subface_fe_values->normal_vector(q);

          // The discontinuous Galerkin method uses 2 types of jumps over edges:
          // undirected and directed jumps. Undirected jumps are dependent only
          // on the order of the numbering of cells. Directed jumps are dependent
          // on the direction of the flow. Thus the flow-dependent terms below are
          // only calculated if the edge is an inflow edge.
          const bool inflow = ((current_u * scratch.subface_fe_values->normal_vector(q)) < 0.);

          for (unsigned int i = 0; i < stress_dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < stress_dofs_per_cell; ++j)
            {
              data.local_matrix(i,j)
              -= (inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.face_phi[i] * scratch.face_phi[j])
                  :
                  0.)
                 * scratch.subface_fe_values->JxW(q);

              data.local_matrices_int_ext[face_no * GeometryInfo<dim>::max_children_per_face + subface_no](i,j)
              += (inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.face_phi[i] * scratch.neighbor_face_phi[j])
                  :
                  0.)
                 * scratch.subface_fe_values->JxW(q);

              data.local_matrices_ext_int[face_no * GeometryInfo<dim>::max_children_per_face + subface_no](i,j)
              -= (!inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.neighbor_face_phi[i] * scratch.face_phi[j])
                  :
                  0.)
                 * scratch.subface_fe_values->JxW(q);

              data.local_matrices_ext_ext[face_no * GeometryInfo<dim>::max_children_per_face + subface_no](i,j)
              += (!inflow
                  ?
                  sim.time_step * normal_velocity
                  * (scratch.neighbor_face_phi[i] * scratch.neighbor_face_phi[j])
                  :
                  0.)
                 * scratch.subface_fe_values->JxW(q);
            }
          }
        }
      }
    }
  }


  template <int dim>
  void
  ElasticityHandler<dim>::
  copy_local_to_global_deviatoric_stress_system (const internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
  {
    sim.current_constraints.distribute_local_to_global (data.local_matrix,
                                                        data.local_rhs,
                                                        data.local_dof_indices,
                                                        sim.system_matrix,
                                                        sim.system_rhs);

    if (parameters.use_discontinuous_stress_discretization)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::max_children_per_face * GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (data.assembled_matrices[f])
        {
          for (unsigned int i=0; i<data.local_dof_indices.size(); ++i)
            for (unsigned int j=0; j<data.neighbor_dof_indices[f].size(); ++j)
            {
              sim.system_matrix.add (data.local_dof_indices[i],
                                     data.neighbor_dof_indices[f][j],
                                     data.local_matrices_int_ext[f](i,j));
              sim.system_matrix.add (data.neighbor_dof_indices[f][j],
                                     data.local_dof_indices[i],
                                     data.local_matrices_ext_int[f](j,i));
            }

          for (unsigned int i=0; i<data.neighbor_dof_indices[f].size(); ++i)
            for (unsigned int j=0; j<data.neighbor_dof_indices[f].size(); ++j)
              sim.system_matrix.add (data.neighbor_dof_indices[f][i],
                                     data.neighbor_dof_indices[f][j],
                                     data.local_matrices_ext_ext[f](i,j));
        }
      }
    }
  }


  template <int dim>
  void
  ElasticityHandler<dim>::assemble_and_solve_deviatoric_stress ()
  {
    TimerOutput::Scope timer (sim.computing_timer, "Solve deviatoric stress system");

    if (sim.timestep_number == 0)
    {
      //solve_first_timestep();
      return;
    }

    const FEVariable<dim> &variable = sim.introspection.variable("deviatoric stress");
    const unsigned int block_idx = variable.block_index;

    sim.system_matrix.block(block_idx, block_idx) = 0;
    sim.system_rhs.block(block_idx) = 0;

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    // We have to assemble the term u.grad phi_i * phi_j, which is
    // of total polynomial degree
    //   stokes_deg + 2*tau_deg - 1,
    // where tau_deg = stokes_deg-1. This suggests using a Gauss
    // quadrature formula of order
    //   (3*stokes_deg - 3) / 2
    // rounded up. Considering the possibility that stokes_deg = 1
    // (in which case (3*stokes_deg - 3) / 2 = 0), we simply plus
    // the formula by 1.
    const unsigned int quadrature_degree = 
      parameters.stress_degree + (sim.parameters.stokes_velocity_degree+1)/2;

    const UpdateFlags update_flags = update_values |
                                     update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values;

    const UpdateFlags face_update_flags = update_values |
                                          update_normal_vectors |
                                          update_quadrature_points |
                                          update_JxW_values;

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      internal::Assembly::Scratch::DeviatoricStressSystem<dim> &scratch,
                      internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
    {
      this->local_assemble_deviatoric_stress_system (cell, scratch, data);
    };

    auto copier = [&](const internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data)
    {
      this->copy_local_to_global_deviatoric_stress_system (data);
    };

    const unsigned int stress_dofs_per_cell = 
      variable.fe->dofs_per_cell * variable.multiplicity;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     sim.dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     sim.dof_handler.end()),
         worker,
         copier,
         internal::Assembly::Scratch::
         DeviatoricStressSystem<dim> (sim.finite_element,
                                      *sim.mapping,
                                      QGauss<dim>(quadrature_degree),
                                      (parameters.use_discontinuous_stress_discretization ?
                                       QGauss<dim-1>(quadrature_degree) :
                                       Quadrature<dim-1>()),
                                      update_flags,
                                      face_update_flags,
                                      stress_dofs_per_cell,
                                      sim.introspection.n_compositional_fields),
         internal::Assembly::CopyData::
         DeviatoricStressSystem<dim> (stress_dofs_per_cell,
                                      parameters.use_discontinuous_stress_discretization));

    sim.system_matrix.compress(VectorOperation::add);
    sim.system_rhs.compress(VectorOperation::add);

    // carry out the solving procedure
    const double tolerance = std::max(1e-50,
                                      parameters.linear_solver_tolerance * sim.system_rhs.block(block_idx).l2_norm());

    SolverControl solver_control (5000, tolerance);
    solver_control.enable_history_data();

#ifdef ASPECT_USE_PETSC
    SolverGMRES<LinearAlgebra::Vector> solver (solver_control,
                                               SolverGMRES<LinearAlgebra::Vector>::AdditionalData(
                                                 parameters.gmres_restart_length, true));
#else
    TrilinosWrappers::SolverGMRES solver (solver_control,
                                          TrilinosWrappers::SolverGMRES::AdditionalData(
                                            false, parameters.gmres_restart_length));
#endif

    LinearAlgebra::PreconditionILU preconditioner;
    preconditioner.initialize (sim.system_matrix.block(block_idx, block_idx));

    sim.pcout << "   Solving deviatoric stress field... " << std::flush;

    // create distributed vector (we need all blocks here even though we only
    // solve for the current block, because we only have a ConstraintMatrix
    // for the whole system). current_linearization_point contains our initial guess.
    LinearAlgebra::BlockVector distributed_solution (
      sim.introspection.index_sets.system_partitioning,
      sim.mpi_communicator);
    distributed_solution.block(block_idx) = sim.current_linearization_point.block(block_idx);

    sim.current_constraints.set_zero(distributed_solution);

    // solve the linear system:
    try
    {
      solver.solve (sim.system_matrix.block(block_idx,block_idx),
                    distributed_solution.block(block_idx),
                    sim.system_rhs.block(block_idx),
                    preconditioner);
    }
    // if the solver fails, report the error from processor 0 with some additional
    // information about its location, and throw a quiet exception on all other
    // processors
    catch (const std::exception &exc)
    {
      if (Utilities::MPI::this_mpi_process(sim.mpi_communicator) == 0)
        AssertThrow (false,
                     ExcMessage (std::string("The GMRES solver for deviatoric stress "
                                             "did not converge. It reported the following error:\n\n")
                                 +
                                 exc.what()))
        else
          throw QuietException();
    }

    sim.current_constraints.distribute (distributed_solution);
    sim.solution.block(block_idx) = distributed_solution.block(block_idx);

    // apply WENO limiter if required
    if (parameters.use_weno_limiter)
      apply_weno_limiter();

    // update the current linearization point
    sim.current_linearization_point.block(block_idx) = sim.solution.block(block_idx);

    // print number of iterations and also record it in the
    // statistics file
    sim.pcout << solver_control.last_step()
              << " iterations." << std::endl;
  }


  namespace internal
  {
    void find_big_stencil (const DoFHandler<2>::active_cell_iterator &cell,
                           std::vector<DoFHandler<2>::cell_iterator> &stencil,
                           std::vector<unsigned int>                 &indicators)
    {
      AssertThrow (cell->is_locally_owned(), ExcInternalError());
      AssertThrow (stencil.size() == 9 && indicators.size() == 9, ExcInternalError());

      std::fill (stencil.begin(), stencil.end(), cell->get_dof_handler().end());
      std::fill (indicators.begin(), indicators.end(), numbers::invalid_unsigned_int);

      if (cell->level() == 0)
      {
        /*
         *      ___________________
         *      |     |     |     |
         *      |  6  |  7  |  8  |
         *      |_____|_____|_____|
         *      |     |     |     |
         *      |  3  |  4  |  5  |
         *      |_____|_____|_____|
         *      |     |     |     |
         *      |  0  |  1  |  2  |
         *      |_____|_____|_____|
         */
        stencil[4] = cell;
        stencil[3] = cell->neighbor(0);
        stencil[5] = cell->neighbor(1);
        stencil[1] = cell->neighbor(2);
        stencil[7] = cell->neighbor(3);

        if (stencil[1].state() == IteratorState::valid)
        {
          stencil[0] = stencil[1]->neighbor(0);
          stencil[2] = stencil[1]->neighbor(1);
        }

        if (stencil[7].state() == IteratorState::valid)
        {
          stencil[6] = stencil[7]->neighbor(0);
          stencil[8] = stencil[7]->neighbor(1);
        }
      }
      else
      {
        DoFHandler<2>::cell_iterator parent = cell->parent();

        // Find the cell index of the present cell.
        unsigned int child_index = numbers::invalid_unsigned_int;
        for (unsigned int i=0; i<4; ++i)
          if (cell == parent->child(i))
          {
            child_index = i;
            break;
          }
        AssertThrow (child_index != numbers::invalid_unsigned_int, ExcInternalError());

        // Now fill in the stencil.
        std::vector<DoFHandler<2>::cell_iterator>
        coarse_stencil (4, parent->get_dof_handler().end());
        switch (child_index)
        {
          case 0:
          {
            /*
             *      ___________________
             *      |     |           |
             *      |     |           |
             *      |     |           |
             *      |  2  |     3     |
             *      |     |           |
             *      |     |           |
             *      |_____|___________|
             *      |     |           |
             *      |  0  |     1     |
             *      |_____|___________|
             */
            coarse_stencil[3] = parent;
            coarse_stencil[2] = parent->neighbor(0);
            coarse_stencil[1] = parent->neighbor(2);
            if (coarse_stencil[1].state() == IteratorState::valid)
              coarse_stencil[0] = coarse_stencil[1]->neighbor(0);

            stencil[4] = coarse_stencil[3];
            stencil[5] = coarse_stencil[3];
            stencil[7] = coarse_stencil[3];
            stencil[8] = coarse_stencil[3];
            indicators[4] = 0;
            indicators[5] = 1;
            indicators[7] = 2;
            indicators[8] = 3;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              indicators[0] = 3;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[1] = coarse_stencil[1];
              stencil[2] = coarse_stencil[1];
              indicators[1] = 2;
              indicators[2] = 3;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[3] = coarse_stencil[2];
              stencil[6] = coarse_stencil[2];
              indicators[3] = 1;
              indicators[6] = 3;
            }

            break;
          }
          case 1:
          {
            /*
             *      ___________________
             *      |           |     |
             *      |           |     |
             *      |           |     |
             *      |     2     |  3  |
             *      |           |     |
             *      |           |     |
             *      |___________|_____|
             *      |           |     |
             *      |     0     |  1  |
             *      |___________|_____|
             */
            coarse_stencil[2] = parent;
            coarse_stencil[3] = parent->neighbor(1);
            coarse_stencil[0] = parent->neighbor(2);
            if (coarse_stencil[3].state() == IteratorState::valid)
              coarse_stencil[1] = coarse_stencil[3]->neighbor(2);

            stencil[3] = coarse_stencil[2];
            stencil[4] = coarse_stencil[2];
            stencil[6] = coarse_stencil[2];
            stencil[7] = coarse_stencil[2];
            indicators[3] = 0;
            indicators[4] = 1;
            indicators[6] = 2;
            indicators[7] = 3;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[1] = coarse_stencil[0];
              indicators[0] = 2;
              indicators[1] = 3;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[2] = coarse_stencil[1];
              indicators[2] = 2;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[5] = coarse_stencil[3];
              stencil[8] = coarse_stencil[3];
              indicators[5] = 0;
              indicators[8] = 2;
            }

            break;
          }
          case 2:
          {
            /*
             *      ___________________
             *      |     |           |
             *      |  2  |     3     |
             *      |_____|___________|
             *      |     |           |
             *      |     |           |
             *      |     |           |
             *      |  0  |     1     |
             *      |     |           |
             *      |     |           |
             *      |_____|___________|
             */
            coarse_stencil[1] = parent;
            coarse_stencil[0] = parent->neighbor(0);
            coarse_stencil[3] = parent->neighbor(3);
            if (coarse_stencil[0].state() == IteratorState::valid)
              coarse_stencil[2] = coarse_stencil[0]->neighbor(3);

            stencil[1] = coarse_stencil[1];
            stencil[2] = coarse_stencil[1];
            stencil[4] = coarse_stencil[1];
            stencil[5] = coarse_stencil[1];
            indicators[1] = 0;
            indicators[2] = 1;
            indicators[4] = 2;
            indicators[5] = 3;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[3] = coarse_stencil[0];
              indicators[0] = 1;
              indicators[3] = 3;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[6] = coarse_stencil[2];
              indicators[6] = 1;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[7] = coarse_stencil[3];
              stencil[8] = coarse_stencil[3];
              indicators[7] = 0;
              indicators[8] = 1;
            }

            break;
          }
          case 3:
          {
            /*
             *      ___________________
             *      |           |     |
             *      |     2     |  3  |
             *      |___________|_____|
             *      |           |     |
             *      |           |     |
             *      |           |     |
             *      |     0     |  1  |
             *      |           |     |
             *      |           |     |
             *      |___________|_____|
             */
            coarse_stencil[0] = parent;
            coarse_stencil[1] = parent->neighbor(1);
            coarse_stencil[2] = parent->neighbor(3);
            if (coarse_stencil[1].state() == IteratorState::valid)
              coarse_stencil[3] = coarse_stencil[1]->neighbor(3);

            stencil[0] = coarse_stencil[0];
            stencil[1] = coarse_stencil[0];
            stencil[3] = coarse_stencil[0];
            stencil[4] = coarse_stencil[0];
            indicators[0] = 0;
            indicators[1] = 1;
            indicators[3] = 2;
            indicators[4] = 3;

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[2] = coarse_stencil[1];
              stencil[5] = coarse_stencil[1];
              indicators[2] = 0;
              indicators[5] = 2;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[6] = coarse_stencil[2];
              stencil[7] = coarse_stencil[2];
              indicators[6] = 0;
              indicators[7] = 1;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[8] = coarse_stencil[3];
              indicators[8] = 0;
            }

            break;
          }
          default:
          {
            AssertThrow (false, ExcInternalError());
          }
        }
      }
    }


    void find_big_stencil (const DoFHandler<3>::active_cell_iterator &cell,
                           std::vector<DoFHandler<3>::cell_iterator> &stencil,
                           std::vector<unsigned int>                 &indicators)
    {
      AssertThrow (cell->is_locally_owned(), ExcInternalError());
      AssertThrow (stencil.size() == 27 && indicators.size() == 27, ExcInternalError());

      std::fill (stencil.begin(), stencil.end(), cell->get_dof_handler().end());
      std::fill (indicators.begin(), indicators.end(), numbers::invalid_unsigned_int);

      if (cell->level() == 0)
      {
        /*
         *             ______________________________
         *            /         /         /         /|
         *           /   24    /   25    /   26    / |
         *          /_________/_________/_________/  |
         *         /         /         /         /|  |
         *        /   21    /   22    /   23    / |26|
         *       /_________/_________/_________/  |  |
         *      /         /         /         /|  | /|
         *     /   18    /   19    /   20    / |23|/ |
         *    /_________/_________/_________/  |  |  |
         *    |         |         |         |  | /|17|  
         *    |         |         |         |20|/ |  |
         *    |    18   |    19   |    20   |  |  | /|
         *    |         |         |         | /|14|/ |
         *    |_________|_________|_________|/ |  |  |
         *    |         |         |         |  | /|8 |
         *    |         |         |         |11|/ |  |
         *    |    9    |    10   |    11   |  |  | /
         *    |         |         |         | /|5 |/
         *    |_________|_________|_________|/ |  /
         *    |         |         |         |  | /
         *    |         |         |         |2 |/
         *    |    0    |    1    |    2    |  /
         *    |         |         |         | /
         *    |_________|_________|_________|/
         *
         */
        stencil[13] = cell;
        stencil[12] = cell->neighbor(0);
        stencil[14] = cell->neighbor(1);
        stencil[10] = cell->neighbor(2);
        stencil[16] = cell->neighbor(3);
        stencil[ 4] = cell->neighbor(4);
        stencil[22] = cell->neighbor(5);

        if (stencil[4].state() == IteratorState::valid)
        {
          stencil[3] = stencil[4]->neighbor(0);
          stencil[5] = stencil[4]->neighbor(1);
          stencil[1] = stencil[4]->neighbor(2);
          stencil[7] = stencil[4]->neighbor(3);
        }

        if (stencil[22].state() == IteratorState::valid)
        {
          stencil[21] = stencil[22]->neighbor(0);
          stencil[23] = stencil[22]->neighbor(1);
          stencil[19] = stencil[22]->neighbor(2);
          stencil[25] = stencil[22]->neighbor(3);
        }

        if (stencil[10].state() == IteratorState::valid)
        {
          stencil[ 9] = stencil[10]->neighbor(0);
          stencil[11] = stencil[10]->neighbor(1);
        }

        if (stencil[16].state() == IteratorState::valid)
        {
          stencil[15] = stencil[16]->neighbor(0);
          stencil[17] = stencil[16]->neighbor(1);
        }

        if (stencil[1].state() == IteratorState::valid)
        {
          stencil[0] = stencil[1]->neighbor(0);
          stencil[2] = stencil[1]->neighbor(1);
        }

        if (stencil[7].state() == IteratorState::valid)
        {
          stencil[6] = stencil[7]->neighbor(0);
          stencil[8] = stencil[7]->neighbor(1);
        }

        if (stencil[19].state() == IteratorState::valid)
        {
          stencil[18] = stencil[19]->neighbor(0);
          stencil[20] = stencil[19]->neighbor(1);
        }

        if (stencil[25].state() == IteratorState::valid)
        {
          stencil[24] = stencil[25]->neighbor(0);
          stencil[26] = stencil[25]->neighbor(1);
        }
      }
      else
      {
        DoFHandler<3>::cell_iterator parent = cell->parent();

        // Find the child index of the present cell.
        unsigned int child_index = numbers::invalid_unsigned_int;
        for (unsigned int i=0; i<8; ++i)
          if (cell == parent->child(i))
          {
            child_index = i;
            break;
          }
        AssertThrow (child_index != numbers::invalid_unsigned_int, ExcInternalError());

        // Now fill in the stencil.
        std::vector<DoFHandler<3>::cell_iterator> 
        coarse_stencil (8, parent->get_dof_handler().end());
        switch (child_index)
        {
          case 0:
          {
            /*
             *          ___________________
             *         /     /           /|
             *        /  6  /     7     / |
             *       /     /           /  |
             *      /_____/___________/   |
             *     /  4  /     5     /|   |
             *    /_____/___________/ | 7 |
             *    |     |           | |   |
             *    |     |           | |  /|
             *    |     |           |5| / |
             *    |  4  |     5     | |/3 |
             *    |     |           | |  /
             *    |_____|___________|/| /
             *    |     |           |1|/
             *    |  0  |     1     | /
             *    |_____|___________|/
             */
            coarse_stencil[7] = parent;
            coarse_stencil[6] = parent->neighbor(0);
            coarse_stencil[5] = parent->neighbor(2);
            coarse_stencil[3] = parent->neighbor(4);
            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              coarse_stencil[4] = coarse_stencil[6]->neighbor(2);
              coarse_stencil[2] = coarse_stencil[6]->neighbor(4);
            }
            if (coarse_stencil[5].state() == IteratorState::valid)
              coarse_stencil[1] = coarse_stencil[5]->neighbor(4);
            if (coarse_stencil[4].state() == IteratorState::valid)
              coarse_stencil[0] = coarse_stencil[4]->neighbor(4);

            stencil[13] = coarse_stencil[7];
            stencil[14] = coarse_stencil[7];
            stencil[16] = coarse_stencil[7];
            stencil[17] = coarse_stencil[7];
            stencil[22] = coarse_stencil[7];
            stencil[23] = coarse_stencil[7];
            stencil[25] = coarse_stencil[7];
            stencil[26] = coarse_stencil[7];
            indicators[13] = 0;
            indicators[14] = 1;
            indicators[16] = 2;
            indicators[17] = 3;
            indicators[22] = 4;
            indicators[23] = 5;
            indicators[25] = 6;
            indicators[26] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              indicators[0] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[1] = coarse_stencil[1];
              stencil[2] = coarse_stencil[1];
              indicators[1] = 6;
              indicators[2] = 7;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[3] = coarse_stencil[2];
              stencil[6] = coarse_stencil[2];
              indicators[3] = 5;
              indicators[6] = 7;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[4] = coarse_stencil[3];
              stencil[5] = coarse_stencil[3];
              stencil[7] = coarse_stencil[3];
              stencil[8] = coarse_stencil[3];
              indicators[4] = 4;
              indicators[5] = 5;
              indicators[7] = 6;
              indicators[8] = 7;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[ 9] = coarse_stencil[4];
              stencil[18] = coarse_stencil[4];
              indicators[ 9] = 3;
              indicators[18] = 7;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[10] = coarse_stencil[5];
              stencil[11] = coarse_stencil[5];
              stencil[19] = coarse_stencil[5];
              stencil[20] = coarse_stencil[5];
              indicators[10] = 2;
              indicators[11] = 3;
              indicators[19] = 6;
              indicators[20] = 7;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[12] = coarse_stencil[6];
              stencil[15] = coarse_stencil[6];
              stencil[21] = coarse_stencil[6];
              stencil[24] = coarse_stencil[6];
              indicators[12] = 1;
              indicators[15] = 3;
              indicators[21] = 5;
              indicators[24] = 7;
            }

            break;
          }
          case 1:
          {
            /*
             *          ___________________
             *         /           /     /|
             *        /     6     /  7  / |
             *       /           /     /  |
             *      /___________/_____/   |
             *     /     4     /  5  /|   |
             *    /___________/_____/ | 7 |
             *    |           |     | |   |
             *    |           |     | |  /|
             *    |           |     |5| / |
             *    |     4     |  5  | |/3 |
             *    |           |     | |  /
             *    |___________|_____|/| /
             *    |           |     |1|/
             *    |     0     |  1  | /
             *    |___________|_____|/
             */
            coarse_stencil[6] = parent;
            coarse_stencil[7] = parent->neighbor(1);
            coarse_stencil[4] = parent->neighbor(2);
            coarse_stencil[2] = parent->neighbor(4);
            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              coarse_stencil[0] = coarse_stencil[4]->neighbor(4);
              coarse_stencil[5] = coarse_stencil[4]->neighbor(1);
            }
            if (coarse_stencil[7].state() == IteratorState::valid)
              coarse_stencil[3] = coarse_stencil[7]->neighbor(4);
            if (coarse_stencil[0].state() == IteratorState::valid)
              coarse_stencil[1] = coarse_stencil[0]->neighbor(1);

            stencil[12] = coarse_stencil[6];
            stencil[13] = coarse_stencil[6];
            stencil[15] = coarse_stencil[6];
            stencil[16] = coarse_stencil[6];
            stencil[21] = coarse_stencil[6];
            stencil[22] = coarse_stencil[6];
            stencil[24] = coarse_stencil[6];
            stencil[25] = coarse_stencil[6];
            indicators[12] = 0;
            indicators[13] = 1;
            indicators[15] = 2;
            indicators[16] = 3;
            indicators[21] = 4;
            indicators[22] = 5;
            indicators[24] = 6;
            indicators[25] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[1] = coarse_stencil[0];
              indicators[0] = 6;
              indicators[1] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[2] = coarse_stencil[1];
              indicators[2] = 6;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[3] = coarse_stencil[2];
              stencil[4] = coarse_stencil[2];
              stencil[6] = coarse_stencil[2];
              stencil[7] = coarse_stencil[2];
              indicators[3] = 4;
              indicators[4] = 5;
              indicators[6] = 6;
              indicators[7] = 7;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[5] = coarse_stencil[3];
              stencil[8] = coarse_stencil[3];
              indicators[5] = 4;
              indicators[8] = 6;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[ 9] = coarse_stencil[4];
              stencil[10] = coarse_stencil[4];
              stencil[18] = coarse_stencil[4];
              stencil[19] = coarse_stencil[4];
              indicators[ 9] = 2;
              indicators[10] = 3;
              indicators[18] = 6;
              indicators[19] = 7;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[11] = coarse_stencil[5];
              stencil[20] = coarse_stencil[5];
              indicators[11] = 2;
              indicators[20] = 6;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[14] = coarse_stencil[7];
              stencil[17] = coarse_stencil[7];
              stencil[23] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[14] = 0;
              indicators[17] = 2;
              indicators[23] = 4;
              indicators[26] = 6;
            }

            break;
          }
          case 2:
          {
            /*
             *          ___________________
             *         /  6  /     7     /|
             *        /_____/___________/ |
             *       /     /           /| |
             *      /  4  /     5     / | |
             *     /     /           /  |7|
             *    /_____/___________/   | |
             *    |     |           |   | |
             *    |     |           | 5 |/|
             *    |  4  |     5     |   |3|
             *    |     |           |  /| |
             *    |     |           | / |/
             *    |_____|___________|/1 /
             *    |     |           |  /
             *    |  0  |     1     | /
             *    |_____|___________|/
             */
            coarse_stencil[5] = parent;
            coarse_stencil[4] = parent->neighbor(0);
            coarse_stencil[7] = parent->neighbor(3);
            coarse_stencil[1] = parent->neighbor(4);
            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              coarse_stencil[6] = coarse_stencil[4]->neighbor(3);
              coarse_stencil[0] = coarse_stencil[4]->neighbor(4);
            }
            if (coarse_stencil[7].state() == IteratorState::valid)
              coarse_stencil[3] = coarse_stencil[7]->neighbor(4);
            if (coarse_stencil[6].state() == IteratorState::valid)
              coarse_stencil[2] = coarse_stencil[6]->neighbor(4);
            
            stencil[10] = coarse_stencil[5];
            stencil[11] = coarse_stencil[5];
            stencil[13] = coarse_stencil[5];
            stencil[14] = coarse_stencil[5];
            stencil[19] = coarse_stencil[5];
            stencil[20] = coarse_stencil[5];
            stencil[22] = coarse_stencil[5];
            stencil[23] = coarse_stencil[5];
            indicators[10] = 0;
            indicators[11] = 1;
            indicators[13] = 2;
            indicators[14] = 3;
            indicators[19] = 4;
            indicators[20] = 5;
            indicators[22] = 6;
            indicators[23] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[3] = coarse_stencil[0];
              indicators[0] = 5;
              indicators[3] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[1] = coarse_stencil[1];
              stencil[2] = coarse_stencil[1];
              stencil[4] = coarse_stencil[1];
              stencil[5] = coarse_stencil[1];
              indicators[1] = 4;
              indicators[2] = 5;
              indicators[4] = 6;
              indicators[5] = 7;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[6] = coarse_stencil[2];
              indicators[6] = 5;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[7] = coarse_stencil[3];
              stencil[8] = coarse_stencil[3];
              indicators[7] = 4;
              indicators[8] = 5;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[ 9] = coarse_stencil[4];
              stencil[12] = coarse_stencil[4];
              stencil[18] = coarse_stencil[4];
              stencil[21] = coarse_stencil[4];
              indicators[ 9] = 1;
              indicators[12] = 3;
              indicators[18] = 5;
              indicators[21] = 7;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[15] = coarse_stencil[6];
              stencil[24] = coarse_stencil[6];
              indicators[15] = 1;
              indicators[24] = 5;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[16] = coarse_stencil[7];
              stencil[17] = coarse_stencil[7];
              stencil[25] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[16] = 0;
              indicators[17] = 1;
              indicators[25] = 4;
              indicators[26] = 5;
            }

            break;
          }
          case 3:
          {
            /*
             *          ___________________
             *         /     6     /  7  /|
             *        /___________/_____/ |
             *       /           /     /| |
             *      /     4     /  5  / | |
             *     /           /     /  |7|
             *    /___________/_____/   | |
             *    |           |     |   | |
             *    |           |     | 5 |/|
             *    |     4     |  5  |   |3|
             *    |           |     |  /| |
             *    |           |     | / |/
             *    |___________|_____|/1 /
             *    |           |     |  /
             *    |     0     |  1  | /
             *    |___________|_____|/
             */
            coarse_stencil[4] = parent;
            coarse_stencil[5] = parent->neighbor(1);
            coarse_stencil[6] = parent->neighbor(3);
            coarse_stencil[0] = parent->neighbor(4);
            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              coarse_stencil[7] = coarse_stencil[5]->neighbor(3);
              coarse_stencil[1] = coarse_stencil[5]->neighbor(4);
            }
            if (coarse_stencil[6].state() == IteratorState::valid)
              coarse_stencil[2] = coarse_stencil[6]->neighbor(4);
            if (coarse_stencil[1].state() == IteratorState::valid)
              coarse_stencil[3] = coarse_stencil[1]->neighbor(3);

            stencil[ 9] = coarse_stencil[4];
            stencil[10] = coarse_stencil[4];
            stencil[12] = coarse_stencil[4];
            stencil[13] = coarse_stencil[4];
            stencil[18] = coarse_stencil[4];
            stencil[19] = coarse_stencil[4];
            stencil[21] = coarse_stencil[4];
            stencil[22] = coarse_stencil[4];
            indicators[ 9] = 0;
            indicators[10] = 1;
            indicators[12] = 2;
            indicators[13] = 3;
            indicators[18] = 4;
            indicators[19] = 5;
            indicators[21] = 6;
            indicators[22] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[1] = coarse_stencil[0];
              stencil[3] = coarse_stencil[0];
              stencil[4] = coarse_stencil[0];
              indicators[0] = 4;
              indicators[1] = 5;
              indicators[3] = 6;
              indicators[4] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[2] = coarse_stencil[1];
              stencil[5] = coarse_stencil[1];
              indicators[2] = 4;
              indicators[5] = 6;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[6] = coarse_stencil[2];
              stencil[7] = coarse_stencil[2];
              indicators[6] = 4;
              indicators[7] = 5;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[8] = coarse_stencil[3];
              indicators[8] = 4;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[11] = coarse_stencil[5];
              stencil[14] = coarse_stencil[5];
              stencil[20] = coarse_stencil[5];
              stencil[23] = coarse_stencil[5];
              indicators[11] = 0;
              indicators[14] = 2;
              indicators[20] = 4;
              indicators[23] = 6;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[15] = coarse_stencil[6];
              stencil[16] = coarse_stencil[6];
              stencil[24] = coarse_stencil[6];
              stencil[25] = coarse_stencil[6];
              indicators[15] = 0;
              indicators[16] = 1;
              indicators[24] = 4;
              indicators[25] = 5;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[17] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[17] = 0;
              indicators[26] = 4;
            }

            break;
          }
          case 4:
          {
            /*
             *          ___________________
             *         /     /           /|
             *        /  6  /     7     / |
             *       /     /           /  |
             *      /_____/___________/ 7 |
             *     /  4  /     5     /|  /|
             *    /_____/___________/ | / |
             *    |     |           |5|/  |
             *    |  4  |     5     | |   |
             *    |_____|___________|/| 3 |
             *    |     |           | |   |
             *    |     |           | |  /
             *    |     |           |1| /
             *    |  0  |     1     | |/
             *    |     |           | /
             *    |_____|___________|/
             */
            coarse_stencil[3] = parent;
            coarse_stencil[1] = parent->neighbor(2);
            coarse_stencil[2] = parent->neighbor(0);
            coarse_stencil[7] = parent->neighbor(5);
            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              coarse_stencil[0] = coarse_stencil[1]->neighbor(0);
              coarse_stencil[5] = coarse_stencil[1]->neighbor(5);
            }
            if (coarse_stencil[7].state() == IteratorState::valid)
              coarse_stencil[6] = coarse_stencil[7]->neighbor(0);
            if (coarse_stencil[0].state() == IteratorState::valid)
              coarse_stencil[4] = coarse_stencil[0]->neighbor(5);

            stencil[ 4] = coarse_stencil[3];
            stencil[ 5] = coarse_stencil[3];
            stencil[ 7] = coarse_stencil[3];
            stencil[ 8] = coarse_stencil[3];
            stencil[13] = coarse_stencil[3];
            stencil[14] = coarse_stencil[3];
            stencil[16] = coarse_stencil[3];
            stencil[17] = coarse_stencil[3];
            indicators[ 4] = 0;
            indicators[ 5] = 1;
            indicators[ 7] = 2;
            indicators[ 8] = 3;
            indicators[13] = 4;
            indicators[14] = 5;
            indicators[16] = 6;
            indicators[17] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[0] = coarse_stencil[0];
              stencil[9] = coarse_stencil[0];
              indicators[0] = 3;
              indicators[9] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[ 1] = coarse_stencil[1];
              stencil[ 2] = coarse_stencil[1];
              stencil[10] = coarse_stencil[1];
              stencil[11] = coarse_stencil[1];
              indicators[ 1] = 2;
              indicators[ 2] = 3;
              indicators[10] = 6;
              indicators[11] = 7;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[ 3] = coarse_stencil[2];
              stencil[ 6] = coarse_stencil[2];
              stencil[12] = coarse_stencil[2];
              stencil[15] = coarse_stencil[2];
              indicators[ 3] = 1;
              indicators[ 6] = 3;
              indicators[12] = 5;
              indicators[15] = 7;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[18] = coarse_stencil[4];
              indicators[18] = 3;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[19] = coarse_stencil[5];
              stencil[20] = coarse_stencil[5];
              indicators[19] = 2;
              indicators[20] = 3;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[21] = coarse_stencil[6];
              stencil[24] = coarse_stencil[6];
              indicators[21] = 1;
              indicators[24] = 3;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[22] = coarse_stencil[7];
              stencil[23] = coarse_stencil[7];
              stencil[25] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[22] = 0;
              indicators[23] = 1;
              indicators[25] = 2;
              indicators[26] = 3;
            }

            break;
          }

          case 5:
          {
            /*
             *          ___________________
             *         /           /     /|
             *        /     6     /  7  / |
             *       /           /     /  |
             *      /___________/_____/ 7 |
             *     /     4     /  5  /|  /|
             *    /___________/_____/ | / |
             *    |           |     |5|/  |
             *    |     4     |  5  | |   |
             *    |___________|_____|/| 3 |
             *    |           |     | |   |
             *    |           |     | |  /
             *    |           |     |1| /
             *    |     0     |  1  | |/
             *    |           |     | /
             *    |___________|_____|/
             */
            coarse_stencil[2] = parent;
            coarse_stencil[3] = parent->neighbor(1);
            coarse_stencil[0] = parent->neighbor(2);
            coarse_stencil[6] = parent->neighbor(5);
            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              coarse_stencil[1] = coarse_stencil[0]->neighbor(1);
              coarse_stencil[4] = coarse_stencil[0]->neighbor(5);
            }
            if (coarse_stencil[3].state() == IteratorState::valid)
              coarse_stencil[7] = coarse_stencil[3]->neighbor(5);
            if (coarse_stencil[1].state() == IteratorState::valid)
              coarse_stencil[5] = coarse_stencil[1]->neighbor(5);

            stencil[ 3] = coarse_stencil[2];
            stencil[ 4] = coarse_stencil[2];
            stencil[ 6] = coarse_stencil[2];
            stencil[ 7] = coarse_stencil[2];
            stencil[12] = coarse_stencil[2];
            stencil[13] = coarse_stencil[2];
            stencil[15] = coarse_stencil[2];
            stencil[16] = coarse_stencil[2];
            indicators[ 3] = 0;
            indicators[ 4] = 1;
            indicators[ 6] = 2;
            indicators[ 7] = 3;
            indicators[12] = 4;
            indicators[13] = 5;
            indicators[15] = 6;
            indicators[16] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[ 0] = coarse_stencil[0];
              stencil[ 1] = coarse_stencil[0];
              stencil[ 9] = coarse_stencil[0];
              stencil[10] = coarse_stencil[0];
              indicators[ 0] = 2;
              indicators[ 1] = 3;
              indicators[ 9] = 6;
              indicators[10] = 7;
            }

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[ 2] = coarse_stencil[1];
              stencil[11] = coarse_stencil[1];
              indicators[ 2] = 2;
              indicators[11] = 6;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[ 5] = coarse_stencil[3];
              stencil[ 8] = coarse_stencil[3];
              stencil[14] = coarse_stencil[3];
              stencil[17] = coarse_stencil[3];
              indicators[ 5] = 0;
              indicators[ 8] = 2;
              indicators[14] = 4;
              indicators[17] = 6;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[18] = coarse_stencil[4];
              stencil[19] = coarse_stencil[4];
              indicators[18] = 2;
              indicators[19] = 3;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[20] = coarse_stencil[5];
              indicators[20] = 2;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[21] = coarse_stencil[6];
              stencil[22] = coarse_stencil[6];
              stencil[24] = coarse_stencil[6];
              stencil[25] = coarse_stencil[6];
              indicators[21] = 0;
              indicators[22] = 1;
              indicators[24] = 2;
              indicators[25] = 3;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[23] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[23] = 0;
              indicators[26] = 2;
            }

            break;
          }
          case 6:
          {
            /*
             *          ___________________
             *         /  6  /     7     /|
             *        /_____/___________/ |
             *       /     /           /|7|
             *      /  4  /     5     / | |
             *     /     /           /  |/|
             *    /_____/___________/ 5 | |
             *    |     |           |  /| |
             *    |  4  |     5     | / |3|
             *    |_____|___________|/  | |
             *    |     |           |   | |
             *    |     |           | 1 |/
             *    |  0  |     1     |   /
             *    |     |           |  /
             *    |     |           | /
             *    |_____|___________|/
             */
            coarse_stencil[1] = parent;
            coarse_stencil[0] = parent->neighbor(0);
            coarse_stencil[3] = parent->neighbor(3);
            coarse_stencil[5] = parent->neighbor(5);
            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              coarse_stencil[4] = coarse_stencil[5]->neighbor(0);
              coarse_stencil[7] = coarse_stencil[5]->neighbor(3);
            }
            if (coarse_stencil[0].state() == IteratorState::valid)
              coarse_stencil[2] = coarse_stencil[0]->neighbor(3);
            if (coarse_stencil[4].state() == IteratorState::valid)
              coarse_stencil[6] = coarse_stencil[4]->neighbor(3);

            stencil[ 1] = coarse_stencil[1];
            stencil[ 2] = coarse_stencil[1];
            stencil[ 4] = coarse_stencil[1];
            stencil[ 5] = coarse_stencil[1];
            stencil[10] = coarse_stencil[1];
            stencil[11] = coarse_stencil[1];
            stencil[13] = coarse_stencil[1];
            stencil[14] = coarse_stencil[1];
            indicators[ 1] = 0;
            indicators[ 2] = 1;
            indicators[ 4] = 2;
            indicators[ 5] = 3;
            indicators[10] = 4;
            indicators[11] = 5;
            indicators[13] = 6;
            indicators[14] = 7;

            if (coarse_stencil[0].state() == IteratorState::valid)
            {
              stencil[ 0] = coarse_stencil[0];
              stencil[ 3] = coarse_stencil[0];
              stencil[ 9] = coarse_stencil[0];
              stencil[12] = coarse_stencil[0];
              indicators[ 0] = 1;
              indicators[ 3] = 3;
              indicators[ 9] = 5;
              indicators[12] = 7;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[ 6] = coarse_stencil[2];
              stencil[15] = coarse_stencil[2];
              indicators[ 6] = 1;
              indicators[15] = 5;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[ 7] = coarse_stencil[3];
              stencil[ 8] = coarse_stencil[3];
              stencil[16] = coarse_stencil[3];
              stencil[17] = coarse_stencil[3];
              indicators[ 7] = 0;
              indicators[ 8] = 1;
              indicators[16] = 4;
              indicators[17] = 5;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[18] = coarse_stencil[4];
              stencil[21] = coarse_stencil[4];
              indicators[18] = 1;
              indicators[21] = 3;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[19] = coarse_stencil[5];
              stencil[20] = coarse_stencil[5];
              stencil[22] = coarse_stencil[5];
              stencil[23] = coarse_stencil[5];
              indicators[19] = 0;
              indicators[20] = 1;
              indicators[22] = 2;
              indicators[23] = 3;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[24] = coarse_stencil[6];
              indicators[24] = 1;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[25] = coarse_stencil[7];
              stencil[26] = coarse_stencil[7];
              indicators[25] = 0;
              indicators[26] = 1;
            }

            break;
          }
          case 7:
          {
            /*
             *          ___________________
             *         /     6     /  7  /|
             *        /___________/_____/ |
             *       /           /     /|7|
             *      /     4     /  5  / | |
             *     /           /     /  |/|
             *    /___________/_____/ 5 | |
             *    |           |     |  /| |
             *    |     4     |  5  | / |3|
             *    |___________|_____|/  | |
             *    |           |     |   | |
             *    |           |     | 1 |/
             *    |     0     |  1  |   /
             *    |           |     |  /
             *    |           |     | /
             *    |___________|_____|/
             */
            coarse_stencil[0] = parent;
            coarse_stencil[1] = parent->neighbor(1);
            coarse_stencil[2] = parent->neighbor(3);
            coarse_stencil[4] = parent->neighbor(5);
            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              coarse_stencil[3] = coarse_stencil[1]->neighbor(3);
              coarse_stencil[5] = coarse_stencil[1]->neighbor(5);
            }
            if (coarse_stencil[2].state() == IteratorState::valid)
              coarse_stencil[6] = coarse_stencil[2]->neighbor(5);
            if (coarse_stencil[3].state() == IteratorState::valid)
              coarse_stencil[7] = coarse_stencil[3]->neighbor(5);

            stencil[ 0] = coarse_stencil[0];
            stencil[ 1] = coarse_stencil[0];
            stencil[ 3] = coarse_stencil[0];
            stencil[ 4] = coarse_stencil[0];
            stencil[ 9] = coarse_stencil[0];
            stencil[10] = coarse_stencil[0];
            stencil[12] = coarse_stencil[0];
            stencil[13] = coarse_stencil[0];
            indicators[ 0] = 0;
            indicators[ 1] = 1;
            indicators[ 3] = 2;
            indicators[ 4] = 3;
            indicators[ 9] = 4;
            indicators[10] = 5;
            indicators[12] = 6;
            indicators[13] = 7;

            if (coarse_stencil[1].state() == IteratorState::valid)
            {
              stencil[ 2] = coarse_stencil[1];
              stencil[ 5] = coarse_stencil[1];
              stencil[11] = coarse_stencil[1];
              stencil[14] = coarse_stencil[1];
              indicators[ 2] = 0;
              indicators[ 5] = 2;
              indicators[11] = 4;
              indicators[14] = 6;
            }

            if (coarse_stencil[2].state() == IteratorState::valid)
            {
              stencil[ 6] = coarse_stencil[2];
              stencil[ 7] = coarse_stencil[2];
              stencil[15] = coarse_stencil[2];
              stencil[16] = coarse_stencil[2];
              indicators[ 6] = 0;
              indicators[ 7] = 1;
              indicators[15] = 4;
              indicators[16] = 5;
            }

            if (coarse_stencil[3].state() == IteratorState::valid)
            {
              stencil[ 8] = coarse_stencil[3];
              stencil[17] = coarse_stencil[3];
              indicators[ 8] = 0;
              indicators[17] = 4;
            }

            if (coarse_stencil[4].state() == IteratorState::valid)
            {
              stencil[18] = coarse_stencil[4];
              stencil[19] = coarse_stencil[4];
              stencil[21] = coarse_stencil[4];
              stencil[22] = coarse_stencil[4];
              indicators[18] = 0;
              indicators[19] = 1;
              indicators[21] = 2;
              indicators[22] = 3;
            }

            if (coarse_stencil[5].state() == IteratorState::valid)
            {
              stencil[20] = coarse_stencil[5];
              stencil[23] = coarse_stencil[5];
              indicators[20] = 0;
              indicators[23] = 2;
            }

            if (coarse_stencil[6].state() == IteratorState::valid)
            {
              stencil[24] = coarse_stencil[6];
              stencil[25] = coarse_stencil[6];
              indicators[24] = 0;
              indicators[25] = 1;
            }

            if (coarse_stencil[7].state() == IteratorState::valid)
            {
              stencil[26] = coarse_stencil[7];
              indicators[26] = 0;
            }

            break;
          }
          default:
          {
            AssertThrow (false, ExcInternalError());
          }
        }
      }
    }


    void calculate_linear_polynomial (const Point<2>      &point,
                                      std::vector<double> &values)
    {
      values.resize(4);

      const double x = point[0], y = point[1];
      values[0] = 1;
      values[1] = x;
      values[2] = y;
      values[3] = x * y;
    }


    void calculate_linear_polynomial (const Point<3>      &point,
                                      std::vector<double> &values)
    {
      values.resize(8);

      const double x = point[0], y = point[1], z = point[2];
      values[0] = 1;
      values[1] = x;
      values[2] = y;
      values[3] = z;
      values[4] = y * z;
      values[5] = z * x;
      values[6] = x * y;
      values[7] = x * y * z;
    }


    void calculate_quadratic_polynomial (const Point<2>      &point,
                                         std::vector<double> &values)
    {
      values.resize(9);

      const double x = point[0], y = point[1];
      values[0] = 1;
      values[1] = x;
      values[2] = y;
      values[3] = x * y;
      values[4] = x * x;
      values[5] = y * y;
      values[6] = values[4] * y;            // xxy
      values[7] = values[5] * x;            // xyy
      values[8] = values[4] * values[5];    // xxyy
    }


    void calculate_quadratic_polynomial (const Point<3>      &point,
                                         std::vector<double> &values)
    {
      values.resize(27);

      const double x = point[0], y = point[1], z = point[2];
      values[ 0] = 1;
      values[ 1] = x;
      values[ 2] = y;
      values[ 3] = z;
      values[ 4] = y * z;
      values[ 5] = z * x;
      values[ 6] = x * y;
      values[ 7] = x * x;
      values[ 8] = y * y;
      values[ 9] = z * z;
      values[10] = values[ 4] * x;          // xyz
      values[11] = values[ 7] * y;          // xxy
      values[12] = values[ 7] * z;          // xxz
      values[13] = values[ 8] * z;          // yyz
      values[14] = values[ 8] * x;          // yyx
      values[15] = values[ 9] * x;          // zzx
      values[16] = values[ 9] * y;          // zzy
      values[17] = values[ 7] * values[ 4]; // xxyz
      values[18] = values[ 8] * values[ 5]; // yyzx
      values[19] = values[ 9] * values[ 6]; // zzxy
      values[20] = values[ 8] * values[ 9]; // yyzz
      values[21] = values[ 9] * values[ 7]; // zzxx
      values[22] = values[ 7] * values[ 8]; // xxyy
      values[23] = values[20] * x;          // xyyzz
      values[24] = values[21] * y;          // yzzxx
      values[25] = values[22] * z;          // zxxyy
      values[26] = values[22] * values[ 9]; // xxyyzz
    }


    void 
    assemble_and_solve_linear_polynomial (const std::vector<std::vector<double> >    &polynomial_integrals,
                                          const std::vector<SymmetricTensor<2,2> >   &cell_averages,
                                          const unsigned int                          i,
                                          FullMatrix<double>                         &A,
                                          Vector<double>                             &b,
                                          std::vector<std::vector<Vector<double> > > &coefs)
    {
      static const unsigned int cells_in_stencil[4][4] 
        = { { 0, 1, 3, 4 }, { 1, 2, 4, 5 },
            { 3, 4, 6, 7 }, { 4, 5, 7, 8 } 
          };

      AssertThrow (A.m() == 4 && A.n() == 4 && b.size() == 4, ExcInternalError());

      for (unsigned int j = 0; j < 4; ++j)
      {
        const unsigned int cell_id = cells_in_stencil[i][j];
        A[j][0] = polynomial_integrals[cell_id][0]; // 1
        A[j][1] = polynomial_integrals[cell_id][1]; // x
        A[j][2] = polynomial_integrals[cell_id][2]; // y
        A[j][3] = polynomial_integrals[cell_id][3]; // xy
      }
      A.gauss_jordan();

      for (unsigned int c = 0; c < 3; ++c)
      {
        const TableIndices<2> indices = SymmetricTensor<2,2>::unrolled_to_component_indices(c);

        for (unsigned int j = 0; j < 4; ++j)
          b[j] = cell_averages[cells_in_stencil[i][j]][indices];

        A.vmult(coefs[c][i], b);
      }
    }


    void 
    assemble_and_solve_linear_polynomial (const std::vector<std::vector<double> >    &polynomial_integrals,
                                          const std::vector<SymmetricTensor<2,3> >   &cell_averages,
                                          const unsigned int                          i,
                                          dealii::FullMatrix<double>                 &A,
                                          Vector<double>                             &b,
                                          std::vector<std::vector<Vector<double> > > &coefs)
    {
      static const unsigned int cells_in_stencil[8][8] 
        = { {  0,  1,  3,  4,  9, 10, 12, 13 },
            {  1,  2,  4,  5, 10, 11, 13, 14 },
            {  3,  4,  6,  7, 12, 13, 15, 16 },
            {  4,  5,  7,  8, 13, 14, 16, 17 },
            {  9, 10, 12, 13, 18, 19, 21, 22 },
            { 10, 11, 13, 14, 19, 20, 22, 23 },
            { 12, 13, 15, 16, 21, 22, 24, 25 },
            { 13, 14, 16, 17, 22, 23, 25, 26 }
          };

      AssertThrow (A.m() == 8 && A.n() == 8 && b.size() == 8, ExcInternalError());

      for (unsigned int j = 0; j < 8; ++j)
      {
        const unsigned int cell_id = cells_in_stencil[i][j];
        A[j][0] = polynomial_integrals[cell_id][0];     // 1
        A[j][1] = polynomial_integrals[cell_id][1];     // x
        A[j][2] = polynomial_integrals[cell_id][2];     // y
        A[j][3] = polynomial_integrals[cell_id][3];     // z
        A[j][4] = polynomial_integrals[cell_id][4];     // yz
        A[j][5] = polynomial_integrals[cell_id][5];     // zx
        A[j][6] = polynomial_integrals[cell_id][6];     // xy
        A[j][7] = polynomial_integrals[cell_id][10];    // xyz
      }
      A.gauss_jordan();

      for (unsigned int c = 0; c < 6; ++c)
      {
        const TableIndices<2> indices = SymmetricTensor<2,3>::unrolled_to_component_indices(c);
        for (unsigned int j = 0; j < 8; ++j)
          b[j] = cell_averages[cells_in_stencil[i][j]][indices];

        A.vmult(coefs[c][i], b);
      }
    }


    template <int dim>
    void
    assemble_and_solve_quadratic_polynomial (const std::vector<std::vector<double> >    &polynomial_integrals,
                                             const std::vector<SymmetricTensor<2,dim> > &cell_averages,
                                             FullMatrix<double>                         &A,
                                             Vector<double>                             &b,
                                             std::vector<Vector<double> >               &coefs)
    {
      const unsigned int cells_per_big_stencil = polynomial_integrals.size();
      const unsigned int dofs_per_big_stencil  = polynomial_integrals.size();

      AssertThrow (A.m() == cells_per_big_stencil &&  
                   A.n() == dofs_per_big_stencil && 
                   b.size() == cells_per_big_stencil,
                   ExcInternalError());

      for (unsigned int i = 0; i < cells_per_big_stencil; ++i)
        for (unsigned int j = 0; j < dofs_per_big_stencil; ++j)
          A[i][j] = polynomial_integrals[i][j];

      A.gauss_jordan();

      for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      {
        AssertThrow (coefs[c].size() == dofs_per_big_stencil, ExcInternalError());
        const TableIndices<2> indices = SymmetricTensor<2,dim>::unrolled_to_component_indices(c);

        for (unsigned int i = 0; i < cells_per_big_stencil; ++i)
          b[i] = cell_averages[i][indices];

        A.vmult(coefs[c], b);
      }
    }


    template <int dim>
    double
    calculate_smoothness_indicator (const std::vector<std::vector<double> > &polynomial_integrals,
                                    const std::vector<Vector<double> >      &linear_polynomial_coefs,
                                    const unsigned int                       i);

    template <>
    double
    calculate_smoothness_indicator<2> (const std::vector<std::vector<double> > &polynomial_integrals,
                                       const std::vector<Vector<double> >      &linear_polynomial_coefs,
                                       const unsigned int                       i)
    {
      const std::vector<double> &pol = polynomial_integrals[4];
      const Vector<double> &c = linear_polynomial_coefs[i];
      double beta = c[1] * c[1] * pol[0] +        // c1*c1
                    c[3] * c[3] * pol[5] +        // c3*c3*y*y
                    c[1] * c[3] * pol[2] * 2      // 2*c1*c3*y
                    +
                    c[2] * c[2] * pol[0] +        // c2*c2
                    c[3] * c[3] * pol[4] +        // c3*c3*x*x
                    c[2] * c[3] * pol[1] * 2;     // 2*c2*c3*x

      return beta;
    }


    template <>
    double
    calculate_smoothness_indicator<3> (const std::vector<std::vector<double> > &polynomial_integrals,
                                       const std::vector<Vector<double> >      &linear_polynomial_coefs,
                                       const unsigned int                       i)
    {
      const std::vector<double> &pol = polynomial_integrals[13];
      const Vector<double> &c = linear_polynomial_coefs[i];
      double beta = c[1] * c[1] * pol[ 0] +         // c1*c1
                    c[5] * c[5] * pol[ 9] +         // c5*c5*z*z
                    c[6] * c[6] * pol[ 8] +         // c6*c6*y*y
                    c[7] * c[7] * pol[20] +         // c7*c7*y*y*z*z
                    c[1] * c[5] * pol[ 3] * 2 +     // 2*c1*c5*z
                    c[1] * c[6] * pol[ 2] * 2 +     // 2*c1*c6*y
                    c[1] * c[7] * pol[ 4] * 2 +     // 2*c1*c7*y*z
                    c[5] * c[6] * pol[ 4] * 2 +     // 2*c5*c6*y*z
                    c[5] * c[7] * pol[16] * 2 +     // 2*c5*c7*y*z*z
                    c[6] * c[7] * pol[13] * 2       // 2*c6*c7*y*y*z
                    +
                    c[2] * c[2] * pol[ 0] +         // c2*c2
                    c[4] * c[4] * pol[ 9] +         // c4*c4*z*z
                    c[6] * c[6] * pol[ 7] +         // c6*c6*x*x
                    c[7] * c[7] * pol[21] +         // c7*c7*x*x*z*z
                    c[2] * c[4] * pol[ 3] * 2 +     // 2*c2*c4*z
                    c[2] * c[6] * pol[ 1] * 2 +     // 2*c2*c6*x
                    c[2] * c[7] * pol[ 5] * 2 +     // 2*c2*c7*x*z
                    c[4] * c[6] * pol[ 5] * 2 +     // 2*c4*c6*x*z
                    c[4] * c[7] * pol[15] * 2 +     // 2*c4*c7*x*z*z
                    c[6] * c[7] * pol[12] * 2       // 2*c6*c7*x*x*z
                    +
                    c[3] * c[3] * pol[ 0] +         // c3*c3
                    c[4] * c[4] * pol[ 8] +         // c4*c4*y*y
                    c[5] * c[5] * pol[ 7] +         // c5*c5*x*x
                    c[7] * c[7] * pol[22] +         // c7*c7*x*x*y*y
                    c[3] * c[4] * pol[ 2] * 2 +     // 2*c3*c4*y
                    c[3] * c[5] * pol[ 1] * 2 +     // 2*c3*c5*x
                    c[3] * c[7] * pol[ 6] * 2 +     // 2*c3*c7*x*y
                    c[4] * c[5] * pol[ 6] * 2 +     // 2*c4*c5*x*y
                    c[4] * c[7] * pol[14] * 2 +     // 2*c4*c7*x*y*y
                    c[5] * c[7] * pol[11] * 2;      // 2*c5*c7*x*x*y

      return beta;
    }
  }


  template <int dim>
  void ElasticityHandler<dim>::apply_weno_limiter ()
  {
    const FEVariable<dim> &variable = sim.introspection.variable("deviatoric stress");
    const unsigned int comp0_idx = variable.first_component_index;
    const unsigned int block_idx = variable.block_index;

    QGauss<dim> gauss_quadrature(2);
    FEValues<dim> fe_values (*(sim.mapping), 
                             sim.finite_element,
                             gauss_quadrature,
                             update_values |
                             update_quadrature_points |
                             update_JxW_values);

    // When some cells in the stencil are coarser than the central one, we should 
    // calculate function values and gradients their 'subcells'. Therefore, we need 
    // 2^dim objects of FEValues that work on Gauss quadrature points of subcells.
    std::vector<std::unique_ptr<FEValues<dim> > > fe_subcell_values;
    for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell; ++i)
    {
      std::vector<Point<dim> > points (gauss_quadrature.get_points());
      for (unsigned int q = 0; q < points.size(); ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          points[q][d] += ((i>>d)&1);

        points[q] *= 0.5;
      }

      Quadrature<dim> quadrature (points, gauss_quadrature.get_weights());
      fe_subcell_values.push_back(std::make_unique<FEValues<dim> > (
        *(sim.mapping), sim.finite_element, quadrature, 
        update_values | update_quadrature_points | update_JxW_values));
    }

    const unsigned int n_q_points         = gauss_quadrature.size();
    const unsigned int dofs_per_cell      = sim.finite_element.dofs_per_cell;
    const unsigned int comp_dofs_per_cell = variable.fe->dofs_per_cell;

    const FEValuesExtractors::SymmetricTensor<2> stress_extractor(comp0_idx);
    std::vector<FEValuesExtractors::Scalar> component_extractors;
    for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      component_extractors.push_back(FEValuesExtractors::Scalar(comp0_idx+c));

    std::vector<types::global_dof_index> cell_dof_indices (dofs_per_cell);
    std::vector<types::global_dof_index> comp_dof_indices (comp_dofs_per_cell);

    std::vector<SymmetricTensor<2,dim> > stress_values (n_q_points);

    const unsigned int cells_per_stencil     = Utilities::fixed_power<dim,int>(2);
    const unsigned int cells_per_big_stencil = Utilities::fixed_power<dim,int>(3);
    const unsigned int n_stencils            = cells_per_stencil;
    const unsigned int dofs_per_stencil      = cells_per_stencil;
    const unsigned int dofs_per_big_stencil  = cells_per_big_stencil;

    std::vector<typename DoFHandler<dim>::cell_iterator> big_stencil (cells_per_big_stencil);
    std::vector<unsigned int> subcell_indicators (cells_per_big_stencil);

    std::vector<SymmetricTensor<2,dim> > cell_averages (cells_per_big_stencil);
    std::vector<std::vector<double> > polynomial_integrals (cells_per_big_stencil);
    for (unsigned int i = 0; i < cells_per_big_stencil; ++i)
      polynomial_integrals[i].resize(dofs_per_big_stencil);
    
    std::vector<double> linear_polynomial_values (dofs_per_stencil);
    std::vector<double> quadratic_polynomial_values (dofs_per_big_stencil);

    FullMatrix<double> A1 (cells_per_stencil, cells_per_stencil), 
                       A2 (cells_per_big_stencil, cells_per_big_stencil),
                       A3 (n_stencils, n_stencils), 
                       A_tmp1 (n_stencils, n_stencils),
                       A_tmp2 (n_stencils, n_stencils);
    Vector<double> b1 (cells_per_stencil), b2 (cells_per_big_stencil);
    std::vector<Vector<double> > b3 (SymmetricTensor<2,dim>::n_independent_components);
    for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
      b3[c].reinit (n_q_points);

    std::vector<std::vector<Vector<double> > > 
    linear_polynomial_coefs (SymmetricTensor<2,dim>::n_independent_components);

    std::vector<Vector<double> >
    quadratic_polynomial_coefs (SymmetricTensor<2,dim>::n_independent_components);

    std::vector<Vector<double> >
    linear_combination_coefs (SymmetricTensor<2,dim>::n_independent_components);

    std::vector<std::vector<double> > 
    smoothness_indicators (SymmetricTensor<2,dim>::n_independent_components);

    std::vector<std::vector<double> >
    nonlinear_weights (SymmetricTensor<2,dim>::n_independent_components);

    for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
    {
      linear_polynomial_coefs[c].resize (n_stencils);
      for (unsigned int i = 0; i < n_stencils; ++i)
        linear_polynomial_coefs[c][i].reinit (dofs_per_stencil);

      quadratic_polynomial_coefs[c].reinit (dofs_per_big_stencil);
      linear_combination_coefs[c].reinit (n_stencils);
      smoothness_indicators[c].resize (n_stencils);
      nonlinear_weights[c].resize (n_stencils);
    }

    std::vector<double> phi (comp_dofs_per_cell);

    FullMatrix<double> local_matrix (comp_dofs_per_cell, comp_dofs_per_cell);
    Vector<double> local_rhs (comp_dofs_per_cell),  comp_dof_values (comp_dofs_per_cell);

    LinearAlgebra::BlockVector distributed_solution (
      sim.introspection.index_sets.system_partitioning, 
      sim.mpi_communicator);
    distributed_solution.block(block_idx) = sim.solution.block(block_idx);

    typename DoFHandler<dim>::active_cell_iterator
    cell = sim.dof_handler.begin_active(),
    endc = sim.dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
      {
        // Find the big stencil composed of 3^dim cells (including the target cell).
        internal::find_big_stencil (cell, big_stencil, subcell_indicators);

        // Calculate the quantities in each cell that will be used in calculation of
        // cell averages and the linear polynomials. To improve the condition number 
        // of the coefficient matrix, we use the center of the target cell as the 
        // origin of coordinates.
        const Point<dim> origin = cell->center();
        for (unsigned int i = 0; i < cells_per_big_stencil; ++i)
        {
          cell_averages[i] = 0;
          for (unsigned int j = 0; j < dofs_per_big_stencil; ++j)
            polynomial_integrals[i][j] = 0;

          if (big_stencil[i].state() != IteratorState::valid)
            continue;

          if (subcell_indicators[i] != numbers::invalid_unsigned_int && big_stencil[i]->is_active())
          {
            // case 1: The quantities are calculated in a subcell.
            // The cell must be active with a valid subcell indicator.
            AssertThrow (big_stencil[i]->level() == cell->level()-1, ExcInternalError());
            FEValues<dim> *fev = fe_subcell_values[subcell_indicators[i]].get();
            fev->reinit (big_stencil[i]);
            (*fev)[stress_extractor].get_function_values (sim.solution, stress_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // notice that the determinant of Jacobian should be divided by 2^dim.
              const double JxW = fev->JxW(q) / std::pow(2,dim);

              cell_averages[i] += stress_values[q] * JxW;

              Point<dim> q_point = fev->quadrature_point(q);
              q_point -= origin;
              internal::calculate_quadratic_polynomial (q_point, quadratic_polynomial_values);
              for (unsigned int j = 0; j < dofs_per_big_stencil; ++j)
                polynomial_integrals[i][j] += quadratic_polynomial_values[j] * JxW;
            }

            //cell_averages[i] /= polynomial_integrals[i][0];
          }
          else if ((subcell_indicators[i] == numbers::invalid_unsigned_int && big_stencil[i]->is_active())
                   ||
                   (subcell_indicators[i] != numbers::invalid_unsigned_int && big_stencil[i]->has_children()
                    && big_stencil[i]->child(subcell_indicators[i])->is_active()))
          {
            // case 2: The average stress is calculated in an active cell.
            // 2 possibilities: (a) the cell is active with an invalid subcell indicator;
            //                  (b) the cell is inactive with a valid subcell indicator, and the 
            //                      target child is active.
            typename DoFHandler<dim>::active_cell_iterator target_cell;
            if (subcell_indicators[i] == numbers::invalid_unsigned_int)
            {
              AssertThrow (big_stencil[i]->level() == 0 && big_stencil[i]->is_active(), 
                           ExcInternalError());
              target_cell = big_stencil[i];
            }
            else
            {
              AssertThrow (big_stencil[i]->level() == cell->level()-1, ExcInternalError());
              target_cell = big_stencil[i]->child(subcell_indicators[i]);
            }

            fe_values.reinit (target_cell);
            fe_values[stress_extractor].get_function_values (sim.solution, stress_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double JxW = fe_values.JxW(q);

              cell_averages[i] += stress_values[q] * fe_values.JxW(q);

              Point<dim> q_point = fe_values.quadrature_point(q);
              q_point -= origin;
              internal::calculate_quadratic_polynomial (q_point, quadratic_polynomial_values);
              for (unsigned int j = 0; j < dofs_per_big_stencil; ++j)
                polynomial_integrals[i][j] += quadratic_polynomial_values[j] * JxW;
            }
            //cell_averages[i] /= polynomial_integrals[i][0];
          }
          else if (big_stencil[i]->has_children() &&
                   (subcell_indicators[i] == numbers::invalid_unsigned_int ||
                    (subcell_indicators[i] != numbers::invalid_unsigned_int && 
                     big_stencil[i]->child(subcell_indicators[i])->has_children())))
          {
            // case 3: The average stress is the average of 2^dim cells.
            // The cell must have children, and (a) the subcell indicator is invalid, or
            //                                  (b) the subcell indicator is active, while the target
            //                                      child has children.
            typename DoFHandler<dim>::cell_iterator parent;
            std::vector<typename DoFHandler<dim>::active_cell_iterator> children;
            if (subcell_indicators[i] == numbers::invalid_unsigned_int)
            {
              AssertThrow (big_stencil[i]->level() == 0, ExcInternalError());
              parent = big_stencil[i];
            }
            else
            {
              AssertThrow (big_stencil[i]->level() == cell->level()-1, ExcInternalError());
              parent = big_stencil[i]->child(subcell_indicators[i]);
            }
            children = GridTools::get_active_child_cells<DoFHandler<dim> >(parent);

            for (unsigned int c = 0; c < children.size(); ++c)
            {
              // notice that the child might be artificial, in which case we can't require
              // the dof values.
              if (children[c]->is_artificial())
                goto end_of_loop;

              fe_values.reinit (children[c]);
              fe_values[stress_extractor].get_function_values (sim.solution, stress_values);
              for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const double JxW = fe_values.JxW(q);

                cell_averages[i] += stress_values[q] * fe_values.JxW(q);

                Point<dim> q_point = fe_values.quadrature_point(q);
                q_point -= origin;
                internal::calculate_quadratic_polynomial (q_point, quadratic_polynomial_values);
                for (unsigned int j = 0; j < dofs_per_big_stencil; ++j)
                  polynomial_integrals[i][j] += quadratic_polynomial_values[j] * JxW;
              }
            }

            //cell_averages[i] /= polynomial_integrals[i][0];
          }
          else
          {
            AssertThrow (false, ExcInternalError());
          }
        }

        if (cell->at_boundary())
        {
          // find the valid stencils for boundary cells
          std::set<unsigned int> invalid_stencil_ids;
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f))
              for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_face; ++i)
                invalid_stencil_ids.insert(GeometryInfo<dim>::child_cell_on_face(
                  RefinementCase<dim>::isotropic_refinement, f, i));

          AssertThrow (invalid_stencil_ids.size() < GeometryInfo<dim>::max_children_per_cell,
                       ExcInternalError());

          std::vector<unsigned int> valid_stencil_ids;
          for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell; ++i)
            if (invalid_stencil_ids.find(i) == invalid_stencil_ids.end())
              valid_stencil_ids.push_back(i);

          // solve for the coefficients of linear polynomials on valid stencils...
          for (unsigned int i = 0; i < valid_stencil_ids.size(); ++i)
            internal::assemble_and_solve_linear_polynomial (polynomial_integrals,
                                                            cell_averages,
                                                            valid_stencil_ids[i], A1, b1,
                                                            linear_polynomial_coefs);

          // ...and set the coefficients of linear polynomials on invalid stencils to 0.
          for (auto p = invalid_stencil_ids.begin(); p != invalid_stencil_ids.end(); ++p)
            for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
              for (unsigned int j = 0; j < dofs_per_stencil; ++j)
                linear_polynomial_coefs[c][*p][j] = 0;

          // calculate the smoothness indicator, and pick the stencil with the minimum
          // smoothness indicator as the ENO stencil
          double min_beta = std::numeric_limits<double>::max();
          unsigned int eno_stencil = numbers::invalid_unsigned_int;
          for (unsigned int i = 0; i < valid_stencil_ids.size(); ++i)
          {
            double beta = 0;
            for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
              beta += internal::calculate_smoothness_indicator<dim> (polynomial_integrals,
                                                                     linear_polynomial_coefs[c],
                                                                     valid_stencil_ids[i]);

            if (beta < min_beta)
            {
              min_beta = beta;
              eno_stencil = valid_stencil_ids[i];
            }
          }

          for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
          {
            for (unsigned int i = 0; i < n_stencils; ++i)
            {
              // set the nonlinear weights of the eno stencil to 1...
              if (i == eno_stencil)
                nonlinear_weights[c][i] = 1;
              // ...and the nonlinear weights of other stencils to 0
              else
                nonlinear_weights[c][i] = 0;
            }
          }
        }
        else
        {
          // solve for the coefficients of linear polynomials on stencils.
          for (unsigned int i = 0; i < n_stencils; ++i)
            internal::assemble_and_solve_linear_polynomial (polynomial_integrals, 
                                                            cell_averages, 
                                                            i, A1, b1, 
                                                            linear_polynomial_coefs);

          // solve for the coefficients of quadratic polynomial on the big stencil.
          internal::assemble_and_solve_quadratic_polynomial (polynomial_integrals, 
                                                             cell_averages, 
                                                             A2, b2, 
                                                             quadratic_polynomial_coefs);

          // determine the linear combination of linear polynomials that coincides with 
          // the quadratic polynomial at Gauss quadrature points.
          fe_values.reinit (cell);
          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Point<dim> q_point = fe_values.quadrature_point(q);
            q_point -= origin;
            internal::calculate_linear_polynomial (q_point, linear_polynomial_values);
            for (unsigned int i = 0; i < dofs_per_stencil; ++i)
              A_tmp1[q][i] = linear_polynomial_values[i];

            internal::calculate_quadratic_polynomial (q_point, quadratic_polynomial_values);
            for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
            {
              b3[c][q] = 0;
              for (unsigned int i = 0; i < dofs_per_big_stencil; ++i)
                b3[c][q] += quadratic_polynomial_coefs[c][i] * quadratic_polynomial_values[i];
            }
          }

          for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
          {
            for (unsigned int i = 0; i < n_stencils; ++i)
              for (unsigned int j = 0; j < dofs_per_stencil; ++j)
                A_tmp2[i][j] = linear_polynomial_coefs[c][i][j];

            A_tmp1.mTmult(A3, A_tmp2);
            A3.gauss_jordan();

            A3.vmult(linear_combination_coefs[c], b3[c]);

            // calculate the smoothness indicator
            for (unsigned int i = 0; i < n_stencils; ++i)
              smoothness_indicators[c][i] = 
                internal::calculate_smoothness_indicator<dim> (polynomial_integrals, 
                                                               linear_polynomial_coefs[c],
                                                               i);

            // calculate the nonlinear weights
            for (unsigned int i = 0; i < n_stencils; ++i)
            {
              double denominator = 0;
              for (unsigned int j = 0; j < n_stencils; ++j)
                denominator += Utilities::fixed_power<2,double>(parameters.weno_epsilon + smoothness_indicators[c][i]);

              nonlinear_weights[c][i] = linear_combination_coefs[c][i] / denominator;
            }

            // TODO: we can't use std::accumulate here, otherwise the sum will be 0. why?
            double sum = 0;
            for (unsigned int i = 0; i < n_stencils; ++i)
              sum += nonlinear_weights[c][i];

            for (unsigned int i = 0; i < n_stencils; ++i)
              nonlinear_weights[c][i] /= sum;
          }
        }

        cell->get_dof_indices (cell_dof_indices);
        fe_values.reinit(cell);
        for (unsigned int c = 0; c < SymmetricTensor<2,dim>::n_independent_components; ++c)
        {
          // finally, calculate the values of weighted polynomials at Gauss quadrature points
          // and solve for the DoFs of the cell
          for (unsigned int i = 0, i_comp = 0; i_comp < comp_dofs_per_cell; /*increment at end of loop*/)
          {
            if (sim.finite_element.system_to_component_index(i).first == comp0_idx+c)
            {
              comp_dof_indices[i_comp] = cell_dof_indices[i];
              ++i_comp;
            }
            ++i;
          }

          local_matrix = 0;
          local_rhs = 0;
          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Point<dim> q_point = fe_values.quadrature_point(q);
            q_point -= origin;
            internal::calculate_linear_polynomial (q_point, linear_polynomial_values);

            for (unsigned int i = 0, i_comp = 0; i_comp < comp_dofs_per_cell; /*increment at end of loop*/)
            {
              if (sim.finite_element.system_to_component_index(i).first == comp0_idx+c)
              {
                phi[i_comp] = fe_values[component_extractors[c]].value(i, q);
                ++i_comp;
              }
              ++i;
            }

            double R = 0;
            for (unsigned int i = 0; i < n_stencils; ++i)
            {
              double pol = 0;
              for (unsigned int j = 0; j < dofs_per_stencil; ++j)
                pol += linear_polynomial_coefs[c][i][j] * linear_polynomial_values[j];

              R += pol * nonlinear_weights[c][i];
            }
            
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < comp_dofs_per_cell; ++i)
            {
              local_rhs(i) += R * phi[i] * JxW;
              for (unsigned int j = 0; j < comp_dofs_per_cell; ++j)
                local_matrix(i,j) += phi[i] * phi[j] * JxW;
            }
          }

          local_matrix.gauss_jordan();
          local_matrix.vmult (comp_dof_values, local_rhs);

          for (unsigned int i = 0; i < comp_dofs_per_cell; ++i)
            distributed_solution[comp_dof_indices[i]] = comp_dof_values[i];
        }
end_of_loop:
        continue;
      }

    sim.solution.block(block_idx) = distributed_solution.block(block_idx);
  }


  template <int dim>
  const std::vector<double> &
  ElasticityHandler<dim>::get_elastic_shear_moduli () const
  {
    return parameters.elastic_shear_moduli;
  }


  template <int dim>
  double
  ElasticityHandler<dim>::get_initial_time_step () const
  {
    return parameters.initial_time_step;
  }


  template <int dim>
  bool
  ElasticityHandler<dim>::
  use_discontinuous_stress_discretization () const
  {
    return parameters.use_discontinuous_stress_discretization;
  }
}


// explicit instantiations
namespace aspect
{
  #define INSTANTIATE(dim) \
  template class ElasticityHandler<dim>; \
  \
  namespace MaterialModel \
  { \
    template class ElasticInputs<dim>; \
    template class ElasticOutputs<dim>; \
  } \
  namespace Assemblers \
  { \
    template class ElasticRHSTerm<dim>; \
    template class ElasticRHSBoundaryTerm<dim>; \
  } \
  namespace internal \
  { \
    namespace Assembly \
    { \
      namespace Scratch \
      { \
        template struct DeviatoricStressSystem<dim>; \
      } \
      namespace CopyData \
      { \
        template struct DeviatoricStressSystem<dim>; \
      } \
    } \
  } \

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
