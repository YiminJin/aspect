#ifndef _aspect_elasticity_h
#define _aspect_elasticity_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator/assemblers/interface.h>

#include <deal.II/base/parsed_function.h>

namespace aspect
{
  using namespace dealii;

  namespace MaterialModel
  {
    template <int dim>
    class ElasticInputs : public AdditionalMaterialInputs<dim>
    {
      public:
        ElasticInputs (const unsigned int n_points);

        std::vector<SymmetricTensor<2,dim> > stress_values;
        std::vector<SymmetricTensor<2,dim> > old_stress_values;
        std::vector<SymmetricTensor<2,dim> > old_old_stress_values;
        std::vector<SymmetricTensor<2,dim> > convection_terms;
        std::vector<Tensor<2,dim> >          velocity_gradients;

        void fill (const LinearAlgebra::BlockVector &solution,
                   const LinearAlgebra::BlockVector &old_solution,
                   const LinearAlgebra::BlockVector &old_old_solution,
                   const FEValuesBase<dim>          &fe_values,
                   const Introspection<dim>         &introspection) override;
    };

    template <int dim>
    class ElasticOutputs : public AdditionalMaterialOutputs<dim>
    {
      public:
        std::vector<double> elastic_shear_moduli;

        ElasticOutputs (const unsigned int n_points);

        void average (const MaterialAveraging::AveragingOperation operation,
                      const FullMatrix<double>  &projection_matrix,
                      const FullMatrix<double>  &expansion_matrix);
    };
  }

  namespace internal
  {
    namespace Assembly
    {
      namespace Scratch
      {
        template <int dim>
        struct DeviatoricStressSystem : public ScratchBase<dim>
        {
          DeviatoricStressSystem (const FiniteElement<dim> &fe,
                                  const Mapping<dim>       &mapping,
                                  const Quadrature<dim>    &quadrature,
                                  const Quadrature<dim-1>  &face_quadrature,
                                  const UpdateFlags         update_flags,
                                  const UpdateFlags         face_update_flags,
                                  const unsigned int        stress_dofs_per_cell,
                                  const unsigned int        n_compositional_fields);

          DeviatoricStressSystem (const DeviatoricStressSystem &scratch);

          FEValues<dim>         fe_values;

          std::unique_ptr<FEFaceValues<dim> >    face_fe_values;
          std::unique_ptr<FEFaceValues<dim> >    neighbor_face_fe_values;
          std::unique_ptr<FESubfaceValues<dim> > subface_fe_values;

          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<types::global_dof_index> neighbor_dof_indices;

          std::vector<SymmetricTensor<2,dim> > phi;
          std::vector<SymmetricTensor<2,dim> > u_dot_grad_phi;
          std::vector<SymmetricTensor<2,dim> > face_phi;
          std::vector<SymmetricTensor<2,dim> > neighbor_face_phi;

          std::vector<SymmetricTensor<2,dim> > old_stress_values;
          std::vector<SymmetricTensor<2,dim> > old_old_stress_values;
          std::vector<Tensor<1,dim> >          velocity_values;
          std::vector<Tensor<2,dim> >          velocity_gradients;
          std::vector<Tensor<1,dim> >          mesh_velocity_values;
          std::vector<Tensor<1,dim> >          face_velocity_values;
          std::vector<Tensor<1,dim> >          face_mesh_velocity_values;

          MaterialModel::MaterialModelInputs<dim>  material_model_inputs;
          MaterialModel::MaterialModelOutputs<dim> material_model_outputs;
        };
      }

      namespace CopyData
      {
        template <int dim>
        struct DeviatoricStressSystem : public CopyDataBase<dim>
        {
          DeviatoricStressSystem (const unsigned int stress_dofs_per_cell,
                                  const bool         discontinuous);

          DeviatoricStressSystem (const DeviatoricStressSystem &data);

          FullMatrix<double> local_matrix;

          std::vector<FullMatrix<double> > local_matrices_int_ext;
          std::vector<FullMatrix<double> > local_matrices_ext_int;
          std::vector<FullMatrix<double> > local_matrices_ext_ext;

          Vector<double> local_rhs;

          std::vector<bool> assembled_matrices;

          std::vector<types::global_dof_index> local_dof_indices;

          std::vector<std::vector<types::global_dof_index> > neighbor_dof_indices;
        };
      }
    }
  }

  namespace Assemblers
  {
    template <int dim>
    class ElasticRHSTerm : public Assemblers::Interface<dim>,
                           public SimulatorAccess<dim>
    {
      public:
        void
        execute (internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                 internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override;

        void
        create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &outputs) const override;
    };

    template <int dim>
    class ElasticRHSBoundaryTerm : public Assemblers::Interface<dim>,
                                   public SimulatorAccess<dim>
    {
      public:
        void
        execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                 internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override;

        void
        create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &outputs) const override;
    };
  }

  template <int dim>
  class ElasticityHandler
  {
    public:
      struct Parameters;

      ElasticityHandler (Simulator<dim>   &sim,
                         ParameterHandler &prm);

      void edit_finite_element_variables (std::vector<VariableDeclaration<dim> > &variables);

      void set_assemblers (const SimulatorAccess<dim> &simulator_access,
                           Assemblers::Manager<dim>   &assemblers) const;

      void initialize (ParameterHandler &prm);

      void add_current_constraints (AffineConstraints<double> &constraints);

      void assemble_and_solve_deviatoric_stress ();

      const std::vector<double> &get_elastic_shear_moduli () const;

      double get_initial_time_step () const;

      bool use_discontinuous_stress_discretization () const;

    private:
      void 
      local_assemble_deviatoric_stress_system (const typename DoFHandler<dim>::active_cell_iterator      &cell,
                                               internal::Assembly::Scratch::DeviatoricStressSystem<dim>  &scratch,
                                               internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data);

      void
      local_assemble_deviatoric_stress_system_on_interior_faces (const typename DoFHandler<dim>::active_cell_iterator      &cell,
                                                                 const unsigned int                                         face_no,
                                                                 internal::Assembly::Scratch::DeviatoricStressSystem<dim>  &scratch,
                                                                 internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data);

      void
      local_assemble_deviatoric_stress_system_on_boundary_faces (const typename DoFHandler<dim>::active_cell_iterator      &cell,
                                                                 const unsigned int                                         face_no,
                                                                 internal::Assembly::Scratch::DeviatoricStressSystem<dim>  &scratch,
                                                                 internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data);

      void
      copy_local_to_global_deviatoric_stress_system (const internal::Assembly::CopyData::DeviatoricStressSystem<dim> &data);

      void apply_weno_limiter ();

    public:
      struct Parameters
      {
        Parameters ();

        static void declare_parameters (ParameterHandler &prm);

        void parse_parameters (ParameterHandler &prm,
                               const bool convert_to_years);

        std::vector<double> elastic_shear_moduli;

        double              initial_time_step;

        unsigned int        stress_degree;

        bool                use_discontinuous_stress_discretization;

        double              linear_solver_tolerance;

        unsigned int        gmres_restart_length;

        bool                use_weno_limiter;
        double              weno_epsilon;

        std::set<types::boundary_id> prescribed_stress_boundary_indicators;

        Functions::ParsedFunction<dim> boundary_stress_function;
      };

    private:
      Parameters parameters;

      Simulator<dim> &sim;
  };
}

#endif
