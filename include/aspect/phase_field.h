/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
 */

#ifndef _aspect_phase_field_h
#define _aspect_phase_field_h

#include <aspect/global.h>
#include <aspect/simulator_access.h>
#include <aspect/fe_variable_collection.h>
#include <aspect/material_model/interface.h>

namespace aspect
{
  namespace MaterialModel
  {
    /**
     * Base class for material models to be used with phase field method.
     */
    template <int dim>
    class PhaseFieldModel
    {
      public:
        virtual ~PhaseFieldModel() = default;

        /**
         * Returns the threshold crack driving force ($H_t$) for background
         * material and compositional fields.
         */
        virtual
        std::vector<double>
        get_threshold_crack_driving_forces() const = 0;

        /**
         * Returns the critical energy release rate ($G_c$) for background
         * material and compositional fields.
         */
        virtual
        std::vector<double>
        get_critical_energy_release_rates() const = 0;
    };
  }

  namespace PhaseField
  {
    /**
     * The geometric function $\alpha(\phi)$ in the generic form of the crack
     * surface density functional.
     */
    class GeometricFunction
    {
      public:
        /**
         * Constructor. The generic form of the geometric function is given by
         * $\xi\phi + (1-\xi)phi^2$, where the parameter $\xi$ is in the range
         * of [0,2].
         */
        GeometricFunction(const double xi);

        /**
         * Given the phase-field $\phi$, return the value of the geometric 
         * function, $\alpha(\phi)$.
         */
        double value(const double phase_field) const;

        /**
         * Given the phase-field $\phi$, return the first derivative of the 
         * geometric function, $\alpha'(\phi)$. 
         */
        double first_derivative(const double phase_field) const;

        /**
         * Given the phase-field $\phi$, return the second derivative of the
         * geometric function, $\alpha''(\phi)$.
         */
        double second_derivative(const double phase_field) const;

      private:
        double xi;
    };

    /**
     * The energetic degradation function $g(\phi)$.
     */
    class DegradationFunction
    {
      public:
        /**
         * Constructor. 
         */
        DegradationFunction(const double p,
                            const double m);

        /**
         * Given the phase-field $\phi$, return the value of the degradation 
         * function, $g(\phi)$.
         */
        double value(const double phase_field) const;

        /**
         * Given the phase-field $\phi$, return the first derivative of the 
         * degradation function, $g'(\phi)$.
         */
        double first_derivative(const double phase_field) const;

        /**
         * Given the phase-field $\phi$, return the second derivative of the 
         * degradation function, $g''(\phi)$.
         */
        double second_derivative(const double phase_field) const;

      private:
        double p;
        double m;
    };

    template <int dim>
    struct Parameters
    {
      double length_scale;

      double geometric_normalization_parameter;

      double degradation_curvature_parameter;

      double linear_solver_tolerance;

      unsigned int max_linear_solver_iterations;

      double nonlinear_solver_tolerance;

      unsigned int max_nonlinear_iterations;

      unsigned int max_newton_line_search_iterations;
    };
  }

  template <int dim>
  class PhaseFieldHandler : public SimulatorAccess<dim>
  {
    public:
      PhaseFieldHandler(const Simulator<dim> &sim);

      /**
       * Add the phase fields to the list of variables, which will be used later
       * to set up the introspection object.
       */
      void edit_finite_element_variables(std::vector<VariableDeclaration<dim>> &variables);

      void initialize();

      void
      make_sparsity_pattern(LinearAlgebra::BlockDynamicSparsityPattern &sp);

      double
      solve_timestep(LinearAlgebra::BlockSparseMatrix &system_matrix,
                     LinearAlgebra::BlockVector       &system_rhs,
                     LinearAlgebra::BlockVector       &solution);

      double crack_surface_density(const double          phase_field_value,
                                   const Tensor<1, dim> &phase_field_gradient) const;

      double
      energetic_degradation(const double               phase_field_value,
                            const std::vector<double> &volume_fractions) const;

      const Particle::Manager<dim> &get_associated_particle_manager() const;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);

    private:
      double compute_microforce(const double               phase_field_value,
                                const double               driving_force,
                                const std::vector<double> &volume_fractions) const;

      double compute_microforce_derivative(const double               phase_field_value,
                                           const double               driving_force,
                                           const std::vector<double> &volume_fractions) const;

      double compute_microstress_prefactor(const std::vector<double> &volume_fractions) const;

      void
      assemble_linearized_system(LinearAlgebra::BlockSparseMatrix &system_matrix,
                                 LinearAlgebra::BlockVector       &system_rhs,
                                 const LinearAlgebra::BlockVector &current_solution,
                                 const bool assemble_system_jacobian) const;

      unsigned int
      solve_linearized_system(const LinearAlgebra::BlockSparseMatrix &system_matrix,
                              const LinearAlgebra::BlockVector       &system_rhs,
                              LinearAlgebra::BlockVector             &solution_vector) const;

      double 
      evaluate_merit_function(const LinearAlgebra::BlockVector &test_solution) const;

      struct SystemInformation
      {
        void initialize(const Introspection<dim>     &introspection,
                        const Parameters<dim>        &parameters,
                        const Particle::Manager<dim> &particle_manager);

        unsigned int phase_field_component_index;

        unsigned int phase_field_block_index;

        BlockIndices block_indices;

        struct ParticleDataPositions
        {
          unsigned int              crack_driving_force;
          std::vector<unsigned int> chemical_fields;
        };

        ParticleDataPositions particle_data_positions;
      };

      SystemInformation system_info;

      PhaseField::Parameters<dim> parameters;

      std::unique_ptr<PhaseField::GeometricFunction> geometric_function;

      std::vector<std::unique_ptr<PhaseField::DegradationFunction>> degradation_functions;

      std::vector<double> critical_energy_densities;

      const Particle::Manager<dim> *particle_manager;

      std::vector<types::global_dof_index> vertex_to_dof_indices;
  };
}

#endif
