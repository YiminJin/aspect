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

#include <deal.II/grid/grid_tools.h>
#include <deal.II/particles/particle_handler.h>

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
         * Returns the critical crack driving forces for background
         * material and compositional fields.
         */
        virtual
        std::vector<double>
        get_critical_crack_driving_forces() const = 0;

        /**
         * Returns the critical energy release rates for background
         * material and compositional fields.
         */
        virtual
        std::vector<double>
        get_critical_energy_release_rates() const = 0;

        /**
         * Returns the lower and upper bounds of the phase-field.
         */
        virtual
        std::pair<double, double>
        get_phase_field_range() const;
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
        GeometricFunction(const double l,
                          const double xi,
                          const double c0);

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

        /**
         * Return the length scale $l$.
         */
        double get_length_scale() const;

      /**
       * Return the normalization factor $c_0$.
       */
      double get_normalization_factor() const;

      private:
        const double l;
        const double xi;
        const double c0;
    };

    inline double 
    GeometricFunction::get_length_scale() const
    {
      return l;
    }

    inline double
    GeometricFunction::get_normalization_factor() const
    {
      return c0;
    }


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
        const double p;
        const double m;
    };

    class PhaseFieldProfile
    {
      public:
        /**
         * Constructor.
         */
        PhaseFieldProfile(const GeometricFunction   &a_func,
                          const DegradationFunction &g_func,
                          const double               phi_hat,
                          const unsigned int         n_points = 5000);

        /**
         * Given the distance from the crack center, return the phase-field
         * value.
         */
        double value(const double zeta) const;

        const std::vector<double> &get_coordinate_values() const;

        const std::vector<double> &get_phase_field_values() const;

      private:
        const unsigned int N;
        std::vector<double> coordinate_values;
        std::vector<double> phase_field_values;
    };

    inline const std::vector<double> &
    PhaseFieldProfile::get_coordinate_values() const
    {
      return coordinate_values;
    }

    inline const std::vector<double> &
    PhaseFieldProfile::get_phase_field_values() const
    {
      return phase_field_values;
    }


    class SlipRateNormalizer
    {
      public:
        /**
         * Number of sample points.
         */
        static constexpr unsigned int M = 100;
        
        /**
         * Number of subdivisions for the trapezoidal rule.
         */
        static constexpr unsigned int N = 1000;

        /**
         * Constructor.
         */
        SlipRateNormalizer(const GeometricFunction   &a_func,
                           const DegradationFunction &g_func,
                           const double               phi_max,
                           const double               phi_min);

        /**
         * Given the peak phase-field value, returns the normalization factor.
         */
        double normalization_factor(const double peak_value) const;

      private:
        const double phi_min;
        const double phi_max;
        std::array<double, M> logit_phi_hat;
        std::array<double, M> log_Ih;
    };

    struct SolverParameters
    {
      double linear_solver_tolerance;
      unsigned int max_linear_solver_iterations;
      double nonlinear_solver_tolerance;
      unsigned int max_nonlinear_iterations;
      unsigned int max_newton_line_search_iterations;
    };
  }

  namespace PhaseFieldUtilities
  {
    template <int dim>
    std::vector<std::pair<double, std::vector<double>>>
    interpolate_from_particles_in_crack_zone(const Particles::ParticleHandler<dim> &particle_handler,
                                             const std::vector<Point<dim>>         &positions,
                                             const ComponentMask                   &selected_properties,
                                             const typename Triangulation<dim>::active_cell_iterator &target_cell,
                                             const GridTools::Cache<dim>           &grid_cache,
                                             const std::function<bool(const ArrayView<const double>&)> &is_in_crack_zone);

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

      void
      evolve_phase_field(LinearAlgebra::BlockSparseMatrix &system_matrix,
                         LinearAlgebra::BlockVector       &system_rhs,
                         LinearAlgebra::BlockVector       &solution);

      double crack_surface_density(const double          phase_field_value,
                                   const Tensor<1, dim> &phase_field_gradient) const;

      double
      energetic_degradation(const std::vector<double> &volume_fractions,
                            const double               phase_field_value) const;

      double
      slip_rate_localization_factor(const std::vector<double> &volume_fractions,
                                    const double degradation_function,
                                    const double core_phase_field) const;

      double 
      stationary_crack_driving_force(const std::vector<double> &volume_fractions,
                                     const double               phase_field,
                                     const double               core_phase_field) const;

      std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>>
      get_phase_field_profiles(const double core_phase_field) const;

      const Particle::Manager<dim> &get_associated_particle_manager() const;

      Particle::Manager<dim> &get_associated_particle_manager();

      const GridTools::Cache<dim> &get_grid_cache() const;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);

      mutable boost::signals2::signal<void (const SimulatorAccess<dim> &)> pre_extend_core_phase_field;

    private:
      void
      assemble_phase_field_system(LinearAlgebra::BlockSparseMatrix &system_matrix,
                                  LinearAlgebra::BlockVector       &system_rhs,
                                  const LinearAlgebra::BlockVector &current_solution,
                                  const bool assemble_system_jacobian) const;

      unsigned int
      solve_phase_field_system(const LinearAlgebra::BlockSparseMatrix &system_matrix,
                               const LinearAlgebra::BlockVector       &system_rhs,
                               LinearAlgebra::BlockVector             &solution_vector) const;

      PhaseField::SolverParameters solver_parameters;

      std::unique_ptr<PhaseField::GeometricFunction> geometric_function;

      std::vector<std::unique_ptr<PhaseField::DegradationFunction>> degradation_functions;

      std::vector<std::unique_ptr<PhaseField::SlipRateNormalizer>> slip_rate_normalizers;

      std::vector<double> critical_energy_densities;

      Particle::Manager<dim> *particle_manager;

      std::unique_ptr<GridTools::Cache<dim>> grid_cache;

      std::vector<types::global_dof_index> vertex_to_dof_indices;
  };
}

#endif
