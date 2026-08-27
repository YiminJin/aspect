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

#ifndef _aspect_material_model_phase_field_rsf_h
#define _aspect_material_model_phase_field_rsf_h

#include <aspect/simulator_access.h>
#include <aspect/phase_field.h>
#include <aspect/solution_evaluator.h>
#include <aspect/material_model/interface.h>
#include <aspect/material_model/equation_of_state/multicomponent_incompressible.h>
#include <aspect/material_model/rheology/rate_state_friction.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    class PhaseFieldRSF : public Interface<dim>,
      public PhaseFieldModel<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void 
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                 MaterialModel::MaterialModelOutputs<dim> &out) const override;

        std::vector<double>
        get_critical_crack_driving_forces() const override;

        std::vector<double>
        get_critical_energy_release_rates() const override;
        
        std::pair<double, double>
        get_phase_field_range() const override;

        bool
        is_compressible() const override;

        static
        void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm) override;

      private:
        double 
        calculate_creep_viscosity(const std::vector<double> &volume_fractions,
                                  const double               temperature) const;

        double
        calculate_stress_relaxation_factor(const double creep_viscosity,
                                           const double shear_modulus) const;

        EquationOfState::MulticomponentIncompressible<dim> equation_of_state;

        Rheology::RateStateFriction<dim> rsf_rheology;

        MaterialUtilities::CompositionalAveragingOperation viscosity_averaging;

        double reference_temperature;

        double maximum_viscosity;
        
        double minimum_viscosity;

        double phase_field_activation_threshold;

        double phase_field_normal_lock_threshold;

        std::vector<double> thermal_conductivities;

        std::vector<double> reference_viscosities;

        std::vector<double> thermal_viscosity_exponents;

        std::vector<double> elastic_shear_moduli;

        std::vector<double> cohesions;

        std::vector<double> initial_friction_coefficients;

        std::vector<double> critical_energy_release_rates;

        std::vector<double> radiation_damping_coefficients;

        double initial_time_step;

        bool evolve_phase_field;

        std::unique_ptr<SolutionEvaluator<dim>> solution_evaluator;
    };
  }
}

#endif
