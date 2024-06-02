/*
  Copyright (C) 2019 - 2023 by the authors of the ASPECT code.

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

#ifndef _aspect_material_model_rheology_pariseau_h
#define _aspect_material_model_rheology_pariseau_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    namespace Rheology
    {
      /**
       * Data structure for Pariseau parameters.
       */
      struct PariseauParameters
      {
        double F;
        double G;
        double M;
        double U;
        double V;
        double n;
      };

      template <int dim>
      class Pariseau : public SimulatorAccess<dim>
      {
        public:
          Pariseau();

          /**
           * Declare the parameters this function takes through input files.
           */
          static
          void
          declare_parameters(ParameterHandler &prm);
          
          /**
           * Read the parameters this class declares from the parameter file.
           * If @p expected_n_phases_per_composition points to a vector of
           * unsigned integers, this is considered the number of phases
           * for each compositional field (plus possibly a background field)
           * and this number will be checked against the parsed parameters.
           *
           * @param [in] prm The ParameterHandler to read from.
           * @param expected_n_phases_per_composition Optional list of number of phases.
           */
          void
          parse_parameters(ParameterHandler &prm,
                           const std::unique_ptr<std::vector<unsigned int>> &expected_n_phases_per_composition = nullptr);

          /**
           * Compute the parameters for the Pariseau criterion.
           * If @p n_phase_transitions_per_composition points to a vector of
           * unsigned integers this is considered the number of phase transitions
           * for each compositional field and viscosity will be first computed on
           * each phase and then averaged for each compositional field.
           */
          const PariseauParameters
          compute_pariseau_parameters(const unsigned int composition,
                                      const bool isotropization = false,
                                      const std::vector<double> &phase_function_values = std::vector<double>(),
                                      const std::vector<unsigned int> &n_phase_transitions_per_composition = std::vector<unsigned int>()) const;

          /**
           * Compute the apparent viscosity using the Periseau criterion. 
           * The function returns the value of viscosity and a boolean that 
           * indicates if plastic yielding has occurred or not.
           */
          std::pair<double,bool>
          compute_viscosity(const SymmetricTensor<2,dim>    &effective_strain_rate,
                            const Tensor<1,dim>             &normal_vector,
                            const double                     pressure,
                            const double                     non_yielding_viscosity,
                            const unsigned int               composition,
                            const std::array<double,3>      &weakening_factors,
                            const std::vector<double>       &phase_function_values = std::vector<double>(),
                            const std::vector<unsigned int> &n_phase_transitions_per_composition = std::vector<unsigned int>()) const;

          /**
           * Update the normal vector of the foliation planes.
           */
          void
          fill_reaction_outputs(const MaterialModel::MaterialModelInputs<dim> &in,
                                MaterialModel::MaterialModelOutputs<dim> &out) const;

        private:
          /**
           * List of material constants F;
           */
          std::vector<double> F;

          /**
           * List of material constants G;
           */
          std::vector<double> G;

          /**
           * List of material constants M;
           */
          std::vector<double> M;

          /**
           * List of material constants U;
           */
          std::vector<double> U;

          /**
           * List of material constants V;
           */
          std::vector<double> V;

          /**
           * List of the exponent coefficients n.
           */
          std::vector<double> exponent_coefficients;

          /**
           * Whether to add a plastic damper in the computation
           * of the plastic viscosity.
           */
          bool use_plastic_damper;

          /**
           * Viscosity of a damper used to stabilize plasticity.
           */
          double damper_viscosity;

          /**
           * We cache the evaluator that is necessary to evaluate the old velocity
           * gradients. They are required to compute the time derivative of the 
           * normal vector, but are not provided by the material model.
           * By caching the evaluator, we can avoid recreating it every time we
           * need it.
           */
          mutable std::unique_ptr<FEPointEvaluation<dim, dim>> evaluator;

      };
    }
  }
}

#endif
