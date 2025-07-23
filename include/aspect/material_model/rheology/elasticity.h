/*
  Copyright (C) 2019 - 2024 by the authors of the ASPECT code.

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

#ifndef _aspect_material_model_rheology_elasticity_h
#define _aspect_material_model_rheology_elasticity_h

#include <aspect/global.h>
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

namespace aspect
{
  namespace MaterialModel
  {
    /**
     * Additional inputs for elastic rheology. This class provides 
     * the velocity gradient tensor at each evaluation point, which
     * is used for computing the rotation terms in the objective
     * stress rate.
     */
    template <int dim>
    class ElasticAdditionalInputs : public AdditionalMaterialInputs<dim>
    {
      public:
        /**
         * Constructor. Resize the array of velocity gradient tensors
         * with the given number of evaluation points, and initialize 
         * each entry with signaling NaN.
         */
        ElasticAdditionalInputs (const unsigned int n_points);

        /**
         * Velocity gradient tensors at the given positions.
         */
        std::vector<Tensor<2,dim>> velocity_gradients;

        /**
         * Fill the velocity gradient tensors.
         */
        void fill (const LinearAlgebra::BlockVector &solution,
                   const FEValuesBase<dim>          &fe_values,
                   const Introspection<dim>         &introspection) override;

        void fill (const DataPostprocessorInputs::Vector<dim> &data,
                   const Introspection<dim> &introspection) override;
    };


    /**
     * Additional output fields for the elastic shear modulus and other
     * elastic outputs to be added to the MaterialModel::MaterialModelOutputs
     * structure and filled in the MaterialModel::Interface::evaluate() function.
     */
    template <int dim>
    class ElasticAdditionalOutputs : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        explicit ElasticAdditionalOutputs(const unsigned int n_points);

        std::vector<double> get_nth_output(const unsigned int idx) const override;

        /**
         * Elastic shear moduli at the evaluation points passed to
         * the instance of MaterialModel::Interface::evaluate() that fills
         * the current object.
         */
        std::vector<double> elastic_shear_moduli;

        /**
         * The deviatoric stress of the current timestep.
         */
        std::vector<SymmetricTensor<2,dim>> deviatoric_stress;
    };



    namespace Rheology
    {
      template <int dim>
      class Elasticity : public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Declare the parameters this function takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm);

          void
          create_elastic_additional_inputs (MaterialModel::MaterialModelInputs<dim> &in) const;

          /**
           * Create the two additional material model output objects that contain the
           * elastic shear moduli, elastic viscosity, ratio of computational to elastic timestep,
           * and deviatoric stress of the current timestep and the reaction rates.
           */
          void
          create_elastic_additional_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * Given the stress of the previous time step in the material model inputs @p in,
           * the elastic shear moduli @p elastic_shear_moduli, the creep viscosities
           * @p creep_viscosities, the yield stresses @p yield_stresses, and the 
           * effective viscosities given in the material model outputs object @p out, fill a 
           * material model outputs object with the elastic force terms, viscoelastic strain 
           * rate and viscous dissipation.
           */
          void
          fill_elastic_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                                const std::vector<std::vector<double>> &volume_fractions,
                                const std::vector<std::vector<double>> &creep_viscosities,
                                const std::vector<std::vector<double>> &yield_stresses,
                                MaterialModel::MaterialModelOutputs<dim> &out) const;

          void
          fill_elastic_additional_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                                           const std::vector<std::vector<double>> &volume_fractions,
                                           const std::vector<std::vector<double>> &creep_viscosities,
                                           const std::vector<std::vector<double>> &yield_stresses,
                                           MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * Given the stress of the previous time step in the material model inputs @p in,
           * the elastic shear moduli @p elastic_shear_moduli, the creep viscosities
           * @p creep_viscosities, the yield stresses @p yield_stresses, and the 
           * effective viscosities given in the material model outputs object &p out, compute
           * the update to the elastic stresses of the previous timestep and use it to fill 
           * the reaction rates material model output property in @p out.
           */
          void
          fill_reaction_rates (const MaterialModel::MaterialModelInputs<dim> &in,
                               const std::vector<std::vector<double>> &volume_fractions,
                               const std::vector<std::vector<double>> &creep_viscosities,
                               const std::vector<std::vector<double>> &yield_stresses,
                               MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * Return the values of the elastic shear moduli for each composition used in the
           * rheology model.
           */
          const std::vector<double> &
          get_elastic_shear_moduli () const;

          /**
           * Given the (viscous or visco-plastic) viscosity and the shear modulus, compute the viscoelastic
           * viscosity (eqn 28 in Moresi et al., 2003, J. Comp. Phys.).
           */
          double
          calculate_viscoelastic_viscosity (const double viscosity,
                                            const double shear_modulus) const;

          /**
           * Calculate the effective deviatoric strain rate tensor,
           * which equals the true deviatoric strain rate plus
           * a fictional strain rate which would arise from stored elastic stresses.
           * In ASPECT, this additional strain rate is
           * supported by a fictional body force.
           * This formulation allows the use of an isotropic effective viscosity
           * by ensuring that the resulting strain rate tensor is equal to the
           * total current stress tensor multiplied by a scalar.
           *
           * Stress tensor components @p stress_0_advected represent the stress from the previous
           * timestep $t$ rotated and advected into the current timestep $t+\Delta t_c$.
           * Stress tensor components @p stress_old represent the stress from the previous
           * timestep $t$ advected into the current timestep $t+\Delta t_c$.
           * By the time the viscoelastic strain rate is required to assemble
           * the Stokes system, the stresses have already been rotated and/or advected.
           */
          SymmetricTensor<2,dim>
          calculate_viscoelastic_strain_rate (const unsigned int i,
                                              const MaterialModel::MaterialModelInputs<dim> &in,
                                              const double creep_viscosity,
                                              const double shear_modulus,
                                              const double elastic_timestep) const;

          /**
           * Compute the elastic time step.
           */
          double
          elastic_timestep () const;

        private:
          /**
           * Viscosity of a damper used to stabilize elasticity.
           * A value of 0 Pas is equivalent to not using a damper.
           */
          double elastic_damper_viscosity;

          /**
           * Vector for field elastic shear moduli, read from parameter file.
           */
          std::vector<double> elastic_shear_moduli;

          /**
           * Bool indicating whether to use a fixed material time scale in the
           * viscoelastic rheology for all time steps (if true) or to use the
           * actual (variable) advection time step of the model (if false). Read
           * from parameter file.
           */
          bool use_fixed_elastic_time_step;

          /**
           * Double for fixed elastic time step value, read from parameter file.
           */
          double fixed_elastic_time_step;

          /**
           * A stabilization factor for the elastic stresses that influences how
           * fast elastic stresses adjust to deformation. 1.0 is equivalent to no
           * stabilization, and infinity is equivalent to not applying elastic
           * stresses at all. The factor is multiplied with the computational
           * time step to create a time scale.
           */
          double stabilization_time_scale_factor;

          static constexpr unsigned int n_independent_components = SymmetricTensor<2, dim>::n_independent_components;
      };
    }
  }
}
#endif
