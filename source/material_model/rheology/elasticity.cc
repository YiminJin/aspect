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


#include <aspect/material_model/rheology/elasticity.h>

#include <aspect/utilities.h>
#include <aspect/material_model/visco_plastic.h>
#include <aspect/material_model/viscoelastic.h>
#include <aspect/heating_model/shear_heating.h>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    ElasticAdditionalInputs<dim>::
    ElasticAdditionalInputs (const unsigned int n_points)
      : velocity_gradients(n_points, numbers::signaling_nan<Tensor<2,dim>>())
    {}



    template <int dim>
    void
    ElasticAdditionalInputs<dim>::
    fill (const LinearAlgebra::BlockVector &solution,
          const FEValuesBase<dim>          &fe_values,
          const Introspection<dim>         &introspection)
    {
      AssertDimension(velocity_gradients.size(), fe_values.n_quadrature_points);
      fe_values[introspection.extractors.velocities].get_function_gradients(solution, velocity_gradients);
    }



    template <int dim>
    void
    ElasticAdditionalInputs<dim>::
    fill (const DataPostprocessorInputs::Vector<dim> &data,
          const Introspection<dim> &introspection)
    {
      const unsigned int n_points = data.solution_values.size();
      AssertDimension(velocity_gradients.size(), n_points);

      for (unsigned int q = 0; q < n_points; ++q)
        for (unsigned int d = 0; d < dim; ++d)
          velocity_gradients[q][d] = data.solution_gradients[q][introspection.component_indices.velocities[d]];
    }



    namespace
    {
      std::vector<std::string> make_elastic_additional_outputs_names()
      {
        std::vector<std::string> names;
        names.emplace_back("elastic_shear_modulus");
        return names;
      }
    }

    template <int dim>
    ElasticAdditionalOutputs<dim>::ElasticAdditionalOutputs (const unsigned int n_points)
      :
      NamedAdditionalMaterialOutputs<dim>(make_elastic_additional_outputs_names()),
      elastic_shear_moduli(n_points, numbers::signaling_nan<double>()),
      deviatoric_stress(n_points, numbers::signaling_nan<SymmetricTensor<2,dim>>())
    {}



    template <int dim>
    std::vector<double>
    ElasticAdditionalOutputs<dim>::get_nth_output(const unsigned int idx) const
    {
      (void)idx; // suppress warning in release mode
      AssertIndexRange (idx, 1);
      return elastic_shear_moduli;
    }



    namespace Rheology
    {
      template <int dim>
      void
      Elasticity<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.declare_entry ("Elastic shear moduli", "75.0e9",
                           Patterns::List(Patterns::Double(0.)),
                           "List of elastic shear moduli, $G$, "
                           "for background material and compositional fields, "
                           "for a total of N+1 values, where N is the number of all compositional fields or only "
                           "those corresponding to chemical compositions. "
                           "The default value of 75 GPa is representative of mantle rocks. Units: Pa.");
        prm.declare_entry ("Use fixed elastic time step", "unspecified",
                           Patterns::Selection("true|false|unspecified"),
                           "Select whether the material time scale in the viscoelastic constitutive "
                           "relationship uses the regular numerical time step or a separate fixed "
                           "elastic time step throughout the model run. The fixed elastic time step "
                           "is always used during the initial time step. If a fixed elastic time "
                           "step is used throughout the model run, a stress averaging scheme is "
                           "applied to account for differences with the numerical time step. An "
                           "alternative approach is to limit the maximum time step size so that it "
                           "is equal to the elastic time step. The default value of this parameter is "
                           "'unspecified', which throws an exception during runtime. In order for "
                           "the model to run the user must select 'true' or 'false'.");
        prm.declare_entry ("Fixed elastic time step", "1.e3",
                           Patterns::Double (0.),
                           "The fixed elastic time step $dte$. It is always used during the first "
                           "timestep; afterwards on if 'Used fixed elastic time step' is true. "
                           "Units: years if the 'Use years in output instead of seconds' parameter is set; "
                           "seconds otherwise.");
        prm.declare_entry ("Stabilization time scale factor", "1.",
                           Patterns::Double (1.),
                           "A stabilization factor for the elastic stresses that influences how fast "
                           "elastic stresses adjust to deformation. This value is equal to the "
                           "elastic time step divided by the computational time step. "
                           "The default value of 1.0 may lead to oscillatory motion. "
                           "Increasing this factor to 2.0 can reduce oscillations while "
                           "preserving an immediate elastic response. In complex models the factor "
                           "can be increased further to improve convergence behaviour. "
                           "As the stabilization factor increases, the effective viscosity "
                           "gets smaller, and is balanced by an increasing body force term. "
                           "For composite rheologies that use this formulation of elasticity, "
                           "setting an infinite shear modulus only recovers the nonelastic part of "
                           "the rheology if this stabilization factor is equal to 1.0.");
        prm.declare_entry ("Elastic damper viscosity", "0.0",
                           Patterns::Double (0.),
                           "Viscosity of a viscous damper that acts in parallel with the elastic "
                           "element to stabilize behavior. Units: \\si{\\pascal\\second}");
      }



      template <int dim>
      void
      Elasticity<dim>::parse_parameters (ParameterHandler &prm)
      {
        AssertThrow(this->get_parameters().enable_elasticity == true,
                    ExcMessage("Rheology model elasticity only works if 'Enable elasticity' is set to true"));

        // Retrieve the list of composition names
        std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();

        // Retrieve the list of names of fields that represent chemical compositions, and not, e.g.,
        // plastic strain
        std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();

        // Establish that a background field is required here
        compositional_field_names.insert(compositional_field_names.begin(), "background");
        chemical_field_names.insert(chemical_field_names.begin(),"background");

        Utilities::MapParsing::Options options(chemical_field_names, "Elastic shear moduli");
        options.list_of_allowed_keys = compositional_field_names;

        elastic_shear_moduli = Utilities::MapParsing::parse_map_to_double_array (prm.get("Elastic shear moduli"),
                                                                                 options);

        // The elastic shear moduli must be positive
        bool shear_moduli_are_positive = true;
        for (unsigned int i = 0; i < elastic_shear_moduli.size(); ++i)
          shear_moduli_are_positive = (elastic_shear_moduli[i] > 0.0);
        AssertThrow(shear_moduli_are_positive,
                    ExcMessage("The values of elastic shear moduli must be positive."));

        // Stabilize elasticity through a viscous damper
        elastic_damper_viscosity = prm.get_double("Elastic damper viscosity");

        if (prm.get ("Use fixed elastic time step") == "true")
          use_fixed_elastic_time_step = true;
        else if (prm.get ("Use fixed elastic time step") == "false")
          use_fixed_elastic_time_step = false;
        else
          AssertThrow(false, ExcMessage("'Use fixed elastic time step' must be set to 'true' or 'false'"));

        stabilization_time_scale_factor = prm.get_double ("Stabilization time scale factor");

        fixed_elastic_time_step = prm.get_double ("Fixed elastic time step");
        AssertThrow(fixed_elastic_time_step > 0,
                    ExcMessage("The fixed elastic time step must be greater than zero"));

        if (this->convert_output_to_years())
          fixed_elastic_time_step *= year_in_seconds;

        // When using the visco_plastic or viscoelastic material model,
        // make sure that no damping is applied. Damping could potentially
        // improve stability under rapidly changing dynamics, but
        // so far it has not been necessary.
        AssertThrow(elastic_damper_viscosity == 0.,
                    ExcMessage("The viscoelastic material model and the visco-plastic material model with elasticity enabled require "
                               "that no elastic damping is applied."));
        // An update of the stored stresses is done in an operator splitting step for fields or by the particle property 'elastic stress'.
        AssertThrow(this->get_parameters().use_operator_splitting || (this->get_parameters().mapped_particle_properties).count(this->introspection().compositional_index_for_name("ve_stress_xx")),
                    ExcMessage("The viscoelastic material model and the visco-plastic material model with elasticity enabled require "
                               "operator splitting for stresses tracked on compositional fields or the particle property 'elastic stress' "
                               "for stresses tracked on particles."));
        // If the operator splitting scheme is used, make sure to use its fixed step solver, as we know the update and it should be applied in one step.
        if (this->get_parameters().use_operator_splitting)
          AssertThrow(this->get_parameters().reaction_solver_type == Parameters<dim>::ReactionSolverType::fixed_step,
                      ExcMessage("If the operator splitting scheme is used, its solver should be set to 'fixed step'."));
        if ((this->get_parameters().mapped_particle_properties).count(this->introspection().compositional_index_for_name("ve_stress_xx")))
          AssertThrow(!this->get_parameters().use_operator_splitting,
                      ExcMessage("If stresses are tracked on particles, the stress update is applied by the particle property 'elastic stress' "
                                 "and operator splitting should not be turned on. "));

        // Check that 3+3 in 2D or 6+6 in 3D stress fields exist.
        AssertThrow((this->introspection().get_number_of_fields_of_type(CompositionalFieldDescription::stress) == SymmetricTensor<2,dim>::n_independent_components),
                    ExcMessage("Rheology model Elasticity requires 3 in 2D or 6 in 3D fields of type stress."));

        // Check that the compositional fields representing the viscoelastic
        // stress tensor components are both named correctly and listed in the right order
        // as well as use a discontinuous discretization.
        std::vector<std::string> stress_field_names = this->introspection().get_names_for_fields_of_type(CompositionalFieldDescription::stress);
        std::vector<unsigned int> stress_field_indices = this->introspection().get_indices_for_fields_of_type(CompositionalFieldDescription::stress);

        // The discontinuous element is required to accommodate discontinuous
        // strain rates that feed into the stored stresses.
        const std::vector<bool> use_discontinuous_composition_discretization = this->get_parameters().use_discontinuous_composition_discretization;
        for (auto stress_field : stress_field_indices)
          AssertThrow(use_discontinuous_composition_discretization[stress_field],
                      ExcMessage("The viscoelastic material model and the visco-plastic material model with elasticity enabled require "
                                 "the use of discontinuous elements for compositions that represent stress tensor components."));

        // We require a consecutive range of indices (for example for FEPointEvaluation)
        // to extract the fields representing the viscoelastic stress tensor components,
        // so check that they are listed without interruption by other fields.
        // They do not, however, have to be the first fields listed.
        AssertThrow((stress_field_indices[n_independent_components-1] - stress_field_indices[0]) == n_independent_components-1,
                    ExcMessage("Rheology model Elasticity requires that the compositional fields representing stress tensor components are listed in consecutive order."));

        AssertThrow(stress_field_names[0] == "ve_stress_xx",
                    ExcMessage("Rheology model Elasticity only works if the first "
                               "compositional field representing stress tensor components is called ve_stress_xx."));
        AssertThrow(stress_field_names[1] == "ve_stress_yy",
                    ExcMessage("Rheology model Elasticity only works if the second "
                               "compositional field representing stress tensor components is called ve_stress_yy."));
        if (dim == 2)
          {
            AssertThrow(stress_field_names[2] == "ve_stress_xy",
                        ExcMessage("Rheology model Elasticity only works if the third "
                                   "compositional field representing stress tensor components is called ve_stress_xy."));
          }
        else if (dim == 3)
          {
            AssertThrow(stress_field_names[2] == "ve_stress_zz",
                        ExcMessage("Rheology model Elasticity only works if the third "
                                   "compositional field representing stress tensor components is called ve_stress_zz."));
            AssertThrow(stress_field_names[3] == "ve_stress_xy",
                        ExcMessage("Rheology model Elasticity only works if the fourth "
                                   "compositional field representing stress tensor components is called ve_stress_xy."));
            AssertThrow(stress_field_names[4] == "ve_stress_xz",
                        ExcMessage("Rheology model Elasticity only works if the fifth "
                                   "compositional field representing stress tensor components is called ve_stress_xz."));
            AssertThrow(stress_field_names[5] == "ve_stress_yz",
                        ExcMessage("Rheology model Elasticity only works if the sixth "
                                   "compositional field representing stress tensor components is called ve_stress_yz."));
          }
        else
          AssertThrow(false, ExcNotImplemented());

        // Functionality to average the additional RHS terms over the cell is not implemented.
        // Also, there is no option implemented in this rheology module to project to Q1 the viscosity
        // in the elastic force term for the RHS.
        // Consequently, it is only possible to use elasticity with the Material averaging schemes
        // 'none', 'harmonic average only viscosity', and 'geometric average only viscosity'.
        // TODO: Find a way to include 'project to Q1 only viscosity'.
        AssertThrow((this->get_parameters().material_averaging == MaterialModel::MaterialAveraging::none
                     ||
                     this->get_parameters().material_averaging == MaterialModel::MaterialAveraging::harmonic_average_only_viscosity
                     ||
                     this->get_parameters().material_averaging == MaterialModel::MaterialAveraging::geometric_average_only_viscosity
                     ||
                     this->get_parameters().material_averaging == MaterialModel::MaterialAveraging::default_averaging),
                    ExcMessage("Material models with elasticity can only be used with the material "
                               "averaging schemes 'none', 'harmonic average only viscosity' and "
                               "'geometric average only viscosity'. This parameter ('Material averaging') "
                               "is located within the 'Material model' subsection."));
      }



      template <int dim>
      void
      Elasticity<dim>::create_elastic_additional_inputs (MaterialModel::MaterialModelInputs<dim> &in) const
      {
        if (in.requests_property(MaterialProperties::additional_outputs) ||
            in.requests_property(MaterialProperties::reaction_rates))
          {
            const unsigned int n_points = in.n_evaluation_points();
            if (in.template has_additional_input_object<ElasticAdditionalInputs<dim>>() == false)
              in.additional_inputs.push_back(std::make_unique<ElasticAdditionalInputs<dim>>(n_points));
          }
      }



      template <int dim>
      void
      Elasticity<dim>::create_elastic_additional_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        // Create the ElasticAdditionalOutputs that include the average shear modulus and
        // deviatoric stress of the current timestep.
        if (out.template has_additional_output_object<ElasticAdditionalOutputs<dim>>() == false)
          {
            const unsigned int n_points = out.n_evaluation_points();
            out.additional_outputs.push_back(
              std::make_unique<ElasticAdditionalOutputs<dim>> (n_points));
          }

        // We need to modify the shear heating outputs to correctly account for elastic stresses.
        if (out.template has_additional_output_object<HeatingModel::PrescribedShearHeatingOutputs<dim>>() == false)
          {
            const unsigned int n_points = out.n_evaluation_points();
            out.additional_outputs.push_back(
              std::make_unique<HeatingModel::PrescribedShearHeatingOutputs<dim>> (n_points));
          }

        // Create the ReactionRateOutputs that are necessary for the operator splitting
        // step (either on the fields or directly on the particles)
        // that sets both sets of stresses to the total stress of the
        // previous timestep.
        if (out.template has_additional_output_object<ReactionRateOutputs<dim>>() == false &&
            (this->get_parameters().use_operator_splitting || (this->get_parameters().mapped_particle_properties).count(this->introspection().compositional_index_for_name("ve_stress_xx"))))
          {
            const unsigned int n_points = out.n_evaluation_points();
            out.additional_outputs.push_back(
              std::make_unique<MaterialModel::ReactionRateOutputs<dim>>(n_points, this->n_compositional_fields()));
          }
      }



      template <int dim>
      void
      Elasticity<dim>::fill_elastic_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                                             const std::vector<std::vector<double>> &volume_fractions,
                                             const std::vector<std::vector<double>> &creep_viscosities,
                                             const std::vector<std::vector<double>> &yield_stresses,
                                             MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        // Create a reference to the structure for the elastic outputs.
        // The structure is created during the Stokes assembly.
        const std::shared_ptr<MaterialModel::ElasticOutputs<dim>>
        elastic_out = out.template get_additional_output_object<MaterialModel::ElasticOutputs<dim>>();

        // Create a reference to the structure for the prescribed shear heating outputs.
        // The structure is created during the advection assembly.
        const std::shared_ptr<HeatingModel::PrescribedShearHeatingOutputs<dim>>
        heating_out = out.template get_additional_output_object<HeatingModel::PrescribedShearHeatingOutputs<dim>>();

        if ((elastic_out == nullptr && heating_out == nullptr) ||
            in.requests_property(MaterialProperties::additional_outputs) == false)
          return;

        AssertThrow(yield_stresses.size() == 0 || yield_stresses.size() == in.n_evaluation_points(),
                    ExcMessage("The length of the yield stress array is neither 0 nor equal to "
                               "the number of evaluation points."));

        const std::shared_ptr<const MaterialModel::ElasticAdditionalInputs<dim>> additional_inputs =
          in.template get_additional_input_object<MaterialModel::ElasticAdditionalInputs<dim>>();
        AssertThrow(additional_inputs != nullptr, ExcInternalError());

        // The viscosity should be averaged if material averaging is applied.
        std::vector<double> effective_viscosities;
        if (this->get_parameters().material_averaging != MaterialAveraging::none)
          {
            MaterialModelOutputs<dim> out_copy(out.n_evaluation_points(),
                                               this->introspection().n_compositional_fields);
            out_copy.viscosities = out.viscosities;

            const MaterialAveraging::AveragingOperation averaging_operation_for_viscosity =
              get_averaging_operation_for_viscosity(this->get_parameters().material_averaging);
            MaterialAveraging::average(averaging_operation_for_viscosity,
                                       in.current_cell,
                                       this->introspection().quadratures.velocities,
                                       this->get_mapping(),
                                       in.requested_properties,
                                       out_copy);

            effective_viscosities = out_copy.viscosities;
          }
        else
          effective_viscosities = out.viscosities;

        const unsigned int stress_start_index = this->introspection().compositional_index_for_name("ve_stress_xx");
        const double dte = elastic_timestep();
        const double dtc = this->get_timestep();

        for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
          {
            const SymmetricTensor<2, dim> deviatoric_strain_rate = Utilities::Tensors::consistent_deviator(in.strain_rate[i]);

            const double effective_viscosity = effective_viscosities[i];

            const SymmetricTensor<2,dim> stress_old(
              Utilities::Tensors::to_symmetric_tensor<dim>(&in.composition[i][stress_start_index],
                                                           &in.composition[i][stress_start_index]+n_independent_components));

            const Tensor<2,dim> spin_tensor = 0.5 * (additional_inputs->velocity_gradients[i] - 
                                                     transpose(additional_inputs->velocity_gradients[i]));
            const SymmetricTensor<2,dim> stress_rate_rotation = symmetrize(spin_tensor * Tensor<2,dim>(stress_old) -
                                                                           Tensor<2,dim>(stress_old) * spin_tensor);

            std::vector<SymmetricTensor<2,dim>> elastic_forces(volume_fractions[i].size());
            std::vector<SymmetricTensor<2,dim>> stresses_dtc(volume_fractions[i].size());
            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              {
                // The elastic force is defined as
                // $-2\eta^{eff}((\bm W\cdot\bm\tau_t - \bm\tau_t\cdot\bm W) / 2G + (1 - a) / a * \bm\tau_t / 2\eta)$,
                // where $(\bm W\cdot\bm\tau_t - \bm\tau_t\cdot\bm W) / 2G$ is the rotation rate,
                // and $a$ denotes the relaxation ratio.
                const double relaxation_time = creep_viscosities[i][j] / elastic_shear_moduli[j];
                const double relaxation_ratio_dte = 1.0 - std::exp(-dte / relaxation_time);
                elastic_forces[j] = (-2.0 * effective_viscosity) *
                                    (stress_rate_rotation / (2.0 * elastic_shear_moduli[j]) + 
                                     stress_old * ((1.0 - relaxation_ratio_dte) / (relaxation_ratio_dte * 2.0 * creep_viscosities[i][j])));

                const SymmetricTensor<2,dim> stress_dte = 2.0 * effective_viscosity * deviatoric_strain_rate - elastic_forces[j];
                stresses_dtc[j] = stress_dte;
                if (heating_out != nullptr && dte != dtc)
                  {
                    const double relaxation_ratio_dtc = 1.0 - std::exp(-dtc / relaxation_time);
                    const double ratio = relaxation_ratio_dtc / relaxation_ratio_dte;
                    stresses_dtc[j] = ratio * stress_dte + (1.0 - ratio) * stress_old;

                    if (yield_stresses.size() > 0)
                      {
                        // If the second invariant of $\bm\tau_{t+dtc}$ exceeds the yield stress,
                        // then scale it back onto the yield surface.
                        const double tau_ii = Utilities::Tensors::consistent_second_invariant_of_deviatoric_tensor(stresses_dtc[j]);
                        if (tau_ii > 0.0 && tau_ii > yield_stresses[i][j])
                          stresses_dtc[j] *= yield_stresses[i][j] / tau_ii;
                      }
                  }
              }

            // Precompute the elastic force, since it is required in the computation of 
            // both the elastic output and the heating output.
            SymmetricTensor<2,dim> elastic_force;
            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              elastic_force += volume_fractions[i][j] * elastic_forces[j];

            if (elastic_out != nullptr)
              {
                elastic_out->elastic_force[i] = elastic_force;

                // The viscoelastic strain rate is needed only when the Newton method is selected.
                const typename Parameters<dim>::NonlinearSolver::Kind nonlinear_solver = this->get_parameters().nonlinear_solver;
                if ((nonlinear_solver == Parameters<dim>::NonlinearSolver::iterated_Advection_and_Newton_Stokes) ||
                    (nonlinear_solver == Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes))
                  elastic_out->viscoelastic_strain_rate[i] = deviatoric_strain_rate - elastic_force / (2.0 * effective_viscosity);              }

            if (heating_out != nullptr)
              {
                SymmetricTensor<2,dim> stress_dtc;
                for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
                  stress_dtc += volume_fractions[i][j] * stresses_dtc[j];

                // Compute the viscoplastic strain rate by substracting the elastic strain rate
                // from the total strain rate.
                // The elastic strain rate is defined as
                // $(\bm\tau_{t+dtc} - \bm\tau_t) / (2G dtc) - (\bm W\cdot\bm\tau_t - \bm\tau_t\cdot\bm W) / (2G)$
                // We have assumed isostrain deformation, so the shear moduli should be averaged through
                // harmonic averaging scheme.
                const double average_elastic_shear_modulus = MaterialUtilities::average_value(volume_fractions[i],
                                                                                              elastic_shear_moduli,
                                                                                              MaterialUtilities::harmonic);
                const SymmetricTensor<2,dim> viscoplastic_strain_rate = 
                  deviatoric_strain_rate - (stress_dtc - stress_old + stress_rate_rotation * dtc) / (2.0 * average_elastic_shear_modulus * dtc);

                // Fill the viscplastic dissipation rate.
                heating_out->prescribed_shear_heating_rates[i] = stress_dtc * viscoplastic_strain_rate;
              }
          }
      }



      template <int dim>
      void
      Elasticity<dim>::
      fill_elastic_additional_outputs (const MaterialModel::MaterialModelInputs<dim> &in,
                                       const std::vector<std::vector<double>> &volume_fractions,
                                       const std::vector<std::vector<double>> &creep_viscosities,
                                       const std::vector<std::vector<double>> &yield_stresses,
                                       MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        std::shared_ptr<MaterialModel::ElasticAdditionalOutputs<dim>> elastic_additional_out
          = out.template get_additional_output_object<MaterialModel::ElasticAdditionalOutputs<dim>>();

        if (elastic_additional_out == nullptr ||
            in.requests_property(MaterialProperties::additional_outputs) == false)
          return;

        AssertThrow(in.current_cell.state() == IteratorState::valid,
                    ExcMessage("Trying to fill the elastic additional outputs in an invalid cell."));

        const std::shared_ptr<const MaterialModel::ElasticAdditionalInputs<dim>> additional_inputs
          = in.template get_additional_input_object<MaterialModel::ElasticAdditionalInputs<dim>>();
        AssertThrow(additional_inputs != nullptr, ExcInternalError());

        const unsigned int stress_start_index = this->introspection().compositional_index_for_name("ve_stress_xx");

        // The viscosity should be averaged if material averaging is applied.
        // Here the averaging scheme "project to Q1 (only viscosity)"  is
        // excluded, because there is no way to know the quadrature formula
        // used for evaluation.
        // TODO: find a way to include "project to Q1 (only viscosity)" as well.
        std::vector<double> effective_viscosities;
        if (this->get_parameters().material_averaging != MaterialAveraging::none &&
            this->get_parameters().material_averaging != MaterialAveraging::project_to_Q1 &&
            this->get_parameters().material_averaging != MaterialAveraging::project_to_Q1_only_viscosity)
          {
            MaterialModelOutputs<dim> out_copy(out.n_evaluation_points(),
                                               this->introspection().n_compositional_fields);
            out_copy.viscosities = out.viscosities;

            const MaterialAveraging::AveragingOperation averaging_operation_for_viscosity =
              get_averaging_operation_for_viscosity(this->get_parameters().material_averaging);
            MaterialAveraging::average(averaging_operation_for_viscosity,
                                       in.current_cell,
                                       this->introspection().quadratures.velocities,
                                       this->get_mapping(),
                                       in.requested_properties,
                                       out_copy);

            effective_viscosities = out_copy.viscosities;
          }
        else
          effective_viscosities = out.viscosities;

        const double dte = elastic_timestep();
        const double dtc = (this->get_timestep_number() > 0 ? this->get_timestep() : dte);

        std::vector<SymmetricTensor<2,dim>> stresses_dtc(volume_fractions[0].size());
        for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
          {
            // Get the stress of the previous timestep from the compositional fields.
            const SymmetricTensor<2, dim> stress_old(
              Utilities::Tensors::to_symmetric_tensor<dim>(&in.composition[i][stress_start_index],
                                                           &in.composition[i][stress_start_index]+n_independent_components));

            const double effective_viscosity = out.viscosities[i];

            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              {
                // Compute the total stress at time spot $t+dte$, i.e. the end of the previous
                // elastic time step. This equals to $2\eta^{\mathrm{eff}}\varepsilon^{\mathrm{eff}}$.
                const SymmetricTensor<2,dim> stress_dte =
                  2.0 * effective_viscosity * calculate_viscoelastic_strain_rate(i, in, creep_viscosities[i][j], elastic_shear_moduli[j], dte);

                stresses_dtc[j] = stress_dte;
                if (dte != dtc)
                  {
                    const double relaxation_time = creep_viscosities[i][j] / elastic_shear_moduli[j];
                    const double ratio = (1.0 - std::exp(-dtc / relaxation_time)) / (1.0 - std::exp(-dte / relaxation_time));
                    stresses_dtc[j] = ratio * stress_dte + (1.0 - ratio) * stress_old;

                    if (yield_stresses.size() > 0)
                      {
                        // If the second invariant of the stress at t+dtc exceeds the yield stress, 
                        // then scale it back onto the yield surface.
                        const double tau_ii = Utilities::Tensors::consistent_second_invariant_of_deviatoric_tensor(stresses_dtc[j]);
                        if (tau_ii > 0.0 && tau_ii > yield_stresses[i][j])
                          stresses_dtc[j] *= yield_stresses[i][j] / tau_ii;
                      }
                  }
              }

            SymmetricTensor<2,dim> stress_dtc;
            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              stress_dtc += volume_fractions[i][j] * stresses_dtc[j];

            // Fill the deviatoric stress
            elastic_additional_out->deviatoric_stress[i] = stress_dtc;

            // Fill the elastic shear modulus
            elastic_additional_out->elastic_shear_moduli[i] = MaterialUtilities::average_value(volume_fractions[i],
                                                                                               elastic_shear_moduli,
                                                                                               MaterialUtilities::arithmetic);
          }
      }



      // The following function computes the reaction rates for the operator
      // splitting step that updates the stored stress from $\tau^t$ to 
      // $tau^{t+dtc}$.
      //
      // At the moment when the reaction rates are required (at the beginning of the new
      // time step), the vector 'solution' holds the stress $\tau^t$ that has been advected 
      // into the position $x^{t+dtc}$. This is the same as the vector 'old_solution' holds. 
      // At later moments during the new time step, 'solution' will hold the
      // current_linearization_point instead of the solution of the time step $[t, t+dtc]$.
      //
      // In case fields are used to track the stresses, MaterialModelInputs are based on
      // 'solution' when calling the MaterialModel for the reaction rates. When particles
      // are used, MaterialModelInputs for this function are filled with the old solution
      // (including for the strain rate), except for the compositions that represent the
      // stress tensor components, which are taken directly from the particles in the
      // property plugin by default (although this can be changed from the input file).
      // As the particles are restored to their pre-advection location at the beginning of
      // each nonlinear iteration, their values and positions correspond to the old solution.
      // This means that in both cases we can use 'in' to get to the stress and velocity/strain
      // rate of the time step $[t, t+dtc]$.
      template <int dim>
      void
      Elasticity<dim>::fill_reaction_rates (const MaterialModel::MaterialModelInputs<dim> &in,
                                            const std::vector<std::vector<double>> &volume_fractions,
                                            const std::vector<std::vector<double>> &creep_viscosities,
                                            const std::vector<std::vector<double>> &yield_stresses,
                                            MaterialModel::MaterialModelOutputs<dim> &out) const
      {        
        if (this->get_timestep_number() == 0 ||
            in.requests_property(MaterialProperties::reaction_rates) == false)
          return;

        AssertThrow(yield_stresses.size() == 0 || yield_stresses.size() == in.n_evaluation_points(),
                    ExcMessage("The length of the yield stress array is neither 0 nor equal to "
                               "the number of evaluation points."));

        AssertThrow(in.current_cell.state() == IteratorState::valid,
                    ExcMessage("Trying to fill the reaction rates for elastic stress components in an invalid cell."));

        const std::shared_ptr<const MaterialModel::ElasticAdditionalInputs<dim>> additional_inputs
          = in.template get_additional_input_object<MaterialModel::ElasticAdditionalInputs<dim>>();
        AssertThrow(additional_inputs != nullptr, ExcInternalError());

        const std::shared_ptr<ReactionRateOutputs<dim>> reaction_rate_out
          = out.template get_additional_output_object<ReactionRateOutputs<dim>>();
        AssertThrow(reaction_rate_out != nullptr, ExcInternalError());

        // Set all reaction rates to zero
        // TODO Should this only set those rates to zero
        // that are used to update the stresses instead of all rates?
        // What if other rheologies also fill reaction rates?
        for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
          for (unsigned int c = 0; c < in.composition[i].size(); ++c)
            reaction_rate_out->reaction_rates[i][c] = 0.0;

        const unsigned int stress_start_index = this->introspection().compositional_index_for_name("ve_stress_xx");

        // The viscosity should be averaged if material averaging is applied.
        // Here the averaging scheme "project to Q1 (only viscosity)"  is
        // excluded, because there is no way to know the quadrature formula
        // used for evaluation.
        // TODO: find a way to include "project to Q1 (only viscosity)" as well.
        std::vector<double> effective_viscosities;
        if (this->get_parameters().material_averaging != MaterialAveraging::none &&
            this->get_parameters().material_averaging != MaterialAveraging::project_to_Q1 &&
            this->get_parameters().material_averaging != MaterialAveraging::project_to_Q1_only_viscosity)
          {
            MaterialModelOutputs<dim> out_copy(out.n_evaluation_points(),
                                               this->introspection().n_compositional_fields);
            out_copy.viscosities = out.viscosities;

            const MaterialAveraging::AveragingOperation averaging_operation_for_viscosity =
              get_averaging_operation_for_viscosity(this->get_parameters().material_averaging);
            MaterialAveraging::average(averaging_operation_for_viscosity,
                                       in.current_cell,
                                       this->introspection().quadratures.velocities,
                                       this->get_mapping(),
                                       in.requested_properties,
                                       out_copy);

            effective_viscosities = out_copy.viscosities;
          }
        else
          effective_viscosities = out.viscosities;

        // Get the elastic time step $dte$ and the computational time step $dtc$.
        // Note that the time step has been updated before calling this function,
        // so we need to request for the old time step.
        double dte = 0.0, dtc = 0.0;
        if (this->get_timestep_number() > 1)
          {
            dtc = this->get_old_timestep();
            if (use_fixed_elastic_time_step)
              dte = fixed_elastic_time_step;
            else
              dte = dtc;
          }
        else
          {
            // At $t=0$, the computational time step always equals to
            // the elastic time step.
            dte = fixed_elastic_time_step;
            dtc = dte;
          }

        std::vector<SymmetricTensor<2,dim>> stress_updates(volume_fractions[0].size());
        for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
          {
            // Get the stress of the previous timestep from the compositional fields.
            const SymmetricTensor<2, dim> stress_old(
              Utilities::Tensors::to_symmetric_tensor<dim>(&in.composition[i][stress_start_index],
                                                           &in.composition[i][stress_start_index]+n_independent_components));

            const double effective_viscosity = effective_viscosities[i];

            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              {
                // Compute the total stress at time spot $t+dte$, i.e. the end of the previous
                // elastic time step. This equals to $2\eta^{\mathrm{eff}}\varepsilon^{\mathrm{eff}}$.
                const SymmetricTensor<2,dim> stress_dte =
                  2.0 * effective_viscosity * calculate_viscoelastic_strain_rate(i, in, creep_viscosities[i][j], elastic_shear_moduli[j], dte);

                SymmetricTensor<2,dim> stress_dtc = stress_dte;
                if (dte != dtc)
                  {
                    const double relaxation_time = creep_viscosities[i][j] / elastic_shear_moduli[j];
                    const double ratio = (1.0 - std::exp(-dtc / relaxation_time)) / (1.0 - std::exp(-dte / relaxation_time));
                    stress_dtc = ratio * stress_dte + (1.0 - ratio) * stress_old;

                    if (yield_stresses.size() > 0)
                      {
                        // If the second invariant of the stress at t+dtc exceeds the yield stress, 
                        // then scale it back onto the yield surface.
                        const double tau_ii = Utilities::Tensors::consistent_second_invariant_of_deviatoric_tensor(stress_dtc);
                        if (tau_ii > 0.0 && tau_ii > yield_stresses[i][j])
                          stress_dtc *= yield_stresses[i][j] / tau_ii;
                      }
                  }

                stress_updates[j] = stress_dtc - stress_old;
              }

            // Fill reaction rates.
            // During this timestep, the reaction rates will be multiplied with
            // the current timestep size to turn the rate of change into a change.
            // Therefore, we divide it by the current timestep.
            SymmetricTensor<2,dim> stress_rate;
            for (unsigned int j = 0; j < volume_fractions[i].size(); ++j)
              stress_rate += volume_fractions[i][j] * stress_updates[j];
            stress_rate /= this->get_timestep();

            Utilities::Tensors::unroll_symmetric_tensor_into_array(stress_rate,
                                                                   &reaction_rate_out->reaction_rates[i][stress_start_index],
                                                                   &reaction_rate_out->reaction_rates[i][stress_start_index]+n_independent_components);
          }
      }



      template <int dim>
      double
      Elasticity<dim>::elastic_timestep () const
      {
        // The elastic time step ($\Delta t_el$, dte) is equal to the numerical time step if the time step number
        // is greater than 0 and the parameter 'use_fixed_elastic_time_step' is set to false.
        // On the first (0) time step, the elastic time step is always equal to the value
        // specified in 'fixed_elastic_time_step', which is also used in all subsequent time
        // steps if 'use_fixed_elastic_time_step' is set to true.
        //
        // We also use this parameter when we are still *before* the first time step,
        // i.e., if the time step number is numbers::invalid_unsigned_int.
        if (use_fixed_elastic_time_step && this->get_timestep_number() > 0 && this->simulator_is_past_initialization())
          AssertThrow(fixed_elastic_time_step >= this->get_timestep(), ExcMessage("The elastic timestep has to be equal to or bigger than the numerical timestep"));

        const double dte = ( ( this->get_timestep_number() > 0 &&
                               this->simulator_is_past_initialization() &&
                               use_fixed_elastic_time_step == false )
                             ?
                             this->get_timestep() * stabilization_time_scale_factor
                             :
                             fixed_elastic_time_step);
        return dte;
      }



      template <int dim>
      const std::vector<double> &
      Elasticity<dim>::get_elastic_shear_moduli () const
      {
        return elastic_shear_moduli;
      }



      template <int dim>
      double
      Elasticity<dim>::
      calculate_viscoelastic_viscosity (const double viscosity,
                                        const double shear_modulus) const
      {
        const double dte = elastic_timestep();
        const double lambda = viscosity / shear_modulus;
        return (1.0 - std::exp(-dte / lambda)) * viscosity;
      }



      template <int dim>
      SymmetricTensor<2,dim>
      Elasticity<dim>::
      calculate_viscoelastic_strain_rate(const unsigned int i,
                                         const MaterialModel::MaterialModelInputs<dim> &in,
                                         const double creep_viscosity,
                                         const double shear_modulus,
                                         const double elastic_timestep) const
      {
        const std::shared_ptr<const MaterialModel::ElasticAdditionalInputs<dim>> additional_inputs
          = in.template get_additional_input_object<MaterialModel::ElasticAdditionalInputs<dim>>();
        AssertThrow(additional_inputs != nullptr, ExcInternalError());

        // The viscoelastic strain rate is defined as
        // $\mathrm{dev}(\bm\varepsilon) + (\bm W\cdot\bm\tau_t - \bm\tau_t\cdot\bm W) / 2G + (1 - a) / a * \bm\tau_t / 2\eta$,
        // where the relaxation ratio $a$ is given by $1 - exp(-dte * G / eta)$.
        SymmetricTensor<2,dim> viscoelastic_strain_rate = Utilities::Tensors::consistent_deviator(in.strain_rate[i]);

        const unsigned int stress_start_index = this->introspection().compositional_index_for_name("ve_stress_xx");
        const SymmetricTensor<2,dim> stress_old(
          Utilities::Tensors::to_symmetric_tensor<dim>(&in.composition[i][stress_start_index],
                                                       &in.composition[i][stress_start_index+n_independent_components]));
        const Tensor<2,dim> spin_tensor = 0.5 * (additional_inputs->velocity_gradients[i] -
                                                 transpose(additional_inputs->velocity_gradients[i]));
        viscoelastic_strain_rate += symmetrize(spin_tensor * Tensor<2,dim>(stress_old) -
                                               Tensor<2,dim>(stress_old) * spin_tensor)
                                    / (2.0 * shear_modulus);

        const double relaxation_ratio = 1.0 - std::exp(-elastic_timestep * shear_modulus / creep_viscosity);
        viscoelastic_strain_rate += stress_old * ((1.0 - relaxation_ratio) / (relaxation_ratio * 2.0 * creep_viscosity));

        return viscoelastic_strain_rate;
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  template class ElasticAdditionalInputs<dim>; \
  template class ElasticAdditionalOutputs<dim>; \
  \
  namespace Rheology \
  { \
    template class Elasticity<dim>; \
  }

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
