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

#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/utilities.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/newton.h>
#include <aspect/simulator.h>
#include <aspect/postprocess/visualization.h>
#include <aspect/postprocess/particles.h>
#include <aspect/geometry_model/box.h>
#include <aspect/plugins.h>
#include <boost/math/tools/roots.hpp>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>

namespace aspect
{
  namespace MaterialModel
  {



    template <int dim>
    bool PhaseFieldFault<dim>::uses_adiabatic_friction_pressure() const
    {
      return use_adiabatic_pressure_in_fault_friction;
    }


    // -----------------------------------------------------------------------------
    // Material-model interface
    // -----------------------------------------------------------------------------

    template <int dim>
    void
    PhaseFieldFault<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      EquationOfStateOutputs<dim> eos_outputs(
        this->introspection().n_chemical_composition_fields() + 1);

      for (unsigned int i = 0; i < in.n_evaluation_points(); ++i)
        {
          const std::vector<double> volume_fractions =
            MaterialUtilities::compute_only_composition_fractions(
              in.composition[i],
              this->introspection().chemical_composition_field_indices());

          // Fill in the equation-of-state outputs
          equation_of_state.evaluate(in, i, eos_outputs);

          out.densities[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.densities, MaterialUtilities::arithmetic);
          out.thermal_expansion_coefficients[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.thermal_expansion_coefficients,
            MaterialUtilities::arithmetic);
          out.specific_heat[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.specific_heat_capacities,
            MaterialUtilities::arithmetic);
          out.thermal_conductivities[i] = MaterialUtilities::average_value(
            volume_fractions, thermal_conductivities, MaterialUtilities::arithmetic);
          out.compressibilities[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.compressibilities,
            MaterialUtilities::arithmetic);
          out.entropy_derivative_pressure[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.entropy_derivative_pressure,
            MaterialUtilities::arithmetic);
          out.entropy_derivative_temperature[i] = MaterialUtilities::average_value(
            volume_fractions, eos_outputs.entropy_derivative_temperature,
            MaterialUtilities::arithmetic);

          if (in.requests_property(MaterialProperties::viscosity))
            {
              // The ordinary Stokes assembler uses eta_ve for the current strain
              // rate; the reconstructed-fault assembler supplies the frozen stress.
              const double G = MaterialUtilities::average_value(
                volume_fractions, elastic_shear_moduli, viscosity_averaging);
              const double eta = compute_creep_viscosity(volume_fractions, in.temperature[i]);
              const double time_step = (this->get_timestep_number() > 0
                                        ? this->get_timestep()
                                        : initial_time_step);
              const MaxwellCoefficients coefficients =
                compute_maxwell_coefficients(eta, G, time_step);
              out.viscosities[i] = coefficients.eta_ve;
            }
        }
    }



    template <int dim>
    bool PhaseFieldFault<dim>::is_compressible() const
    {
      return equation_of_state.is_compressible();
    }



    template <int dim>
    std::vector<double>
    PhaseFieldFault<dim>::get_critical_crack_driving_forces() const
    {
      const unsigned int n_comp = elastic_shear_moduli.size();
      std::vector<double> critical_crack_driving_forces(n_comp);
      for (unsigned int j = 0; j < n_comp; ++j)
        critical_crack_driving_forces[j] = cohesions[j] * cohesions[j] / (2.0 * elastic_shear_moduli[j]);

      return critical_crack_driving_forces;
    }


    
    template <int dim>
    std::vector<double>
    PhaseFieldFault<dim>::get_critical_energy_release_rates() const
    {
      return critical_energy_release_rates;
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::get_phase_field_activation_threshold() const
    {
      return phase_field_activation_threshold;
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::get_phase_field_upper_admissibility_threshold() const
    {
      return 0.99;
    }



    template <int dim>
    void PhaseFieldFault<dim>::set_reconstructed_fault_background_traction_property(
      const unsigned int property_index, const unsigned int correction_property)
    {
      const auto &manager = this->get_reconstructed_fault_manager();
      if (property_index != numbers::invalid_unsigned_int)
        {
          const auto &info = manager.get_property_information();
          AssertThrow(property_index < info.size() && info[property_index].n_components == 2,
                      ExcMessage("Background fault tractions require a two-component property."));
          for (const auto &fault : manager.get_faults())
            for (unsigned int v=0; v<fault.n_vertices(); ++v)
              for (unsigned int c=0; c<2; ++c)
                AssertThrow(fault.property_value_is_initialized(v,info[property_index].position+c)
                            && std::isfinite(fault.get_properties(v)[info[property_index].position+c]),
                            ExcMessage("Background fault tractions must be initialized and finite."));
        }
      if (correction_property != numbers::invalid_unsigned_int)
        {
          const auto &info = manager.get_property_information();
          AssertThrow(property_index != numbers::invalid_unsigned_int
                      && correction_property < info.size() && info[correction_property].n_components == 3,
                      ExcMessage("Background shear correction requires (a,b,d) and a selected background."));
          for (const auto &fault : manager.get_faults())
            for (unsigned int v=0; v<fault.n_vertices(); ++v)
              {
                const auto position=info[correction_property].position;
                for (unsigned int c=0; c<3; ++c)
                  AssertThrow(fault.property_value_is_initialized(v,position+c)
                              && std::isfinite(fault.get_properties(v)[position+c]),
                              ExcMessage("Background correction coefficients must be initialized and finite."));
                AssertThrow(fault.get_properties(v)[position+2]>0.0,
                            ExcMessage("Background correction denominator must be positive."));
              }
        }
      background_traction_property = property_index;
      background_shear_correction_property = correction_property;
    }


    template <int dim>
    std::pair<double,double> PhaseFieldFault<dim>::reconstructed_fault_background_tractions(
      const unsigned int f, const unsigned int segment, const double xi) const
    {
      if (background_traction_property == numbers::invalid_unsigned_int) return {0.,0.};
      const auto &manager=this->get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(f);
      const auto left=fault.get_properties(segment), right=fault.get_properties(segment+1);
      const auto interpolate=[&](const unsigned int i) {return (1-xi)*left[i]+xi*right[i];};
      const auto position=manager.get_property_information()[background_traction_property].position;
      double shear=interpolate(position);
      if (background_shear_correction_property != numbers::invalid_unsigned_int)
        {
          const auto p=manager.get_property_information()[background_shear_correction_property].position;
          // The fixed rational field is evaluated at the quadrature coordinate;
          // a Q1 interpolation of its nodal values is a different prestress.
          shear -= interpolate(p)+interpolate(p+1)/interpolate(p+2);
        }
      return {shear,interpolate(position+1)};
    }


    template <int dim>
    bool PhaseFieldFault<dim>::is_mature_frictional_fault() const
    {
      return mature_frictional_fault;
    }


    template <int dim>
    void PhaseFieldFault<dim>::set_boundary_normalization_completion_file(const std::string &path)
    {
      AssertThrow(!path.empty() && mature_frictional_fault && !evolve_phase_field,
                  ExcMessage("Boundary normalization completion requires a file and a frozen mature fault."));
      AssertThrow(boundary_normalization_completion_file.empty()
                  || boundary_normalization_completion_file==path,
                  ExcMessage("Boundary normalization completion must remain fixed during a run."));
      if (boundary_normalization_completion_file.empty())
        {
          boundary_normalization_completion_file=path;
          normalization_value_cache.valid=false;
        }
    }



    template <int dim>
    double
    PhaseFieldFault<dim>::minimum_fault_slip_rate() const
    {
      return fault_friction.get_minimum_slip_rate();
    }



    // -----------------------------------------------------------------------------
    // Material parameters and parsing
    // -----------------------------------------------------------------------------



    template <int dim>
    void
    PhaseFieldFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field fault");
        {
          EquationOfState::MulticomponentIncompressible<dim>::declare_parameters(prm);
          Rheology::FaultFriction<dim>::declare_parameters(prm);

          // Equation of state parameters
          prm.declare_entry("Thermal conductivities", "3.0",
                            Patterns::List(Patterns::Double(0)),
                            "List of thermal conductivities, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\watt\\per\\meter\\per\\kelvin}.");

          // Reference and minimum/maximum values
          prm.declare_entry("Reference temperature", "293",
                            Patterns::Double(0),
                            "The reference temperature $T_0$ in the power-law viscosity formula. "
                            "Units: \\si{\\kelvin}.");

          prm.declare_entry("Maximum viscosity", "1.e25",
                            Patterns::Double(0),
                            "Upper cutoff for the power-law viscosity. Units: \\si{\\pascal\\second}.");

          prm.declare_entry("Minimum viscosity", "1.e17",
                            Patterns::Double(0),
                            "Lower cutoff for the power-law viscosity. Units: \\si{\\pascal\\second}.");

          prm.declare_entry("Viscosity averaging scheme", "harmonic",
                            Patterns::Selection("arithmetic|harmonic|geometric|maximum composition"),
                            "When more than one compositional field is present at a point "
                            "with different viscosities, we need to come up with an average "
                            "viscosity at that point. Select a weighted harmonic, arithmetic, "
                            "geometric, or maximum composition.");

          prm.declare_entry("Phase field activation threshold", "0.1",
                            Patterns::Double(0, 1),
                            "Value of the phase-field damage variable above which frictional slip and "
                            "rate-and-state fault physics become active. Material points with damage "
                            "below this threshold are treated as intact and the fault friction law is "
                            "not applied. This parameter is used to avoid numerical noise when the "
                            "phase-field variable is small and the fracture is not yet fully developed. "
                            "The value of this parameter should be between 0 and 1.");

          prm.declare_entry("Initial time step", "1.",
                            Patterns::Double(0),
                            "The initial time step size. It is used for evolving the stress at the "
                            "zeroth time step. Note that if an initial distribution of slip rate is "
                            "provided, then it will be assumed that the modeling starts with steady "
                            "slip state, in which case it is recommended to set the initial time step "
                            "to a very large value to be consistent with the slip state. "
                            "Otherwise, it would be easier for the local return-mapping to fail. "
                            " Units: years if the 'Use years instead of seconds' "
                            "parameter is set; seconds otherwise.");

          // Rheological parameters
          prm.declare_entry("Reference viscosities", "1.e24",
                            Patterns::List(Patterns::Double(0)),
                            "List of the reference viscosity, $\\eta_0$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal}.");

          prm.declare_entry("Thermal viscosity exponents", "0.0",
                            Patterns::List(Patterns::Double(0)),
                            "List of the temperature dependences of viscosity, $\\beta$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: none.");

          prm.declare_entry("Elastic shear moduli", "1e10",
                            Patterns::List(Patterns::Double(0)),
                            "List of elastic shear moduli, $G$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal}.");

          prm.declare_entry("Cohesions", "1.e7",
                            Patterns::List(Patterns::Double(0)),
                            "List of cohesions, $C$, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. Units: \\si{\\pascal}.");

          prm.declare_entry("Initial friction coefficients", "0.6",
                            Patterns::List(Patterns::Double(0)),
                            "List of the initial friction coefficients, $\\mu_{\\text{init}}$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: none.");

          prm.declare_entry("Critical energy release rates", "1.e5",
                            Patterns::List(Patterns::Double(0)),
                            "List of the critical energy release rates, $G_c$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\joule\\per\\square\\meter}.");

          prm.declare_entry("Radiation damping coefficients", "",
                            Patterns::List(Patterns::Double(0)),
                            "List of the rediation damping coefficients, $\\eta^d$, "
                            "for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of all compositional fields or only "
                            "those corresponding to chemical compositions. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\pascal\\second\\per\\meter}.");

          prm.declare_entry("Phase field normal lock threshold", "0.5",
                            Patterns::Double(0, 1),
                            "Value of the phase-field damage variable above which the fault normal "
                            "vector is considered fully developed and its orientation is frozen. "
                            "Below this threshold the fault normal may still evolve according to the "
                            "local stress state, while above this value the stored normal direction "
                            "is used to define the slip plane. This parameter helps stabilize the "
                            "fault geometry once the fracture is sufficiently developed. The value "
                            "should be between 0 and 1.");

          prm.declare_entry("Use adiabatic pressure in fault friction", "false",
                            Patterns::Bool(),
                            "Use the adiabatic-model pressure as the complete normal pressure "
                            "in the reconstructed-fault friction term. If false, use the dynamic "
                            "pressure minus the deviatoric normal traction.");

          prm.declare_entry("Evolve phase field", "true",
                            Patterns::Bool(),
                            "Whether to evolve the phase field during the simulation. If set to "
                            "false, then the crack driving force and the direction vectors will be "
                            "frozen after initialization. This is useful when conducting benchmarks "
                            "with pre-existing faults.");

          prm.declare_entry("Fault constitutive mode", "cohesive",
                            Patterns::Selection("cohesive|mature frictional"),
                            "Mature frictional removes cohesive force/storage for a permanently "
                            "prescribed phase profile. Requires Evolve phase field=false, fixed "
                            "reconstructed geometry and a fresh compatible initialization.");

          prm.declare_entry("I h integration backend", "remote points",
                            Patterns::Selection("remote points|cell intervals"),
                            "Normal-profile integration backend. Cell intervals reuses affine "
                            "Box ray/cell geometry, not phase values; unsupported maps retain "
                            "the remote-point reference implementation.");
          prm.declare_entry("I h quadrature tolerance", "1e-8",
                            Patterns::Double(0),
                            "Relative tolerance used to compare the four- and eight-point "
                            "normal-profile quadrature rules.");

          prm.declare_entry("I h surface quadrature subdivisions", "1",
                            Patterns::Integer(1),
                            "Number of equal panels per fault element for the consistent Q1 "
                            "normalization projection. Each panel uses three Gauss points. "
                            "Increase this to resolve bulk-cell-scale tangential variation of "
                            "the profile integrals; it does not change normal-profile tolerances. "
                            "Boundary completion input must use these same ordered profile origins.");

          prm.declare_entry("I h tail tolerance", "1e-8",
                            Patterns::Double(0),
                            "Relative integral tolerance used to terminate each normal-profile tail.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    PhaseFieldFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      normalization_value_cache.valid = false;
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Phase field fault");
        {
          // Equation of state parameters
          equation_of_state.initialize_simulator(this->get_simulator());
          equation_of_state.parse_parameters(prm);

          // Fault-friction parameters
          fault_friction.initialize_simulator(this->get_simulator());
          fault_friction.parse_parameters(prm);

          // Reference and minimum/maximum values
          reference_temperature = prm.get_double("Reference temperature");
          maximum_viscosity     = prm.get_double("Maximum viscosity");
          minimum_viscosity     = prm.get_double("Minimum viscosity");

          viscosity_averaging = MaterialUtilities::parse_compositional_averaging_operation("Viscosity averaging scheme", prm);

          phase_field_activation_threshold  = prm.get_double("Phase field activation threshold");
          phase_field_normal_lock_threshold = prm.get_double("Phase field normal lock threshold");
          AssertThrow(phase_field_activation_threshold <= phase_field_normal_lock_threshold,
                      ExcMessage("The phase field normal lock threshold must be greater than or equal to "
                                 "the phase field activation threshold."));

          initial_time_step = prm.get_double("Initial time step");
          if (this->convert_output_to_years())
            initial_time_step *= year_in_seconds;

          evolve_phase_field = prm.get_bool("Evolve phase field");
          mature_frictional_fault = prm.get("Fault constitutive mode") == "mature frictional";
          AssertThrow(!mature_frictional_fault || (!evolve_phase_field
                        && this->get_parameters().reconstruct_faults),
                      ExcMessage("Mature frictional mode requires reconstructed faults and Evolve phase field=false."));
          use_adiabatic_pressure_in_fault_friction =
            prm.get_bool("Use adiabatic pressure in fault friction");
          normalization_quadrature_tolerance =
            prm.get_double("I h quadrature tolerance");
          normalization_surface_subdivisions =
            prm.get_integer("I h surface quadrature subdivisions");
          use_cell_normalization_profiles = prm.get("I h integration backend") == "cell intervals";
          normalization_tail_tolerance =
            prm.get_double("I h tail tolerance");
          AssertThrow(numbers::is_finite(normalization_quadrature_tolerance)
                      && normalization_quadrature_tolerance > 0.0
                      && numbers::is_finite(normalization_tail_tolerance)
                      && normalization_tail_tolerance > 0.0,
                      ExcMessage("The I_h quadrature and tail tolerances must be positive."));

          // Make options file for parsing maps to double arrays
          std::vector<std::string> compositional_field_names = this->introspection().get_composition_names();
          compositional_field_names.insert(compositional_field_names.begin(), "background");

          std::vector<std::string> chemical_field_names = this->introspection().chemical_composition_field_names();
          chemical_field_names.insert(chemical_field_names.begin(), "background");

          Utilities::MapParsing::Options options(chemical_field_names, "Thermal conductivities");
          options.list_of_allowed_keys = compositional_field_names;

          thermal_conductivities = Utilities::MapParsing::parse_map_to_double_array(prm.get("Thermal conductivities"), options);

          options.property_name = "Reference viscosities";
          reference_viscosities = Utilities::MapParsing::parse_map_to_double_array(prm.get("Reference viscosities"), options);

          options.property_name = "Thermal viscosity exponents";
          thermal_viscosity_exponents = Utilities::MapParsing::parse_map_to_double_array(prm.get("Thermal viscosity exponents"), options);

          options.property_name = "Elastic shear moduli";
          elastic_shear_moduli = Utilities::MapParsing::parse_map_to_double_array(prm.get("Elastic shear moduli"), options);

          AssertThrow(numbers::is_finite(minimum_viscosity) && minimum_viscosity > 0.0,
                      ExcMessage("The minimum viscosity of the phase field fault material model "
                                 "must be finite and positive."));
          AssertThrow(numbers::is_finite(maximum_viscosity)
                      && maximum_viscosity >= minimum_viscosity,
                      ExcMessage("The maximum viscosity of the phase field fault material model "
                                 "must be finite and no smaller than the minimum viscosity."));
          AssertThrow(numbers::is_finite(initial_time_step) && initial_time_step > 0.0,
                      ExcMessage("The initial time step of the phase field fault material model "
                                 "must be finite and positive."));
          for (const double viscosity : reference_viscosities)
            AssertThrow(numbers::is_finite(viscosity) && viscosity > 0.0,
                        ExcMessage("Every reference viscosity of the phase field fault material "
                                   "model must be finite and positive."));
          for (const double shear_modulus : elastic_shear_moduli)
            AssertThrow(numbers::is_finite(shear_modulus) && shear_modulus > 0.0,
                        ExcMessage("Every elastic shear modulus of the phase field fault material "
                                   "model must be finite and positive."));

          options.property_name = "Cohesions";
          cohesions = Utilities::MapParsing::parse_map_to_double_array(prm.get("Cohesions"), options);

          options.property_name = "Initial friction coefficients";
          initial_friction_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Initial friction coefficients"), options);

          options.property_name = "Critical energy release rates";
          critical_energy_release_rates = Utilities::MapParsing::parse_map_to_double_array(prm.get("Critical energy release rates"), options);

          options.property_name = "Radiation damping coefficients";
          radiation_damping_coefficients = Utilities::MapParsing::parse_map_to_double_array(prm.get("Radiation damping coefficients"), options);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

namespace aspect
{
namespace MaterialModel
  {

    // Material-model registration

    ASPECT_REGISTER_MATERIAL_MODEL(PhaseFieldFault,
                                   "phase field fault",
                                   "")
  }
}
