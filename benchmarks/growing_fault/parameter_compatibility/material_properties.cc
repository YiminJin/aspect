/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

#include <iomanip>

namespace aspect::Postprocess
{
  /** Sample the selected material through the ordinary evaluate interface.
   * All three public material inputs must construct exactly the same type. */
  template <int dim>
  class FaultMaterialProperties : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string, std::string> execute(TableHandler &) override
      {
        const auto &model = this->get_material_model();
        AssertThrow(typeid(model) == typeid(MaterialModel::PhaseFieldFault<dim>),
                    ExcMessage("The ordinary material factory did not construct PhaseFieldFault."));
        const unsigned int n_fields = this->introspection().n_compositional_fields;
        MaterialModel::MaterialModelInputs<dim> inputs(1, n_fields);
        MaterialModel::MaterialModelOutputs<dim> outputs(1, n_fields);
        inputs.temperature[0] = 320.;
        inputs.pressure[0] = 0.;
        inputs.position[0] = Point<dim>();
        for (unsigned int c = 0; c < n_fields; ++c)
          inputs.composition[0][c] = 0.1 * (c + 1);
        model.create_additional_named_outputs(outputs);
        model.evaluate(inputs, outputs);

        const auto &dependence = model.get_model_dependence();
        using namespace MaterialModel;
        AssertThrow(dependence.viscosity == NonlinearDependence::none &&
                    dependence.compressibility == NonlinearDependence::none &&
                    dependence.specific_heat == NonlinearDependence::none &&
                    dependence.thermal_conductivity == NonlinearDependence::none &&
                    !model.is_compressible(), ExcMessage("Unexpected fault material dependence."));
        AssertThrow(outputs.compressibilities[0] == 0. &&
                    outputs.entropy_derivative_pressure[0] == 0. &&
                    outputs.entropy_derivative_temperature[0] == 0. &&
                    outputs.reaction_terms[0] == std::vector<double>(n_fields, 0.),
                    ExcMessage("Nonzero fault compressibility, entropy derivative or reaction."));

        std::ostringstream sample;
        sample << std::setprecision(17) << this->get_timestep_number() << ' '
               << outputs.densities[0] << ' ' << outputs.specific_heat[0] << ' '
               << outputs.thermal_conductivities[0] << ' ' << outputs.thermal_expansion_coefficients[0]
               << ' ' << outputs.viscosities[0] << ' ' << dependence.density;
        const auto named = outputs.template get_additional_output_object<NamedAdditionalMaterialOutputs<dim>>();
        if (named)
          for (unsigned int i = 0; i < named->get_names().size(); ++i)
            sample << ' ' << named->get_names()[i] << '=' << named->get_nth_output(i)[0];
        return {"Fault material sample", sample.str()};
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(FaultMaterialProperties, "fault material properties",
                                "Sample material properties and identify the installed fault plugin.")
}
