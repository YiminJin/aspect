/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/rheology/friction_normal_stress.h>
#include <aspect/adiabatic_conditions/ascii_data.h>
#include <aspect/geometry_model/interface.h>

namespace aspect::MaterialModel::Rheology
{
  void FrictionNormalStress::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material model");
    prm.declare_entry("Friction normal stress","model default",
                      Patterns::Selection("model default|adiabatic pressure|resolved normal traction|prescribed constant"),
                      "Source of fault compression in Pa. Adiabatic pressure supplies the complete normal "
                      "stress from ASPECT's adiabatic conditions at the fault. Model default preserves "
                      "resolved normal traction on mature faults and prescribed constant compression on growing strips.");
    prm.leave_subsection();
  }

  FrictionNormalStress FrictionNormalStress::parse(const ParameterHandler &prm, const Source model_default)
  {
    FrictionNormalStress result;
    const auto choice = prm.get({"Material model"},"Friction normal stress");
    if (choice == "adiabatic pressure")
      {
        result.source = Source::adiabatic_pressure;
        result.adiabatic_model = prm.get({"Adiabatic conditions model"},"Model name");
        if (result.adiabatic_model == "function")
          {
            const std::vector<std::string> path = {"Adiabatic conditions model","Function"};
            const auto expressions = Utilities::split_string_list(prm.get(path,"Function expression"),';');
            AssertThrow(expressions.size()==3, ExcMessage("Expected temperature, pressure and density in the adiabatic function."));
            result.function_definition = prm.get(path,"Variable names") + "\n" +
                                         prm.get(path,"Function constants") + "\n" + expressions[1];
          }
        else if (result.adiabatic_model == "compute profile")
          {
            const std::vector<std::string> path = {"Adiabatic conditions model","Compute profile"};
            result.table_size = prm.get_integer(path,"Number of points");
            result.time_dependent_table = prm.get_bool(path,"Use surface condition function");
          }
      }
    else
      {
        result.source = model_default;
        AssertThrow(choice == "model default" ||
                    (choice == "resolved normal traction" && model_default == Source::resolved_traction) ||
                    (choice == "prescribed constant" && model_default == Source::prescribed_constant),
                    ExcMessage("The selected Friction normal stress is incompatible with this fault material."));
      }
    return result;
  }

  bool FrictionNormalStress::uses_adiabat() const
  {
    return source == Source::adiabatic_pressure;
  }

  template <int dim>
  std::string FrictionNormalStress::checkpoint_profile(const SimulatorAccess<dim> &sim) const
  {
    if (!uses_adiabat()) return {};
    if (adiabatic_model == "function") return adiabatic_model + "\n" + function_definition;
    const auto &adiabat = sim.get_adiabatic_conditions();
    AssertThrow(adiabat.is_initialized(), ExcMessage("The adiabatic pressure profile must be initialized before checkpoint comparison."));
    std::ostringstream out;
    out << adiabatic_model << '\n' << std::setprecision(17);
    if (const auto *ascii = dynamic_cast<const AdiabaticConditions::AsciiData<dim> *>(&adiabat))
      {
        const unsigned int column = ascii->get_column_index_from_name("pressure");
        for (const double depth : ascii->get_interpolation_point_coordinates())
          out << depth << ' ' << ascii->get_data_component(Point<1>(depth),column) << '\n';
      }
    else
      {
        AssertThrow(table_size>1 && !time_dependent_table,
                    ExcMessage("Checkpointing with adiabatic fault friction requires a static function, ASCII table or computed profile."));
        for (unsigned int i=0; i<table_size; ++i)
          {
            const double depth = sim.get_geometry_model().maximal_depth()*i/(table_size-1);
            out << depth << ' ' << adiabat.pressure(sim.get_geometry_model().representative_point(depth)) << '\n';
          }
      }
    return out.str();
  }

  template std::string FrictionNormalStress::checkpoint_profile(const SimulatorAccess<2> &) const;
  template std::string FrictionNormalStress::checkpoint_profile(const SimulatorAccess<3> &) const;
}
