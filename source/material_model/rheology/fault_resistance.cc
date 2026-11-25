/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/rheology/fault_resistance.h>
#include <deal.II/base/parameter_handler.h>

namespace aspect::MaterialModel::Rheology
{
  namespace
  {
    auto &registered_laws()
    {
      static std::map<std::string,std::pair<FaultResistance::Declare,FaultResistance::Create>> laws;
      return laws;
    }
  }

  void FaultResistance::register_law(const std::string &name,Declare declare,Create create)
  {
    AssertThrow(registered_laws().emplace(name,std::make_pair(declare,create)).second,
                ExcMessage("Duplicate fault resistance law: "+name));
  }

  void FaultResistance::declare_parameters(ParameterHandler &prm,const Route route)
  {
    std::string names;
    for (const auto &[name,law]:registered_laws())
      {
        if (!names.empty()) names+="|";
        names+=name;
        law.first(prm,route);
      }
    prm.declare_entry("Resistance law","rate and state",Patterns::Selection(names),
                      "Surface resistance law evaluated on the active material trace or growing strips.");
  }

  std::unique_ptr<FaultResistance> FaultResistance::create(const ParameterHandler &prm,const Route route)
  {
    return registered_laws().at(prm.get("Resistance law")).second(prm,route);
  }
}
