/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/fault_parameters.h>
#include <aspect/material_model/growing_fault.h>
#include <aspect/material_model/phase_field_fault.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace aspect::MaterialModel::FaultParameters
{
  namespace
  {
    using Path = std::vector<std::string>;
    const std::string trace = "Phase field fault", strips = "Growing fault", simple = "Simple model";

    struct SharedInput
    {
      bool numeric;
      std::map<std::string,std::string> values;
    };

    struct ExplicitInputs
    {
      std::map<Path,SharedInput> shared;
      std::map<std::string,std::set<std::string>> restricted;
    };

    // deal.II 9.6 exposes explicit assignments via actions, including values
    // equal to defaults. Each handler owns its observations through its actions.
    // The weak index keeps independent handlers and their lifetimes separate.
    std::shared_ptr<ExplicitInputs> inputs(const ParameterHandler &prm)
    {
      static std::map<const ParameterHandler *,std::weak_ptr<ExplicitInputs>> handlers;
      for (auto it=handlers.begin(); it!=handlers.end();)
        if (it->second.expired())
          it=handlers.erase(it);
        else
          ++it;
      auto &entry=handlers[&prm];
      if (auto state=entry.lock())
        return state;
      auto state=std::make_shared<ExplicitInputs>();
      entry=state;
      return state;
    }

    // ParameterHandler's documented JSON contains the original names, subsection
    // hierarchy and validation patterns. Walk only the two small fault schemas,
    // so an independently compiled law can use ordinary declare_entry().
    void collect_entries(const boost::property_tree::ptree &section, const Path &parent,
                         std::map<Path,bool> &entries)
    {
      for (const auto &[name,node]:section)
        {
          Path path=parent;
          path.push_back(name);
          if (const auto pattern=node.get_optional<std::string>("pattern_description"))
            entries[path]=pattern->find("[Double ")==0 || pattern->find("[Integer ")==0;
          else
            collect_entries(node,path,entries);
        }
    }

    // Both operations use paths relative to Material model. Leave the caller
    // in its original subsection, including for nested law coefficients.
    template <typename Operation>
    void at_entry(ParameterHandler &prm,const std::string &section,const Path &path,
                  const Operation &operation)
    {
      prm.enter_subsection(section);
      for (auto it=path.begin(); it!=std::prev(path.end()); ++it)
        prm.enter_subsection(*it);
      operation(path.back());
      for (unsigned int i=0; i<path.size(); ++i)
        prm.leave_subsection();
    }

    void observe(ParameterHandler &prm,const std::string &section,const Path &path,
                 const bool shared,const bool numeric)
    {
      const auto state=inputs(prm);
      at_entry(prm,section,path,[&](const auto &name)
      {
        prm.add_action(name,[=](const std::string &value)
        {
          if (shared)
            {
              auto &input=state->shared[path];
              input.numeric=numeric;
              input.values[section]=value;
            }
          else
            state->restricted[section].insert(boost::algorithm::join(path," / "));
        },false);
      });
    }
  }

  template <int dim>
  void observe_material_inputs(ParameterHandler &prm)
  {
    ParameterHandler declarations;
    PhaseFieldFault<dim>::declare_parameters(declarations);
    GrowingFault::Material<dim>::declare_parameters(declarations);
    std::stringstream json;
    declarations.print_parameters(json,ParameterHandler::JSON);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(json,tree);
    const auto &material=tree.get_child("Material model");
    std::map<Path,bool> trace_entries,strip_entries;
    collect_entries(material.get_child(trace), {},trace_entries);
    collect_entries(material.get_child(strips), {},strip_entries);
    prm.enter_subsection("Material model");
    for (const auto &[path,numeric]:trace_entries)
      observe(prm,trace,path,strip_entries.count(path),numeric);
    for (const auto &[path,numeric]:strip_entries)
      observe(prm,strips,path,trace_entries.count(path),numeric);
    observe(prm,simple, {"Viscosity"},true,true);
    prm.leave_subsection();
  }

  bool explicitly_set(const ParameterHandler &prm,const std::string &section,const std::string &name)
  {
    const auto state=inputs(prm);
    const auto input=state->shared.find({name});
    return input!=state->shared.end() && input->second.values.count(section);
  }

  void resolve(ParameterHandler &prm,const bool growing)
  {
    const auto state=inputs(prm);
    const auto &active=growing?strips:trace;
    const auto &other=growing?trace:strips;
    const auto &incompatible=state->restricted[other];
    AssertThrow(incompatible.empty(),ExcMessage("Material model / "+other+" / "+*incompatible.begin()+
                                                (growing?" requires mature trace mechanics.":" requires growing strip mechanics.")));

    prm.enter_subsection("Material model");
    for (auto &[path,input]:state->shared)
      {
        auto first=input.values.find(trace);
        if (first==input.values.end())
          first=input.values.begin();
        for (const auto &[section,value]:input.values)
          {
            const bool equal=(value==first->second ||
                              (input.numeric && Utilities::string_to_double(value)==Utilities::string_to_double(first->second)));
            const auto name=boost::algorithm::join(path," / ");
            AssertThrow(equal,ExcMessage("Conflicting Material model / "+first->first+" / "+name+
                                         " and Material model / "+section+" / "+name+"."));
          }
        // Simple historically supplies the mature bulk viscosity. Growing
        // mechanics and external law constructors read their active subsection.
        const auto &target=(!growing && path==Path {"Viscosity"}) ? simple : active;
        if (!input.values.count(target))
          {
            at_entry(prm,target,path,[&](const auto &name)
            {
              prm.set(name,first->second);
            });
            // set() also ran the observation action. This copy was not supplied.
            input.values.erase(target);
          }
      }
    prm.leave_subsection();
  }

  template void observe_material_inputs<2>(ParameterHandler &);
  template void observe_material_inputs<3>(ParameterHandler &);
}
