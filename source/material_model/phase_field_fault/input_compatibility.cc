/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/input_compatibility.h>
#include <deal.II/base/utilities.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace aspect::MaterialModel::FaultInputCompatibility
{
  using namespace dealii;

  namespace
  {
    using Path = std::vector<std::string>;
    const std::string trace = "Phase field fault", strips = "Growing fault", simple = "Simple model";

    struct Entry
    {
      std::set<std::string> declared_in;
      std::map<std::string,std::string> supplied;
      std::unique_ptr<const Patterns::PatternBase> pattern;
      std::string external_pattern;
    };

    struct MaterialInputs
    {
      std::map<Path,Entry> entries;
      std::set<std::string> completed_law_declarations;
    };

    // Actions own the observations for the lifetime of their parameter handler.
    std::shared_ptr<MaterialInputs> inputs(const ParameterHandler &prm)
    {
      static std::map<const ParameterHandler *,std::weak_ptr<MaterialInputs>> handlers;
      for (auto it=handlers.begin(); it!=handlers.end();)
        if (it->second.expired())
          it=handlers.erase(it);
        else
          ++it;
      auto &entry=handlers[&prm];
      if (auto state=entry.lock())
        return state;
      auto state=std::make_shared<MaterialInputs>();
      entry=state;
      return state;
    }

    // ParameterHandler's XML and get_current_path() encode each non-alphanumeric
    // byte as _hh. Split the encoded path before restoring literal punctuation.
    std::string decode_name(const std::string &encoded)
    {
      std::string name;
      for (unsigned int i=0; i<encoded.size(); ++i)
        if (encoded[i]=='_')
          {
            name+=static_cast<char>(std::stoul(encoded.substr(i+1,2),nullptr,16));
            i+=2;
          }
        else
          name+=encoded[i];
      return name;
    }

    Entry &observe(ParameterHandler &prm, const std::string &section, const Path &path)
    {
      const auto state=inputs(prm);
      auto &entry=state->entries[path];
      if (entry.declared_in.insert(section).second)
        prm.add_action(path.back(),[state,path,section](const std::string &value)
        {
          state->entries.at(path).supplied[section]=value;
        },false);
      return entry;
    }

    // The public deal.II 9.6 declaration callback predates named section
    // iterators. Its XML format retains entry boundaries even for names
    // containing periods, and stores the machine-readable validation pattern.
    void observe_external_entries(ParameterHandler                  &prm,
                                  const std::string                 &section,
                                  const boost::property_tree::ptree &tree,
                                  const Path                        &parent,
                                  const MaterialInputs              &state)
    {
      for (const auto &[encoded,node]:tree)
        {
          const auto name=decode_name(encoded);
          Path path=parent;
          path.push_back(name);
          if (const auto description=node.get_optional<std::string>("pattern_description"))
            {
              const auto entry=state.entries.find(path);
              if (entry==state.entries.end() || !entry->second.declared_in.count(section))
                observe(prm, section, path).external_pattern=*description;
            }
          else if (!node.get_optional<std::string>("alias"))
            {
              prm.enter_subsection(name);
              observe_external_entries(prm, section, node, path, state);
              prm.leave_subsection();
            }
        }
    }

    // A plugin may supply its own PatternBase. deal.II's generic factory
    // aborts on an unknown pattern, including one nested in a list. Ask the
    // List factory for its declared separators only after confirming that
    // the scalar leaf is one of the two numeric types we compare.
    std::unique_ptr<Patterns::PatternBase> external_numeric_pattern(const std::string &description)
    {
      const std::string list_prefix="[List of <";
      std::size_t start=0;
      while (description.compare(start,list_prefix.size(),list_prefix)==0)
        start+=list_prefix.size();
      const auto end=description.find(']',start);
      if (end==std::string::npos)
        return nullptr;
      const auto leaf=description.substr(start,end-start+1);
      std::unique_ptr<Patterns::PatternBase> number=Patterns::Double::create(leaf);
      if (!number)
        number=Patterns::Integer::create(leaf);
      if (number && start!=0)
        return Patterns::List::create(description);
      return number;
    }

    bool equal_values(const Patterns::PatternBase &pattern,
                      const std::string           &a,
                      const std::string           &b)
    {
      if (a==b)
        return true;
      if (dynamic_cast<const Patterns::Double *>(&pattern))
        return Patterns::Tools::Convert<double>::to_value(a)==Patterns::Tools::Convert<double>::to_value(b);
      if (dynamic_cast<const Patterns::Integer *>(&pattern))
        return Patterns::Tools::Convert<long int>::to_value(a)==Patterns::Tools::Convert<long int>::to_value(b);
      if (const auto *list=dynamic_cast<const Patterns::List *>(&pattern))
        {
          const auto left=Utilities::split_string_list(a,list->get_separator());
          const auto right=Utilities::split_string_list(b,list->get_separator());
          return left.size()==right.size() && std::equal(left.begin(),left.end(),right.begin(),[&](const auto &l,const auto &r)
          {
            return equal_values(list->get_base_pattern(),l,r);
          });
        }
      return false;
    }
  }

  void observe_entry(ParameterHandler            &prm,
                     const std::string           &name,
                     const Patterns::PatternBase &pattern)
  {
    const auto components=Utilities::split_string_list(prm.get_current_path(),".");
    // A law can also be declared on its own; only the two material scopes
    // and the historical Simple viscosity participate in cross-route aliases.
    if (components.size()<2 || decode_name(components[0])!="Material model")
      return;
    const auto section=decode_name(components[1]);
    if (section!=trace && section!=strips && section!=simple)
      return;
    Path path;
    for (auto it=components.begin()+2; it!=components.end(); ++it)
      path.push_back(decode_name(*it));
    path.push_back(name);
    auto &entry=observe(prm,section,path);
    if (!entry.pattern)
      entry.pattern=pattern.clone();
  }

  void observe_simple_viscosity(ParameterHandler &prm)
  {
    // Only after the plugin declaration loop is Simple's upstream entry
    // guaranteed to exist. Capture explicit assignments, including its default.
    prm.enter_subsection("Material model");
    prm.enter_subsection("Simple model");
    observe_entry(prm, "Viscosity", Patterns::Double(0.));
    prm.leave_subsection();
    prm.leave_subsection();
  }

  void declare_entry(ParameterHandler            &prm,
                     const std::string           &name,
                     const std::string           &value,
                     const Patterns::PatternBase &pattern,
                     const std::string           &documentation)
  {
    prm.declare_entry(name,value,pattern,documentation);
    observe_entry(prm, name, pattern);
  }

  void observe_external_coefficients(ParameterHandler &prm)
  {
    const auto state=inputs(prm);
    const auto path=prm.get_current_path();
    const auto components=Utilities::split_string_list(path,".");
    if (components.size()!=2 || decode_name(components[0])!="Material model")
      return;
    const auto current=decode_name(components[1]);
    if ((current!=trace && current!=strips) || !state->completed_law_declarations.insert(current).second ||
        state->completed_law_declarations.size()!=2)
      return;

    // Only a process loading additional resistance laws needs this discovery.
    // Each opaque callback has already run once per route on the real handler.
    std::stringstream xml;
    prm.print_parameters(xml,ParameterHandler::XML | ParameterHandler::KeepDeclarationOrder);
    boost::property_tree::ptree document;
    boost::property_tree::read_xml(xml,document);
    const auto &root=document.get_child("ParameterHandler");
    prm.leave_subsection();
    const std::string sections[] = {trace,strips};
    for (const auto &section : sections)
      {
        prm.enter_subsection(section);
        observe_external_entries(prm, section, root.get_child(prm.get_current_path()), {}, *state);
        prm.leave_subsection();
      }
    prm.enter_subsection(current);
  }

  void resolve(ParameterHandler &prm,const bool growing)
  {
    const auto state=inputs(prm);
    const auto &active=growing?strips:trace;
    prm.enter_subsection("Material model");
    const bool historical_growth = growing && prm.get("Model name")!="phase field fault";

    for (auto &[path,entry]:state->entries)
      {
        // Simple viscosity never entered the Maxwell law of the historical
        // growing material. The public material accepts it as a viscosity alias.
        auto values=entry.supplied;
        if (historical_growth)
          values.erase(simple);
        if (values.empty())
          continue;
        const auto name=boost::algorithm::join(path," / ");
        auto first=values.find(trace);
        if (first==values.end())
          first=values.begin();
        AssertThrow(entry.declared_in.count(active),
                    ExcMessage("Material model / "+first->first+" / "+name+
                               (growing?" requires mature trace mechanics.":" requires growing strip mechanics.")));
        for (const auto &[section,value]:values)
          if (value!=first->second)
            {
              // Single-spelling inputs (including custom SDK patterns) go
              // directly to their registered reader. Only a duplicate with
              // different text needs a typed comparison.
              if (!entry.pattern)
                entry.pattern=external_numeric_pattern(entry.external_pattern);
              AssertThrow(entry.pattern && equal_values(*entry.pattern,value,first->second),
                          ExcMessage("Conflicting Material model / "+first->first+" / "+name+
                                     " and Material model / "+section+" / "+name+"."));
            }

        const auto &target=(!growing && path==Path {"Viscosity"}) ? simple : active;
        if (!values.count(target))
          {
            prm.enter_subsection(target);
            for (auto it=path.begin(); it!=std::prev(path.end()); ++it)
              prm.enter_subsection(*it);
            prm.set(path.back(),first->second);
            for (unsigned int i=0; i<path.size(); ++i)
              prm.leave_subsection();
            // set() also ran the observation action. This copy was not supplied.
            entry.supplied.erase(target);
          }
      }
    prm.leave_subsection();
  }
}
