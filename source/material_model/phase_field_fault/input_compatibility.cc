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
    const std::string canonical = "Phase field fault", strips = "Growing fault", simple = "Simple model";

    struct Entry
    {
      bool mature = false;
      bool growing = false;
      std::set<std::string> declared_in;
      std::map<std::string,std::string> supplied;
      std::map<std::string,std::unique_ptr<const Patterns::PatternBase>> patterns;
      std::map<std::string,std::string> descriptions;
      std::string documentation;
      bool validate_mature_range = false;
      bool keep_native_growing_value = false;
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

    Entry &observe(ParameterHandler  &prm,
                   const std::string &section,
                   const Path        &path,
                   const Use          use)
    {
      const auto state=inputs(prm);
      auto &entry=state->entries[path];
      if (use != Use::growing)
        entry.mature = true;
      if (use != Use::mature)
        entry.growing = true;
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
                                  MaterialInputs                    &state)
    {
      for (const auto &[encoded,node]:tree)
        {
          const auto name=decode_name(encoded);
          Path path=parent;
          path.push_back(name);
          if (const auto description=node.get_optional<std::string>("pattern_description"))
            {
              const auto known = state.entries.find(path);
              auto &entry = (known == state.entries.end() || !known->second.declared_in.count(section)) ?
                            observe(prm, section, path, section == strips ? Use::growing : Use::mature) :
                            known->second;
              entry.descriptions[section] = *description;
              if (section == canonical || !entry.declared_in.count(canonical))
                entry.documentation = node.get<std::string>("documentation", "");
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

    // Unlike pattern_factory(), each scalar factory returns null for an
    // unknown SDK type. A nested non-numeric or custom pattern can instead
    // be retained exactly by using this module's typed declare_entry.
    std::unique_ptr<Patterns::PatternBase> standard_pattern(const std::string &description)
    {
      auto pattern = external_numeric_pattern(description);
      if (!pattern)
        pattern = Patterns::Bool::create(description);
      if (!pattern)
        pattern = Patterns::Selection::create(description);
      if (!pattern)
        pattern = Patterns::MultipleSelection::create(description);
      if (!pattern)
        pattern = Patterns::Anything::create(description);
      if (!pattern)
        pattern = Patterns::FileName::create(description);
      if (!pattern)
        pattern = Patterns::DirectoryName::create(description);
      return pattern;
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

    // Paths passed to ParameterHandler::get are relative to Material model.
    Path parameter_group(const std::string &section, const Path &path)
    {
      Path group;
      if (!section.empty())
        group.push_back(section);
      group.insert(group.end(), path.begin(), std::prev(path.end()));
      return group;
    }

    std::string parameter_name(const std::string &section, const Path &path)
    {
      auto components = parameter_group(section, path);
      components.push_back(path.back());
      return "Material model / " + boost::algorithm::join(components, " / ");
    }

    void copy_value(ParameterHandler  &prm,
                    const Path        &path,
                    Entry             &entry,
                    const std::string &destination,
                    const std::string &value)
    {
      const Path group = parameter_group(destination, path);
      if (prm.get(group, path.back()) == value)
        return;
      for (const auto &subsection : group)
        prm.enter_subsection(subsection);
      prm.set(path.back(), value);
      for (unsigned int i=0; i<group.size(); ++i)
        prm.leave_subsection();
      // set() also ran the observation action. This copy was not supplied.
      entry.supplied.erase(destination);
    }
  }

  void observe_entry(ParameterHandler            &prm,
                     const std::string           &name,
                     const Patterns::PatternBase &pattern)
  {
    // A pre-existing SDK callback declared each native route separately.
    // Union those declarations before installing compatibility spellings.
    const auto components = Utilities::split_string_list(prm.get_current_path(), ".");
    Use use = Use::both;
    if (components.size() >= 2 && decode_name(components[0]) == "Material model")
      {
        const auto section = decode_name(components[1]);
        if (section == strips)
          use = Use::growing;
        else if (section == canonical)
          use = Use::mature;
      }
    observe_entry(prm, use, name, pattern);
  }

  void observe_entry(ParameterHandler            &prm,
                     const Use                    use,
                     const std::string           &name,
                     const Patterns::PatternBase &pattern)
  {
    const auto components=Utilities::split_string_list(prm.get_current_path(),".");
    // A law can also be declared on its own. Only the fault material scopes,
    // historical normal-stress selector and Simple viscosity have aliases.
    if (components.empty() || decode_name(components[0])!="Material model")
      return;
    const std::string section=components.size()>1 ? decode_name(components[1]) : "";
    if (!section.empty() && section!=canonical && section!=strips && section!=simple)
      return;
    Path path;
    for (unsigned int i=2; i<components.size(); ++i)
      path.push_back(decode_name(components[i]));
    path.push_back(name);
    auto &entry=observe(prm,section,path,use);
    if (!entry.patterns[section])
      entry.patterns[section] = pattern.clone();
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
    prm.declare_entry(name, value, pattern, documentation);
    observe_entry(prm, name, pattern);
  }

  void declare_entry(ParameterHandler            &prm,
                     const Use                    use,
                     const std::string           &name,
                     const std::string           &value,
                     const Patterns::PatternBase &pattern,
                     const std::string           &documentation)
  {
    prm.declare_entry(name,value,pattern,documentation);
    observe_entry(prm, use, name, pattern);
  }

  void observe_external_coefficients(ParameterHandler &prm)
  {
    const auto state=inputs(prm);
    const auto path=prm.get_current_path();
    const auto components=Utilities::split_string_list(path,".");
    if (components.size()!=2 || decode_name(components[0])!="Material model")
      return;
    const auto current=decode_name(components[1]);
    if ((current!=canonical && current!=strips) || !state->completed_law_declarations.insert(current).second ||
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
    const std::string sections[] = {canonical,strips};
    for (const auto &section : sections)
      {
        prm.enter_subsection(section);
        observe_external_entries(prm, section, root.get_child(prm.get_current_path()), {}, *state);
        prm.leave_subsection();
      }

    for (auto &[path, entry] : state->entries)
      {
        if (!entry.growing || !entry.declared_in.count(strips))
          continue;
        const bool strip_only = !entry.declared_in.count(canonical);
        const bool different_patterns = !strip_only &&
                                        entry.descriptions.at(canonical) != entry.descriptions.at(strips);
        if (!strip_only && !different_patterns)
          {
            // A custom validator's text need not encode its limits. Preserve
            // the native SDK value even if two raw descriptions are identical.
            if (!entry.patterns[canonical])
              entry.patterns[canonical] = standard_pattern(entry.descriptions.at(canonical));
            entry.keep_native_growing_value = !entry.patterns[canonical];
            continue;
          }

        if (different_patterns)
          {
            // A shared law coefficient can have disjoint valid ranges on the
            // two formulations. Keep the mature validator for use after the
            // phase method is known. Growing still has its native declaration.
            auto &mature = entry.patterns[canonical];
            if (!mature)
              mature = standard_pattern(entry.descriptions.at(canonical));
            if (!mature)
              {
                // deal.II exposes only the description of a raw SDK declaration.
                // An unregistered custom pattern cannot be cloned from text.
                // Preserve both original entries and their active-route readers.
                entry.keep_native_growing_value = true;
                continue;
              }
            entry.validate_mature_range = true;
          }

        const std::string previous = strip_only ? strips : canonical;
        const auto default_value = prm.get(parameter_group(previous, path), path.back());
        const auto group = parameter_group(canonical, path);
        for (const auto &subsection : group)
          prm.enter_subsection(subsection);

        // With two different ranges, syntax alone cannot decide which range
        // applies. The active-route validation in resolve() is authoritative.
        // A growing-only input can expose its native range immediately.
        auto &strip_pattern = entry.patterns[strips];
        if (!strip_pattern)
          strip_pattern = standard_pattern(entry.descriptions.at(strips));
        const Patterns::Anything any;
        const Patterns::PatternBase &public_pattern = (strip_only && strip_pattern) ? *strip_pattern : any;
        prm.declare_entry(path.back(), default_value, public_pattern, entry.documentation);
        if (strip_only)
          observe(prm, canonical, path, Use::growing);
        for (unsigned int i=0; i<group.size(); ++i)
          prm.leave_subsection();
      }
    prm.enter_subsection(current);
  }

  std::string normal_stress_source(const std::string &choice, const bool growing)
  {
    if (choice == "model default")
      return growing ? "prescribed constant" : "resolved normal traction";
    return choice;
  }

  void resolve(ParameterHandler &prm,
               const bool        growing,
               const bool        historical_strip_inputs)
  {
    const auto state=inputs(prm);
    prm.enter_subsection("Material model");

    for (auto &[path,entry]:state->entries)
      {
        // Simple viscosity never entered the Maxwell law of the historical
        // growing material. The public material accepts it as a viscosity alias.
        auto values=entry.supplied;
        if (growing && historical_strip_inputs)
          values.erase(simple);
        if (values.empty())
          {
            // Only the active phase method is known at read time. In particular
            // the two Maxwell coefficients and RSF laws have different defaults.
            if (growing && entry.growing && entry.declared_in.count(strips) && !entry.keep_native_growing_value)
              copy_value(prm, path, entry, canonical,
                         prm.get(parameter_group(strips, path), path.back()));
            continue;
          }
        if (path == Path {"Friction normal stress"})
          for (auto &source : values)
            source.second = normal_stress_source(source.second, growing);
        auto first=values.find(canonical);
        if (first==values.end())
          first=values.begin();
        AssertThrow(growing ? entry.growing : entry.mature,
                    ExcMessage(parameter_name(first->first, path)+
                               (growing?" requires mature trace mechanics.":" requires growing strip mechanics.")));
        for (const auto &[section,value]:values)
          if (value!=first->second)
            {
              // Single-spelling inputs (including custom SDK patterns) go
              // directly to their registered reader. Only a duplicate with
              // different text needs a typed comparison.
              const auto &scope = growing && entry.declared_in.count(strips) ? strips : canonical;
              auto &pattern = entry.patterns[scope];
              if (!pattern)
                pattern = external_numeric_pattern(entry.descriptions[scope]);
              AssertThrow(pattern && equal_values(*pattern,value,first->second),
                          ExcMessage("Conflicting "+parameter_name(first->first, path)+
                                     " and "+parameter_name(section, path)+"."));
            }

        if (entry.validate_mature_range)
          {
            const auto &pattern = entry.patterns.at(growing ? strips : canonical);
            AssertThrow(!pattern || pattern->match(first->second),
                        ExcMessage(parameter_name(first->first, path) +
                                   (growing ? " does not match the growing strip pattern " :
                                    " does not match the mature trace pattern ") +
                                   pattern->description(Patterns::PatternBase::Text) + "."));
          }
        if (!values.count(canonical) && !(growing && entry.keep_native_growing_value))
          copy_value(prm, path, entry, canonical, first->second);
        // Apply external validation and preserve the resolved legacy values
        // in ASPECT's parameter output for consumers of the older spellings.
        if (growing && entry.declared_in.count(strips) && !values.count(strips))
          copy_value(prm, path, entry, strips, first->second);
        if (!growing && path==Path {"Viscosity"} && !values.count(simple))
          copy_value(prm, path, entry, simple, first->second);
      }
    prm.leave_subsection();
  }
}
