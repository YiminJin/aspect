/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/input_compatibility.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/material_model/rheology/fault_resistance.h>
#include <aspect/geometry_model/box.h>
#include <aspect/adiabatic_conditions/interface.h>

namespace aspect::GrowingFault
{
  namespace
  {
    // The v1 archive interleaved RSF age with each physical strip.
    // Decode this representation only when restoring that format.
    struct ArchivedRsfStrip : SectionHistory
    {
      double age;
      template <class Archive> void serialize(Archive &ar, const unsigned int)
      {
        ar &age &slip &angle &rate &resistance &formed;
      }
    };
  }

  template <int dim>
  void StripMaterialHistory<dim>::declare_parameters(ParameterHandler &prm)
  {
    using MaterialModel::FaultInputCompatibility::declare_entry;
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    declare_entry(prm, "Prescribed fault compression", "", Patterns::List(Patterns::Double(0.), 0, 1),
                  "Positive compression in Pa used by the signed fault resistance law. An empty entry "
                  "uses the explicitly supplied Surface pressure from legacy growing fault inputs.");
    declare_entry(prm, "Seed tangent angle", "0", Patterns::Double(),
                  "Angle in degrees counterclockwise from x.");
    declare_entry(prm, "Forward seed tip", "0,0", Patterns::List(Patterns::Double(), 2, 2),
                  "Physical tip of the initial contiguous seed in metres.");
    declare_entry(prm, "Section quadrature order", "6", Patterns::Integer(2),
                  "Gauss points per strip subinterval and chord.");
    declare_entry(prm, "Shear modulus", "1e6", Patterns::Double(0),
                  "Shear modulus in Pa.");
    declare_entry(prm, "Viscosity", "1e8", Patterns::Double(0),
                  "Maxwell viscosity in Pa s.");
    declare_entry(prm, "Cohesion", "1000", Patterns::Double(0),
                  "Cohesive strength in Pa.");
    MaterialModel::Rheology::FaultResistance::declare_parameters(prm, MaterialModel::Rheology::FaultResistance::Route::growing);
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  StripMaterialHistory<dim>::StripMaterialHistory(const SimulatorAccess<dim> &simulator,
                                                  ParameterHandler           &prm,
                                                  const unsigned int          phase_field_index)
    : simulator(simulator),
      phase_field_index(phase_field_index)
  {
    const auto &p = simulator.get_parameters();
    const auto &energy = p.surface_energy_model;
    using NormalStress = MaterialModel::Rheology::FrictionNormalStress;
    normal_stress = NormalStress::parse(prm, NormalStress::Source::prescribed_constant);
    const bool legacy_compression = !prm.get("Surface pressure").empty();
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    prescribed_compression = normal_stress.uses_adiabat() ? 0. : p.surface_pressure;
    AssertThrow(!normal_stress.uses_adiabat() || prm.get("Prescribed fault compression").empty(),
                ExcMessage("Adiabatic friction supplies the complete pressure; omit Prescribed fault compression."));
    if (!prm.get("Prescribed fault compression").empty())
      {
        prescribed_compression = prm.get_double("Prescribed fault compression");
        AssertThrow(!legacy_compression || prescribed_compression == p.surface_pressure,
                    ExcMessage("Conflicting Prescribed fault compression and Surface pressure in growing fault input."));
      }
    AssertThrow(normal_stress.uses_adiabat() || (std::isfinite(prescribed_compression) && prescribed_compression > 0.),
                ExcMessage("Growing fault requires a finite, positive prescribed fault compression in Pa."));
    seed_angle_degrees = prm.get_double("Seed tangent angle");
    const auto coordinates = Utilities::string_to_double(Utilities::split_string_list(prm.get("Forward seed tip")));
    seed_tip_position = Point<2>(coordinates[0], coordinates[1]);
    quadrature_order = prm.get_integer("Section quadrature order");
    const double G = prm.get_double("Shear modulus");
    const double strength = prm.get_double("Cohesion");
    const double Kf = energy.fracture_energy / (energy.geometry.normalization() * energy.length_scale);
    AssertThrow(strength > 0., ExcMessage("Growing fault requires positive cohesion."));
    point_law = {G, prm.get_double("Viscosity"), Kf * 2.*G / (strength * strength)};
    AssertThrow(G > 0. && point_law.eta > 0., ExcMessage("Growing fault requires positive Maxwell scales."));
    resistance_law = MaterialModel::Rheology::FaultResistance::create(prm, MaterialModel::Rheology::FaultResistance::Route::growing);
    resistance_law->validate_midpoint();
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void StripMaterialHistory<dim>::accept_endpoint(const double duration)
  {
    if (trial)
      {
        for (unsigned int k = 0; k < trial->sections.size(); ++k)
          resistance_law->accept_rate(k, trial->sections[k].rate, duration);
        accepted = std::move(*trial);
        trial.reset();
        last_balance = trial_balance;
        balance_available = true;
      }
  }

  template <int dim>
  void StripMaterialHistory<dim>::initialize_native_history()
  {
    if constexpr (dim == 2)
      {
        AssertThrow(dynamic_cast<const GeometryModel::Box<dim>*>(&simulator.get_geometry_model()) &&
                    simulator.get_geometry_model().get_periodic_boundary_pairs().empty(),
                    ExcMessage("Growing fault requires a fixed nonperiodic box."));
        geometry = std::make_unique<MaterialSections>(simulator.get_triangulation(), simulator.get_mpi_communicator(), seed_angle_degrees, quadrature_order);
        accepted.sections.resize(geometry->edges.size() - 1);
        resistance_law->initialize(accepted.sections.size());
        for (const auto &[cell_id, rule] : geometry->cells)
          accepted.cells[cell_id].resize(rule.quadrature.size());
        if (!checkpoint_bytes.empty())
          {
            std::istringstream input(checkpoint_bytes);
            aspect::iarchive ar(input);
            unsigned int saved_quadrature_order;
            double saved_seed_angle;
            std::vector<double> edges;
            std::vector<double> signature;
            ar >> signature;
            AssertThrow(signature == checkpoint_parameters(), ExcMessage("Growing fault physical parameters differ from the checkpoint."));
            ar >> saved_quadrature_order >> saved_seed_angle >> edges >> accepted.step >> accepted.time;
            if (restore_v1)
              {
                AssertThrow(resistance_law->name() == "rate and state", ExcMessage("The v1 growing fault checkpoint requires RSF."));
                std::vector<ArchivedRsfStrip> records;
                ar >> records;
                std::vector<double> ages;
                accepted.sections.assign(records.begin(), records.end());
                for (const auto &record : records)
                  ages.push_back(record.age);
                resistance_law->restore_state(std::move(ages));
              }
            else
              {
                std::string name;
                std::vector<double> state;
                ar >> name >> accepted.sections >> state;
                AssertThrow(name == resistance_law->name(), ExcMessage("Growing fault checkpoint resistance law changed."));
                resistance_law->restore_state(std::move(state));
              }
            AssertThrow(saved_quadrature_order == quadrature_order && saved_seed_angle == seed_angle_degrees && edges == geometry->edges && accepted.sections.size() + 1 == edges.size(),
                        ExcMessage("Incompatible native material strip checkpoint."));
            std::map<std::string, std::pair<std::array<Point<2>, 4>, std::vector<PointHistory>>> points;
            ar >> points;
            for (const auto &cell : simulator.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  auto &history = accepted.cells.at(cell->id());
                  const auto &saved_cell = points.at(cell->id().to_string());
                  for (const auto v : cell->vertex_indices())
                    AssertThrow(saved_cell.first[v] == cell->vertex(v), ExcMessage("Material point locations differ from the fixed checkpoint mesh."));
                  const auto &saved = saved_cell.second;
                  AssertThrow(saved.size() == history.size(), ExcMessage("Incompatible native quadrature checkpoint."));
                  history = saved;
                }
            checkpoint_bytes.clear();
          }
      }
  }

  template <int dim>
  SectionPressures StripMaterialHistory<dim>::section_pressures() const
  {
    require_native_history();
    SectionPressures result;
    result.values.assign(accepted.sections.size(), prescribed_compression);
    if constexpr (dim == 2)
      if (normal_stress.uses_adiabat())
        {
          const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(simulator.get_geometry_model());
          std::vector<double> angles;
          std::vector<unsigned int> formed;
          for (const auto &section : accepted.sections)
            {
              angles.push_back(section.angle);
              formed.push_back(section.formed);
            }
          result.stations = geometry->pressure_stations(seed_tip_position, box.get_origin() + box.get_extents(), angles, formed);
          for (unsigned int k = 0; k < result.values.size(); ++k)
            {
              const auto &point = result.stations[k];
              const double pressure = simulator.get_adiabatic_conditions().pressure(point);
              if (!(std::isfinite(pressure) && pressure > 0.))
                {
                  std::ostringstream message;
                  message << "Invalid adiabatic fault compression " << pressure << " Pa at strip " << k
                          << " position " << point << " m. Friction requires finite positive compression.";
                  AssertThrow(false, ExcMessage(message.str()));
                }
              result.values[k] = pressure;
            }
        }
    return result;
  }

  template <int dim>
  void StripMaterialHistory<dim>::begin_attempt()
  {
    trial.reset();
    if (!geometry)
      initialize_native_history();
  }

  template <int dim>
  void StripMaterialHistory<dim>::stage_candidate(History &&candidate, const Balance &ledger)
  {
    trial = std::make_unique<History>(std::move(candidate));
    trial_balance = ledger;
  }

  template <int dim>
  std::vector<double> StripMaterialHistory<dim>::checkpoint_parameters() const
  {
    const auto &energy = simulator.get_parameters().surface_energy_model;
    std::vector<double> signature = {energy.fracture_energy, energy.length_scale, simulator.get_parameters().phase_discontinuous_penalty,
                                     prescribed_compression, seed_tip_position[0], seed_tip_position[1], point_law.G, point_law.eta, point_law.m
                                    };
    const auto parameters = resistance_law->parameters();
    signature.insert(signature.end(), parameters.begin(), parameters.end());
    return signature;
  }

  template <int dim>
  void StripMaterialHistory<dim>::save(std::map<std::string, std::string> &status) const
  {
    if (!geometry)
      return;
    const auto pressure_profile = normal_stress.checkpoint_profile(simulator);
    using CellRecord = std::pair<std::array<Point<2>, 4>, std::vector<PointHistory>>;
    std::map<std::string, CellRecord> local;
    if constexpr(dim == 2)
      for (const auto &cell : simulator.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto &record = local[cell->id().to_string()];
            for (const auto v : cell->vertex_indices())
              record.first[v] = cell->vertex(v);
            record.second = accepted.cells.at(cell->id());
          }
    auto parts = Utilities::MPI::gather(simulator.get_mpi_communicator(), local);
    if (Utilities::MPI::this_mpi_process(simulator.get_mpi_communicator()) == 0)
      {
        std::map<std::string, CellRecord> all;
        for (auto &part : parts)
          all.merge(part);
        std::ostringstream out;
        {
          aspect::oarchive ar(out);
          ar << checkpoint_parameters() << quadrature_order << seed_angle_degrees << geometry->edges << accepted.step
             << accepted.time << resistance_law->name() << accepted.sections << resistance_law->state() << all;
        }
        status["Native growing fault v2"] = out.str();
        if (normal_stress.uses_adiabat())
          {
            std::ostringstream bytes;
            {
              aspect::oarchive archive(bytes);
              archive << pressure_profile << accepted.friction_pressures.stations
                      << accepted.friction_pressures.values;
            }
            status["Growing friction pressure profile"] = bytes.str();
          }
        if (balance_available)
          {
            std::ostringstream bytes;
            {
              aspect::oarchive archive(bytes);
              archive << last_balance;
            }
            status["Growing fault accepted balance v1"] = bytes.str();
          }
      }
  }

  template <int dim>
  void StripMaterialHistory<dim>::load(const std::map<std::string, std::string> &status)
  {
    AssertThrow(status.count("Native growing fault v1") || status.count("Native growing fault v2"),
                ExcMessage("Checkpoint has no native growing fault history for the selected route."));
    const auto profile = status.find("Growing friction pressure profile");
    AssertThrow((profile != status.end()) == normal_stress.uses_adiabat(),
                ExcMessage("Growing fault friction pressure source differs from the checkpoint."));
    if (profile != status.end())
      {
        std::istringstream bytes(profile->second);
        aspect::iarchive archive(bytes);
        std::string definition;
        archive >> definition >> accepted.friction_pressures.stations >> accepted.friction_pressures.values;
        AssertThrow(definition == normal_stress.checkpoint_profile(simulator),
                    ExcMessage("Growing fault friction pressure profile differs from the checkpoint."));
      }
    restore_v1 = status.find("Native growing fault v2") == status.end();
    checkpoint_bytes = status.at(restore_v1 ? "Native growing fault v1" : "Native growing fault v2");
    const auto entry = status.find("Growing fault accepted balance v1");
    balance_available = entry != status.end();
    if (balance_available)
      {
        std::istringstream bytes(entry->second);
        aspect::iarchive archive(bytes);
        archive >> last_balance;
      }
  }

  template class StripMaterialHistory<2>;
  template class StripMaterialHistory<3>;
}
