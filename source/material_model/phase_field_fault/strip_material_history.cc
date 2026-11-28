/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/input_compatibility.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/material_model/rheology/fault_resistance.h>
#include <aspect/geometry_model/box.h>
#include <aspect/adiabatic_conditions/interface.h>

#include <deal.II/fe/fe_values.h>

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
    using MaterialModel::FaultInputCompatibility::Use;
    prm.enter_subsection("Material model");
    const char *const input_sections[] = {"Phase field fault", "Growing fault"};
    for (const auto *section : input_sections)
      {
        prm.enter_subsection(section);
        declare_entry(prm, Use::growing, "Prescribed fault compression", "", Patterns::List(Patterns::Double(0.), 0, 1),
                      "Positive compression in Pa used by the signed fault resistance law. An empty entry "
                      "uses the explicitly supplied Surface pressure from legacy growing fault inputs.");
        declare_entry(prm, Use::growing, "Initial deviatoric stress", "0, 0, 0", Patterns::List(Patterns::Double(), 3, 3),
                      "Uniform Cartesian deviatoric stress on a fresh start, in Pa, ordered xx, yy, xy. "
                      "Plane strain retains zz = -(xx + yy). A restart restores the saved point stresses.");
        declare_entry(prm, Use::growing, "Seed tangent angle", "0", Patterns::Double(),
                      "Angle in degrees counterclockwise from x.");
        declare_entry(prm, Use::growing, "Forward seed tip", "0,0", Patterns::List(Patterns::Double(), 2, 2),
                      "Physical tip of the initial contiguous seed in metres.");
        declare_entry(prm, Use::growing, "Target strip width", "0", Patterns::Double(0.),
                      "Maximum material-strip spacing in metres. A positive width establishes the physical "
                      "stations independently of bulk refinement. Zero derives the spacing from the finest "
                      "bulk cell on a fixed mesh.");
        declare_entry(prm, Use::growing, "Section quadrature order", "6", Patterns::Integer(2),
                      "Gauss points per strip subinterval and chord. Remeshing requires at least four points "
                      "to integrate the phase-weighted history moments.");
        declare_entry(prm, Use::growing, "Cohesion", "1000", Patterns::Double(0),
                      "Cohesive strength in Pa.");
        prm.leave_subsection();
      }

    // Keep the historical strip defaults, including defaults of external laws.
    // The resolver selects them only for a midpoint fault phase.
    prm.enter_subsection("Growing fault");
    declare_entry(prm, "Shear modulus", "1e6", Patterns::Double(0),
                  "Shear modulus in Pa.");
    declare_entry(prm, "Viscosity", "1e8", Patterns::Double(0),
                  "Maxwell viscosity in Pa s.");
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
    // Both the strip construction and pre-solve fault marking use this box.
    // The parent material has already required zero initial topography.
    AssertThrow(Plugins::plugin_type_matches<GeometryModel::Box<dim>>(simulator.get_geometry_model()) &&
                simulator.get_geometry_model().get_periodic_boundary_pairs().empty(),
                ExcMessage("Growing fault requires a fixed nonperiodic box."));
    using NormalStress = MaterialModel::Rheology::FrictionNormalStress;
    normal_stress = NormalStress::parse(prm, NormalStress::Source::prescribed_constant);
    const bool legacy_compression = !prm.get("Surface pressure").empty();
    prm.enter_subsection("Material model");
    prm.enter_subsection("Phase field fault");
    prescribed_compression = normal_stress.uses_adiabat() ? 0. : p.surface_pressure;
    AssertThrow(!normal_stress.uses_adiabat() || prm.get("Prescribed fault compression").empty(),
                ExcMessage("Adiabatic friction supplies the complete pressure; omit Prescribed fault compression."));
    if (!prm.get("Prescribed fault compression").empty())
      {
        prescribed_compression = prm.get_double("Prescribed fault compression");
        AssertThrow(!legacy_compression || prescribed_compression == p.surface_pressure,
                    ExcMessage("Conflicting Prescribed fault compression and Surface pressure. "
                               "Material model / Phase field fault / Prescribed fault compression "
                               "(legacy Material model / Growing fault / Prescribed fault compression) "
                               "must agree with Surface pressure when both are supplied."));
      }
    AssertThrow(normal_stress.uses_adiabat() || (std::isfinite(prescribed_compression) && prescribed_compression > 0.),
                ExcMessage("Growing fault requires a finite, positive prescribed fault compression in Pa."));
    const auto stress = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial deviatoric stress")));
    initial_deviatoric_stress[0][0] = stress[0];
    initial_deviatoric_stress[1][1] = stress[1];
    // A zero in-plane trace retains the positive zero of the original fresh history.
    initial_deviatoric_stress[2][2] = 0. - (stress[0] + stress[1]);
    initial_deviatoric_stress[0][1] = stress[2];
    AssertThrow(std::isfinite(initial_deviatoric_stress[2][2]) && std::isfinite(stress[2]),
                ExcMessage("Initial deviatoric stress must have finite Cartesian components including zz."));
    seed_angle_degrees = prm.get_double("Seed tangent angle");
    const auto coordinates = Utilities::string_to_double(Utilities::split_string_list(prm.get("Forward seed tip")));
    seed_tip_position = Point<2>(coordinates[0], coordinates[1]);
    requested_strip_width = prm.get_double("Target strip width");
    quadrature_order = prm.get_integer("Section quadrature order");
    const double G = prm.get_double("Shear modulus");
    const double strength = prm.get_double("Cohesion");
    const double Kf = energy.fracture_energy / (energy.geometry.normalization() * energy.length_scale);
    AssertThrow(strength > 0., ExcMessage("Growing fault requires positive cohesion."));
    point_law = {G, prm.get_double("Viscosity"), Kf * 2.*G / (strength * strength)};
    AssertThrow(G > 0. && point_law.eta > 0., ExcMessage("Growing fault requires positive Maxwell scales."));
    // The SDK callback reads the same scoped declaration it registered for
    // the growing route, including its own defaults, patterns and actions.
    prm.leave_subsection();
    prm.enter_subsection("Growing fault");
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
        geometry = std::make_unique<MaterialSections>(simulator.get_triangulation(), simulator.get_mpi_communicator(), seed_angle_degrees, quadrature_order, requested_strip_width);
        accepted.sections.resize(geometry->edges.size() - 1);
        resistance_law->initialize(accepted.sections.size());
        PointHistory initial_point;
        if (checkpoint_bytes.empty())
          initial_point.stress = initial_deviatoric_stress;
        for (const auto &[cell_id, rule] : geometry->cells)
          accepted.cells[cell_id].resize(rule.quadrature.size(), initial_point);
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
            // Material IDs refer to these physical intervals. The bulk mesh
            // target and the requested upper bound on strip spacing may
            // change if the realized partition stays identical.
            AssertThrow(edges == geometry->edges,
                        ExcMessage("Physical strip edges differ from the growing fault checkpoint."));
            AssertThrow(saved_quadrature_order == quadrature_order && saved_seed_angle == seed_angle_degrees,
                        ExcMessage("Incompatible native material strip checkpoint."));
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
            AssertThrow(accepted.sections.size() + 1 == edges.size(),
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


  namespace
  {
    constexpr unsigned int tensor_size = Stress::n_independent_components;
    constexpr unsigned int signed_index = 2 * tensor_size;
    constexpr unsigned int absolute_index = signed_index + 1;
    using Components = std::array<double,absolute_index+1>;

    Components components(const PointHistory &point)
    {
      Components result;
      for (unsigned int i=0; i<tensor_size; ++i)
        {
          result[i] = point.stress.access_raw_entry(i);
          result[tensor_size+i] = point.cohesive_strain.access_raw_entry(i);
        }
      result[signed_index] = point.signed_shear;
      result[absolute_index] = point.absolute_shear;
      return result;
    }

    /** Reconstruction on one old cell/strip intersection in parent reference
     * coordinates. Stress is affine in physical area, while cohesive strain
     * and shear use a phase-weighted affine basis. The constant Cartesian
     * Jacobian cancels from the local projection equations. */
    struct StripPatch
    {
      Point<2> center, phase_center;
      Components mean, dx, dy;

      PointHistory evaluate(const Point<2> &point, const double phase) const
      {
        Components values;
        for (unsigned int i=0; i<values.size(); ++i)
          {
            const auto d = point-(i<tensor_size ? center : phase_center);
            values[i] = (i<tensor_size ? 1. : phase)*(mean[i]+dx[i]*d[0]+dy[i]*d[1]);
          }
        PointHistory result;
        for (unsigned int i=0; i<tensor_size; ++i)
          {
            result.stress.access_raw_entry(i) = values[i];
            result.cohesive_strain.access_raw_entry(i) = values[tensor_size+i];
          }
        result.signed_shear = values[signed_index];
        result.absolute_shear = values[absolute_index];
        return result;
      }

      template <class Archive> void serialize(Archive &ar, const unsigned int)
      {
        ar &center &phase_center &mean &dx &dy;
      }
    };

    // The minimum of an affine function on the convex intersection occurs
    // at a box vertex or at a strip edge crossing a box edge.
    std::vector<Point<2>> strip_corners(const std::array<double,4> &station,
                                        const double                lower,
                                        const double                upper)
    {
      std::vector<Point<2>> points;
      for (const auto v : GeometryInfo<2>::vertex_indices())
        if (station[v] >= lower && station[v] <= upper)
          points.push_back(GeometryInfo<2>::unit_cell_vertex(v));
      for (unsigned int line=0; line<GeometryInfo<2>::lines_per_cell; ++line)
        {
          const unsigned int v = GeometryInfo<2>::line_to_cell_vertices(line,0);
          const unsigned int w = GeometryInfo<2>::line_to_cell_vertices(line,1);
          const Point<2> a = GeometryInfo<2>::unit_cell_vertex(v), b = GeometryInfo<2>::unit_cell_vertex(w);
          for (const double edge : {lower,upper})
            if (edge > std::min(station[v],station[w]) && edge < std::max(station[v],station[w]))
              points.push_back(a + ((edge-station[v])/(station[w]-station[v]))*(b-a));
        }
      return points;
    }

    std::vector<double> accepted_phase_on_rule(const SimulatorAccess<2>                           &simulator,
                                               const Triangulation<2>::cell_iterator              &cell,
                                               const Quadrature<2>                                &rule,
                                               const unsigned int                                  phase_index,
                                               const unsigned int                                  history_step)
    {
      const auto component = simulator.introspection().component_indices.compositional_fields[phase_index];
      const auto &field = (history_step == simulator.get_timestep_number() ?
                           simulator.get_solution() : simulator.get_old_solution());
      FEValues<2> values(simulator.get_mapping(),simulator.get_fe(),rule,update_values);
      values.reinit(cell->as_dof_handler_iterator(simulator.get_dof_handler()));
      std::vector<double> phi(rule.size());
      values[FEValuesExtractors::Scalar(component)].get_function_values(field,phi);
      for (double &value : phi)
        value = physical_phase(value);
      return phi;
    }

    StripPatch fit_strip_patch(const Quadrature<2>             &rule,
                               const std::vector<PointHistory> &history,
                               const std::vector<double>       &phase,
                               const unsigned int               first,
                               const unsigned int               end,
                               const std::vector<Point<2>>     &corners)
    {
      StripPatch fit{};
      // Enforce the constant and two linear test moments on this physical
      // cut. Stress is affine; cohesive strain and shear use phase times an
      // affine function. They then vanish on intact material. In particular
      // cohesive energy remains integrable as phase tends to zero. Form the
      // phase-weighted mass and the native history load directly, avoiding
      // division by the tiny phase values near a crack's edge.
      for (const bool weight_by_phase : {false,true})
        {
          const unsigned int begin_i = weight_by_phase ? tensor_size : 0;
          const unsigned int end_i = weight_by_phase ? fit.mean.size() : tensor_size;
          long double mass = 0.;
          std::array<long double,2> centroid{};
          std::array<long double,Components{}.size()> sum{}, moment_x{}, moment_y{};
          for (unsigned int q=first; q<end; ++q)
            {
              const long double w = rule.weight(q);
              const long double weight = w*(weight_by_phase ? phase[q] : 1.);
              mass += weight;
              for (unsigned int d=0; d<2; ++d)
                centroid[d] += weight*rule.point(q)[d];
              const auto data = components(history[q]);
              for (unsigned int i=begin_i; i<end_i; ++i)
                sum[i] += w*data[i];
            }
          if (mass == 0.)
            continue;
          auto &center = weight_by_phase ? fit.phase_center : fit.center;
          for (unsigned int d=0; d<2; ++d)
            {
              centroid[d] /= mass;
              center[d] = centroid[d];
            }
          for (unsigned int i=begin_i; i<end_i; ++i)
            {
              sum[i] /= mass;
              fit.mean[i] = sum[i];
            }
          long double xx=0., xy=0., yy=0.;
          for (unsigned int q=first; q<end; ++q)
            {
              const long double w = rule.weight(q);
              const long double factor = weight_by_phase ? phase[q] : 1.;
              const long double weight = w*factor;
              const long double x = rule.point(q)[0]-centroid[0], y = rule.point(q)[1]-centroid[1];
              const auto data = components(history[q]);
              xx += weight*x*x;
              xy += weight*x*y;
              yy += weight*y*y;
              for (unsigned int i=begin_i; i<end_i; ++i)
                {
                  moment_x[i] += w*x*(data[i]-factor*sum[i]);
                  moment_y[i] += w*y*(data[i]-factor*sum[i]);
                }
            }
          const long double det = xx*yy-xy*xy;
          if (det > 64.*std::numeric_limits<long double>::epsilon()*xx*yy)
            for (unsigned int i=begin_i; i<end_i; ++i)
              {
                fit.dx[i] = (yy*moment_x[i]-xy*moment_y[i])/det;
                fit.dy[i] = (xx*moment_y[i]-xy*moment_x[i])/det;
              }
        }

      // Accumulated absolute shear must dominate both signs of the net
      // shear. Limit slopes only if either bound would fail on this polygon.
      // Reconstruct net shear directly, including when the two accumulated
      // directions nearly cancel. The limiter preserves both strip integrals.
      double scale = 1.;
      for (const auto &corner : corners)
        {
          const auto d = corner-fit.phase_center;
          for (double sign : {-1.,1.})
            {
              const double margin = fit.mean[absolute_index]+sign*fit.mean[signed_index];
              const double increment = (fit.dx[absolute_index]+sign*fit.dx[signed_index])*d[0]
                                       +(fit.dy[absolute_index]+sign*fit.dy[signed_index])*d[1];
              if (increment < 0. && margin < -increment)
                scale = std::min(scale, std::max(0.,margin)/(-increment));
            }
        }
      for (const unsigned int i : {signed_index,absolute_index})
        {
          fit.dx[i] *= scale;
          fit.dy[i] *= scale;
        }
      return fit;
    }
  }

  template <int dim>
  void StripMaterialHistory<dim>::validate_refinement() const
  {
    AssertThrow(std::isfinite(requested_strip_width) && requested_strip_width > 0.,
                ExcMessage("Growing fault mesh refinement requires a positive Target strip width."));
    AssertThrow(simulator.get_parameters().coarsening_fraction == 0.,
                ExcMessage("Growing fault mesh refinement requires zero Coarsening fraction."));
    // A Cartesian Q2 phase times two affine factors and the cut-chord
    // Jacobian has degree at most seven along the outer strip coordinate.
    AssertThrow(quadrature_order >= 4,
                ExcMessage("Growing fault remeshing requires Section quadrature order >= 4 for phase-weighted history transfer."));
  }

  template <int dim>
  double StripMaterialHistory<dim>::incoming_seed_coordinate() const
  {
    return sections().coordinate(seed_tip_position) - simulator.get_parameters().surface_energy_model.length_scale;
  }

  template <int dim>
  Point<2> StripMaterialHistory<dim>::incoming_seed_anchor() const
  {
    const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(simulator.get_geometry_model());
    const auto tangent = seed_tangent();
    Point<2> tip = seed_tip_position;
    double distance = simulator.get_parameters().surface_energy_model.length_scale;
    for (unsigned int d=0; d<2; ++d)
      {
        const double lower = box.get_origin()[d];
        const double upper = lower + box.get_extents()[d];
        const double tolerance = 64.*std::numeric_limits<double>::epsilon()*box.get_extents()[d]/box.get_repetitions()[d];
        AssertThrow(tip[d] >= lower-tolerance && tip[d] <= upper+tolerance,
                    ExcMessage("Forward seed tip lies outside the mesh."));
        tip[d] = std::clamp(tip[d],lower,upper);
        if (tangent[d] != 0.)
          {
            const double wall = tangent[d] > 0. ? lower : upper;
            distance = std::min(distance, (tip[d]-wall)/tangent[d]);
          }
      }
    return tip - distance*tangent;
  }

  template <int dim>
  void StripMaterialHistory<dim>::prepare_history_transfer(parallel::distributed::Triangulation<dim> &mesh)
  {
    if constexpr (dim == 2)
      {
        unsigned int flags = 0;
        for (const auto &cell : mesh.active_cell_iterators())
          if (cell->is_locally_owned())
            flags = std::max(flags, cell->coarsen_flag_set() ? 2u : cell->refine_flag_set() ? 1u : 0u);
        flags = Utilities::MPI::max(flags,simulator.get_mpi_communicator());
        AssertThrow(flags != 2, ExcMessage("Growing fault history supports refinement only. Set Coarsening fraction = 0."));
        if (!flags || !geometry)
          return;
        validate_refinement();
        AssertThrow(transfer_handle == numbers::invalid_unsigned_int, ExcInternalError());
        trial.reset();
        transfer_handle = mesh.register_data_attach(
          [this](const auto &cell, const CellStatus status)
        {
          const auto &old = accepted.cells.at(cell->id());
          if (status == CellStatus::cell_will_persist)
            return dealii::Utilities::pack(old, false);
          AssertThrow(status == CellStatus::cell_will_be_refined, ExcInternalError());
          const auto &rule = geometry->cells.at(cell->id());
          const auto phase = accepted_phase_on_rule(simulator,cell,rule.quadrature,phase_field_index,accepted.step);
          std::array<double,4> station;
          for (const unsigned int v : cell->vertex_indices())
            station[v] = geometry->coordinate(cell->vertex(v));
          std::map<unsigned int,StripPatch> fits;
          unsigned int first = 0;
          while (first < old.size())
            {
              const unsigned int k = rule.section[first];
              unsigned int end = first+1;
              while (end < old.size() && rule.section[end] == k)
                ++end;
              fits.emplace(k,fit_strip_patch(rule.quadrature,old,phase,first,end,
                                             strip_corners(station,geometry->edges[k],geometry->edges[k+1])));
              first = end;
            }
          return dealii::Utilities::pack(fits, false);
        }, true);
      }
  }

  template <int dim>
  void StripMaterialHistory<dim>::finish_history_transfer(parallel::distributed::Triangulation<dim> &mesh)
  {
    if constexpr (dim == 2)
      {
        if (transfer_handle == numbers::invalid_unsigned_int)
          return;
        // Source station coordinates are fixed by the material resolution.
        // Rebuild only the intersections and rank-local ownership.
        auto refined = std::make_unique<MaterialSections>(mesh, simulator.get_mpi_communicator(),
                                                          seed_angle_degrees, quadrature_order, requested_strip_width);
        AssertThrow(refined->edges == geometry->edges, ExcMessage("Growing fault refinement changed the physical strip partition."));
        geometry = std::move(refined);
        accepted.cells.clear();
        mesh.notify_ready_to_unpack(transfer_handle,
                                    [this](const auto &cell, const CellStatus status, const auto &bytes)
        {
          if (status == CellStatus::cell_will_persist)
            accepted.cells.emplace(cell->id(), dealii::Utilities::unpack<std::vector<PointHistory>>(bytes.begin(),bytes.end(),false));
          else
            {
              AssertThrow(status == CellStatus::cell_will_be_refined, ExcInternalError());
              const auto fits = dealii::Utilities::unpack<std::map<unsigned int,StripPatch>>(bytes.begin(),bytes.end(),false);
              for (unsigned int child=0; child<cell->n_children(); ++child)
                {
                  const auto id = cell->child(child)->id();
                  const auto &rule = geometry->cells.at(id);
                  const auto phase = accepted_phase_on_rule(simulator,cell->child(child),rule.quadrature,
                                                            phase_field_index,accepted.step);
                  auto &values = accepted.cells[id];
                  values.reserve(rule.quadrature.size());
                  for (unsigned int q=0; q<rule.quadrature.size(); ++q)
                    {
                      // The distributed p4est triangulation refines Cartesian
                      // quads isotropically. Use the parent's recorded case to
                      // map the new material point into its old reconstruction.
                      const auto parent_point = GeometryInfo<2>::child_to_cell_coordinates(rule.quadrature.point(q),
                                                                                           child,
                                                                                           cell->refinement_case());
                      values.push_back(fits.at(rule.section[q]).evaluate(parent_point,phase[q]));
                    }
                }
            }
        });
        transfer_handle = numbers::invalid_unsigned_int;
        AssertThrow(accepted.cells.size() == geometry->cells.size(), ExcMessage("Growing mesh transfer missed native cell records."));
      }
  }

  template <int dim>
  Point<2> StripMaterialHistory<dim>::phase_supported_tip(const std::vector<double>         &angles,
                                                          const std::vector<unsigned int>   &formed,
                                                          const LinearAlgebra::BlockVector  &solution) const
  {
    Point<2> tip=seed_tip_position;
    if constexpr (dim==2)
      {
        const auto &sections=this->sections();
        const double seed_xi=sections.coordinate(tip);
        std::vector<double> stations;
        for (unsigned int k=0; k<formed.size(); ++k)
          stations.push_back(.5*(sections.edges[k]+sections.edges[k+1]));
        const auto beyond=std::upper_bound(stations.begin(),stations.end(),seed_xi)-stations.begin();
        if (beyond==static_cast<std::ptrdiff_t>(stations.size()) || !formed[beyond])
          return tip;

        const auto graph=sections.accepted_graph(tip,angles,formed);
        const auto &fe=simulator.get_fe();
        const auto c=simulator.introspection().component_indices.compositional_fields[phase_field_index];
        const auto &element=fe.base_element(fe.component_to_base_index(c).first);
        std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
        const unsigned int anchor=std::lower_bound(stations.begin(),stations.end(),incoming_seed_coordinate())-stations.begin();
        for (unsigned int k=anchor; k<formed.size() && (k<beyond || formed[k]); ++k)
          {
            const auto position=graph.at(stations[k]);
            double trace=0.;
            for (const auto &cell : simulator.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned() && cell->bounding_box().point_inside(position,1.e-12))
                {
                  const auto unit=simulator.get_mapping().transform_real_to_unit_cell(cell,position);
                  if (!GeometryInfo<2>::is_inside_unit_cell(unit,1.e-12)) continue;
                  cell->get_dof_indices(ids);
                  double value=0.;
                  for (unsigned int j=0; j<element.n_dofs_per_cell(); ++j)
                    value+=element.shape_value(j,unit)*solution[ids[fe.component_to_system_index(c,j)]];
                  trace=std::max(trace,value);
                }
            // A formed station carries the directional kernel forward only
            // while the maximum of its two DG phase traces supports the tip.
            constexpr double mechanical_tip_phase = 0.05;
            if (Utilities::MPI::max(trace,simulator.get_mpi_communicator()) < mechanical_tip_phase) break;
            if (stations[k]>seed_xi) tip=position;
          }
      }
    return tip;
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
        if (requested_strip_width > 0.)
          status["Growing fault target strip width"] = Utilities::to_string(requested_strip_width);
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
