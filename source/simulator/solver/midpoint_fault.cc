/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/simulator/solver/midpoint_system.h>
#include <aspect/simulator/solver/midpoint_fault.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <numeric>
#include <aspect/simulator.h>
#include <aspect/gravity_model/interface.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools_constraints.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/lac/sparse_direct.h>

namespace aspect::GrowingFault
{
  template <int dim>
  void parse_midpoint_wall_velocity(ParameterHandler &prm, Parameters<dim> &parameters)
  {
    prm.enter_subsection("Boundary velocity model");
    const auto entries = Utilities::split_string_list(prm.get("Prescribed velocity boundary indicators"));
    bool uses_function = false;
    for (const auto &entry : entries)
      {
        const auto parts = Utilities::split_string_list(entry, ':');
        AssertThrow(parts.size() == 2, ExcMessage("Expected a boundary indicator and velocity plugin separated by ':'."));
        AssertThrow(parts[1] == "function" || parts[1] == "zero velocity",
                    ExcMessage("Growing fault cannot evaluate wall velocity plugin '" + parts[1] +
                               "' at the attempted midpoint."));
        uses_function |= parts[1] == "function";
      }
    if (uses_function)
      {
        prm.enter_subsection("Function");
        const auto coordinates = Utilities::Coordinates::string_to_coordinate_system(prm.get("Coordinate system"));
        auto function = std::make_shared<Functions::ParsedFunction<dim>>(dim);
        function->parse_parameters(prm);
        const double unit = parameters.convert_to_years ? year_in_seconds : 1.;
        // Growth requires a Cartesian box, for which the release velocity
        // function rejects spherical unit vectors. Its arguments may still
        // be expressed in depth or spherical coordinates.
        parameters.midpoint_wall_velocity =
          [function, coordinates, unit](const GeometryModel::Interface<dim> &geometry,
                                        const Point<dim>                   &position,
                                        const double                        seconds)
        {
          if (function->get_time() != seconds / unit)
            function->set_time(seconds / unit);
          const auto point = geometry.cartesian_to_other_coordinates(position, coordinates);
          const auto arguments = Utilities::convert_array_to_point<dim>(point.get_coordinates());
          Tensor<1, dim> velocity;
          for (unsigned int d = 0; d < dim; ++d)
            velocity[d] = function->value(arguments, d) / unit;
          return velocity;
        };
        prm.leave_subsection();
      }
    prm.leave_subsection();
  }

  template void parse_midpoint_wall_velocity(ParameterHandler &, Parameters<2> &);
  template void parse_midpoint_wall_velocity(ParameterHandler &, Parameters<3> &);

  namespace
  {
    // The 2-D displacement has zero out-of-plane strain. Its 3-D deviator
    // includes zz, which contributes to native stress, energy and tangent.
    Stress plane_strain(const SymmetricTensor<2, 2> &d)
    {
      Stress embedded;
      embedded[0][0] = d[0][0];
      embedded[1][1] = d[1][1];
      embedded[0][1] = d[0][1];
      return deviator(embedded);
    }

    // Interpolate at the attempted midpoint in the same boundary order and
    // component masks as Simulator::compute_current_velocity_boundary_constraints.
    // Leave the ordinary boundary manager and its endpoint clock untouched.
    AffineConstraints<double> midpoint_velocity_constraints(const SimulatorAccess<2> &sim, const double time)
    {
      const auto &manager = sim.get_boundary_velocity_manager();
      const auto &intro = sim.introspection();
      AffineConstraints<double> walls;
      const auto &mapping = sim.get_mapping();
      const auto &dofs = sim.get_dof_handler();
      for (const auto id : manager.get_zero_boundary_velocity_indicators())
        VectorTools::interpolate_boundary_values(mapping, dofs, id, Functions::ZeroFunction<2>(intro.n_components),
                                                 walls, intro.component_masks.velocities);
      const auto &tangential = manager.get_tangential_boundary_velocity_indicators();
      if (!tangential.empty())
        VectorTools::compute_no_normal_flux_constraints(dofs, intro.component_indices.velocities[0], tangential,
                                                        walls, mapping);

      const auto &ids = manager.get_active_plugin_boundary_indicators();
      const auto &names = manager.get_active_plugin_names();
      for (const auto id : manager.get_prescribed_boundary_velocity_indicators())
        {
          unsigned int functions = 0;
          for (unsigned int i = 0; i < names.size(); ++i)
            if (ids[i] == id && names[i] == "function")
              ++functions;
          Utilities::VectorFunctionFromVelocityFunctionObject<2> velocity(intro.n_components,
                                                                          [&](const Point<2> &point)
          {
            return functions == 0 ? Tensor<1, 2>() :
                   functions * sim.get_parameters().midpoint_wall_velocity(sim.get_geometry_model(), point, time);
          });
          VectorTools::interpolate_boundary_values(mapping, dofs, id, velocity, walls, manager.get_component_mask(id));
        }
      walls.close();
      return walls;
    }

  }

  MidpointSystem::MidpointSystem(const SimulatorAccess<2>         &simulator,
                                 const StripMaterialHistory<2>    &m,
                                 const LinearAlgebra::BlockVector &solution)
    : sim(simulator),
      material(m),
      comm(sim.get_mpi_communicator()),
      h(.5 * sim.get_timestep()),
      G(m.maxwell().G),
      B(m.maxwell().relaxation(h)),
      Kf(sim.get_parameters().surface_energy_model.fracture_energy / (sim.get_parameters().surface_energy_model.geometry.normalization()
                                                                      *sim.get_parameters().surface_energy_model.length_scale)),
      field(m.phase_field()),
      phase_block(sim.introspection().block_indices.compositional_fields[field]),
      nv(solution.block(0).size()),
      phase_begin(nv + solution.block(1).size()),
      n(phase_begin + solution.block(phase_block).size()),
      fracture(sim, field),
      friction_pressures(m.section_pressures())
  {
    AssertThrow(h > 0., ExcMessage("Midpoint mechanics needs a positive stage duration."));
    // Both admissibility bounds depend on this attempt's half-duration and
    // hold on every rank. Let the time-step manager retry from the same parent.
    if (h * G > m.maxwell().eta)
      throw ExcNonlinearSolverNoConvergence();
    for (const double pressure : friction_pressures.values)
      if (!m.friction().admits_duration(h, pressure))
        throw ExcNonlinearSolverNoConvergence();
    const auto &old = sim.get_old_solution();
    const auto &fe = sim.get_fe();
    const auto &intro = sim.introspection();
    const auto c = intro.component_indices.compositional_fields[field];
    const auto offset = solution.get_block_indices().block_start(phase_block);
    initial.assign(n, 0.);
    for (const unsigned int block :
    {
      0u, 1u, phase_block
    })
    for (const auto i : intro.index_sets.system_partitioning[block])
      {
        const Index block_begin = block == 0 ? 0 : block == 1 ? nv : phase_begin;
        const double stage_scale = block == 0 ? h : block == 1 ? 1. / G : 1.;
        initial[block_begin + i] = old.block(block)[i] * stage_scale;
      }
    Utilities::MPI::sum(initial, comm, initial);
    lower = initial;
    upper = lower;
    for (Index i = phase_begin; i < n; ++i)
      {
        upper[i] = .5 * (1. + lower[i]);
        AssertThrow(lower[i] <= upper[i], ExcMessage("Infeasible incoming Bernstein obstacle in growing fault."));
      }
    mass.assign(n, 0.);
    for (const auto &[row, value] : fracture.mass)
      mass[phase_begin + row] = value;
    Utilities::MPI::sum(mass, comm, mass);

    const double parent_time = m.history().step == 0 ? sim.get_parameters().start_time : m.history().time;
    const auto walls = midpoint_velocity_constraints(sim, parent_time + h);
    const auto &constraints = sim.get_current_constraints();
    const auto boundary = DoFTools::extract_boundary_dofs(sim.get_dof_handler(), intro.component_masks.velocities);
    unsigned int invalid = 0;
    for (const auto i : boundary)
      if (intro.index_sets.system_partitioning[0].is_element(i))
        invalid |= !constraints.is_constrained(i) || !walls.is_constrained(i);
    std::map<Index, double> owned_constraints;
    for (const auto &line : constraints.get_lines())
      {
        if (line.index < nv && intro.index_sets.system_partitioning[0].is_element(line.index))
          {
            invalid |= !line.entries.empty();
            const double velocity = walls.is_constrained(line.index) ? walls.get_inhomogeneity(line.index) : line.inhomogeneity;
            owned_constraints[line.index] = h * velocity;
          }
        if (line.index >= nv && line.index < phase_begin && intro.index_sets.system_partitioning[1].is_element(line.index - nv))
          {
            invalid |= !line.entries.empty();
            owned_constraints[line.index] = line.inhomogeneity / G;
          }
        if (line.index >= offset && line.index < offset + n - phase_begin)
          invalid = 1;
      }
    AssertThrow(!Utilities::MPI::max(invalid, comm), ExcMessage("Midpoint mechanics requires fully prescribed wall velocity with simple prescribed values, at most one pressure datum and unconstrained DG phase."));
    for (auto part : Utilities::MPI::all_gather(comm, owned_constraints))
      fixed.merge(part);
    auto datum = fixed.lower_bound(nv);
    if (datum == fixed.end())
      datum = fixed.emplace(nv, 0.).first;
    AssertThrow(std::next(datum) == fixed.end(), ExcMessage("Midpoint fault supports one prescribed physical pressure datum."));
    const double gauge = initial[datum->first] - datum->second;
    for (Index i = nv; i < phase_begin; ++i)
      initial[i] -= gauge;
    for (const auto &[i, v] : fixed)
      initial[i] = v;

    unsigned int nonzero_gravity = 0;
    double flux = 0., flux_scale = 0.;
    const auto &sections = m.sections();
    for (const auto &s : m.history().sections)
      {
        angles.push_back(s.angle);
        formed.push_back(s.formed);
      }
    eligible.assign(angles.size(), 0);
    std::vector<Index> ids(fe.n_dofs_per_cell());
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &rule = sections.cells.at(cell->id());
          FEValues<2> values(sim.get_mapping(), fe, rule.quadrature, update_values | update_gradients | update_JxW_values | update_quadrature_points);
          values.reinit(cell);
          cell->get_dof_indices(ids);
          auto &data = cells[cell->id()];
          std::vector<unsigned int> local;
          for (unsigned int j = 0; j < ids.size(); ++j)
            if (ids[j] < phase_begin || fe.system_to_component_index(j).first == c)
              {
                local.push_back(j);
                data.indices.push_back(ids[j] < phase_begin ? ids[j] : phase_begin + ids[j] - offset);
              }
          for (unsigned int q = 0; q < rule.quadrature.size(); ++q)
            {
              Sample p {rule.section[q], values.quadrature_point(q), values.JxW(q), 0., 0., {}};
              nonzero_gravity |= sim.get_gravity_model().gravity_vector(p.position) != Tensor<1, 2>();
              for (const auto j : local)
                {
                  p.shapes.push_back({plane_strain(values[intro.extractors.velocities].symmetric_gradient(j, q)),
                                      values[intro.extractors.velocities].divergence(j, q), values[intro.extractors.pressure].value(j, q),
                                      values.shape_value_component(j, q, c)
                                     });
                  if (ids[j] < nv)
                    {
                      const double term = p.area * p.shapes.back().div * (initial[ids[j]] / h);
                      flux += term;
                      flux_scale += std::abs(term);
                    }
                  if (ids[j] >= phase_begin)
                    p.old_phase += p.shapes.back().phase * old[ids[j]];
                }
              data.points.push_back(std::move(p));
            }
        }
    const unsigned int gravity_owners = Utilities::MPI::sum(nonzero_gravity, comm);
    AssertThrow(gravity_owners == 0, ExcMessage("Growing fault mechanics requires zero gravity at every native integration point; nonzero on " +
                                                std::to_string(gravity_owners) + " of " + std::to_string(Utilities::MPI::n_mpi_processes(comm)) + " ranks."));
    flux = Utilities::MPI::sum(flux, comm);
    flux_scale = Utilities::MPI::sum(flux_scale, comm);
    // Sum of all Q1 continuity tests is the constant test. Its integral
    // equals the prescribed net wall flux regardless of the interior values.
    AssertThrow(std::abs(flux) <= 1.e-12 * flux_scale, ExcMessage("Incompatible prescribed net velocity flux: " + Utilities::to_string(flux) + " m^2/s."));
  }

  MidpointSystem::Evaluation MidpointSystem::assemble_stage(const Values &stage_fields,
                                                            const bool   tangent,
                                                            const bool   publish) const
  {
    const auto &maxwell = material.maxwell();
    const auto &parent = material.history();
    const auto &geometry = material.sections();
    const unsigned int n_strips = angles.size();
    std::vector<Stress> planes(n_strips);
    Values lengths(n_strips);
    for (unsigned int k = 0; k < n_strips; ++k)
      {
        planes[k] = geometry.shear(angles[k]);
        lengths[k] = geometry.length(k, angles[k]);
      }
    Evaluation evaluation;
    evaluation.residual.assign(n, 0.);
    // Integrate free rates and mobilities with physical area/current length.
    // The root then supplies one signed traction to every point in the strip.
    SectionTransfer transfer(n_strips);
    struct Trial
    {
      Stress strain_increment;
      double divergence = 0., scaled_pressure = 0., phase = 0.;
    };
    std::map<CellId, std::vector<Trial>> trials;
    for (const auto &[id, cell] : cells)
      {
        auto &states = trials[id];
        for (unsigned int q = 0; q < cell.points.size(); ++q)
          {
            const auto &point = cell.points[q];
            const auto k = point.strip;
            Trial trial;
            for (unsigned int j = 0; j < cell.indices.size(); ++j)
              {
                const double v = stage_fields[cell.indices[j]];
                trial.strain_increment += v * point.shapes[j].strain;
                trial.divergence += v * point.shapes[j].div;
                trial.scaled_pressure += v * point.shapes[j].pressure;
                trial.phase += v * point.shapes[j].phase;
              }
            const auto free = maxwell.response(parent.cells.at(id)[q], trial.strain_increment, trial.phase, h, planes[k], 0.);
            transfer.add(k, point.area / lengths[k], free.rate, free.fraction / (G * B * h));
            states.push_back(trial);
          }
      }
    transfer.sum(comm);
    for (unsigned int k = 0; k < n_strips; ++k)
      evaluation.roots.push_back(material.friction().solve_strip(k, transfer.free_rate[k], transfer.mobility[k],
                                                                 h, friction_pressures.values[k]));
    // In exact arithmetic the weighted sum of the returned point rates is V.
    // In cold creep even the rounding error of the scalar resistance can be
    // much larger than V/M. Recover that low part of the traction before
    // forming the native endpoint tensors and shear inventories.
    Values traction_remainder(n_strips);
    if (publish)
      {
        std::vector<long double> native_rate(n_strips), magnitude(n_strips);
        std::vector<unsigned int> count(n_strips);
        for (const auto &[id, cell] : cells)
          for (unsigned int q = 0; q < cell.points.size(); ++q)
            {
              const auto &point = cell.points[q];
              const auto k = point.strip;
              const auto &trial = trials.at(id)[q];
              const auto &old = parent.cells.at(id)[q];
              const auto &root = evaluation.roots[k];
              const auto response = maxwell.response(old, trial.strain_increment, trial.phase, h, planes[k], root.resistance);
              const long double weight = point.area / lengths[k];
              native_rate[k] += weight * response.rate;
              magnitude[k] += weight * (std::abs(response.rate) + 2.*std::abs(root.resistance) * response.fraction / (G * B * h)
                                        + 4.*std::abs(old.cohesive_strain * planes[k]) / h);
              ++count[k];
            }
        Utilities::MPI::sum(native_rate, comm, native_rate);
        Utilities::MPI::sum(magnitude, comm, magnitude);
        Utilities::MPI::sum(count, comm, count);
        for (unsigned int k = 0; k < n_strips; ++k)
          {
            const long double defect = evaluation.roots[k].rate - native_rate[k];
            // n positive-weight additions, plus the 32-epsilon root certificate.
            const long double budget = (64. + 4.*count[k]) * std::numeric_limits<double>::epsilon()
                                       * (magnitude[k] + std::abs(evaluation.roots[k].rate));
            AssertThrow(std::abs(defect) <= budget, ExcMessage("Native transfer disagrees with the signed section root."));
            if (transfer.mobility[k] > 0.)
              traction_remainder[k] = defect / transfer.mobility[k];
          }
      }
    std::vector<unsigned int> admitted(n_strips, 0);
    unsigned int detached = 0;
    if (publish)
      {
        evaluation.candidate.sections = parent.sections;
        if (!friction_pressures.stations.empty())
          evaluation.candidate.friction_pressures = friction_pressures;
      }
    for (const auto &[id, cell] : cells)
      {
        const auto &states = trials.at(id);
        const auto &history = parent.cells.at(id);
        const unsigned int size = cell.indices.size();
        FullMatrix<double> block(tangent ? size : 0, tangent ? size : 0);
        for (unsigned int q = 0; q < cell.points.size(); ++q)
          {
            const auto &point = cell.points[q];
            const auto k = point.strip;
            const auto &slip_plane = planes[k];
            const auto &trial = states[q];
            const auto &old = history[q];
            const double f = evaluation.roots[k].resistance;
            const auto response = maxwell.response(old, trial.strain_increment, trial.phase, h, slip_plane, f, traction_remainder[k]);
            if (publish)
              {
                // The converged stage lies halfway between the accepted parent
                // and the candidate endpoint for both native tensors.
                PointHistory endpoint_state {{2.*response.stress - old.stress, 2.*response.cohesive_strain - old.cohesive_strain},
                  old.signed_shear + 2.*h * response.rate, old.absolute_shear + 2.*h *std::abs(response.rate)
                };
                evaluation.candidate.cells[id].push_back(endpoint_state);
                admitted[k] |= (formed[k] || eligible[k]) && trial.phase > 0.;
                detached |= !formed[k] && !eligible[k] && (endpoint_state.absolute_shear != 0. || endpoint_state.cohesive_strain.norm() != 0.);
                const double phase1 = 2.*trial.phase - point.old_phase;
                const double cohesive = maxwell.cohesive_energy(endpoint_state.cohesive_strain, phase1);
                const double change = cohesive - maxwell.cohesive_energy(old.cohesive_strain, point.old_phase);
                auto &b = evaluation.ledger;
                b.bulk += point.area * (endpoint_state.stress * endpoint_state.stress) / (4.*G);
                b.cohesive += point.area * cohesive;
                b.storage_change += point.area * ((endpoint_state.stress * endpoint_state.stress - old.stress * old.stress) / (4.*G) + change);
                b.bulk_loss += point.area * 2.*h * (response.stress * response.stress) / (2.*maxwell.eta);
                const auto reaction = 2.*(1. - response.fraction) * response.force * slip_plane;
                b.cohesive_loss += point.area * 2.*h * G / maxwell.eta * (reaction * response.cohesive_strain);
                b.adjoint_defect += point.area * 2.*h * f * response.rate;
                b.compliance_remainder += point.area * (reaction * (endpoint_state.cohesive_strain - old.cohesive_strain) - change - response.driving * (phase1 - point.old_phase));
              }
            const auto resistance_derivative = maxwell.derivative(response, {}, 0., h, slip_plane, 1.);
            for (unsigned int j = 0; j < size; ++j)
              {
                const auto &v = point.shapes[j];
                const auto row = cell.indices[j];
                evaluation.residual[row] += point.area * (response.stress * v.strain / G
                                                          - trial.scaled_pressure * v.div
                                                          - v.pressure * trial.divergence
                                                          - v.phase * response.driving / Kf);
                if (!tangent)
                  continue;
                const auto field_derivative = maxwell.derivative(response, v.strain, v.phase, h, slip_plane, 0.);
                // The last row represents df = condensed slope * integral(dw|f).
                // Its Schur complement is the cross-cell, cross-rank derivative.
                evaluation.jacobian[ {n + k, row}] -= evaluation.roots[k].slope * point.area / lengths[k] * field_derivative.rate;
                evaluation.jacobian[ {row, n + k}] += point.area * (resistance_derivative.stress * v.strain / G - v.phase * resistance_derivative.driving / Kf);
                for (unsigned int i = 0; i < size; ++i)
                  {
                    const auto &test = point.shapes[i];
                    block(i, j) += point.area * (field_derivative.stress * test.strain / G
                                                 - v.pressure * test.div
                                                 - test.pressure * v.div
                                                 - test.phase * field_derivative.driving / Kf);
                  }
              }
          }
        if (tangent)
          for (unsigned int i = 0; i < size; ++i)
            for (unsigned int j = 0; j < size; ++j)
              if (block(i, j) != 0.)
                evaluation.jacobian[ {cell.indices[i], cell.indices[j]}] += block(i, j);
      }
    const auto endpoint = [&](const Index i)
    {
      return 2.*stage_fields[i] - lower[i];
    };
    for (const auto &[row, value] : fracture.mass)
      {
        const auto i = phase_begin + row;
        evaluation.residual[i] += value;
        if (publish)
          {
            evaluation.ledger.fracture += Kf * value * endpoint(i);
            evaluation.ledger.storage_change += Kf * value * (endpoint(i) - lower[i]);
          }
      }
    for (const auto &[indices, value] : fracture.stiffness)
      {
        const auto i = phase_begin + indices.first, j = phase_begin + indices.second;
        evaluation.residual[i] += value * stage_fields[j];
        if (tangent)
          evaluation.jacobian[ {i, j}] += value;
        if (publish)
          {
            evaluation.ledger.fracture += .5 * Kf * value * endpoint(i) * endpoint(j);
            evaluation.ledger.storage_change += .5 * Kf * value * (endpoint(i) * endpoint(j) - lower[i] * lower[j]);
          }
      }
    Utilities::MPI::sum(evaluation.residual, comm, evaluation.residual);
    double squared = 0.;
    for (Index i = 0; i < phase_begin; ++i)
      if (!fixed.count(i))
        squared += evaluation.residual[i] * evaluation.residual[i];
    evaluation.ledger.mechanical = std::sqrt(squared);
    for (Index i = phase_begin; i < n; ++i)
      evaluation.ledger.kkt = std::max(evaluation.ledger.kkt, std::abs(PhaseField::projected_gradient(stage_fields[i], evaluation.residual[i], mass[i], lower[i], upper[i])));
    if (publish)
      {
        AssertThrow(!Utilities::MPI::max(detached, comm), ExcMessage("Detached exterior cohesive history requires a formation law."));
        Utilities::MPI::max(admitted, comm, admitted);
        auto &b = evaluation.ledger;
        for (auto *v :
             {
               &b.bulk, &b.cohesive, &b.storage_change, &b.bulk_loss, &b.cohesive_loss, &b.adjoint_defect, &b.compliance_remainder, &b.fracture
             })
          *v = Utilities::MPI::sum(*v, comm);
        for (unsigned int k = 0; k < n_strips; ++k)
          {
            auto &s = evaluation.candidate.sections[k];
            s.angle = angles[k];
            s.formed = formed[k] || admitted[k];
            s.rate = evaluation.roots[k].rate;
            s.resistance = evaluation.roots[k].resistance;
            s.slip += 2.*h * s.rate;
            b.friction_work += 2.*h * lengths[k] * s.rate * s.resistance;
          }
        b.adjoint_defect -= b.friction_work;
        for (const auto &[i, v] : fixed)
          if (i < nv) b.prescribed_velocity_work += 2.*G * v * evaluation.residual[i];
        for (Index i = phase_begin; i < n; ++i)
          b.phase_work += Kf * (endpoint(i) - lower[i]) * evaluation.residual[i];
        evaluation.candidate.step = parent.step + 1;
        evaluation.candidate.time = sim.get_time();
      }
    return evaluation;
  }

  void MidpointSystem::restore_continuity(Values &x) const
  {
    Evaluation projection;
    projection.residual.assign(n, 0.);
    for (const auto &[id, cell] : cells)
      {
        const unsigned int size = cell.indices.size();
        FullMatrix<double> block(size, size);
        for (const auto &p : cell.points)
          {
            double divergence = 0.;
            for (unsigned int j = 0; j < size; ++j)
              divergence += p.shapes[j].div * x[cell.indices[j]];
            for (unsigned int i = 0; i < size; ++i)
              {
                const auto &v = p.shapes[i];
                if (cell.indices[i] >= phase_begin)
                  continue;
                projection.residual[cell.indices[i]] -= p.area * v.pressure * divergence;
                for (unsigned int j = 0; j < size; ++j)
                  if (cell.indices[j] < phase_begin)
                    {
                      const auto &u = p.shapes[j];
                      // 3-D deviatoric product plus div-div/3 recovers
                      // the full symmetric gradient of the Stokes metric.
                      block(i, j) += p.area * (2.*B * (v.strain * u.strain + v.div * u.div / 3.) - v.div * u.pressure - v.pressure * u.div);
                    }
              }
          }
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size; ++j)
            if (block(i, j) != 0.)
              projection.jacobian[ {cell.indices[i], cell.indices[j]}] += block(i, j);
      }
    Utilities::MPI::sum(projection.residual, comm, projection.residual);
    const auto delta = solve_linearized_obstacle(projection, x);
    for (Index i = 0; i < phase_begin; ++i)
      x[i] += delta[i];
  }

  void MidpointSystem::use_phase_search_metric(Evaluation &e) const
  {
    for (auto it = e.jacobian.begin(); it != e.jacobian.end();)
      if (it->first.first >= phase_begin && it->first.first < n)
        it = e.jacobian.erase(it);
      else ++it;
    for (const auto &[ij, value] : fracture.stiffness)
      e.jacobian[ {phase_begin + ij.first, phase_begin + ij.second}] += value;
    for (const auto &[i, value] : fracture.mass)
      e.jacobian[ {phase_begin + i, phase_begin + i}] += value;
  }

  MidpointSystem::Values MidpointSystem::solve_linearized_obstacle(const Evaluation &e,
                                                                   const Values     &x) const
  {
    // Gather sparse element blocks only for the direct solve. Strip borders
    // keep its graph sparse; eliminating their linear increments gives the
    // exact condensed tangent without forming any dense strip outer products.
    const auto parts = Utilities::MPI::gather(comm, e.jacobian);
    Values update(n);
    std::string error;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      try
        {
          const Index rows = n + angles.size();
          Matrix entries;
          for (const auto &part : parts)
            for (const auto &[ij, v] : part)
              entries[ij] += v;
          for (Index i = n; i < rows; ++i)
            entries[ {i, i}] = 1.;
          DynamicSparsityPattern graph(rows, rows);
          for (const auto &[ij, v] : entries)
            if (v != 0.) graph.add(ij.first, ij.second);
          for (Index i = 0; i < rows; ++i)
            graph.add(i, i);
          SparsityPattern sp;
          sp.copy_from(graph);
          // Solve the linearized VI to feasibility before taking a nonlinear
          // step. A row leaves the working set only when its multiplier has
          // the wrong sign. Shorten a search to the first blocking bound.
          std::map<Index, double> constraints;
          for (const auto &[i, v] : fixed)
            constraints[i] = v - x[i];
          for (Index i = phase_begin; i < n; ++i)
            if (x[i] == lower[i] || x[i] == upper[i])
              constraints[i] = 0.;
          Vector<double> feasible(rows), result(rows), rhs(rows);
          SparseMatrix<double> matrix(sp);
          bool solved = false;
          for (Index pass = 0; pass < 4 * (n - phase_begin) + 20; ++pass)
            {
              matrix = 0.;
              rhs = 0.;
              for (Index i = 0; i < n; ++i)
                rhs[i] = -e.residual[i];
              for (const auto &[ij, v] : entries)
                if (v != 0. && !constraints.count(ij.first))
                  {
                    const auto column = constraints.find(ij.second);
                    if (column == constraints.end())
                      matrix.set(ij.first, ij.second, v);
                    else rhs[ij.first] -= v * column->second;
                  }
              for (const auto &[i, v] : constraints)
                {
                  matrix.set(i, i, 1.);
                  rhs[i] = v;
                }
              SparseDirectUMFPACK direct;
              direct.initialize(matrix);
              direct.vmult(result, rhs);
              double alpha = 1., target = 0.;
              Index blocking = n;
              for (Index i = phase_begin; i < n; ++i)
                if (!constraints.count(i))
                  {
                    const double lo = lower[i] - x[i], hi = upper[i] - x[i];
                    if (result[i] >= lo && result[i] <= hi)
                      continue;
                    const double bound = result[i] < lo ? lo : hi;
                    const double step = (bound - feasible[i]) / (result[i] - feasible[i]);
                    if (step < alpha)
                      {
                        alpha = std::max(0., step);
                        blocking = i;
                        target = bound;
                      }
                  }
              if (blocking < n)
                {
                  feasible.sadd(1. - alpha, alpha, result);
                  feasible[blocking] = target;
                  constraints[blocking] = target;
                  continue;
                }
              feasible = result;
              Vector<double> gradient(n);
              std::copy(e.residual.begin(), e.residual.end(), gradient.begin());
              for (const auto &[ij, v] : entries)
                if (ij.first >= phase_begin && ij.first < n)
                  gradient[ij.first] += v * result[ij.second];
              Index release = n;
              double largest = 1.e-12;
              for (const auto &[i, v] : constraints)
                if (i >= phase_begin && lower[i] != upper[i])
                  {
                    const double violation = (v == lower[i] - x[i] ? -gradient[i] : gradient[i]) / mass[i];
                    if (violation > largest)
                      {
                        largest = violation;
                        release = i;
                      }
                  }
              if (release == n)
                {
                  solved = true;
                  break;
                }
              constraints.erase(release);
            }
          AssertThrow(solved, ExcMessage("The linearized midpoint obstacle failed to find a feasible working set."));
          std::copy_n(feasible.begin(), n, update.begin());
        }
      catch (const std::exception &ex)
        {
          error = ex.what();
        }
    error = Utilities::MPI::broadcast(comm, error);
    AssertThrow(error.empty(), ExcMessage("Sparse midpoint correction failed: " + error));
    return Utilities::MPI::broadcast(comm, update);
  }

  void MidpointSystem::write_candidate_fields(const Values               &x,
                                              LinearAlgebra::BlockVector &solution) const
  {
    const auto &intro = sim.introspection();
    const auto c = intro.component_indices.compositional_fields[field];
    const auto offset = solution.get_block_indices().block_start(phase_block);
    std::vector<Index> ids(sim.get_fe().n_dofs_per_cell());
    unsigned int invalid = 0;
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ids);
          PhaseField::BernsteinPatch<2> patch;
          for (unsigned int j = 0; j < patch.size(); ++j)
            {
              const auto i = phase_begin + ids[sim.get_fe().component_to_system_index(c, j)] - offset;
              patch[j] = 2.*x[i] - lower[i];
              invalid |= x[i] < lower[i] || x[i] > upper[i];
            }
          const auto bounds = PhaseField::enclosure<2>(patch);
          const double tolerance = 128.*std::numeric_limits<double>::epsilon();
          invalid |= bounds.first < -tolerance || bounds.second > 1. + tolerance;
        }
    AssertThrow(!Utilities::MPI::max(invalid, comm), ExcMessage("Midpoint endpoint failed phase bounds or irreversibility."));
    for (const unsigned int b :
    {
      0u, 1u, phase_block
    })
    {
      LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[b], comm);
      for (const auto j : owned.locally_owned_elements())
        if (b == 0)
          owned[j] = x[j] / h;
        else if (b == 1)
          owned[j] = G * x[nv + j];
        else
          owned[j] = 2.*x[phase_begin + j] - lower[phase_begin + j];
      owned.compress(VectorOperation::insert);
      solution.block(b) = owned;
    }
  }

  namespace
  {
    // Confirm the observable a consumer can reconstruct from stored native
    // increments, including roundoff in adding an increment to old shear.
    void check_published_section_rates(const MidpointSystem &system, const History &candidate)
    {
      const unsigned int n = candidate.sections.size();
      std::vector<long double> rates(n), inventory(n);
      for (const auto &[id, cell] : system.cells)
        for (unsigned int q = 0; q < cell.points.size(); ++q)
          {
            const auto &p = cell.points[q];
            const unsigned int k = p.strip;
            const long double old = system.material.history().cells.at(id)[q].signed_shear;
            const long double next = candidate.cells.at(id)[q].signed_shear;
            const long double weight = p.area / (2.*system.h * system.material.sections().length(k, candidate.sections[k].angle));
            rates[k] += weight * (next - old);
            inventory[k] += weight * (std::abs(next) + std::abs(old));
          }
      Utilities::MPI::sum(rates, system.comm, rates);
      Utilities::MPI::sum(inventory, system.comm, inventory);
      for (unsigned int k = 0; k < n; ++k)
        {
          const double rate = candidate.sections[k].rate;
          const long double error = std::abs(rates[k] - rate);
          const long double budget = 32.*std::numeric_limits<double>::epsilon() * inventory[k];
          if (error > budget || (rate != 0. && error >= std::abs(rate)))
            {
              std::ostringstream message;
              message << "Unresolved native slip rate on strip " << k << ": stored increments give " << rates[k]
                      << " m/s, signed root " << rate << " m/s, transfer roundoff budget " << budget << " m/s.";
              AssertThrow(false, ExcMessage(message.str()));
            }
        }
    }
  }

  static void solve_midpoint(const SimulatorAccess<2>      &sim,
                             const StripMaterialHistory<2> &material,
                             LinearAlgebra::BlockVector    &solution,
                             History                       &next,
                             Balance                       &balance)
  {
    TimerOutput::Scope timer(sim.get_computing_timer(), "Coupled midpoint fault");
    MidpointSystem system(sim, material, solution);
    auto x = system.initial;
    system.freeze_direction_neighborhood();
    system.restore_continuity(x);
    double initial_merit = 0.;
    bool use_positive_metric = true;
    unsigned int direction_updates = 0;
    const auto &settings = sim.get_parameters();
    for (unsigned int iteration = 0; iteration < settings.max_nonlinear_iterations; ++iteration)
      {
        auto stage = system.assemble_stage(x, true, false);
        sim.get_pcout() << "   Midpoint iteration " << iteration << ", mechanics " << stage.ledger.mechanical
                        << ", projected phase " << stage.ledger.kkt << std::endl;
        if (stage.ledger.mechanical <= settings.nonlinear_tolerance && stage.ledger.kkt <= 1.e-7)
          {
            if (system.select_prospective_planes(x))
              {
                if (++direction_updates > 12)
                  throw ExcNonlinearSolverNoConvergence();
                continue;
              }
            // Bind publication to a fresh evaluation on the final plane map.
            auto returned = system.assemble_stage(x, true, true);
            if (returned.ledger.mechanical > settings.nonlinear_tolerance || returned.ledger.kkt > 1.e-7)
              throw ExcNonlinearSolverNoConvergence();
            check_published_section_rates(system, returned.candidate);
            system.write_candidate_fields(x, solution);
            next = std::move(returned.candidate);
            balance = returned.ledger;
            balance.iterations = iteration;
            return;
          }
        // Globalize a newly activated phase obstacle with the convex
        // fracture-plus-mass metric. A decade of residual reduction switches
        // back to the full physical Newton matrix for the remaining solve.
        const double merit = std::max(stage.ledger.mechanical / settings.nonlinear_tolerance, stage.ledger.kkt / 1.e-7);
        if (iteration == 0)
          initial_merit = merit;
        if (merit <= std::max(10., .1 * initial_merit))
          use_positive_metric = false;
        if (use_positive_metric)
          system.use_phase_search_metric(stage);
        const auto d = system.solve_linearized_obstacle(stage, x);
        for (unsigned int j = 0; j < x.size(); ++j)
          {
            x[j] += d[j];
            if (j >= system.phase_begin)
              x[j] = std::clamp(x[j], system.lower[j], system.upper[j]);
          }
      }
    throw ExcNonlinearSolverNoConvergence();
  }
  template <int dim>
  void validate_midpoint_material(const Parameters<dim>               &p,
                                  const MaterialModel::Interface<dim> &model)
  {
    const auto &methods = p.compositional_field_methods;
    const auto selected = std::find(methods.begin(), methods.end(), Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase);
    if (selected == methods.end())
      return;
    const auto *provider = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(&model);
    AssertThrow(provider && provider->uses_growing_strips(), ExcMessage("Midpoint fault phase requires a material providing the Maxwell/AT1 midpoint response."));
    const auto field = provider->strip_history().phase_field();
    p.surface_energy_model.validate();
    AssertThrow(&p.surface_energy_model.geometry == &PhaseField::CrackGeometry::select("AT1") &&
                p.names_of_compositional_fields[selected - methods.begin()] == p.surface_energy_model.composition,
                ExcMessage("Midpoint fault phase requires the configured AT1 surface energy on the selected phase field."));
    AssertThrow(dim == 2 && !p.enable_elasticity &&
                p.nonlinear_solver == Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes &&
                field == static_cast<unsigned int>(selected - methods.begin()) &&
                std::count(p.compositional_field_methods.begin(), p.compositional_field_methods.end(),
                           Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase) == 1 &&
                p.nonlinear_solver_failure_strategy != Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep &&
                p.pressure_normalization == "no",
                ExcMessage("Growing fault requires a single midpoint fault phase, plane strain, native elasticity off, single Advection, iterated Newton Stokes, and no pressure normalization. Nonconvergence must abort or request a retry."));
    AssertThrow(!p.mesh_deformation_enabled && !p.initial_adaptive_refinement && !p.adaptive_refinement_interval &&
                p.additional_refinement_times.empty() && p.stokes_velocity_degree == 2 && !p.use_equal_order_interpolation_for_stokes &&
                !p.use_locally_conservative_discretization && !p.include_melt_transport,
                ExcMessage("Growing fault requires a fixed conforming Q2/Q1 mesh."));
    AssertThrow(!p.use_direct_stokes_solver, ExcMessage("Midpoint fault mechanics requires separate velocity and pressure blocks."));
    AssertThrow(!p.run_postprocessors_on_nonlinear_iterations, ExcMessage("Midpoint fault output requires an accepted step; iteration postprocessing is unsupported."));
    AssertThrow(!p.enable_prescribed_dilation &&
                (p.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::incompressible ||
                 p.formulation_mass_conservation == Parameters<dim>::Formulation::MassConservation::ask_material_model),
                ExcMessage("Midpoint fault mechanics requires incompressible continuity without prescribed dilation."));
  }
  template void validate_midpoint_material(const Parameters<2> &,
                                           const MaterialModel::Interface<2> &);
  template void validate_midpoint_material(const Parameters<3> &,
                                           const MaterialModel::Interface<3> &);

  SolverControl advance_midpoint(const SimulatorAccess<2>    &sim,
                                 MaterialModel::Interface<2> &model,
                                 LinearAlgebra::BlockVector  &solution)
  {
    auto &fault = dynamic_cast<MaterialModel::PhaseFieldFault<2> &>(model);
    AssertThrow(fault.uses_growing_strips(),
                ExcMessage("Midpoint fault phase requires a material providing the Maxwell/AT1 midpoint response."));
    auto &material = *fault.strips;
    SolverControl control(sim.get_parameters().max_nonlinear_iterations, sim.get_parameters().nonlinear_tolerance);
    material.begin_attempt();
    const auto field = material.phase_field();
    if (sim.get_timestep_number() == 0)
      {
        // Initialize the physical Q2 field with the inherited whole-cell
        // scaling. An admissible polynomial keeps every input coefficient.
        const auto &intro = sim.introspection();
        const auto c = intro.component_indices.compositional_fields[field], b = intro.block_indices.compositional_fields[field];
        const auto offset = solution.get_block_indices().block_start(b);
        LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[b], sim.get_mpi_communicator());
        std::vector<types::global_dof_index> ids(sim.get_fe().n_dofs_per_cell());
        unsigned int invalid = 0;
        for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(ids);
              std::vector<double> patch(9);
              for (unsigned int j = 0; j < 9; ++j)
                patch[j] = solution[ids[sim.get_fe().component_to_system_index(c, j)]];
              invalid |= !PhaseField::limit_bernstein<2>(patch, std::accumulate(patch.begin(), patch.end(), 0.) / 9.);
              for (unsigned int j = 0; j < 9; ++j)
                owned[ids[sim.get_fe().component_to_system_index(c, j)] - offset] = patch[j];
            }
        AssertThrow(!Utilities::MPI::max(invalid, sim.get_mpi_communicator()), ExcMessage("Invalid initial phase mean."));
        owned.compress(VectorOperation::insert);
        solution.block(b) = owned;
        control.check(0, 0.);
        return control;
      }
    AssertThrow(material.history().step + 1 == sim.get_timestep_number(), ExcMessage("Growing fault accepted parent clock mismatch."));
    History next;
    Balance ledger;
    solve_midpoint(sim, material, solution, next, ledger);
    material.stage_candidate(std::move(next), ledger);
    control.check(ledger.iterations, ledger.mechanical);
    return control;
  }

}
