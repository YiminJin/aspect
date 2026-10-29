/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_system_internal.h>
#include <deal.II/fe/fe_values.h>
#include <chrono>
#include <numeric>
namespace aspect
{
  namespace CohesiveInterface
  {
    namespace
    {
      void
      add_native (System::Field &to, const System::Field &from, double weight)
      {
        for (unsigned b = 0; b < to.n_blocks (); ++b)
          for (const auto i : to.block (b).locally_owned_elements ())
            to.block (b)[i] += weight * from.block (b)[i];
      }
      double
      bond (const Parameters &law, const PointState &s)
      {
        return s.damage > 0 && s.damage < 1
                   ? (1 - s.damage) * std::pow (s.jump - s.reference, 2)
                         / (2 * (2 * law.Gc / (law.peak * law.peak)) * s.damage)
                   : 0.;
      }
    }

    std::shared_ptr<System::Trial>
    System::Implementation::trial (const State &old, std::shared_ptr<Prepared> p,
                                   const Field &incoming_solution)
    {
      const auto start = std::chrono::steady_clock::now ();
      const double dt = system.get_timestep (), beta = 1 / (1 + law.G / law.eta * dt);
      const unsigned n = p->geometry.modes (), nq = p->points.size ();
      CohesiveEvolution::Incoming anchor;
      anchor.memory.assign (n, 0.);
      anchor.history = p->incoming_points;
      Vector<double> rate (n);
      for (unsigned j = 0; j < old.rate.size () && j < n; ++j)
        rate[j] = old.rate[j];
      const auto predictor
          = CohesiveEvolution::continuation_predictor (p->surface, anchor, dt, rate);
      auto result = std::make_shared<Trial> ();
      result->prepared = p;
      result->surface = CohesiveEvolution::solve (p->surface, anchor, dt, 0., &predictor.first,
                                                  &predictor.second);
      solve_seconds
          += std::chrono::duration<double> (std::chrono::steady_clock::now () - start).count ();
      auto &r = result->surface;
      auto s = std::make_shared<State> (old);
      s->geometry = p->geometry;
      s->time = system.get_time ();
      s->step = old.step + 1;
      s->publications = old.publications + 1;
      s->identity_clock = old.identity_clock + 1;
      AssertThrow (std::abs (s->time - old.time - dt) < 1e-12 * std::max (1., std::abs (s->time)),
                   ExcMessage ("Native cohesive physical clock differs from accepted anchor."));
      s->jump.resize (n, 0.);
      s->memory.resize (n, 0.);
      s->rate.resize (n, 0.);
      s->offset.resize (nq, 0.);
      for (unsigned j = 0; j < n; ++j)
        {
          s->jump[j] += r.increment[j];
          s->memory[j] = beta * (p->incoming_memory[j] + r.increment[j]);
          s->rate[j] = r.increment[j] / dt;
        }
      std::vector<PointState> history (nq);
      std::vector<std::array<double, 3>> diagnostic (nq);
      for (unsigned q = 0; q < nq; ++q)
        {
          history[q] = r.point[q].state;
          double pn = law.p0 + p->surface.normal_offset[q];
          for (unsigned j = 0; j < n; ++j)
            pn += beta * p->surface.Nm (q, j + 1) * r.increment[j];
          diagnostic[q] = { r.point[q].q, r.point[q].resistance, pn };
          AssertThrow (
              history[q].damage >= anchor.history[q].damage && history[q].damage <= 1.
                  && history[q].theta > 0.,
              ExcMessage ("Native surface history violates irreversibility/admissibility."));
        }
      unsigned count = 0;
      bool gap = false;
      unsigned q0 = 0;
      for (unsigned piece = 0; piece < s->geometry.pieces ().size (); ++piece)
        {
          const unsigned end = q0 + 3 * s->geometry.pieces ()[piece].cells;
          bool formed = piece < old.formed;
          for (unsigned q = q0; q < end; ++q)
            formed |= history[q].damage != 0. || history[q].jump != 0. || history[q].reference != 0.
                      || history[q].relaxed_jump != 0. || history[q].absolute_slip != 0.;
          if (formed)
            {
              AssertThrow (!gap, ExcMessage ("Noncontiguous formed material cannot be admitted by "
                                             "the straight-extension topology."));
              count = piece + 1;
            }
          else
            gap = true;
          q0 = end;
        }
      s->formed = count;
      s->registry_publications = old.registry_publications + (count > old.formed);
      unsigned formed_q = 0;
      for (unsigned piece = 0; piece < count; ++piece)
        formed_q += 3 * s->geometry.pieces ()[piece].cells;
      for (unsigned q = s->ids.size (); q < formed_q; ++q)
        s->ids.push_back ("admit:" + std::to_string (s->identity_clock) + ":" + std::to_string (q));
      s->virgin_theta = old.virgin_theta + law.V0 * dt / law.Dc;
      // Metadata belongs to the candidate view; solve() supplies it before commit.
      s->manager = manager (*s, history, diagnostic);
      auto smooth = owned ();
      *smooth = 0.;
      add_native (*smooth, *p->base, 1.);
      for (unsigned j = 0; j < n; ++j)
        if (r.increment[j] != 0.)
          add_native (*smooth, *p->columns[j].response, r.increment[j]);
      smooth->compress (VectorOperation::insert);
      auto du = ghosted ();
      *du = *smooth;
      const unsigned pb = system.introspection ().block_indices.pressure;
      auto native = owned ();
      *native = *smooth;
      native->block (system.introspection ().block_indices.velocities) /= dt;
      native->block (pb) *= beta;
      result->native_solution = ghosted ();
      *result->native_solution = *native;
      const auto history_start = std::chrono::steady_clock::now ();
      s->bulk = old.bulk->advance (*du, dt);
      const auto source = old.bulk->check_advance (*s->bulk, *du, dt, QGauss<2> (6));
      history_seconds
          += std::chrono::duration<double> (std::chrono::steady_clock::now () - history_start)
                 .count ();
      const auto sided = traces (*s, *result->native_solution);
      double raw_contact = std::numeric_limits<double>::infinity ();
      for (const auto &trace : sided)
        raw_contact = std::min ({ raw_contact, trace[0], trace[1] });
      AssertThrow (raw_contact > 0.,
                   ExcMessage ("Native physical sided trace lost closed contact."));
      // Independent native physical work and full-tensor storage ledger. No
      // reduced-response-span energy is substituted for the general history.
      double eold = 0., enew = 0., be = 0., work = 0., trap = 0., div2 = 0.;
      const auto &intro = system.introspection ();
      const auto &fe = system.get_fe ();
      auto residual = owned (), measure = owned ();
      *residual = 0.;
      *measure = 0.;
      std::vector<unsigned> local;
      for (unsigned i = 0; i < fe.n_dofs_per_cell (); ++i)
        if (intro.is_stokes_component (fe.system_to_component_index (i).first))
          local.push_back (i);
      std::vector<types::global_dof_index> ids (fe.n_dofs_per_cell ()), stokes (local.size ());
      for (const auto &cell : system.get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            const auto rule
                = p->geometry.quadrature (cell->vertex (0), cell->vertex (3), bulk_order);
            FEValues<2> v (system.get_mapping (), fe, rule,
                           update_values | update_gradients | update_quadrature_points
                               | update_JxW_values);
            v.reinit (cell);
            const auto before = physical_stress (old, cell, rule),
                       after = physical_stress (*s, cell, rule);
            std::vector<SymmetricTensor<2, 2>> ep (rule.size ());
            std::vector<double> dp (rule.size ()), div (rule.size ()), pn (rule.size ()),
                po (rule.size ());
            v[intro.extractors.velocities].get_function_symmetric_gradients (*p->particular, ep);
            v[intro.extractors.velocities].get_function_divergences (*p->particular, dp);
            v[intro.extractors.velocities].get_function_divergences (*du, div);
            v[intro.extractors.pressure].get_function_values (*result->native_solution, pn);
            v[intro.extractors.pressure].get_function_values (incoming_solution, po);
            cell->get_dof_indices (ids);
            for (unsigned i = 0; i < local.size (); ++i)
              stokes[i] = ids[local[i]];
            Vector<double> load (local.size ()), row_measure (local.size ());
            for (unsigned q = 0; q < rule.size (); ++q)
              {
                const double w = v.JxW (q);
                eold += w * (before[q] * before[q]) / (4 * law.G);
                enew += w * (after[q] * after[q]) / (4 * law.G);
                const auto change = after[q] - before[q];
                be += w * (change * change) / (4 * law.G);
                work += w * (after[q] * dev3 (ep[q]) - (law.p0 + pn[q]) * dp[q]);
                trap += .5 * w
                        * ((before[q] + after[q]) * dev3 (ep[q])
                           - (2 * law.p0 + pn[q] + po[q]) * dp[q]);
                double total_div = div[q];
                for (const auto &m : p->geometry.lifting (v.quadrature_point (q)))
                  total_div += r.increment[m.index] * trace (m.gradient);
                div2 += w * total_div * total_div;
                for (unsigned i = 0; i < local.size (); ++i)
                  {
                    load[i] += w
                               * (after[q]
                                      * dev3 (v[intro.extractors.velocities].symmetric_gradient (
                                          local[i], q))
                                  - pn[q] * v[intro.extractors.velocities].divergence (local[i], q)
                                  - pressure_scaling * total_div
                                        * v[intro.extractors.pressure].value (local[i], q));
                    row_measure[i]
                        += w
                           * (dev3 (v[intro.extractors.velocities].symmetric_gradient (local[i], q))
                                  .norm ()
                              + (pressure_scaling / law.G)
                                    * std::abs (v[intro.extractors.pressure].value (local[i], q)));
                  }
              }
            system.get_current_constraints ().distribute_local_to_global (load, stokes, *residual);
            system.get_current_constraints ().distribute_local_to_global (row_measure, stokes,
                                                                          *measure);
          }
      residual->compress (VectorOperation::add);
      measure->compress (VectorOperation::add);
      double mechanical_weak = 0.;
      types::global_dof_index block_start = 0;
      for (unsigned b = 0; b < residual->n_blocks ();
           block_start += residual->block (b).size (), ++b)
        for (const auto i : residual->block (b).locally_owned_elements ())
          if (!system.get_current_constraints ().is_constrained (i + block_start)
              && measure->block (b)[i] > 0.)
            mechanical_weak = std::max (mechanical_weak,
                                        std::abs (residual->block (b)[i]) / measure->block (b)[i]);
      mechanical_weak = Utilities::MPI::max (mechanical_weak, system.get_mpi_communicator ());
      AssertThrow (
          mechanical_weak <= 1e-8,
          ExcMessage (
              "Native independently reconstructed mechanical/continuity weak residual failed: "
              + std::to_string (mechanical_weak) + " Pa."));
      for (double *value : { &eold, &enew, &be, &work, &trap, &div2 })
        *value = Utilities::MPI::sum (*value, system.get_mpi_communicator ());
      AssertThrow (
          std::isfinite (eold) && std::isfinite (enew) && std::isfinite (be) && std::isfinite (work)
              && std::isfinite (trap) && std::isfinite (div2),
          ExcMessage (
              "Native physical storage/work or incompressibility diagnostic is nonfinite."));
      double bold = 0., bnew = 0., fracture = 0., friction = 0., radiation = 0., surface_rem = 0.,
             damage = 0., absjump = 0., full = 0., peak_increment = 0.;
      for (unsigned q = 0; q < nq; ++q)
        {
          const double w = p->points[q].weight;
          const auto &a = anchor.history[q];
          const auto &b = history[q];
          bold += w * bond (law, a);
          bnew += w * bond (law, b);
          fracture += w * law.Gc * (b.damage - a.damage);
          friction += w * r.point[q].friction;
          radiation += w * r.point[q].radiation;
          surface_rem += w
                         * (r.point[q].q * (b.jump - a.jump) - (bond (law, b) - bond (law, a))
                            - law.Gc * (b.damage - a.damage));
          damage += w * b.damage;
          absjump += w * std::abs (b.jump);
          full += w * (b.damage == 1.);
          peak_increment = std::max (peak_increment, b.damage - a.damage);
        }
      const double maxwell = 2 * law.G * dt / law.eta * enew;
      const double ledger = work
                            - (enew - eold + bnew - bold + fracture + friction + radiation + maxwell
                               + be + surface_rem);
      AssertThrow (std::abs (ledger) < 1e-10 * std::max (1., std::abs (work)),
                   ExcMessage ("Native physical BE work ledger failed: " + std::to_string (ledger)
                               + " J/m."));
      auto &d = s->diagnostics;
      d.clear ();
      const auto old_rejections = old.diagnostics.find ("clock_outer_rejections");
      d["clock_outer_rejections"]
          = (old_rejections == old.diagnostics.end () ? 0. : old_rejections->second)
            + clock_attempts - 1;
      d["clock_roundoff_bound_s"] = clock_bound (*s);
      d["time_s"] = s->time;
      d["dt_s"] = dt;
      d["bulk_J_m"] = enew;
      d["bond_J_m"] = bnew;
      d["bulk_increment_J_m"] = enew - eold;
      const auto previous_energy = old.diagnostics.find ("bulk_J_m");
      const double integration_change
          = previous_energy == old.diagnostics.end () ? 0. : eold - previous_energy->second;
      AssertThrow (std::abs (integration_change) <= 1e-10 * std::max (1., std::abs (eold)),
                   ExcMessage ("Native geometry quadrature repartition changed stored energy "
                               "beyond the retained transfer allowance."));
      d["quadrature_repartition_J_m"] = integration_change;
      d["bond_increment_J_m"] = bnew - bold;
      d["fracture_increment_J_m"] = fracture;
      d["friction_J_m"] = friction;
      d["radiation_J_m"] = radiation;
      d["maxwell_J_m"] = maxwell;
      d["BE_bulk_J_m"] = be;
      d["endpoint_work_J_m"] = work;
      d["trapezoid_work_J_m"] = trap;
      d["surface_remainder_J_m"] = surface_rem;
      d["ledger_error_J_m"] = ledger;
      d["physical_remainder_J_m"]
          = trap - (enew - eold + bnew - bold + fracture + friction + radiation + maxwell);
      d["physical_remainder_with_quadrature_J_m"]
          = d["physical_remainder_J_m"] - integration_change;
      d["weak_error_Pa"] = r.weak;
      d["bulk_residual_linf_N_m"] = residual->linfty_norm ();
      d["incompressibility_L2_m"] = std::sqrt (div2);
      d["mechanical_weak_Pa"] = mechanical_weak;
      d["source_error_Pa"] = source.first;
      d["source_scaled"] = source.second;
      d["minimum_normal_Pa"] = r.contact;
      d["minimum_raw_normal_Pa"] = raw_contact;
      d["minimum_symmetric_tangent_Pa_m"] = r.curvature;
      d["integrated_damage_m"] = damage;
      d["integrated_absolute_jump_m2"] = absjump;
      d["fully_decohered_length_m"] = full;
      d["maximum_damage_increment"] = peak_increment;
      d["surface_iterations"] = r.iterations;
      d["free_dimensions"] = r.free;
      result->state = std::move (s);
      return result;
    }

    std::shared_ptr<System::Trial>
    System::solve (const Field &particular, const Field &incoming, const BulkSolve &inverse)
    {
      auto &d = *data;
      AssertThrow (d.accepted && !d.solving && d.active_trial.expired (),
                   ExcMessage ("Native cohesive solve has an unresolved trial."));
      d.solving = true;
      ++d.clock_attempts;
      try
        {
          const auto &s = *d.accepted;
          d.validate (s);
          AssertThrow (s.step + 1 == get_timestep_number (),
                       ExcMessage ("Native cohesive publication clock differs from Simulator."));
          const auto sampled = d.sample (s);
          const double direction = sampled.second;
          const int epoch = static_cast<int> (std::floor ((s.time - .410 + 1e-12) / d.cadence));
          bool refresh = s.candidate_epoch != epoch || s.candidate_anchor != s.formed;
          unsigned number
              = std::max (static_cast<unsigned> (std::ceil (d.candidate_extent / d.spacing)),
                          static_cast<unsigned> (s.geometry.pieces ().size () - s.formed));
          unsigned retries = 0, activation_retries = 0;
          std::shared_ptr<Trial> result;
          while (true)
            {
              Geometry geometry = s.geometry;
              auto metadata = s.metadata;
              if (refresh)
                {
                  std::vector<Geometry::Piece> prefix (s.geometry.pieces ().begin (),
                                                       s.geometry.pieces ().begin () + s.formed);
                  geometry = Geometry (prefix, d.support);
                  metadata.resize (s.formed);
                  for (unsigned i = 0; i < number; ++i)
                    {
                      geometry.append (direction, d.spacing, d.R);
                      Implementation::Metadata m;
                      m.selected_time = s.time;
                      m.preferred = direction;
                      metadata.push_back (m);
                    }
                }
              auto prepared = d.prepare (s, std::move (geometry), inverse);
              d.forcing (*prepared, s, particular, inverse);
              result = d.trial (s, prepared, incoming);
              if (result->state->formed > s.formed && !refresh)
                {
                  refresh = true;
                  ++activation_retries;
                  continue;
                }
              if (result->state->formed == prepared->geometry.pieces ().size ())
                {
                  AssertThrow (++retries <= 4 && 2 * number * d.spacing <= 1.5,
                               ExcMessage ("Native candidate completion budget exhausted; accepted "
                                           "histories remain unchanged."));
                  number *= 2;
                  refresh = true;
                  continue;
                }
              auto &next = *result->state;
              next.metadata = std::move (metadata);
              for (unsigned i = s.formed; i < next.formed; ++i)
                next.metadata[i].admitted_time = next.time;
              next.candidate_epoch = epoch;
              next.candidate_anchor = s.formed;
              const double length = std::accumulate (
                  prepared->points.begin (), prepared->points.begin () + 3 * d.formed_cells (next),
                  0., [] (double a, const Geometry::Point &p) { return a + p.weight; });
              AssertThrow (
                  length <= d.max_length,
                  ExcMessage ("Native cohesive formed-length stopping condition reached: trial "
                              + std::to_string (length) + " m, bound "
                              + std::to_string (d.max_length) + " m, time "
                              + std::to_string (next.time)
                              + " s. Accepted histories remain unchanged."));
              auto &diag = next.diagnostics;
              diag["persistent_length_m"] = length;
              diag["formed_pieces"] = next.formed;
              diag["newly_formed_pieces"] = next.formed - s.formed;
              diag["activation_retries"] = activation_retries;
              diag["extension_retries"] = retries;
              diag["selected_preference_deg"] = direction * 180 / numbers::PI;
              diag["selection_clock_s"] = s.time;
              diag["sample_xx_Pa"] = sampled.first[0][0];
              diag["sample_yy_Pa"] = sampled.first[1][1];
              diag["sample_xy_Pa"] = sampled.first[0][1];
              diag["transfer_stress_Pa"] = prepared->transfer_stress;
              diag["metadata_commit_work_J_m"] = 0.;
              diag["metadata_commit_fracture_J_m"] = 0.;
              diag["setup_seconds_cumulative"] = d.setup_seconds;
              diag["surface_seconds_cumulative"] = d.solve_seconds;
              diag["history_seconds_cumulative"] = d.history_seconds;
              diag["bulk_solves_cumulative"] = d.bulk_solves;
              diag["cache_hits_cumulative"] = d.cache_hits;
              d.validate (next);
              break;
            }
          result->parent = d.accepted;
          d.active_trial = result;
          d.solving = false;
          return result;
        }
      catch (...)
        {
          d.solving = false;
          throw;
        }
    }
  }
}
