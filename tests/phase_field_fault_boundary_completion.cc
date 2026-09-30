/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "phase_field_fault_test_access.h"
#include <aspect/geometry_model/box.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace
  {
    template <int dim>
    void
    prescribe_boundary_test_phase (const SimulatorAccess<dim> &sim,
                                   AffineConstraints<double> &constraints)
    {
      if (std::getenv ("ASPECT_TEST_BOUNDARY_H_DRIVEN"))
        return;
      const auto &faults = sim.get_reconstructed_fault_manager ().get_prescribed_faults ();
      const auto &box
        = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>> (sim.get_geometry_model ());
      const auto lower = box.get_origin (), upper = lower + box.get_extents ();
      const auto boundary = [&] (const Point<dim> &p)
      {
        for (unsigned int d = 0; d < dim; ++d)
          if (p[d] == lower[d] || p[d] == upper[d])
            return true;
        return false;
      };
      const auto profile = sim.get_phase_field_handler ().get_phase_field_profiles (.6);
      const auto &fe = sim.get_fe ();
      const auto component = sim.introspection ().variable ("phase_field").first_component_index;
      std::vector<types::global_dof_index> dofs (fe.n_dofs_per_cell ());
      for (const auto &cell : sim.get_dof_handler ().active_cell_iterators ())
        if (!cell->is_artificial ())
          {
            cell->get_dof_indices (dofs);
            for (unsigned int j = 0; j < dofs.size (); ++j)
              if (fe.system_to_component_index (j).first == component
                  && constraints.can_store_line (dofs[j]) && !constraints.is_constrained (dofs[j]))
                {
                  const auto x = sim.get_mapping ().transform_unit_to_real_cell (
                                   cell, fe.get_unit_support_points ()[j]);
                  double distance = std::numeric_limits<double>::max ();
                  for (const auto &fault : faults)
                    for (unsigned int i = 0; i + 1 < fault.vertices.size (); ++i)
                      {
                        const auto direction = fault.vertices[i + 1] - fault.vertices[i];
                        double xi = (x - fault.vertices[i]) * direction / direction.norm_square ();
                        if (i != 0 || !boundary (fault.vertices.front ()))
                          xi = std::max (0., xi);
                        if (i + 2 != fault.vertices.size () || !boundary (fault.vertices.back ()))
                          xi = std::min (1., xi);
                        distance = std::min (distance, x.distance (fault.vertices[i] + xi * direction));
                      }
                  constraints.add_line (dofs[j]);
                  constraints.set_inhomogeneity (dofs[j], profile.front ()->value (distance));
                }
          }
    }
    template <int dim>
    void
    connect_boundary_test (SimulatorSignals<dim> &signals)
    {
      signals.post_constraints_creation.connect (&prescribe_boundary_test_phase<dim>);
    }
    ASPECT_REGISTER_SIGNALS_CONNECTOR (connect_boundary_test<2>, connect_boundary_test<3>)
  }

  namespace Postprocess
  {
    template <int dim>
    class VerifyBoundaryCompletion : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string, std::string>
        execute (TableHandler &) override
        {
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &> (
                          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>> (
                            this->get_material_model ()));
          using Access = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
          Access::initialize_cohesive_state_from_initial_fields (model);
          const auto &ih = Access::compute_normalization_integrals (model);
          auto &manager = this->get_reconstructed_fault_manager ();
          auto &surface = this->get_reconstructed_fault_surface_system ();
          for (const auto &contact : manager.get_boundary_contacts ())
            if (contact.influence_length > 0.)
              {
                const double b = contact.normal * contact.inward_boundary_normal;
                const auto point = contact.position - .1 * contact.influence_length * contact.inward_tangent
                                   + std::copysign (.5 * contact.transverse_extent, b) * contact.normal;
                const auto association = manager.project_to_bulk_source (point);
                AssertThrow (association.active && association.fault_index == contact.fault_index
                             && association.xi == (contact.endpoint == 0 ? 0. : 1.)
                             && association.segment_index
                             == (contact.endpoint == 0 ? 0 : manager.get_fault (contact.fault_index).n_cells () - 1),
                             ExcMessage ("Physical wedge did not retain its endpoint association."));
              }
          using Vector = typename ReconstructedFaultSurfaceSystem<dim>::FaultVector;
          Vector V (ih.size ()), direction (ih.size ());
          for (unsigned int f = 0; f < ih.size (); ++f)
            {
              V[f].assign (ih[f].size (), 2e-6);
              direction[f].assign (ih[f].size (), 0.);
              direction[f].front () = 1e-6;
              direction[f].back () = .7e-6;
              for (const auto value : ih[f])
                AssertThrow (std::isfinite (value) && value > 0., ExcMessage ("Invalid completed I_h."));
            }
          surface.linearize_surface_system (this->get_solution (), V);
          Vector action;
          surface.apply_surface_jacobian (direction, action);
          constexpr double epsilon = .125;
          auto plus = V, minus = V;
          for (unsigned int f = 0; f < V.size (); ++f)
            for (unsigned int i = 0; i < V[f].size (); ++i)
              {
                plus[f][i] += epsilon * direction[f][i];
                minus[f][i] -= epsilon * direction[f][i];
              }
          const auto rp = surface.evaluate_surface_residual (this->get_solution (), plus);
          const auto rm = surface.evaluate_surface_residual (this->get_solution (), minus);
          double error = 0., scale = 0.;
          for (unsigned int f = 0; f < V.size (); ++f)
            for (unsigned int i = 0; i < V[f].size (); ++i)
              {
                error = std::max (
                          error, std::abs (action[f][i] + (rp.values[f][i] - rm.values[f][i]) / (2 * epsilon)));
                scale = std::max (scale, std::abs (action[f][i]));
              }
          AssertThrow (scale > 0. && error < 1e-8 * scale,
                       ExcMessage ("Free-endpoint K_V finite difference failed."));
          const double k_error = error / scale;
          const auto owned = [&] ()
          {
            return LinearAlgebra::BlockVector (this->introspection ().index_sets.system_partitioning,
                                               this->get_mpi_communicator ());
          };
          const auto ghosted = [&] (const LinearAlgebra::BlockVector &input)
          {
            LinearAlgebra::BlockVector result (
              this->introspection ().index_sets.system_partitioning,
              this->introspection ().index_sets.system_relevant_partitioning,
              this->get_mpi_communicator ());
            result = input;
            return result;
          };
          auto pressure = owned ();
          pressure = 0.;
          pressure.block (this->introspection ().block_indices.pressure) = 1e6;
          this->get_current_constraints ().set_zero (pressure);
          pressure.compress (VectorOperation::insert);
          surface.apply_G (ghosted (pressure), action);
          auto state_plus = owned (), state_minus = owned ();
          state_plus = this->get_solution ();
          state_minus = this->get_solution ();
          state_plus.add (epsilon, pressure);
          state_minus.add (-epsilon, pressure);
          const auto gp = surface.evaluate_surface_residual (ghosted (state_plus), V);
          const auto gm = surface.evaluate_surface_residual (ghosted (state_minus), V);
          error = 0.;
          scale = 0.;
          for (unsigned int f = 0; f < V.size (); ++f)
            for (unsigned int i = 0; i < V[f].size (); ++i)
              {
                error = std::max (
                          error, std::abs (action[f][i] - (gp.values[f][i] - gm.values[f][i]) / (2 * epsilon)));
                scale = std::max (scale, std::abs (action[f][i]));
              }
          AssertThrow (scale > 0. && error < 1e-8 * scale,
                       ExcMessage ("Continued surface G finite difference failed."));
          const double g_error = error / scale;
          auto &bulk = this->get_reconstructed_fault_stokes_coupling ();
          bulk.linearize_B (this->get_solution ());
          auto B = owned (), Bp = owned (), Bm = owned ();
          bulk.apply_B (direction, B);
          bulk.evaluate_slip_dependent_bulk_residual (this->get_solution (), plus, Bp);
          bulk.evaluate_slip_dependent_bulk_residual (this->get_solution (), minus, Bm);
          Bp.add (-1., Bm);
          Bp *= 1. / (2 * epsilon);
          Bp.add (1., B);
          AssertThrow (B.l2_norm () > 0. && Bp.l2_norm () < 1e-8 * B.l2_norm (),
                       ExcMessage ("Continued bulk B finite difference failed."));
          this->get_pcout () << std::setprecision (17) << "Boundary coupling verified: K=" << k_error
                             << ", G=" << g_error << ", B=" << Bp.l2_norm () / B.l2_norm () << std::endl;
          if (Utilities::MPI::this_mpi_process (this->get_mpi_communicator ()) == 0)
            {
              std::ofstream out (this->get_output_directory () + "boundary_state.csv");
              out << std::setprecision (17) << "fault,node,x,y,I_h\n";
              for (unsigned int f = 0; f < ih.size (); ++f)
                for (unsigned int i = 0; i < ih[f].size (); ++i)
                  out << f << ',' << i << ',' << manager.get_fault (f).vertex (i)[0] << ','
                      << manager.get_fault (f).vertex (i)[1] << ',' << ih[f][i] << '\n';
            }
          return { "Boundary completion:", "verified" };
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR (
      VerifyBoundaryCompletion, "verify fault boundary completion",
      "Check completed normalization and free-endpoint K, B and G derivatives.")
  }
}
