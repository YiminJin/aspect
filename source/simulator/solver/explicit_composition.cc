/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/simulator/solver/explicit_composition.h>
#include <aspect/simulator/solver/dg_transport.h>
#include <aspect/parameters.h>
#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/initial_topography_model/zero_topography.h>
#include <aspect/material_model/interface.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace Transport
  {
    template <int dim>
    CompositionInflow<dim>
    parse_explicit_composition_parameters(ParameterHandler &prm, const Parameters<dim> &p)
    {
      using Method = typename Parameters<dim>::AdvectionFieldMethod;
      if (std::find(p.compositional_field_methods.begin(), p.compositional_field_methods.end(),
                    Method::explicit_dg_ssprk3) == p.compositional_field_methods.end())
        return {};

      using Scheme=typename Parameters<dim>::NonlinearSolver;
      AssertThrow(!p.use_operator_splitting && !p.enable_elasticity && !p.include_melt_transport &&
                  (p.nonlinear_solver==Scheme::single_Advection_no_Stokes ||
                   p.nonlinear_solver==Scheme::single_Advection_single_Stokes ||
                   p.nonlinear_solver==Scheme::single_Advection_iterated_Stokes),
                  ExcMessage("SSPRK3 fields require single Advection on a Cartesian box without operator splitting, melt or elasticity."));
      AssertThrow(p.stokes_velocity_degree==2,
                  ExcMessage("SSPRK3 composition transport is qualified for Stokes velocity degree 2."));
      AssertThrow(p.initial_adaptive_refinement==0 && p.adaptive_refinement_interval==0 && p.additional_refinement_times.empty(),
                  ExcMessage("SSPRK3 fields require a fixed mesh."));
      for (unsigned int c=0; c<p.n_compositional_fields; ++c)
        if (p.compositional_field_methods[c] == Method::explicit_dg_ssprk3)
          {
            const auto type=p.composition_descriptions[c].type;
            AssertThrow(p.use_discontinuous_composition_discretization[c] && p.composition_degrees[c]==2 &&
                        !p.use_limiter_for_discontinuous_composition_solution[c] &&
                        (type==CompositionalFieldDescription::generic || type==CompositionalFieldDescription::chemical_composition ||
                         type==CompositionalFieldDescription::phase_field),
                        ExcMessage("SSPRK3 requires a scalar DGQ2 tracer or phase without the release limiter: "+p.names_of_compositional_fields[c]));
          }

      // The release boundary manager updates its plugins at the simulator clock.
      // Parse an independent evaluator of the selected function for the RK clocks.
      prm.enter_subsection("Boundary composition model");
      if (prm.get("Fixed composition boundary indicators").empty())
        {
          // A closed box can have no incoming face. An actual inflow without a
          // prescribed value is diagnosed collectively during the stage.
          prm.leave_subsection();
          return {};
        }
      AssertThrow(prm.get("List of model names")=="function" && prm.get("List of model operators")=="add",
                  ExcMessage("SSPRK3 fields require the function boundary composition model with the add operator."));
      prm.enter_subsection("Function");
      AssertThrow(prm.get("Coordinate system")=="cartesian",ExcMessage("SSPRK3 inflow requires Cartesian coordinates."));
      // This evaluator is private to the explicit solver. Each evaluation
      // receives its stage clock, including when an attempt is repeated.
      auto function=std::make_shared<Functions::ParsedFunction<dim>>(p.n_compositional_fields);
      function->parse_parameters(prm);
      const double unit=p.convert_to_years ? year_in_seconds : 1.;
      prm.leave_subsection();
      prm.leave_subsection();
      return [function,unit](const Point<dim> &point,unsigned int field,double seconds)
      {
        const double time=seconds/unit;
        if (function->get_time()!=time)
          function->set_time(time);
        return function->value(point,field);
      };
    }

    template <int dim>
    void advance_composition(const SimulatorAccess<dim> &sim, const unsigned int field, LinearAlgebra::BlockVector &solution)
    {
      const auto &p=sim.get_parameters();
      AssertThrow(Plugins::plugin_type_matches<GeometryModel::Box<dim>>(sim.get_geometry_model()) &&
                  Plugins::plugin_type_matches<InitialTopographyModel::ZeroTopography<dim>>(sim.get_initial_topography_model()) &&
                  sim.get_geometry_model().get_periodic_boundary_pairs().empty() && !p.mesh_deformation_enabled,
                  ExcMessage("Explicit DG composition requires an undeformed nonperiodic Cartesian box."));
      const auto &intro=sim.introspection();
      const auto comm=sim.get_mpi_communicator();
      // A strong interior prescription cannot be imposed only at the end of
      // the RK step. Inspect global indices of this block, including on retries.
      const unsigned int block=intro.block_indices.compositional_fields[field];
      const auto first=solution.get_block_indices().block_start(block);
      const auto end=first+solution.block(block).size();
      const auto lines=sim.get_current_constraints().get_lines();
      const bool constrained=std::any_of(lines.begin(),lines.end(),[&](const auto &line)
      {
        return first<=line.index && line.index<end;
      });
      AssertThrow(!Utilities::MPI::max(static_cast<unsigned int>(constrained),comm),
                  ExcMessage("SSPRK3 cannot advance prescribed DoFs on field "+intro.name_for_compositional_index(field)));

      // The implicit composition RHS tests a whole-step reaction increment at
      // this linearization point and quadrature. Only the selected row matters.
      const QGauss<dim> rule(p.composition_degrees[field]+(p.stokes_velocity_degree+1)/2);
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),rule,update_values|update_gradients|update_quadrature_points);
      MaterialModel::MaterialModelInputs<dim> in(rule.size(),intro.n_compositional_fields);
      MaterialModel::MaterialModelOutputs<dim> out(rule.size(),intro.n_compositional_fields);
      in.requested_properties=MaterialModel::MaterialProperties::reaction_terms;
      const auto &material=sim.get_material_model();
      const auto &iterate=sim.get_current_linearization_point();
      bool reacts=false;
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            in.reinit(fe,cell,intro,iterate);
            material.fill_additional_material_model_inputs(in,iterate,fe,intro);
            material.evaluate(in,out);
            for (const auto &increments : out.reaction_terms)
              reacts |= increments[field]!=0.;
          }
      AssertThrow(!Utilities::MPI::max(static_cast<unsigned int>(reacts),comm),
                  ExcMessage("SSPRK3 pure advection cannot omit a reaction increment for "+intro.name_for_compositional_index(field)));
      const auto &fixed=sim.get_fixed_composition_boundary_indicators();
      const ScalarInflow<dim> exterior=[&](types::boundary_id id,const Point<dim> &point,const double t)
      {
        if (!fixed.count(id))
          return std::numeric_limits<double>::quiet_NaN();
        return p.explicit_composition_inflow(point,field,t);
      };

      const auto candidate=advect_scalar(sim,field,exterior,p.composition_descriptions[field].type==CompositionalFieldDescription::phase_field);
      LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[block],comm);
      const auto &system=sim.get_fe();
      std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            const auto &values=candidate.at(cell->id());
            for (unsigned int i=0; i<values.size(); ++i)
              {
                const auto local=system.component_to_system_index(intro.component_indices.compositional_fields[field],i);
                owned[indices[local]-solution.get_block_indices().block_start(block)]=values[i];
              }
          }
      owned.compress(VectorOperation::insert);
      solution.block(block)=owned;
    }

    template CompositionInflow<2> parse_explicit_composition_parameters(ParameterHandler &, const Parameters<2> &);
    template CompositionInflow<3> parse_explicit_composition_parameters(ParameterHandler &, const Parameters<3> &);
    template void advance_composition(const SimulatorAccess<2> &, unsigned int, LinearAlgebra::BlockVector &);
    template void advance_composition(const SimulatorAccess<3> &, unsigned int, LinearAlgebra::BlockVector &);
  }
}
