/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/mesh_refinement/growing_fault.h>
#include <aspect/mesh_refinement/fault_damage.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/geometry_model/box.h>
#include <aspect/reconstructed_fault/onset_score.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/simulator_signals.h>

namespace aspect::MeshRefinement
{
  namespace
  {
    using Box2 = BoundingBox<2>;

    // Minimum Euclidean distance to the closed cell footprint. Test segment
    // intersection first. A disjoint closest pair includes a vertex of
    // either the box or the segment.
    double distance_to_segment(const Box2 &box, const Point<2> &a, const Point<2> &b)
    {
      const auto &[lo,hi] = box.get_boundary_points();
      const auto direction = b-a;
      double first = 0., last = 1.;
      for (unsigned int d=0; d<2; ++d)
        if (direction[d] != 0.)
          {
            const double u=(lo[d]-a[d])/direction[d], v=(hi[d]-a[d])/direction[d];
            first=std::max(first,std::min(u,v));
            last=std::min(last,std::max(u,v));
          }
        else if (a[d]<lo[d] || a[d]>hi[d])
          last=-1.;
      if (first<=last)
        return 0.;
      double result = std::min(box.signed_distance(a),box.signed_distance(b));
      for (unsigned int i=0; i<4 && direction.norm_square()>0.; ++i)
        {
          const Point<2> corner((i&1)?hi[0]:lo[0],(i&2)?hi[1]:lo[1]);
          const double t=std::clamp(((corner-a)*direction)/direction.norm_square(),0.,1.);
          result=std::min(result,corner.distance(a+t*direction));
        }
      return result;
    }

    // Clip the physical box to the two half planes of the forward cone.
    // Padding is applied to the resulting polygon by Euclidean distance.
    std::vector<Point<2>> forward_sector(const Box2 &box, const Point<2> &tip,
                                         const Tensor<1,2> &tangent)
    {
      const auto &[lo,hi]=box.get_boundary_points();
      std::vector<Point<2>> polygon {lo,Point<2>(hi[0],lo[1]),hi,Point<2>(lo[0],hi[1])};
      const Tensor<1,2> normal({-tangent[1],tangent[0]});
      for (const double side :
      {
        -1.,1.
      })
      {
        std::vector<Point<2>> clipped;
        const double angle = aspect::GrowingFault::maximum_turn_angle;
        const auto inward = std::sin(angle)*tangent + side*std::cos(angle)*normal;
        for (unsigned int i=0; i<polygon.size(); ++i)
          {
            const auto &a=polygon[i], &b=polygon[(i+1)%polygon.size()];
            const double u=(a-tip)*inward, v=(b-tip)*inward;
            if (u>=0.) clipped.push_back(a);
            if ((u<0.) != (v<0.))
              clipped.push_back(a+u/(u-v)*(b-a));
          }
        polygon=std::move(clipped);
      }
      return polygon;
    }

    bool within_sector(const Box2 &box, const std::vector<Point<2>> &polygon, const double margin)
    {
      if (polygon.empty()) return false;
      bool inside=true;
      for (unsigned int i=0; i<polygon.size(); ++i)
        {
          const auto &a=polygon[i], &b=polygon[(i+1)%polygon.size()];
          if (distance_to_segment(box,a,b)<=margin) return true;
          const auto edge=b-a, delta=box.center()-a;
          inside &= edge[0]*delta[1]-edge[1]*delta[0]>=0.;
        }
      return inside;
    }

    bool overlaps(const Box2 &a, const Box2 &b)
    {
      const auto &[x,y]=a.get_boundary_points();
      const auto &[u,v]=b.get_boundary_points();
      return x[0]<v[0] && u[0]<y[0] && x[1]<v[1] && u[1]<y[1];
    }

    double cell_width(const Box2 &box)
    {
      return std::max(box.side_length(0),box.side_length(1));
    }

    template <int dim>
    const aspect::GrowingFault::StripMaterialHistory<dim> &material(const SimulatorAccess<dim> &sim)
    {
      const auto *fault=dynamic_cast<const MaterialModel::PhaseFieldFault<dim> *>(&sim.get_material_model());
      AssertThrow(dim==2 && fault && fault->uses_growing_strips(),
                  ExcMessage("Growing fault refinement requires the two-dimensional midpoint strip material."));
      return fault->strip_history();
    }
  }

  template <int dim>
  typename GrowingFault<dim>::Region GrowingFault<dim>::fault_region(const bool candidate) const
  {
    Region region;
    if constexpr (dim==2)
      {
        const auto &state=material(*this);
        const auto &box=dynamic_cast<const GeometryModel::Box<2> &>(this->get_geometry_model());
        const auto lo=box.get_origin(), hi=lo+box.get_extents();
        const auto &seed=state.seed_tip();
        const auto tangent=state.seed_tangent();
        region.tip=seed;

        // A step returned by Stokes lives in solution. Before the next solve,
        // the accepted parent lives in old_solution. During early refinement
        // some inputs skip FE initialization altogether.
        const bool has_field = initialized_phase || this->get_parameters().resume_computation;
        const auto &fe = this->get_fe();
        const auto field = state.phase_field();
        const auto component = this->introspection().component_indices.compositional_fields[field];
        const auto &element = fe.base_element(fe.component_to_base_index(component).first);
        AssertThrow(element.n_dofs_per_cell()==9, ExcMessage("Growing fault refinement requires DGQ2 Bernstein phase."));
        const auto &solution = candidate || !state.has_native_history() || state.history().step == this->get_timestep_number()
                               ? this->get_solution() : this->get_old_solution();
        std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
        const auto anchor=state.incoming_seed_anchor();
        const bool physical_trial=candidate && this->get_timestep_number()>0 && state.staged_sections();
        std::vector<FaultDamage::Cell> active, accepted_seed;
        unsigned int coarse_candidates = 0;
        for (const auto &cell : this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              FaultDamage::Cell data {cell->id(),cell->bounding_box(),{}};
              if (has_field)
                {
                  cell->get_dof_indices(dofs);
                  for (unsigned int i=0; i<data.phase.size(); ++i)
                    data.phase[i]=solution[dofs[fe.component_to_system_index(component,i)]];
                }
              else
                {
                  // Sample the prescribed initializer at the element's own
                  // interpolation sites. The actual limited FE seed is checked
                  // again after the last initial refinement/zero-duration solve.
                  std::vector<Vector<double>> values(9,Vector<double>(1));
                  for (unsigned int i=0; i<values.size(); ++i)
                    {
                      const auto position=this->get_mapping().transform_unit_to_real_cell(cell,element.get_generalized_support_points()[i]);
                      values[i][0]=this->get_initial_composition_manager().initial_composition(position,field);
                    }
                  std::vector<double> coefficients(9);
                  element.convert_generalized_support_point_values_to_dof_values(values,coefficients);
                  std::copy(coefficients.begin(),coefficients.end(),data.phase.begin());
                }
              if (physical_trial && distance_to_segment(data.bounds,anchor,seed)==0.)
                {
                  auto parent=data;
                  for (unsigned int i=0; i<parent.phase.size(); ++i)
                    parent.phase[i]=this->get_old_solution()[dofs[fe.component_to_system_index(component,i)]];
                  accepted_seed.push_back(std::move(parent));
                }
              if (FaultDamage::may_contain_damage(data.phase))
                {
                  coarse_candidates |= cell_width(data.bounds)>bulk_target;
                  active.push_back(std::move(data));
                }
            }

        const auto &comm = this->get_mpi_communicator();

        // Freeze the incoming material source on the accepted one-sided DG
        // trace. An earlier disconnected island on that line may reach the
        // contour in this trial. It must not become the new seed.
        std::optional<FaultDamage::Source> source;
        if (physical_trial)
          source=FaultDamage::seed_on_line(accepted_seed,anchor,seed,tangent,comm).source;

        std::optional<Point<2>> rear;
        if (Utilities::MPI::max(coarse_candidates,comm))
          {
            AssertThrow(!physical_trial || source,
                        ExcMessage("Coarse phase may reach the damage contour but its accepted seed has no resolved anchor."));
            const auto connected = FaultDamage::connected_component(active,anchor,seed,tangent,comm,source);
            if (connected.anchored)
              {
                rear=connected.rear;
                for (const auto &[id,bounds] : connected.cells)
                  if (cell_width(bounds)>bulk_target)
                    region.damage.push_back(bounds);
              }
            else
              {
                AssertThrow(!has_field, ExcMessage("Coarse phase may reach the damage contour but the incoming "
                                                   "seed has no resolved anchor. Check its position and initial phase."));
                // An early interpolant can miss a narrow seed. Give the
                // initialization a finer mesh and recheck the physical field.
                region.damage.emplace_back(std::make_pair(lo,hi));
              }
          }
        else
          {
            const auto line=FaultDamage::seed_on_line(active,anchor,seed,tangent,comm,source);
            if (line.anchored)
              rear=line.rear;
          }

        region.band= {rear.value_or(anchor),seed};
        if (state.has_native_history())
          {
            const auto &records=candidate && state.staged_sections() ?
                                *state.staged_sections() : state.history().sections;
            std::vector<unsigned int> formed;
            std::vector<double> angles;
            for (const auto &s : records)
              {
                formed.push_back(s.formed);
                angles.push_back(s.angle);
              }
            const auto &sections=state.sections();
            const auto graph=sections.accepted_graph(seed,angles,formed);
            for (unsigned int i=0; i<records.size(); ++i)
              {
                const double xi=.5*(sections.edges[i]+sections.edges[i+1]);
                if (xi<=sections.coordinate(seed)) continue;
                if (!formed[i]) break;
                region.band.push_back(graph.at(xi));
              }
            if (coverage==Coverage::disk)
              {
                // Ordinary scheduled marking can run after publication but
                // before advance_time rotates the FE vectors.
                region.tip=state.phase_supported_tip(angles,formed,solution);
              }
          }
        if (coverage==Coverage::sector)
          region.sector=forward_sector(Box2({lo,hi}),seed,tangent);
        else if (coverage==Coverage::disk)
          region.tip_radius=restore_radius;
      }
    return region;
  }

  template <int dim>
  bool GrowingFault<dim>::mark_region(const Region &region, const bool mark) const
  {
    unsigned int count=0;
    if constexpr (dim==2)
      {
        const double margin=profile_half_width;
        for (const auto &cell : this->get_triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto box=cell->bounding_box();
              if (cell_width(box)<=bulk_target) continue;
              bool needed=(region.tip_radius>0. && box.signed_distance(region.tip)<=region.tip_radius)
                          || within_sector(box,region.sector,margin);
              for (unsigned int i=1; !needed && i<region.band.size(); ++i)
                needed=distance_to_segment(box,region.band[i-1],region.band[i])<=margin;
              if (!needed)
                needed=std::any_of(region.damage.begin(),region.damage.end(),
                                   [&](const auto &miss)
                {
                  return overlaps(box,miss);
                });
              if (needed)
                {
                  ++count;
                  if (mark)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();
                    }
                }
            }
      }
    return Utilities::MPI::sum(count,this->get_mpi_communicator())>0;
  }

  template <int dim>
  void GrowingFault<dim>::tag_additional_cells() const
  {
    // ASPECT already flags every cell on its global refinement passes, before
    // the DoFHandler or phase field can be used.
    if (this->get_dof_handler().n_dofs()!=0)
      mark_region(requested ? *requested : fault_region(false),true);
  }

  template <int dim>
  bool GrowingFault<dim>::needs_further_refinement() const
  {
    return requested && mark_region(*requested,false);
  }

  template <int dim>
  void GrowingFault<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Mesh refinement");
    prm.enter_subsection("Growing fault");
    prm.declare_entry("Coverage","tip disk",Patterns::Selection("tip disk|forward sector|damage threshold"),
                      "Resolve the formed band and seed-connected damage. The tip disk replenishes the "
                      "direction-search neighborhood. The forward sector covers the possible seed-relative "
                      "directions before loading. Damage threshold omits the forward reserve.");
    prm.declare_entry("Maximum cell edge","0",Patterns::Double(0.),
                      "Maximum physical length in metres of the longest edge of a cell in the refined region. "
                      "Zero inherits the material's Target strip width. Changing this setting preserves strip identities.");
    prm.declare_entry("Tip disk trigger margin","1",Patterns::Double(0.),
                      "Extra clearance, in phase length scales, outside the mechanical direction-search radius "
                      "before the tip disk requests preventive refinement.");
    prm.declare_entry("Tip disk replenishment margin","1",Patterns::Double(0.),
                      "Additional clearance, in phase length scales, outside the trigger radius when refining the disk. "
                      "Must be positive.");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void GrowingFault<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Mesh refinement");
    prm.enter_subsection("Growing fault");
    const auto mode=prm.get("Coverage");
    coverage=mode=="tip disk" ? Coverage::disk : mode=="forward sector" ? Coverage::sector : Coverage::reactive;
    bulk_target=prm.get_double("Maximum cell edge");
    const auto &model=this->get_parameters().surface_energy_model;
    profile_half_width=model.compact_profile_half_width();
    const double mechanical_radius=aspect::GrowingFault::direction_search_radius(model.length_scale);
    trigger_radius=mechanical_radius+prm.get_double("Tip disk trigger margin")*model.length_scale;
    restore_radius=trigger_radius+prm.get_double("Tip disk replenishment margin")*model.length_scale;
    prm.leave_subsection();
    prm.leave_subsection();
    AssertThrow(dim==2,ExcMessage("Growing fault refinement requires two dimensions."));
    AssertThrow(std::isfinite(restore_radius) && restore_radius>trigger_radius && trigger_radius>=mechanical_radius,
                ExcMessage("The tip disk must cover the direction-search radius and replenish beyond its trigger radius."));
  }

  template <int dim>
  void GrowingFault<dim>::initialize()
  {
    const auto &state=material(*this);
    state.validate_refinement();
    if (bulk_target==0.)
      bulk_target=state.target_strip_width();
    // Roundoff is relative to this physical size and applies identically to
    // marking, the reserve audit and the coarse-damage screen.
    bulk_target *= 1.+1.e-12;
    // On restart the simulator grants additional levels only for scheduled
    // times already passed by the restored clock (Simulator::run).
    const auto check_reachable = [this](const unsigned int additional_levels)
    {
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const unsigned int levels=this->get_parameters().initial_global_refinement
                                +this->get_parameters().initial_adaptive_refinement + additional_levels;
      for (unsigned int d=0; d<dim; ++d)
        AssertThrow(std::ldexp(box.get_extents()[d]/box.get_repetitions()[d],-static_cast<int>(levels))<=bulk_target,
                    ExcMessage("Maximum cell edge is unreachable at the allowed refinement level."));
    };
    if (this->get_parameters().resume_computation)
      this->get_signals().post_resume_load_user_data.connect([this,check_reachable](const auto &)
      {
        const auto &times=this->get_parameters().additional_refinement_times;
        const auto end=std::lower_bound(times.begin(),times.end(),this->get_time()+this->get_timestep());
        check_reachable(std::distance(times.begin(),end));
      });
    else
      check_reachable(0);
    this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim> &)
    {
      initialized_phase=true;
    });
  }

  template <int dim>
  bool GrowingFault<dim>::needs_coverage(const Region &region) const
  {
    if (!region.damage.empty())
      return true;
    if (coverage==Coverage::reactive)
      return false;
    if (coverage==Coverage::disk)
      {
        Region guard=region;
        guard.tip_radius=trigger_radius;
        return mark_region(guard,false);
      }
    return mark_region(region,false);
  }

  template <int dim>
  bool GrowingFault<dim>::needs_initial_refinement() const
  {
    requested.reset();
    auto seed=fault_region(true);
    if (!needs_coverage(seed))
      return false;
    this->get_pcout() << "Growing fault initial coverage: refine initialized seed" << std::endl;
    requested=std::move(seed);
    return true;
  }

  template <int dim>
  TimeStepping::Reaction GrowingFault<dim>::assess_endpoint() const
  {
    using TimeStepping::Reaction;
    requested.reset();
    const auto &state=material(*this);
    const bool physical=this->get_timestep_number()>0;
    if (physical && !state.staged_sections())
      return Reaction::advance; // A nonlinear failure has no candidate to inspect.
    auto next=fault_region(true);
    if (!needs_coverage(next))
      return Reaction::advance;

    const auto exposed_cells=next.damage.size();
    const bool unresolved=exposed_cells>0;
    bool parent_exposed=false;
    if (unresolved && physical)
      {
        const auto parent=fault_region(false);
        parent_exposed=!parent.damage.empty();
        // Include any different coarse footprints of the old state to make
        // the transferred parent and the trial's extension both resolvable.
        next.damage.insert(next.damage.end(),parent.damage.begin(),parent.damage.end());
      }
    this->get_pcout() << (unresolved && physical ?
                          (parent_exposed ? "Growing fault unresolved accepted damage: refine parent and repeat" :
                           "Growing fault missed extension: refine and repeat") :
                          "Growing fault preventive refinement: refine and advance")
                      << " at t=" << this->get_time() << " s, tip " << next.tip
                      << ", connected coarse cells " << exposed_cells << std::endl;
    requested=std::move(next);
    return unresolved && physical ? Reaction::refine_and_repeat_step : Reaction::refine_and_advance;
  }

  ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(GrowingFault, "growing fault",
                                            "Resolve the formed strip band and a forward region on fixed Cartesian "
                                            "boxes. Select the growing fault refinement timestep model to follow "
                                            "the advancing front and refine before repeating a missed extension.")
}
