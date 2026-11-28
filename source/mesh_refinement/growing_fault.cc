/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/mesh_refinement/growing_fault.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/geometry_model/box.h>

#include <deal.II/fe/fe_values.h>
#include <queue>

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
        const auto inward=tangent+side*normal;
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
        // Trace the seed back to its first wall, including an inclined seed.
        double distance=std::numeric_limits<double>::max();
        for (unsigned int d=0; d<2; ++d)
          if (tangent[d]!=0.)
            distance=std::min(distance,(seed[d]-(tangent[d]>0.?lo[d]:hi[d]))/tangent[d]);
        region.band= {seed-distance*tangent,seed};
        region.tip=seed;
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
                const auto &phase = candidate || state.history().step == this->get_timestep_number() ?
                                    this->get_solution() : this->get_old_solution();
                region.tip=state.phase_supported_tip(angles,formed,phase);
              }
          }
        if (coverage==Coverage::sector)
          region.sector=forward_sector(Box2({lo,hi}),seed,tangent);
        else if (coverage==Coverage::disk)
          region.tip_radius=4.*this->get_parameters().surface_energy_model.length_scale;
      }
    return region;
  }

  template <int dim>
  bool GrowingFault<dim>::mark_region(const Region &region, const bool mark) const
  {
    unsigned int count=0;
    if constexpr (dim==2)
      {
        const double resolution=material(*this).target_strip_width();
        AssertThrow(resolution>0.,ExcMessage("Growing fault refinement requires a positive Target strip width."));
        const double margin=2.*this->get_parameters().surface_energy_model.length_scale;
        for (const auto &cell : this->get_triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto box=cell->bounding_box();
              if (cell_width(box)<=resolution*(1.+1.e-12)) continue;
              bool needed=(region.tip_radius>0. && box.signed_distance(region.tip)<=region.tip_radius)
                          || within_sector(box,region.sector,margin);
              for (unsigned int i=1; !needed && i<region.band.size(); ++i)
                needed=distance_to_segment(box,region.band[i-1],region.band[i])<=margin;
              if (!needed)
                needed=std::any_of(region.missed_cells.begin(),region.missed_cells.end(),
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
                      "Resolve the formed band to the material's target strip width. The tip disk "
                      "restores four phase lengths when fewer than three remain. The forward sector "
                      "covers all seed-relative directions within 45 degrees before loading. Damage "
                      "threshold requests extra coverage only after an unresolved extension.");
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
    prm.leave_subsection();
    prm.leave_subsection();
    AssertThrow(dim==2,ExcMessage("Growing fault refinement requires two dimensions."));
  }

  template <int dim>
  std::vector<BoundingBox<2>> GrowingFault<dim>::unresolved_damage() const
  {
    std::vector<Box2> missed;
    if constexpr (dim==2)
      {
        const auto &state=material(*this);
        // Resolve connectivity on a 4-by-4 physical sampling lattice per cell.
        // First screen only coarse cells, since the distributed component
        // search is needed only if the .01 contour crossed a coarse sample.
        constexpr unsigned int n=4;
        constexpr double contour=.01, minimum_increment=1.e-4;
        std::vector<Point<2>> points;
        for (unsigned int y=0; y<n; ++y)
          for (unsigned int x=0; x<n; ++x)
            points.emplace_back((x+.5)/n,(y+.5)/n);
        const auto comm=this->get_mpi_communicator();
        FEValues<2> fe(this->get_mapping(),this->get_fe(),Quadrature<2>(points),update_values);
        const auto phase=this->introspection().extractors.compositional_fields[state.phase_field()];
        std::vector<double> trial(points.size()), old(points.size());
        const auto &box=dynamic_cast<const GeometryModel::Box<2> &>(this->get_geometry_model());
        const auto origin=box.get_origin();
        const auto subdivisions=box.get_repetitions();
        const unsigned int levels=this->get_triangulation().n_global_levels();
        const double size=state.target_strip_width()*(1.+1.e-12);
        using Coordinate=std::int64_t;
        std::array<double,2> unit;
        for (unsigned int d=0; d<2; ++d)
          unit[d]=std::ldexp(box.get_extents()[d]/(n*subdivisions[d]),1-static_cast<int>(levels));
        // Integer intervals on the common finest dyadic lattice avoid
        // coordinate-roundoff decisions at hanging and partition faces.
        // The prescribed seed may be an interior patch. Its backward tangent
        // reaches the wall, but an active island elsewhere on that ray is not
        // connected to this seed. Anchor at the incoming station one length
        // behind the physical tip, clipped to the box as in the strip tracker.
        const Point<2> seed_tip = state.seed_tip();
        const auto tangent = state.seed_tangent();
        double incoming_length = this->get_parameters().surface_energy_model.length_scale;
        for (unsigned int d=0; d<2; ++d)
          if (tangent[d]!=0.)
            {
              const double wall = origin[d]+(tangent[d]>0. ? 0. : box.get_extents()[d]);
              incoming_length = std::min(incoming_length,(seed_tip[d]-wall)/tangent[d]);
            }
        const Point<2> anchor = seed_tip - incoming_length*tangent;
        using Tile=std::array<Coordinate,6>; // x0,y0,x1,y1,crossing,seed
        std::vector<Tile> active;
        unsigned int crossing=0;
        for (unsigned int pass=0; pass<2; ++pass)
          {
            for (const auto &cell : this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto bounds=cell->bounding_box();
                  const bool coarse=cell_width(bounds)>size;
                  if (pass==0 && !coarse) continue;
                  fe.reinit(cell);
                  fe[phase].get_function_values(this->get_solution(),trial);
                  if (coarse || pass==1)
                    fe[phase].get_function_values(this->get_old_solution(),old);
                  const auto &lo=bounds.get_boundary_points().first;
                  std::array<Coordinate,2> base;
                  for (unsigned int d=0; d<2; ++d)
                    base[d]=std::llround((lo[d]-origin[d])/unit[d]);
                  const Coordinate stride=Coordinate(1)<<(levels-1-cell->level());
                  for (unsigned int q=0; q<points.size(); ++q)
                    if (trial[q]>=contour)
                      {
                        const bool new_damage=coarse && old[q]<contour && trial[q]-old[q]>=minimum_increment;
                        if (new_damage) crossing=1;
                        if (pass==0) continue;
                        const auto x=base[0]+(q%n)*stride, y=base[1]+(q/n)*stride;
                        const Box2 tile(
                        {
                          Point<2>(origin[0]+x*unit[0],origin[1]+y*unit[1]),
                                Point<2>(origin[0]+(x+stride)*unit[0],origin[1]+(y+stride)*unit[1])
                        });
                        const bool seed = old[q]>=contour && distance_to_segment(tile,anchor,seed_tip)==0.;
                        active.push_back({{x,y,x+stride,y+stride,new_damage,seed}});
                      }
                }
            if (!Utilities::MPI::max(crossing,comm)) return missed;
          }

        std::vector<Tile> tiles;
        for (auto &part : Utilities::MPI::all_gather(comm,active))
          tiles.insert(tiles.end(),part.begin(),part.end());
        std::vector<std::vector<unsigned int>> neighbors(tiles.size());
        // Sweep the two ordered traces of each physical face. Strict interval
        // overlap admits hanging faces and excludes corner-only contacts.
        for (unsigned int d=0; d<2; ++d)
          {
            using Interval=std::tuple<Coordinate,Coordinate,unsigned int>;
            std::map<Coordinate,std::array<std::vector<Interval>,2>> faces;
            const unsigned int e=1-d;
            for (unsigned int i=0; i<tiles.size(); ++i)
              for (unsigned int f=0; f<2; ++f)
                faces[tiles[i][d+2*f]][f].emplace_back(tiles[i][e],tiles[i][e+2],i);
            for (auto &[coordinate,sides] : faces)
              {
                (void)coordinate;
                for (auto &side : sides) std::sort(side.begin(),side.end());
                unsigned int i=0,j=0;
                while (i<sides[0].size() && j<sides[1].size())
                  {
                    const auto [a,b,k]=sides[0][i];
                    const auto [c,d_end,l]=sides[1][j];
                    if (std::max(a,c)<std::min(b,d_end))
                      {
                        neighbors[k].push_back(l);
                        neighbors[l].push_back(k);
                      }
                    if (b<=d_end) ++i;
                    if (d_end<=b) ++j;
                  }
              }
          }
        std::queue<unsigned int> queue;
        std::vector<bool> visited(tiles.size(),false);
        // Use one source. Seeding every piece of the incoming segment would
        // merge disconnected components when the accepted trace has a gap.
        // The physical tile coordinates break distance ties across MPI ranks.
        unsigned int source = numbers::invalid_unsigned_int;
        double nearest = std::numeric_limits<double>::max();
        for (unsigned int i=0; i<tiles.size(); ++i)
          if (tiles[i][5])
            {
              const auto &tile=tiles[i];
              const Point<2> center(origin[0]+.5*(tile[0]+tile[2])*unit[0],
                                    origin[1]+.5*(tile[1]+tile[3])*unit[1]);
              const double distance=center.distance_square(anchor);
              if (distance<nearest || (distance==nearest && tile<tiles[source]))
                {
                  nearest=distance;
                  source=i;
                }
            }
        if (source!=numbers::invalid_unsigned_int)
          {
            visited[source]=true;
            queue.push(source);
          }
        while (!queue.empty())
          {
            const auto i=queue.front();
            queue.pop();
            const auto &t=tiles[i];
            if (t[4])
              missed.emplace_back(std::make_pair(Point<2>(origin[0]+t[0]*unit[0],origin[1]+t[1]*unit[1]),
                                                 Point<2>(origin[0]+t[2]*unit[0],origin[1]+t[3]*unit[1])));
            for (const auto j : neighbors[i])
              if (!visited[j])
                {
                  visited[j]=true;
                  queue.push(j);
                }
          }
      }
    return missed;
  }

  template <int dim>
  TimeStepping::Reaction GrowingFault<dim>::assess_endpoint() const
  {
    using TimeStepping::Reaction;
    requested.reset();
    const auto &state=material(*this);
    if (this->get_timestep_number()>0 && !state.staged_sections())
      return Reaction::advance; // A nonlinear failure has no candidate to inspect.
    auto next=fault_region(true);
    if (this->get_timestep_number()>0)
      next.missed_cells=unresolved_damage();
    const bool miss=!next.missed_cells.empty();
    if (!miss && coverage==Coverage::reactive) return Reaction::advance;
    if (!miss)
      {
        const double restore_radius=next.tip_radius;
        if (coverage==Coverage::disk)
          next.tip_radius=3.*this->get_parameters().surface_energy_model.length_scale;
        if (!mark_region(next,false))
          return Reaction::advance;
        next.tip_radius=restore_radius;
      }
    this->get_pcout() << (miss ? "Growing fault missed extension: refine and repeat" :
                          "Growing fault preventive refinement: refine and advance")
                      << " at t=" << this->get_time() << " s, tip " << next.tip
                      << ", newly crossed coarse samples " << next.missed_cells.size() << std::endl;
    requested=std::move(next);
    return miss ? Reaction::refine_and_repeat_step : Reaction::refine_and_advance;
  }

  ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(GrowingFault, "growing fault",
                                            "Resolve the formed strip band and a forward region on fixed Cartesian "
                                            "boxes. Select the growing fault refinement timestep model to follow "
                                            "the advancing front and refine before repeating a missed extension.")
}
