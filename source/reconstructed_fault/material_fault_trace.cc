/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/reconstructed_fault/material_fault_trace.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

namespace aspect
{
  template <int dim>
  void MaterialFaultTrace<dim>::initialize(const Point<dim> &origin, const Point<dim> &extent,
                                           const std::vector<Point<dim>> &vertices,
                                           const double rate, const double support, const bool interior)
  {
    AssertThrow(dim == 2 && vertices.size() >= 2, ExcMessage("A material fault trace requires at least two ordered vertices in 2-D."));
    const auto direction = vertices.back()-vertices.front();
    const unsigned int axis = std::abs(direction[0]) > std::abs(direction[1]) ? 0 : 1;
    const double length = direction.norm();
    // A face entered as a decimal may round differently from origin + extent.
    const auto face_tolerance = [&](const unsigned int d)
    {
      return 4.*std::numeric_limits<double>::epsilon() *
             std::max({std::abs(origin[d]), std::abs(origin[d]+extent[d]), extent[d]});
    };
    AssertThrow(length > 0. && support > 0., ExcMessage("Degenerate fault seed or normal support."));
    if (!interior)
      for (unsigned int contact=0; contact<2; ++contact)
        {
          const auto &vertex = contact == 1 ? vertices.back() : vertices.front();
          contact_face_axes[contact] = dim;
          double best = 0.;
          for (unsigned int d=0; d<dim; ++d)
            {
              const double outward = contact == 1 ? direction[d] : -direction[d];
              const double face = origin[d] + (outward > 0. ? extent[d] : 0.);
              if (std::abs(vertex[d]-face) <= face_tolerance(d) && std::abs(outward) > best)
                {
                  contact_face_axes[contact] = d;
                  best = std::abs(outward);
                }
            }
          AssertThrow(contact_face_axes[contact] < dim,ExcMessage("The boundary-spanning seed requires contacts on two box faces."));
        }
    seed_tangent = direction/length;
    seed_normal[0] = -seed_tangent[1];
    seed_normal[1] = seed_tangent[0];
    initial_half_width = support;
    material_labels.clear();
    for (const auto &vertex : vertices)
      {
        for (unsigned int d=0; d<dim; ++d)
          AssertThrow(interior ? (vertex[d]>origin[d] && vertex[d]<origin[d]+extent[d]) :
                      (vertex[d] >= origin[d]-face_tolerance(d) && vertex[d] <= origin[d]+extent[d]+face_tolerance(d)),
                      ExcMessage("A seed vertex is outside its supported box region."));
        AssertThrow(std::abs((vertex-vertices.front())*seed_normal) <= face_tolerance(axis) &&
                    (material_labels.empty() || (vertex-material_labels.back().position)*seed_tangent > 0.),
                    ExcMessage("Initial material vertices must be straight and strictly ordered."));
        material_labels.push_back({vertex,0.,rate,support==std::numeric_limits<double>::max() ? 0. : support});
      }
    rebuild_segments();
  }

  template <int dim>
  void MaterialFaultTrace<dim>::replace_material_labels(std::vector<Label> labels)
  {
    AssertThrow(labels.size()==material_labels.size(),ExcMessage("Material fault topology changed."));
    // The existing checkpoint format uses zero at every vertex for a full
    // ray. A finite radius is material history and may have stretched since
    // initialization. Validate its convention before installing any record.
    const bool full_ray = initial_half_width == std::numeric_limits<double>::max();
    for (const auto &label : labels)
      AssertThrow(full_ray ? label.half_width == 0. :
                  (std::isfinite(label.half_width) && label.half_width > 0.),
                  ExcMessage("Checkpoint normal support convention differs from the configured trace "
                             "or contains an invalid material radius."));
    material_labels=std::move(labels);
    rebuild_segments();
  }

  template <int dim>
  void MaterialFaultTrace<dim>::advect_with_accepted_velocity(const SimulatorAccess<dim> &sim, const double dt)
  {
    const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    std::vector<Point<dim>> vertices;
    for (const auto &label:material_labels) vertices.push_back(label.position);
    GridTools::Cache<dim> cache(sim.get_triangulation(),sim.get_mapping());
    Utilities::MPI::RemotePointEvaluation<dim> lookup;
    lookup.reinit(cache,vertices);
    AssertThrow(lookup.all_points_found(),ExcMessage("A material fault vertex left the mesh."));
    const unsigned int first_velocity_component=sim.introspection().component_indices.velocities[0];
    const auto velocity=VectorTools::point_values<dim>(lookup,sim.get_dof_handler(),sim.get_old_solution(),VectorTools::EvaluationFlags::avg,first_velocity_component);
    const auto velocity_gradient=VectorTools::point_gradients<dim>(lookup,sim.get_dof_handler(),sim.get_old_solution(),VectorTools::EvaluationFlags::avg,first_velocity_component);
    for (unsigned int i=0; i<vertices.size(); ++i)
      {
        const auto edge=vertices[std::min<unsigned int>(i+1,vertices.size()-1)]-vertices[i==0 ? 0 : i-1];
        Tensor<1,dim> vertex_normal;
        vertex_normal[0]=-edge[1]/edge.norm();
        vertex_normal[1]=edge[0]/edge.norm();
        Tensor<2,dim> F=unit_symmetric_tensor<dim>();
        F+=dt*velocity_gradient[i];
        AssertThrow(determinant(F)>0.,ExcMessage("Material support reversed orientation."));
        if (initial_half_width!=std::numeric_limits<double>::max())
          // Pulling the old normal through F^{-T} gives the inverse normal stretch.
          material_labels[i].half_width/=(transpose(invert(F))*vertex_normal).norm();
        material_labels[i].position+=dt*velocity[i];
        for (unsigned int d=0; d<dim; ++d)
          AssertThrow(material_labels[i].position[d]>box.get_origin()[d] &&
                      material_labels[i].position[d]<box.get_origin()[d]+box.get_extents()[d],
                      ExcMessage("A finite material fault reached the box boundary."));
      }
    rebuild_segments();
  }

  template <int dim>
  void MaterialFaultTrace<dim>::rebuild_segments()
  {
    // A trace may turn through any angle, but cannot fold back onto
    // itself or acquire a crossing without a change of fault topology.
    const auto cross=[](const Tensor<1,dim> &a,const Tensor<1,dim> &b)
    {
      return a[0]*b[1]-a[1]*b[0];
    };
    segments.clear();
    arclength = {0.};
    for (unsigned int j=0; j+1<material_labels.size(); ++j)
      {
        const auto edge = material_labels[j+1].position-material_labels[j].position;
        const double length = edge.norm();
        AssertThrow(std::isfinite(length) && length>0.,ExcMessage("Degenerate material fault edge."));
        const auto &start=material_labels[j].position;
        const auto &end=material_labels[j+1].position;
        if (j>0)
          {
            const auto previous=start-material_labels[j-1].position;
            AssertThrow(cross(previous,edge)!=0. || previous*edge>0.,
                        ExcMessage("Adjacent material fault edges overlap."));
          }
        for (unsigned int k=0; k+1<j; ++k)
          {
            const auto &a=material_labels[k].position;
            const auto &b=material_labels[k+1].position;
            const auto separate=[](const double u,const double v,const double x,const double y)
            {
              return std::max(u,v)<std::min(x,y) || std::max(x,y)<std::min(u,v);
            };
            if (separate(a[0],b[0],start[0],end[0]) || separate(a[1],b[1],start[1],end[1]))
              continue;
            const auto same_side=[](const double x,const double y)
            {
              return (x>0. && y>0.) || (x<0. && y<0.);
            };
            AssertThrow(same_side(cross(edge,a-start),cross(edge,b-start)) ||
                        same_side(cross(b-a,start-a),cross(b-a,end-a)),
                        ExcMessage("Nonadjacent material fault edges intersect."));
          }
        const auto segment_tangent = edge/length;
        Tensor<1,dim> segment_normal;
        segment_normal[0] = -segment_tangent[1];
        segment_normal[1] = segment_tangent[0];
        segments.push_back({arclength.back(),length,segment_tangent,segment_normal,outer_product(segment_tangent,segment_normal)});
        arclength.push_back(arclength.back()+length);
      }
  }

  template <int dim>
  double MaterialFaultTrace<dim>::normal_support_radius(const unsigned int segment_index, const double xi) const
  {
    if (initial_half_width==std::numeric_limits<double>::max())
      return std::numeric_limits<double>::infinity();
    const double left=material_labels[segment_index].half_width;
    return left+xi*(material_labels[segment_index+1].half_width-left);
  }

  template <int dim>
  const typename MaterialFaultTrace<dim>::Segment &MaterialFaultTrace<dim>::segment(const unsigned int j) const
  {
    return segments[j];
  }

  namespace
  {
    // Maximize the actual one-sided DGQ2 face polynomial in the seed window.
    // At an interior face vertex use the interval on its higher-coordinate side.
    template <int dim>
    Point<dim> find_ridge_contact_on_face(const SimulatorAccess<dim> &sim, unsigned int field,
                                          const LinearAlgebra::BlockVector &fields, const Point<dim> &seed,
                                          unsigned int axis, double radius)
    {
      const unsigned int along=1-axis;
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
      const auto upper=box.get_origin()+box.get_extents();
      const bool high=std::abs(seed[axis]-upper[axis])<std::abs(seed[axis]-box.get_origin()[axis]);
      std::vector<Point<dim>> nodes(3);
      for (unsigned int i=0; i<3; ++i)
        {
          nodes[i][axis]=high ? 1. : 0.;
          nodes[i][along]=.5*i;
        }
      FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),Quadrature<dim>(nodes),update_values);
      std::vector<double> values(3);
      std::vector<std::pair<double,double>> candidates;
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned() && cell->face(2*axis+high)->at_boundary())
          {
            const double a=cell->vertex(0)[along], b=cell->vertex(GeometryInfo<dim>::vertices_per_cell-1)[along];
            const double left=std::max(a,seed[along]-radius), right=std::min(b,seed[along]+radius);
            if (left>=right) continue;
            fe.reinit(cell);
            fe[sim.introspection().extractors.compositional_fields[field]].get_function_values(fields,values);
            const double c=2.*(values[0]+values[2]-2.*values[1]), slope=values[2]-values[0]-c;
            const auto add=[&](const double x)
            {
              const double u=(x-a)/(b-a);
              candidates.emplace_back(values[0]+u*(slope+c*u),x);
            };
            add(left);
            if (right<b || b==upper[along]) add(right);
            if (c<0.)
              {
                const double x=a-(b-a)*slope/(2.*c);
                if (x>left && x<right) add(x);
              }
            // A flat maximal interval has no uniquely observed crossing.
            if (std::abs(c)+std::abs(slope)<1.e-13) add(.5*(left+right));
          }
      const auto shards=Utilities::MPI::all_gather(sim.get_mpi_communicator(),candidates);
      double peak=-std::numeric_limits<double>::infinity();
      for (const auto &shard:shards)
        for (const auto &entry:shard) peak=std::max(peak,entry.first);
      double left=upper[along], right=box.get_origin()[along];
      for (const auto &shard:shards)
        for (const auto &entry:shard)
          if (entry.first>=peak-1.e-13)
            {
              left=std::min(left,entry.second);
              right=std::max(right,entry.second);
            }
      AssertThrow(peak>.1 && right-left<=1.e-8*box.get_extents()[along] &&
                  std::abs(left-seed[along])<radius,
                  ExcMessage("Unresolved or ambiguous stationary phase maximum on the seed contact face."));
      Point<dim> result=seed;
      result[along]=left;
      result[axis]=high ? upper[axis] : box.get_origin()[axis];
      return result;
    }
  }

  template <int dim>
  void MaterialFaultTrace<dim>::fit_boundary_spanning_seed_to_phase_ridge(const SimulatorAccess<dim> &sim, const unsigned int phase_field,
                                                                          const LinearAlgebra::BlockVector &fields)
  {
    const unsigned int n = material_labels.size();
    const QGauss<dim> rule(sim.get_fe().degree+1);
    FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),rule,update_values|update_JxW_values|update_quadrature_points);
    std::vector<double> phase(rule.size()), matrix(n*n), rhs(n);
    const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    const auto upper=box.get_origin()+box.get_extents();
    double total_weight = 0., cell_margin = 0., ridge_radius=0.;
    // Each bulk observation contributes JxW max(phase-0.1,0) to the Q1
    // least-squares fit of the signed normal displacement from the seed.
    // Estimate the observed transverse extent before taking any centroids.
    // Missing exterior phase on a cropped normal would bias its first moment.
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[sim.introspection().extractors.compositional_fields[phase_field]].get_function_values(fields,phase);
          for (unsigned int q=0; q<rule.size(); ++q)
            if (phase[q]>.1)
              ridge_radius=std::max(ridge_radius,std::abs((fe.quadrature_point(q)-material_labels.front().position)*seed_normal));
        }
    ridge_radius=std::min(initial_half_width,Utilities::MPI::max(ridge_radius,sim.get_mpi_communicator()));
    double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_margin = std::max(cell_margin,cell->diameter());
          fe.reinit(cell);
          fe[sim.introspection().extractors.compositional_fields[phase_field]].get_function_values(fields,phase);
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              minimum = std::min(minimum,phase[q]);
              maximum = std::max(maximum,phase[q]);
              const double weight = fe.JxW(q)*std::max(0.,phase[q]-.1);
              if (weight == 0.) continue;
              const auto &x = fe.quadrature_point(q);
              const auto foot = project(x);
              const unsigned int j = foot.first_vertex;
              const Point<dim> center = material_labels[j].position + foot.xi*segments[j].length*segments[j].tangent;
              bool cropped=false;
              for (unsigned int d=0; d<dim; ++d)
                {
                  const double reach=ridge_radius*std::abs(seed_normal[d]);
                  cropped |= center[d]-reach < box.get_origin()[d] || center[d]+reach > upper[d];
                }
              if (cropped) continue;
              if (center.distance(x) > initial_half_width+cell->diameter()) continue;
              const double N[2] = {1.-foot.xi,foot.xi};
              total_weight += weight;
              for (unsigned int i=0; i<2; ++i)
                {
                  rhs[j+i] += weight*N[i]*((x-center)*segments[j].normal);
                  for (unsigned int k=0; k<2; ++k)
                    matrix[(j+i)*n+j+k] += weight*N[i]*N[k];
                }
            }
        }
    const auto comm = sim.get_mpi_communicator();
    if (Utilities::MPI::max(maximum,comm)-Utilities::MPI::min(minimum,comm) < 1.e-12) return;
    Utilities::MPI::sum(matrix,comm,matrix);
    Utilities::MPI::sum(rhs,comm,rhs);
    total_weight = Utilities::MPI::sum(total_weight,comm);
    AssertThrow(total_weight>0. || n==2,ExcMessage("No two-sided interior phase support for ridge fitting."));
    const double radius=initial_half_width+Utilities::MPI::max(cell_margin,comm);
    const auto first_contact=find_ridge_contact_on_face(sim,phase_field,fields,material_labels.front().position,contact_face_axes[0],radius);
    const auto last_contact=find_ridge_contact_on_face(sim,phase_field,fields,material_labels.back().position,contact_face_axes[1],radius);
    FullMatrix<double> system(n,n);
    Vector<double> b(n), offsets(n);
    for (unsigned int j=0; j<n; ++j)
      {

        b[j] = rhs[j]/(total_weight>0. ? total_weight : 1.);
        for (unsigned int k=0; k<n; ++k) system(j,k) = matrix[j*n+k]/(total_weight>0. ? total_weight : 1.);
      }
    // Add the unit-weight squared second differences of the vertex
    // offsets to the normalized data misfit before prescribing the contacts.
    const double second[3] = {1.,-2.,1.};
    for (unsigned int j=1; j+1<n; ++j)
      for (unsigned int a=0; a<3; ++a)
        for (unsigned int k=0; k<3; ++k)
          system(j-1+a,j-1+k) += second[a]*second[k];
    // Constrain the normal component of each measured face displacement
    // during the fit. The second-difference penalty spans the cropped rows.
    for (const unsigned int j :
    {
      0u,n-1
    })
    {
      for (unsigned int k=0; k<n; ++k) system(j,k)=0.;
      system(j,j)=1.;
      b[j]=((j==0 ? first_contact : last_contact)-material_labels[j].position)*seed_normal;
    }
    system.gauss_jordan();
    system.vmult(offsets,b);
    for (unsigned int j=1; j+1<n; ++j)
      {
        AssertThrow(std::abs(offsets[j]) <= radius,ExcMessage("Fitted label left its phase reconstruction region."));
        material_labels[j].position += offsets[j]*seed_normal;
      }
    material_labels.front().position=first_contact;
    material_labels.back().position=last_contact;
    for (unsigned int j=1; j<n; ++j)
      AssertThrow((material_labels[j].position-material_labels[j-1].position)*seed_tangent > 0.,
                  ExcMessage("Ridge fit reversed the order of the material labels."));
    for (const auto &label : material_labels)
      AssertThrow(sim.get_geometry_model().point_is_in_domain(label.position),ExcMessage("Fitted fault left the physical box."));
    rebuild_segments();
  }

  template <int dim>
  typename MaterialFaultTrace<dim>::Projection MaterialFaultTrace<dim>::project(const Point<dim> &point) const
  {
    Projection closest {0,0.,false};
    double best = std::numeric_limits<double>::infinity();
    for (unsigned int j=0; j<segments.size(); ++j)
      {
        const auto &seg = segments[j];
        const auto offset = point-material_labels[j].position;
        const double s = offset*seg.tangent;
        const double bounded = std::clamp(s,0.,seg.length);
        // Two segments can share the closest vertex outside their normal fans.
        // Use its stored coordinates on both sides so a roundoff tie retains
        // the earlier material segment on every MPI partition.
        const auto closest_point = (s <= 0. ? material_labels[j].position :
                                    s >= seg.length ? material_labels[j+1].position :
                                    material_labels[j].position + s*seg.tangent);
        const double distance = point.distance_square(closest_point);
        const double tol = 32.*std::numeric_limits<double>::epsilon()*arclength.back();
        const bool active = s >= -tol && s <= seg.length+tol && std::abs(offset*seg.normal) <= normal_support_radius(j,bounded/seg.length);
        if ((active && !closest.active) || (active == closest.active && distance < best))
          {
            closest = {j,bounded/seg.length,active};
            best = distance;
          }
      }
    return closest;
  }

  template <int dim>
  Quadrature<dim> MaterialFaultTrace<dim>::cut_quadrature(const Point<dim> &lower,const Point<dim> &upper) const
  {
    // On a Cartesian cell each segment's finite normal strip is the
    // intersection of four half planes. Where strips overlap, nearest
    // distance changes at the two bisectors of their supporting lines.
    struct Line
    {
      double a,b,c;
      double value(const Point<dim> &p) const
      {
        return a*p[0]+b*p[1]+c;
      }
    };
    using Polygon=std::vector<Point<dim>>;
    const auto extent=upper-lower;
    const double epsilon=64.*std::numeric_limits<double>::epsilon()*extent.norm();
    const auto equation=[&](const Tensor<1,dim> &direction,const Point<dim> &origin)
    {
      return Line {direction[0] *extent[0],direction[1] *extent[1],direction *(lower-origin)};
    };
    std::vector<Line> cuts,normals;
    for (unsigned int j=0; j<segments.size(); ++j)
      {
        const auto station=equation(segments[j].tangent,material_labels[j].position);
        const auto normal_distance=equation(segments[j].normal,material_labels[j].position);
        const auto range=[](const Line &line)
        {
          return std::make_pair(line.c+std::min(0.,line.a)+std::min(0.,line.b),
                                line.c+std::max(0.,line.a)+std::max(0.,line.b));
        };
        const auto s=range(station), d=range(normal_distance);
        const double radius=std::max(normal_support_radius(j,0.),normal_support_radius(j,1.));
        if (s.second<=0. || s.first>=segments[j].length || d.second<=-radius || d.first>=radius)
          continue;
        for (const double face :
        {
          0.,segments[j].length
          })
        if (s.first<face && s.second>face) cuts.push_back({station.a,station.b,station.c-face});
        const double slope=(material_labels[j+1].half_width-material_labels[j].half_width)/segments[j].length;
        for (const double side :
        {
          -1.,1.
        })
        {
          const Line edge
          {
            normal_distance.a-side *slope *station.a, normal_distance.b-side *slope *station.b,
                              normal_distance.c-side*(material_labels[j].half_width+slope*station.c)
          };
          const auto limits=range(edge);
          if (std::isfinite(radius) && limits.first<0. && limits.second>0.) cuts.push_back(edge);
        }
        for (const auto &other:normals)
          for (const double sign :
          {
            -1.,1.
          })
        cuts.push_back({normal_distance.a+sign*other.a,normal_distance.b+sign*other.b,normal_distance.c+sign*other.c});
        normals.push_back(normal_distance);
      }
    if (cuts.empty()) return {};
    Polygon square(4);
    square[1][0]=square[2][0]=square[2][1]=square[3][1]=1.;
    std::vector<Polygon> pieces= {square};
    for (const auto &line:cuts)
      {
        const unsigned int old_size=pieces.size();
        for (unsigned int i=0; i<old_size; ++i)
          {
            const auto &piece=pieces[i];
            double low=0.,high=0.;
            for (const auto &p:piece)
              {
                low=std::min(low,line.value(p));
                high=std::max(high,line.value(p));
              }
            if (low>=-epsilon || high<=epsilon) continue;
            Polygon positive,negative;
            auto prev=piece.back();
            double vp=line.value(prev);
            for (const auto &p:piece)
              {
                const double v=line.value(p);
                if ((vp<0. && v>0.) || (vp>0. && v<0.))
                  {
                    auto cross=prev+(vp/(vp-v))*(p-prev);
                    positive.push_back(cross);
                    negative.push_back(cross);
                  }
                if (v>=0.) positive.push_back(p);
                if (v<=0.) negative.push_back(p);
                prev=p;
                vp=v;
              }
            pieces[i]=std::move(positive);
            pieces.push_back(std::move(negative));
          }
      }
    if (pieces.size()==1) return {};
    static const QWitherdenVincentSimplex<2> triangle(5);
    std::vector<Point<dim>> points;
    std::vector<double> weights;
    for (const auto &piece:pieces)
      for (unsigned int k=1; k+1<piece.size(); ++k)
        {
          const auto u=piece[k]-piece[0], v=piece[k+1]-piece[0];
          const double det=std::abs(u[0]*v[1]-u[1]*v[0]);
          if (det==0.) continue;
          for (unsigned int q=0; q<triangle.size(); ++q)
            {
              points.push_back(piece[0]+triangle.point(q)[0]*u+triangle.point(q)[1]*v);
              weights.push_back(triangle.weight(q)*det);
            }
        }
    return Quadrature<dim>(points,weights);
  }

  template <int dim>
  double MaterialFaultTrace<dim>::Projection::interpolate(const double a, const double b) const
  {
    return (1.-xi)*a + xi*b;
  }

  template <int dim>
  void MaterialFaultTrace<dim>::accept_section_rates(const std::vector<double> &rates, const double dt)
  {
    for (unsigned int j=0; j<material_labels.size(); ++j)
      {
        auto &label = material_labels[j];
        label.slip += dt*rates[j];
        label.rate = rates[j];
      }
  }

  template <int dim>
  const std::vector<typename MaterialFaultTrace<dim>::Label> &MaterialFaultTrace<dim>::labels() const
  {
    return material_labels;
  }

#define INSTANTIATE(dim) template class MaterialFaultTrace<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
