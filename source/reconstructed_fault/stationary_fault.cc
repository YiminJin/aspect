/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/reconstructed_fault/stationary_fault.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

namespace aspect
{
  template <int dim>
  void StationaryFault<dim>::initialize(const Point<dim> &origin, const Point<dim> &extent,
                                        const std::vector<Point<dim>> &vertices,
                                        const double age, const double rate, const double support, const bool interior)
  {
    AssertThrow(dim == 2 && vertices.size() >= 2, ExcMessage("A stationary fault requires at least two ordered vertices in 2-D."));
    const auto direction = vertices.back()-vertices.front();
    const unsigned int axis = std::abs(direction[0]) > std::abs(direction[1]) ? 0 : 1;
    const double length = direction.norm();
    // A face entered as a decimal may round differently from origin + extent.
    const auto face_tolerance = [&](const unsigned int d)
    {
      return 4.*std::numeric_limits<double>::epsilon() *
             std::max({std::abs(origin[d]), std::abs(origin[d]+extent[d]), extent[d]});
    };
    AssertThrow(length > 0. && support > 0., ExcMessage("Degenerate stationary seed or support."));
    if (!interior)
      for (unsigned int end=0; end<2; ++end)
        {
          const auto &vertex = end ? vertices.back() : vertices.front();
          contact_axes[end] = dim;
          double best = 0.;
          for (unsigned int d=0; d<dim; ++d)
            {
              const double outward = end ? direction[d] : -direction[d];
              const double face = origin[d] + (outward > 0. ? extent[d] : 0.);
              if (std::abs(vertex[d]-face) <= face_tolerance(d) && std::abs(outward) > best)
                {
                  contact_axes[end] = d;
                  best = std::abs(outward);
                }
            }
          AssertThrow(contact_axes[end] < dim,ExcMessage("The stationary fault must span the Box between boundary contacts."));
        }
    tangent = direction/length;
    normal[0] = -tangent[1];
    normal[1] = tangent[0];
    half_width = support;
    if (interior && support==std::numeric_limits<double>::max())
      {
        half_width=0.;
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            Point<dim> corner=origin;
            for (unsigned int d=0; d<dim; ++d)
              if (v & (1u<<d)) corner[d]+=extent[d];
            half_width=std::max(half_width,std::abs((corner-vertices.front())*normal));
          }
        half_width+=64.*std::numeric_limits<double>::epsilon()*std::max({1.,length,vertices.front().norm(),vertices.back().norm()});
      }
    accepted.clear();
    for (const auto &vertex : vertices)
      {
        for (unsigned int d=0; d<dim; ++d)
          AssertThrow(interior ? (vertex[d]>origin[d] && vertex[d]<origin[d]+extent[d]) :
                      (vertex[d] >= origin[d]-face_tolerance(d) && vertex[d] <= origin[d]+extent[d]+face_tolerance(d)),
                      ExcMessage("A seed vertex is outside its supported box region."));
        AssertThrow(std::abs((vertex-vertices.front())*normal) <= face_tolerance(axis) &&
                    (accepted.empty() || (vertex-accepted.back().position)*tangent > 0.),
                    ExcMessage("Stationary seed labels must be straight and strictly ordered."));
        accepted.push_back({vertex,age,0.,rate,half_width,0.,0.});
      }
    rebuild_segments();
  }

  template <int dim>
  void StationaryFault<dim>::set_labels(std::vector<Label> labels)
  {
    AssertThrow(labels.size()==accepted.size(),ExcMessage("Material fault topology changed."));
    accepted=std::move(labels);
    rebuild_segments();
  }

  template <int dim>
  void StationaryFault<dim>::advect(const SimulatorAccess<dim> &sim, const double dt)
  {
    const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    std::vector<Point<dim>> vertices;
    for (const auto &label:accepted) vertices.push_back(label.position);
    GridTools::Cache<dim> cache(sim.get_triangulation(),sim.get_mapping());
    Utilities::MPI::RemotePointEvaluation<dim> lookup;
    lookup.reinit(cache,vertices);
    AssertThrow(lookup.all_points_found(),ExcMessage("A material fault vertex left the mesh."));
    const unsigned int first=sim.introspection().component_indices.velocities[0];
    const auto u=VectorTools::point_values<dim>(lookup,sim.get_dof_handler(),sim.get_old_solution(),VectorTools::EvaluationFlags::avg,first);
    const auto grad=VectorTools::point_gradients<dim>(lookup,sim.get_dof_handler(),sim.get_old_solution(),VectorTools::EvaluationFlags::avg,first);
    for (unsigned int i=0; i<vertices.size(); ++i)
      {
        const auto edge=vertices[std::min<unsigned int>(i+1,vertices.size()-1)]-vertices[i==0 ? 0 : i-1];
        Tensor<1,dim> n;
        n[0]=-edge[1]/edge.norm();
        n[1]=edge[0]/edge.norm();
        Tensor<2,dim> F=unit_symmetric_tensor<dim>();
        F+=dt*grad[i];
        AssertThrow(determinant(F)>0.,ExcMessage("Material support reversed orientation."));
        accepted[i].half_width/=(transpose(invert(F))*n).norm();
        accepted[i].position+=dt*u[i];
        for (unsigned int d=0; d<dim; ++d)
          AssertThrow(accepted[i].position[d]>box.get_origin()[d] &&
                      accepted[i].position[d]<box.get_origin()[d]+box.get_extents()[d],
                      ExcMessage("A finite material fault reached the box boundary."));
      }
    rebuild_segments();
  }

  template <int dim>
  void StationaryFault<dim>::rebuild_segments()
  {
    segments.clear();
    arclength = {0.};
    for (unsigned int j=0; j+1<accepted.size(); ++j)
      {
        const auto edge = accepted[j+1].position-accepted[j].position;
        AssertThrow(edge*tangent > 0.,ExcMessage("Ridge fit reversed the order of the material labels."));
        const auto t = edge/edge.norm();
        Tensor<1,dim> n;
        n[0] = -t[1];
        n[1] = t[0];
        segments.push_back({arclength.back(),edge.norm(),t,n,outer_product(t,n)});
        arclength.push_back(arclength.back()+edge.norm());
      }
  }

  template <int dim>
  const typename StationaryFault<dim>::Segment &StationaryFault<dim>::segment(const unsigned int j) const
  {
    return segments[j];
  }

  namespace
  {
    // Maximize the actual one-sided DGQ2 face polynomial in the seed window.
    // At an interior face vertex use the interval on its higher-coordinate side.
    template <int dim>
    Point<dim> ridge_contact(const SimulatorAccess<dim> &sim, unsigned int field,
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
  void StationaryFault<dim>::fit(const SimulatorAccess<dim> &sim, const unsigned int field,
                                 const LinearAlgebra::BlockVector &fields)
  {
    const unsigned int n = accepted.size();
    const QGauss<dim> rule(sim.get_fe().degree+1);
    FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),rule,update_values|update_JxW_values|update_quadrature_points);
    std::vector<double> phase(rule.size()), matrix(n*n), rhs(n);
    const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    const auto upper=box.get_origin()+box.get_extents();
    double total = 0., cell_margin = 0., ridge_radius=0.;
    // Estimate the observed transverse extent before taking any centroids.
    // Missing exterior phase on a cropped normal would bias its first moment.
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe.reinit(cell);
          fe[sim.introspection().extractors.compositional_fields[field]].get_function_values(fields,phase);
          for (unsigned int q=0; q<rule.size(); ++q)
            if (phase[q]>.1)
              ridge_radius=std::max(ridge_radius,std::abs((fe.quadrature_point(q)-accepted.front().position)*normal));
        }
    ridge_radius=std::min(half_width,Utilities::MPI::max(ridge_radius,sim.get_mpi_communicator()));
    double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_margin = std::max(cell_margin,cell->diameter());
          fe.reinit(cell);
          fe[sim.introspection().extractors.compositional_fields[field]].get_function_values(fields,phase);
          for (unsigned int q=0; q<rule.size(); ++q)
            {
              minimum = std::min(minimum,phase[q]);
              maximum = std::max(maximum,phase[q]);
              const double weight = fe.JxW(q)*std::max(0.,phase[q]-.1);
              if (weight == 0.) continue;
              const auto &x = fe.quadrature_point(q);
              const auto foot = project(x);
              const unsigned int j = foot.left;
              const Point<dim> center = accepted[j].position + foot.xi*segments[j].length*segments[j].tangent;
              bool cropped=false;
              for (unsigned int d=0; d<dim; ++d)
                {
                  const double reach=ridge_radius*std::abs(normal[d]);
                  cropped |= center[d]-reach < box.get_origin()[d] || center[d]+reach > upper[d];
                }
              if (cropped) continue;
              if (center.distance(x) > half_width+cell->diameter()) continue;
              const double N[2] = {1.-foot.xi,foot.xi};
              total += weight;
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
    total = Utilities::MPI::sum(total,comm);
    AssertThrow(total>0. || n==2,ExcMessage("No two-sided interior phase support for ridge fitting."));
    const double radius=half_width+Utilities::MPI::max(cell_margin,comm);
    const auto bottom=ridge_contact(sim,field,fields,accepted.front().position,contact_axes[0],radius);
    const auto top=ridge_contact(sim,field,fields,accepted.back().position,contact_axes[1],radius);
    FullMatrix<double> system(n,n);
    Vector<double> b(n), offsets(n);
    for (unsigned int j=0; j<n; ++j)
      {

        b[j] = rhs[j]/(total>0. ? total : 1.);
        for (unsigned int k=0; k<n; ++k) system(j,k) = matrix[j*n+k]/(total>0. ? total : 1.);
      }
    // Donor ridge coefficient one on second differences of normal offsets.
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
      b[j]=((j==0 ? bottom : top)-accepted[j].position)*normal;
    }
    system.gauss_jordan();
    system.vmult(offsets,b);
    for (unsigned int j=1; j+1<n; ++j)
      {
        AssertThrow(std::abs(offsets[j]) <= radius,ExcMessage("Fitted label left its phase reconstruction region."));
        accepted[j].position += offsets[j]*normal;
      }
    accepted.front().position=bottom;
    accepted.back().position=top;
    for (const auto &label : accepted)
      AssertThrow(sim.get_geometry_model().point_is_in_domain(label.position),ExcMessage("Fitted fault left the physical box."));
    rebuild_segments();
  }

  template <int dim>
  typename StationaryFault<dim>::Projection StationaryFault<dim>::project(const Point<dim> &point) const
  {
    Projection closest {0,0.,false};
    double best = std::numeric_limits<double>::infinity();
    for (unsigned int j=0; j<segments.size(); ++j)
      {
        const auto &seg = segments[j];
        const auto offset = point-accepted[j].position;
        const double s = offset*seg.tangent;
        const double bounded = std::clamp(s,0.,seg.length);
        // Two segments can share the closest vertex outside their normal fans.
        // Use its stored coordinates on both sides so a roundoff tie retains
        // the earlier material segment on every MPI partition.
        const auto closest_point = (s <= 0. ? accepted[j].position :
                                    s >= seg.length ? accepted[j+1].position :
                                    accepted[j].position + s*seg.tangent);
        const double distance = point.distance_square(closest_point);
        const double tol = 32.*std::numeric_limits<double>::epsilon()*arclength.back();
        const bool active = s >= -tol && s <= seg.length+tol && std::abs(offset*seg.normal) <= accepted[j].half_width+(bounded/seg.length)*(accepted[j+1].half_width-accepted[j].half_width);
        if ((active && !closest.active) || (active == closest.active && distance < best))
          {
            closest = {j,bounded/seg.length,active};
            best = distance;
          }
      }
    return closest;
  }

  template <int dim>
  Quadrature<dim> StationaryFault<dim>::cut_quadrature(const Point<dim> &lower,const Point<dim> &upper) const
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
        const auto station=equation(segments[j].tangent,accepted[j].position);
        const auto normal_distance=equation(segments[j].normal,accepted[j].position);
        const auto range=[](const Line &line)
        {
          return std::make_pair(line.c+std::min(0.,line.a)+std::min(0.,line.b),
                                line.c+std::max(0.,line.a)+std::max(0.,line.b));
        };
        const auto s=range(station), d=range(normal_distance);
        const double radius=std::max(accepted[j].half_width,accepted[j+1].half_width);
        if (s.second<=0. || s.first>=segments[j].length || d.second<=-radius || d.first>=radius)
          continue;
        for (const double face :
        {
          0.,segments[j].length
          })
        if (s.first<face && s.second>face) cuts.push_back({station.a,station.b,station.c-face});
        const double slope=(accepted[j+1].half_width-accepted[j].half_width)/segments[j].length;
        for (const double side :
        {
          -1.,1.
        })
        {
          const Line edge
          {
            normal_distance.a-side *slope *station.a, normal_distance.b-side *slope *station.b,
                              normal_distance.c-side*(accepted[j].half_width+slope*station.c)
          };
          const auto limits=range(edge);
          if (limits.first<0. && limits.second>0.) cuts.push_back(edge);
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
  double StationaryFault<dim>::Projection::interpolate(const double a, const double b) const
  {
    return (1.-xi)*a + xi*b;
  }

  template <int dim>
  void StationaryFault<dim>::accept(const std::vector<double> &rates, const double dt,
                                    const MaterialModel::Rheology::RateAndStateFriction &law)
  {
    for (unsigned int j=0; j<accepted.size(); ++j)
      {
        auto &label = accepted[j];
        label.age = law.advance_age(rates[j],label.age,dt);
        label.slip += dt*rates[j];
        label.rate = rates[j];
      }
  }

  template <int dim>
  const std::vector<typename StationaryFault<dim>::Label> &StationaryFault<dim>::labels() const
  {
    return accepted;
  }

#define INSTANTIATE(dim) template class StationaryFault<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
