/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/reconstructed_fault/stationary_fault.h>

namespace aspect
{
  template <int dim>
  void StationaryFault<dim>::initialize(const Point<dim> &origin, const Point<dim> &extent,
                                        const std::vector<Point<dim>> &vertices,
                                        const double age, const double rate, const double support)
  {
    AssertThrow(dim == 2 && vertices.size() >= 2, ExcMessage("A stationary fault requires at least two ordered vertices in 2-D."));
    const auto direction = vertices.back()-vertices.front();
    const unsigned int axis = std::abs(direction[0]) > std::abs(direction[1]) ? 0 : 1;
    const unsigned int transverse = 1-axis;
    const double length = direction.norm();
    // A face entered as a decimal may round differently from origin + extent.
    const auto face_tolerance = [&](const unsigned int d)
    {
      return 4.*std::numeric_limits<double>::epsilon() *
             std::max({std::abs(origin[d]), std::abs(origin[d]+extent[d]), extent[d]});
    };
    AssertThrow(length > 0. && support > 0., ExcMessage("Degenerate stationary seed or support."));
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
    slip_gradient = outer_product(tangent,normal);
    width = extent[transverse];
    half_width = support;
    accepted.clear();
    for (const auto &vertex : vertices)
      {
        for (unsigned int d=0; d<dim; ++d)
          AssertThrow(vertex[d] >= origin[d]-face_tolerance(d) && vertex[d] <= origin[d]+extent[d]+face_tolerance(d),
                      ExcMessage("A seed vertex is outside the box."));
        AssertThrow(std::abs((vertex-vertices.front())*normal) <= face_tolerance(axis) &&
                    (accepted.empty() || (vertex-accepted.back().position)*tangent > 0.),
                    ExcMessage("Stationary seed labels must be straight and strictly ordered."));
        accepted.push_back({vertex,age,0.,rate});
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

  template <int dim>
  void StationaryFault<dim>::fit(const SimulatorAccess<dim> &sim, const unsigned int field,
                                 const LinearAlgebra::BlockVector &fields)
  {
    const unsigned int n = accepted.size();
    const QGauss<dim> rule(sim.get_fe().degree+1);
    FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),rule,update_values|update_JxW_values|update_quadrature_points);
    std::vector<double> phase(rule.size()), matrix(n*n), rhs(n);
    double total = 0., minimum = 1., maximum = 0.;
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
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
    FullMatrix<double> system(n,n);
    Vector<double> b(n), offsets(n);
    for (unsigned int j=0; j<n; ++j)
      {
        AssertThrow(matrix[j*n+j] > 0.,ExcMessage("A seed label has no active FE ridge support."));
        b[j] = rhs[j]/total;
        for (unsigned int k=0; k<n; ++k) system(j,k) = matrix[j*n+k]/total;
      }
    // Donor ridge coefficient one on second differences of normal offsets.
    const double second[3] = {1.,-2.,1.};
    for (unsigned int j=1; j+1<n; ++j)
      for (unsigned int a=0; a<3; ++a)
        for (unsigned int k=0; k<3; ++k)
          system(j-1+a,j-1+k) += second[a]*second[k];
    system.gauss_jordan();
    system.vmult(offsets,b);
    const auto seed = accepted;
    for (unsigned int j=0; j<n; ++j)
      accepted[j].position += offsets[j]*normal;
    // Intersect each fitted end segment with its original physical contact face.
    for (unsigned int end=0; end<2; ++end)
      {
        const unsigned int j = end ? n-1 : 0, neighbor = end ? j-1 : 1, axis = contact_axes[end];
        const auto edge = accepted[neighbor].position-accepted[j].position;
        AssertThrow(edge[axis] != 0.,ExcMessage("The fitted trace grazes its contact face."));
        accepted[j].position += (seed[j].position[axis]-accepted[j].position[axis])/edge[axis]*edge;
      }
    rebuild_segments();
  }

  template <int dim>
  bool StationaryFault<dim>::resolves_segments(const Point<dim> &lower, const Point<dim> &upper) const
  {
    const double a = (lower-accepted.front().position)*tangent;
    const double b = (upper-accepted.front().position)*tangent;
    const double tolerance = 64.*std::numeric_limits<double>::epsilon()*arclength.back();
    const auto next = std::upper_bound(arclength.begin()+1,arclength.end()-1,std::min(a,b)+tolerance);
    return next == arclength.end()-1 || *next >= std::max(a,b)-tolerance;
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
        const double distance = (offset-bounded*seg.tangent).norm_square();
        const double tol = 32.*std::numeric_limits<double>::epsilon()*arclength.back();
        const bool active = s >= -tol && s <= seg.length+tol && std::abs(offset*seg.normal) <= half_width;
        if ((active && !closest.active) || (active == closest.active && distance < best))
          {
            closest = {j,bounded/seg.length,active};
            best = distance;
          }
      }
    return closest;
  }

  template <int dim>
  double StationaryFault<dim>::Projection::interpolate(const double a, const double b) const
  {
    return (1.-xi)*a + xi*b;
  }

  template <int dim>
  double StationaryFault<dim>::localization() const
  {
    return 1./width;
  }

  template <int dim>
  const Tensor<2,dim> &StationaryFault<dim>::slip_dyad() const
  {
    return slip_gradient;
  }

  template <int dim>
  const Tensor<1,dim> &StationaryFault<dim>::normal_vector() const
  {
    return normal;
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
