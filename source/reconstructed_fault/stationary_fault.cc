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
                                        const double age, const double rate)
  {
    AssertThrow(dim == 2 && vertices.size() >= 2, ExcMessage("A stationary fault requires at least two ordered vertices in 2-D."));
    const auto direction = vertices.back()-vertices.front();
    const unsigned int axis = std::abs(direction[0]) > std::abs(direction[1]) ? 0 : 1;
    const unsigned int transverse = 1-axis;
    const double length = std::abs(direction[axis]);
    AssertThrow(length > 0. && direction[transverse] == 0. &&
                std::min(vertices.front()[axis],vertices.back()[axis]) == origin[axis] &&
                std::max(vertices.front()[axis],vertices.back()[axis]) == origin[axis]+extent[axis] &&
                vertices.front()[transverse] >= origin[transverse] &&
                vertices.front()[transverse] <= origin[transverse]+extent[transverse],
                ExcMessage("The stationary fault must span the Box parallel to a coordinate axis."));
    tangent = direction/length;
    normal[0] = -tangent[1];
    normal[1] = tangent[0];
    slip_gradient = outer_product(tangent,normal);
    width = extent[transverse];
    accepted.clear();
    arclength.clear();
    for (const auto &vertex : vertices)
      {
        const double s = (vertex-vertices.front())*tangent;
        AssertThrow(vertex[transverse] == vertices.front()[transverse] &&
                    (arclength.empty() || s > arclength.back()),
                    ExcMessage("Stationary fault labels must be straight and strictly ordered."));
        accepted.push_back({vertex,age,0.,rate});
        arclength.push_back(s);
      }
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
    const double s = std::clamp((point-accepted.front().position)*tangent,0.,arclength.back());
    const auto right = std::upper_bound(arclength.begin()+1,arclength.end()-1,s);
    const unsigned int j = std::distance(arclength.begin(),right)-1;
    return {j,(s-arclength[j])/(arclength[j+1]-arclength[j])};
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
