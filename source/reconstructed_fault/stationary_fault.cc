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
                                        const double age, const double rate, const double slip)
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
        accepted.push_back({vertex,age,slip,rate});
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
  std::vector<double> StationaryFault<dim>::solve_rates(const std::vector<BulkPoint> &points,
                                                        const MaterialModel::Rheology::MatureFaultFriction &law,
                                                        const double kappa, const double minimum_speed,
                                                        const MPI_Comm comm) const
  {
    const unsigned int n = accepted.size();
    std::vector<double> rates(n);
    for (unsigned int j=0; j<n; ++j)
      rates[j] = accepted[j].rate;

    // The derivative of the localized shear stress with respect to V is
    // -2 kappa (h/I) S:S = -kappa h/I for S=sym(t x n).
    const double stiffness = kappa*localization();
    for (unsigned int iteration=0; iteration<100; ++iteration)
      {
        std::vector<double> rhs(n), diagonal(n), upper(n-1), scale(n);
        for (const auto &point : points)
          {
            const unsigned int j = point.foot.left;
            const double N[2] = {1.-point.foot.xi,point.foot.xi};
            const double speed = point.foot.interpolate(rates[j],rates[j+1]);
            const double age = point.foot.interpolate(accepted[j].age,accepted[j+1].age);
            const auto resistance = law.resistance(speed,age,point.compression);
            const double residual = point.driving_shear - stiffness*speed - resistance.first;
            const double derivative = stiffness + resistance.second;
            for (unsigned int i=0; i<2; ++i)
              {
                const double test = point.weight*N[i];
                rhs[j+i] += test*residual;
                diagonal[j+i] += test*N[i]*derivative;
                scale[j+i] += test*std::max(std::abs(point.driving_shear),std::abs(resistance.first));
              }
            upper[j] += point.weight*N[0]*N[1]*derivative;
          }

        for (auto *row :
             {
               &rhs,&diagonal,&upper,&scale
             })
          Utilities::MPI::sum(*row,comm,*row);
        double residual = 0.;
        for (unsigned int j=0; j<n; ++j)
          {
            AssertThrow(scale[j] > 0.,ExcMessage("A stationary fault label has no resolved bulk support."));
            residual = std::max(residual,std::abs(rhs[j])/scale[j]);
          }
        if (residual <= 64.*std::numeric_limits<double>::epsilon())
          return rates;

        // Eliminate the symmetric positive tridiagonal Q1 tangent in place.
        for (unsigned int j=1; j<n; ++j)
          {
            AssertThrow(diagonal[j-1] > 0.,ExcMessage("Singular stationary fault tangent."));
            const double multiplier = upper[j-1]/diagonal[j-1];
            diagonal[j] -= multiplier*upper[j-1];
            rhs[j] -= multiplier*rhs[j-1];
          }
        AssertThrow(diagonal.back() > 0.,ExcMessage("Singular stationary fault tangent."));
        rhs.back() /= diagonal.back();
        for (unsigned int j=n-1; j>0; --j)
          rhs[j-1] = (rhs[j-1]-upper[j-1]*rhs[j])/diagonal[j-1];

        double fraction = 1.;
        for (unsigned int j=0; j<n; ++j)
          if (rates[j]+rhs[j] < minimum_speed)
            fraction = std::min(fraction,.9*(rates[j]-minimum_speed)/(-rhs[j]));
        AssertThrow(fraction > 0.,ExcMessage("No admissible positive sliding solution on the stationary fault."));
        for (unsigned int j=0; j<n; ++j)
          rates[j] += fraction*rhs[j];
      }
    AssertThrow(false,ExcMessage("Stationary fault Q1 balance did not converge."));
    return rates;
  }

  template <int dim>
  void StationaryFault<dim>::accept(const std::vector<double> &rates, const double dt,
                                    const MaterialModel::Rheology::MatureFaultFriction &law)
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

  template <int dim>
  void StationaryFault<dim>::save(oarchive &archive) const
  {
    archive << accepted.size() << width;
    for (const auto &label : accepted)
      archive << label.position << label.age << label.slip << label.rate;
  }

  template <int dim>
  void StationaryFault<dim>::load(iarchive &archive)
  {
    std::size_t n;
    double saved_width;
    archive >> n >> saved_width;
    AssertThrow(n == accepted.size() && saved_width == width,ExcMessage("Incompatible stationary fault geometry."));
    auto restored = accepted;
    for (auto &label : restored)
      {
        Point<dim> position;
        archive >> position >> label.age >> label.slip >> label.rate;
        AssertThrow(position == label.position && label.age > 0. && label.rate > 0. && std::isfinite(label.slip),
                    ExcMessage("Incompatible stationary fault label in checkpoint."));
      }
    accepted.swap(restored);
  }

#define INSTANTIATE(dim) template class StationaryFault<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
