#include "profile.h"
#include "geometry.h"
#include "runtime.h"
#include "bp3_model.h"
#include <aspect/phase_field.h>
#include <deal.II/base/quadrature_lib.h>
#include <functional>

namespace BP3
{
  namespace
  {
    double hermite(double t, double length, double left, double right,
                   double left_slope, double right_slope)
    {
      return (2*t*t*t-3*t*t+1)*left + (-2*t*t*t+3*t*t)*right
             + length*((t*t*t-2*t*t+t)*left_slope+(t*t*t-t*t)*right_slope);
    }
  }

  double LoadingProfile::cumulative(const double r) const
  {
    const double a=std::abs(r);
    if (a>=support) return r<0 ? 0. : 1.;
    const unsigned int i=std::upper_bound(radius.begin(),radius.end(),a)-radius.begin()-1;
    const double length=radius[i+1]-radius[i];
    const double value=hermite((a-radius[i])/length,length,integral[i],integral[i+1],slope[i],slope[i+1]);
    return .5 + (r<0 ? -.5 : .5)*value/integral.back();
  }

  template <int dim>
  const LoadingProfile &loading_profile(const aspect::SimulatorAccess<dim> &sim)
  {
    // A BP3 run has one immutable stationary profile. Preparation is local and
    // occurs on first use, after the phase-field handler has initialized. No
    // observer or external table must run before a boundary query.
    static LoadingProfile cache;
    if (!cache.radius.empty()) return cache;
    const auto &handler=sim.get_phase_field_handler();
    const auto profiles=handler.get_phase_field_profiles(geometry().peak_phase);
    AssertThrow(!profiles.empty(),dealii::ExcMessage("BP3 requires a stationary material profile."));
    const auto &coordinates=profiles[0]->get_coordinate_values();
    std::vector<double> fractions(profiles.size(),0.);
    fractions[0]=1.;
    // Completion and the single symmetric loading primitive require identical
    // material profiles/laws. Check every profile knot and interval midpoint.
    for(unsigned int j=1;j<profiles.size();++j)
      {
        AssertThrow(profiles[j]->get_coordinate_values()==coordinates
                    && profiles[j]->get_phase_field_values()==profiles[0]->get_phase_field_values(),
                    dealii::ExcMessage("BP3 requires identical stationary profiles in all materials."));
        std::vector<double> other(profiles.size(),0.);other[j]=1.;
        for(unsigned int i=0;i<coordinates.size();++i)
          for(const double x:{coordinates[i],i ? .5*(coordinates[i-1]+coordinates[i]) : 0.})
            {
              const double phi=profiles[0]->value(x);
              AssertThrow(handler.energetic_degradation(fractions,phi)==handler.energetic_degradation(other,phi),
                          dealii::ExcMessage("BP3 requires a composition-independent loading degradation law."));
            }
      }
    const auto h=[&](double x)
      {
        const double g=handler.energetic_degradation(fractions,profiles[0]->value(x));
        AssertThrow(g>0. && g<=1.,dealii::ExcMessage("Invalid BP3 loading degradation."));
        return 1./g-1.; // Same h as the live slip-rate localization numerator.
      };
    const dealii::QGauss<1> quadrature(8);
    const auto gauss=[&](double a,double b)
      {
        double result=0.;
        for(unsigned int q=0;q<quadrature.size();++q)
          result+=(b-a)*quadrature.weight(q)*h(a+(b-a)*quadrature.point(q)[0]);
        return result;
      };
    double estimate=0.;
    for(unsigned int i=1;i<coordinates.size();++i) estimate+=gauss(coordinates[i-1],coordinates[i]);
    AssertThrow(estimate>0. && std::isfinite(estimate),dealii::ExcMessage("Invalid BP3 loading normalization."));
    LoadingProfile prepared;
    prepared.support=coordinates.back();
    prepared.radius.push_back(0.);prepared.integral.push_back(0.);prepared.slope.push_back(h(0.));
    std::function<void(double,double,unsigned int)> append;
    append=[&](double a,double b,unsigned int depth)
      {
        const double middle=.5*(a+b),left=gauss(a,middle),right=gauss(middle,b),value=left+right;
        const double ha=h(a),hb=h(b);
        bool accurate=std::abs(value-gauss(a,b))<=1e-13*estimate*(b-a)/prepared.support;
        for(const double t:{.25,.5,.75})
          accurate=accurate && std::abs(hermite(t,b-a,0.,value,ha,hb)-gauss(a,a+t*(b-a)))<=2e-14*estimate;
        if (!accurate)
          {
            AssertThrow(depth<20,dealii::ExcMessage("BP3 loading primitive failed integration/interpolation accuracy."));
            append(a,middle,depth+1);append(middle,b,depth+1);
          }
        else
          {
            prepared.radius.push_back(b);prepared.integral.push_back(prepared.integral.back()+value);
            prepared.slope.push_back(hb);
          }
      };
    for(unsigned int i=1;i<coordinates.size();++i) append(coordinates[i-1],coordinates[i],0);
    cache=std::move(prepared);
    return cache;
  }
  template const LoadingProfile &loading_profile(const aspect::SimulatorAccess<2> &);
  template const LoadingProfile &loading_profile(const aspect::SimulatorAccess<3> &);
}

namespace aspect { namespace BP3Restore
{
  template <int dim>
  Tensor<1,dim> loading(const SimulatorAccess<dim> &sim,const Point<dim> &p)
  {
    const double rate=-BP3::Vp*(BP3::loading_profile(sim).cumulative(BP3::signed_normal(p[0],p[1]))-.5);
    Tensor<1,dim> u;
    u[0]=rate*BP3::geometry().tangent[0];u[1]=rate*BP3::geometry().tangent[1];
    return u;
  }
  template Tensor<1,2> loading(const SimulatorAccess<2> &,const Point<2> &);
  template Tensor<1,3> loading(const SimulatorAccess<3> &,const Point<3> &);
}}
