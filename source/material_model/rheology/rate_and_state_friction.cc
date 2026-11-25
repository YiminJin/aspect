/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/global.h>
#include <deal.II/base/parameter_handler.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace Rheology
    {
      FaultTraction
      RateAndStateFriction::resistance(const double speed, const double age, const double compression) const
      {
        // Use log(z) to avoid overflow for large reference friction / direct effect.
        AssertThrow(speed > 0. && age > 0. && compression > 0., ExcMessage("Regularized mature friction requires positive speed, age and compression."));
        const double log_z = std::log(speed) - std::log(2.*reference_speed)
                             + (reference_coefficient + evolution_effect * std::log(age*reference_speed/slip_distance))/direct_effect;
        const double small = std::exp(-std::abs(log_z));
        const double asinh_z = log_z > 0 ? log_z + std::log1p(std::hypot(1.,small)) : std::asinh(small);
        const double ratio = (log_z > 0 ? 1. : small)/std::hypot(1.,small);
        return {compression *direct_effect *asinh_z + damping*speed,
                compression *direct_effect *ratio/speed + damping
               };
      }

      double
      RateAndStateFriction::advance_age(const double speed, const double age, const double dt) const
      {
        AssertThrow(speed >= 0. && age > 0. && dt >= 0., ExcMessage("Invalid aging inputs."));
        const double x = speed*dt/slip_distance;
        return age*std::exp(-x) + dt*(x == 0. ? 1. : -std::expm1(-x)/x);
      }

      SectionRoot RateAndStateFriction::solve_strip(const double free_rate, const double mobility,
                                                    const double age, const double h, const double pressure) const
      {
        const double Dc=slip_distance, a=direct_effect, b=evolution_effect;
        AssertThrow(pressure>0. && h>0. && age>0. && mobility>=0. && damping>=pressure*h*b/Dc,
                    ExcMessage("Growing fault requires a monotone regularized section resistance at this duration."));
        const double scale=2.*reference_speed*std::exp(-(reference_coefficient+b*std::log((age+h)*reference_speed/Dc))/a);
        AssertThrow(scale>0. && std::isfinite(scale),ExcMessage("Section creep scale exceeds floating-point range."));
        if (free_rate==0.)
          {
            const long double compliance=scale/(static_cast<long double>(pressure)*a+damping*scale);
            const long double condensed=1.L/(mobility+compliance);
            AssertThrow(condensed<=std::numeric_limits<double>::max(),ExcMessage("Locked section tangent exceeds floating-point range."));
            return {0.,0.,static_cast<double>(condensed)};
          }
        const double target=std::abs(free_rate), sign=std::copysign(1.,free_rate);
        long double lo=0., hi=std::asinh(static_cast<long double>(target)/scale), y=.5L*hi;
        for (unsigned int iteration=0; iteration<100; ++iteration)
          {
            const double speed=mobility==0. ? target : double(scale*std::sinh(y));
            AssertThrow(speed>0.,ExcMessage("Section rate is below floating-point resolution."));
            const double theta=advance_age(speed,age,h);
            const double x=speed*h/Dc;
            const double kernel=x<1e-3 ? -.5+x*(1./3.+x*(-.125+x*(1./30.+x*(-1./144.+x/840.))))
                                : (std::expm1(-x)+x*std::exp(-x))/(x*x);
            const double age_derivative=h/Dc*(h*kernel-age*std::exp(-x));
            long double traction, condensed, step_scale;
            // At extreme creep speed df/dV may exceed DBL_MAX. V df/dV is
            // finite. Form both the Newton step and its Schur coefficient there.
            if (static_cast<long double>(speed)*std::numeric_limits<double>::max()<=pressure*a*2.)
              {
                const long double v=speed;
                const long double zlog=std::log(v/(2.L*reference_speed))
                                       +(reference_coefficient+b*std::log(theta*reference_speed/Dc))/static_cast<long double>(a);
                const long double small=std::exp(-std::abs(zlog));
                const long double friction=zlog>0 ? zlog+std::log1p(std::hypot(1.L,small)) : std::asinh(small);
                traction=pressure*a*friction+damping*v;
                const long double log_slope=pressure*a*(zlog>0 ? 1.L : small)/std::hypot(1.L,small)
                                            *(1.L+v*b*age_derivative/(a*theta))+v*damping;
                const long double denominator=v+mobility*log_slope;
                condensed=log_slope/denominator;
                step_scale=denominator*scale/v;
              }
            else
              {
                const auto r=resistance(speed,theta,pressure);
                traction=r.traction;
                const double slope=r.speed_derivative+(r.speed_derivative-damping)*speed*b/(a*theta)*age_derivative;
                if (mobility>1. && slope>std::numeric_limits<double>::max()/mobility)
                  {
                    condensed=1.L/(mobility+1.L/slope);
                    step_scale=(1.L+static_cast<long double>(mobility)*slope)*scale;
                  }
                else
                  {
                    condensed=slope/(1.+mobility*slope);
                    step_scale=(1.+mobility*slope)*scale;
                  }
              }
            const long double residual=static_cast<long double>(speed)+mobility*traction-target;
            if (std::abs(residual)<=32.*std::numeric_limits<double>::epsilon()*target)
              {
                AssertThrow(std::abs(condensed)<=std::numeric_limits<double>::max(),ExcMessage("Section tangent exceeds floating-point range."));
                return {sign*speed,static_cast<double>(sign*traction),static_cast<double>(condensed)};
              }
            if (residual>0.) hi=y;
            else lo=y;
            const long double trial=y-residual/(step_scale*std::cosh(y));
            y=trial>lo && trial<hi ? trial : .5L*(lo+hi);
          }
        AssertThrow(false,ExcMessage("Signed section RSF root failed its roundoff certificate."));
        return {};
      }

      SectionRoot RateStateHistory::solve_strip(const unsigned int i,const double free,const double mobility,const double h,const double pressure) const
      {
        return law.solve_strip(free,mobility,ages[i],h,pressure);
      }

      void RateStateHistory::validate_midpoint() const
      {
        AssertThrow(law.damping>0. || law.evolution_effect==0.,
                    ExcMessage("The growing fault monotonicity bound requires positive damping for nonzero state evolution."));
      }

      bool RateStateHistory::admits_duration(const double h,const double pressure) const
      {
        return law.damping>=pressure*h*law.evolution_effect/law.slip_distance;
      }

      std::vector<double> RateStateHistory::parameters() const
      {
        return {law.reference_coefficient,law.direct_effect,law.evolution_effect,law.reference_speed,law.slip_distance,law.damping};
      }

      RateStateHistory::RateStateHistory(const RateAndStateFriction &parameters,const double age,
                                         std::vector<double> supplied)
        : law(parameters),initial_age(age),ages(std::move(supplied))
      {
        AssertThrow(initial_age>0. && law.direct_effect>0. && law.reference_speed>0. && law.slip_distance>0.,
                    ExcMessage("RSF requires positive material, time and speed scales."));
      }

      void RateStateHistory::initialize(const unsigned int count)
      {
        if (ages.empty()) ages.assign(count,initial_age);
        AssertThrow(ages.size()==count && std::all_of(ages.begin(),ages.end(),[](double v)
        {
          return v>0.;
        }),
        ExcMessage("Expected one positive RSF age per material identity."));
      }

      FaultTraction RateStateHistory::at_speed(const unsigned int i,const double xi,const double speed,const double pressure) const
      {
        return law.resistance(speed,(1.-xi)*ages[i]+xi*ages[i+1],pressure);
      }

      void RateStateHistory::accept_rate(const unsigned int i,const double rate,const double dt)
      {
        ages[i]=law.advance_age(std::abs(rate),ages[i],dt);
      }

      void RateStateHistory::restore_state(std::vector<double> values)
      {
        AssertThrow(values.size()==ages.size() && std::all_of(values.begin(),values.end(),[](double v)
        {
          return v>0.;
        }),
        ExcMessage("Invalid RSF age history in checkpoint."));
        ages=std::move(values);
      }
    }
  }
}

namespace aspect::MaterialModel::Rheology
{
  namespace
  {
    void declare_rsf(ParameterHandler &prm, const FaultResistance::Route route)
    {
      const bool growing=route==FaultResistance::Route::growing;
      prm.declare_entry("Initial age",growing?"10":"1",Patterns::Double(0.),"Initial state age in seconds.");
      prm.declare_entry("Reference friction coefficient","0.6",growing?Patterns::Double():Patterns::Double(0.),"Dimensionless reference coefficient.");
      prm.declare_entry("Direct effect",growing?"0.015":"0.01",Patterns::Double(0.),"Dimensionless direct effect.");
      prm.declare_entry("Evolution effect",growing?"0.02":"0.015",Patterns::Double(0.),"Dimensionless evolution effect.");
      prm.declare_entry("Reference speed",growing?"0.001":"1e-6",Patterns::Double(0.),"Reference slip speed in m/s.");
      prm.declare_entry("Slip distance","0.01",Patterns::Double(0.),"Characteristic slip distance in m.");
      prm.declare_entry("Radiation damping",growing?"1000":"0",Patterns::Double(0.),"Radiation impedance in Pa s/m.");
    }

    std::unique_ptr<FaultResistance> read_rsf(const ParameterHandler &prm,FaultResistance::Route)
    {
      const RateAndStateFriction law= {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),
                                       prm.get_double("Evolution effect"),prm.get_double("Reference speed"),
                                       prm.get_double("Slip distance"),prm.get_double("Radiation damping")
                                      };
      return std::make_unique<RateStateHistory>(law,prm.get_double("Initial age"));
    }
    const auto registered_rsf=(FaultResistance::register_law("rate and state",declare_rsf,read_rsf),true);
  }
}
