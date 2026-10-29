/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_interface.h>
#include <algorithm>
#include <cmath>
namespace aspect { namespace CohesiveInterface
{
  PointTrial evaluate(const Parameters &p,const PointState &old,const double increment,
                      const double pressure,const double dt,const double sign_hint)
  {
    AssertThrow(dt>0 && pressure>0 && old.theta>0 && old.damage>=0 && old.damage<=1,
                ExcMessage("Cohesive interface: invalid accepted state or lost closed contact."));
    PointTrial out;out.state=old;auto &s=out.state;
    const double c0=2*p.Gc/(p.peak*p.peak),df=2*p.Gc/p.peak,beta=1/(1+p.G/p.eta*dt);
    s.jump+=increment;s.relaxed_jump=beta*(old.relaxed_jump+increment);s.absolute_slip+=std::abs(increment);
    const double e=s.jump-old.reference;
    s.damage=std::max(old.damage,std::min(std::abs(e)/df,1.));
    const bool growing=std::abs(e)>df*old.damage && s.damage<1.;
    double dq=0.,da=0.;
    if(s.damage==0.) {out.q=p.peak*sign_hint;dq=-p.peak/df;da=sign_hint/df;}
    else if(s.damage<1.)
      {
        const double stiffness=(1-s.damage)/(c0*s.damage);out.q=stiffness*e;dq=stiffness;
        if(growing) {dq=-p.peak/df;da=std::copysign(1.,e)/df;}
        out.storage=.5*out.q*e;
      }
    out.velocity=increment/dt;const double speed=std::abs(out.velocity),sign=out.velocity==0?0:std::copysign(1.,out.velocity);
    s.theta=(old.theta+p.V0*dt/p.Dc)/(1+speed*dt/p.Dc);
    const double exponential=std::exp((p.mu0+p.evolution*std::log(s.theta))/p.direct);
    const double z=speed/(2*p.V0)*exponential,fraction=z/std::hypot(1.,z);
    const double mu=p.direct*std::asinh(z);
    const double slope=(speed>0?p.direct*fraction/speed:p.direct*exponential/(2*p.V0))
                       -p.evolution*fraction*(dt/p.Dc)/(1+speed*dt/p.Dc);
    const double R=sign*(pressure*mu+p.radiation*speed);
    out.resistance=out.q+s.damage*R;
    out.derivative=dq+da*R+s.damage*(pressure*slope+p.radiation)/dt;
    out.pressure_derivative=s.damage*sign*mu;
    out.friction=dt*s.damage*pressure*mu*speed;out.radiation=dt*s.damage*p.radiation*speed*speed;
    AssertThrow(std::isfinite(out.derivative) && std::isfinite(out.resistance),ExcMessage("Nonfinite interface return."));
    return out;
  }
}}
