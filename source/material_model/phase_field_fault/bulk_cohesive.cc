/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <deal.II/base/exceptions.h>
#include <cmath>

namespace aspect { namespace MaterialModel { namespace BulkCohesive
{
  Compliance compliance(const double phi,const double m)
  {
    AssertThrow(std::isfinite(phi) && phi>=0. && phi<1. && m>0.,
                dealii::ExcMessage("Bulk cohesive phase requires 0<=phi<1; no endpoint floor is defined."));
    const double d=1.-phi;
    return {m*phi*(1.+phi)/(d*d), m*(1.+3.*phi)/(d*d*d),
            6.*m*(1.+phi)/(d*d*d*d)};
  }

  Response update(const Parameters &p,const State &incoming,const Tensor &increment,
                  const double phi,const double dt)
  {
    AssertThrow(p.G>0. && p.eta>0. && p.m>0. && p.resistance>=0.
                && std::isfinite(dt) && dt>0. && phi>=incoming.phi,
                dealii::ExcMessage("Invalid bulk cohesive coefficients, step or physical obstacle."));
    const auto h=compliance(phi,p.m);
    Response r;
    r.state=incoming;
    r.state.phi=phi;
    r.beta=1./(1.+dt*p.G/p.eta);
    r.state.T=r.beta*(incoming.T+2.*p.G*increment);
    const double t0=r.state.T[0][1];
    const double abar=r.beta*incoming.a;
    r.intact=h.h==0.;
    if(r.intact)
      {
        AssertThrow(incoming.a==0.,dealii::ExcMessage("Intact state cannot carry finite cohesive shear."));
        // Minimum-magnitude multiplier in the intact Coulomb interval: the
        // one-sided relaxed return, including at finite dt. No 0/0 occurs.
        const double excess=std::max(0.,std::abs(t0)-p.resistance);
        r.state.a=0.;r.state.q=std::copysign(excess,t0);
        r.sliding=excess>0.;
        if(r.sliding)
          {
            r.dq_dt=1.;r.da_dphi=h.first*r.state.q/p.G;
            r.dq_dphi=-h.first*r.state.q;
          }
      }
    else
      {
        const double numerator=h.h*t0-p.G*(1.+h.h)*abar;
        r.sliding=std::abs(numerator)>p.resistance*h.h;
        if(r.sliding)
          {
            const double sign=std::copysign(1.,numerator);
            r.state.q=(t0-sign*p.resistance)/(1.+h.h);
            r.state.a=h.h*r.state.q/p.G;
            r.da_dt=h.h/(p.G*(1.+h.h));
            r.da_dphi=h.first*r.state.q/(p.G*(1.+h.h));
            r.dq_dt=1./(1.+h.h);
            r.dq_dphi=-h.first*r.state.q/(1.+h.h);
          }
        else
          {
            r.state.a=abar;r.state.q=p.G*abar/h.h;
            r.dq_dphi=-r.state.q*h.first/h.h;
          }
      }
    r.stress=r.state.T;
    r.stress[0][1]-=p.G*r.state.a;
    // In sticking the exact rate is zero, rather than the subtraction of
    // two nearly equal a values. This also preserves cyclic slip histories.
    const double dk=r.sliding ? (1.+dt*p.G/p.eta)*r.state.a-incoming.a : 0.;
    r.w=dk/dt;
    r.state.signed_shear+=dk;
    r.state.absolute_shear+=std::abs(dk);
    r.driving=h.first*r.state.q*r.state.q/(2.*p.G);
    r.driving_dt=h.first*r.state.q*r.dq_dt/p.G;
    r.driving_dphi=(h.second*r.state.q*r.state.q+2.*h.first*r.state.q*r.dq_dphi)/(2.*p.G);
    r.bulk_energy=(r.stress*r.stress)/(4.*p.G);
    r.cohesive_energy=h.h*r.state.q*r.state.q/(2.*p.G);
    r.bulk_dissipation=(r.stress*r.stress)/(2.*p.eta);
    r.cohesive_dissipation=h.h*r.state.q*r.state.q/p.eta;
    r.friction_dissipation=p.resistance*std::abs(r.w);
    AssertThrow(std::isfinite(r.driving) && std::isfinite(r.cohesive_energy)
                && std::isfinite(r.driving_dphi),dealii::ExcMessage("Nonfinite bulk cohesive response."));
    return r;
  }

  StepAccounting account(const Parameters &p,const State &incoming,const Response &r,
                         const Tensor &increment,const double dt)
  {
    const auto hn=compliance(incoming.phi,p.m),hp=compliance(r.state.phi,p.m);
    Tensor old_stress=incoming.T;
    old_stress[0][1]-=p.G*incoming.a;
    const double old_cohesive=hn.h==0. ? 0. : p.G*incoming.a*incoming.a/(2.*hn.h);
    StepAccounting a;
    a.internal_work=r.stress*increment;
    a.storage_change=r.bulk_energy+r.cohesive_energy
                     -(old_stress*old_stress)/(4.*p.G)-old_cohesive;
    a.physical_dissipation=dt*(r.bulk_dissipation+r.cohesive_dissipation+r.friction_dissipation);
    const Tensor difference=r.stress-old_stress;
    a.stress_loss=(difference*difference)/(4.*p.G);
    a.traction_loss=hn.h*std::pow(r.state.q-incoming.q,2)/(2.*p.G);
    a.compliance_work=(hp.h-hn.h)*r.state.q*r.state.q/(2.*p.G);
    a.error=a.internal_work-a.storage_change-a.physical_dissipation
            -a.stress_loss-a.traction_loss-a.compliance_work;
    return a;
  }

  ProfileMoments observe_profile(const Parameters &p,const std::vector<double> &weights,
                                 const std::vector<Response> &samples)
  {
    AssertDimension(weights.size(),samples.size());
    ProfileMoments m;
    for(unsigned int i=0;i<samples.size();++i)
      {
        AssertThrow(weights[i]>0.,dealii::ExcMessage("Bulk observer requires positive normal weights."));
        const auto &r=samples[i];const double h=compliance(r.state.phi,p.m).h;
        m.I+=weights[i]*h;m.elastic_slip+=weights[i]*r.state.a;
        m.rate+=weights[i]*r.w;m.absolute_rate+=weights[i]*std::abs(r.w);
        m.bulk_cohesive+=weights[i]*r.cohesive_energy;
        m.bulk_viscous+=weights[i]*r.cohesive_dissipation;
      }
    m.Z=p.G*m.elastic_slip;
    m.bulk_friction=p.resistance*m.absolute_rate;
    m.surface_friction=p.resistance*std::abs(m.rate);
    m.friction_excess=m.bulk_friction-m.surface_friction;
    if(m.I==0.)
      {
        AssertThrow(m.elastic_slip==0.,dealii::ExcMessage("Intact profile cannot have nonzero elastic slip."));
        return m; // No normalized observer record at I=0.
      }
    m.normalized=true;
    m.surface_cohesive=m.Z*m.Z/(2.*p.G*m.I);
    m.surface_viscous=m.Z*m.Z/(p.eta*m.I);
    for(unsigned int i=0;i<samples.size();++i)
      {
        const auto &r=samples[i];const double h=compliance(r.state.phi,p.m).h;
        const double rho=r.state.a-h/m.I*m.elastic_slip;
        m.residual_moment+=weights[i]*rho;
        if(h==0.)AssertThrow(rho==0.,dealii::ExcMessage("Finite-energy intact residual must vanish."));
        else
          {
            m.residual_cohesive+=weights[i]*p.G*rho*rho/(2.*h);
            m.residual_viscous+=weights[i]*p.G*p.G*rho*rho/(p.eta*h);
          }
      }
    return m;
  }
}}}
