/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_tip.h>
#include <deal.II/base/exceptions.h>
#include <cmath>

namespace aspect
{
  namespace MaterialModel
  {
    namespace BulkCohesive
    {
      namespace Tip
      {
        Response update(const Parameters &p, const State &old, const Tensor &increment,
                        const double phi, const double dt, const Tensor &S, const double f)
        {
          AssertThrow(p.G>0. && p.eta>0. && p.m>0. && dt>0. && phi>=0. && phi<=1.
                      && std::isfinite(f) && std::abs(S*S-.5)<1e-13,
                      dealii::ExcMessage("Invalid fixed-reference bulk tip state or direction."));
          AssertThrow((old.cohesive_strain-2.*(old.cohesive_strain*S)*S).norm()<1e-12
                      && (old.phi!=0. || old.cohesive_strain.norm()==0.),
                      dealii::ExcMessage("Bulk tip memory must remain in the prescribed Mode-II space; intact A=0."));
          const double beta=1./(1.+dt*p.G/p.eta);
          const double c=p.m*phi*(1.+phi), d=(1.-phi)*(1.-phi), denominator=c+d;
          const double cp=p.m*(1.+2.*phi), dp=-2.*(1.-phi), denominator_derivative=cp+dp;
          Response r;
          r.fraction=c/denominator;
          r.fraction_derivative=(cp*d-c*dp)/(denominator*denominator);
          r.driving_factor=p.m*(1.+3.*phi)*(1.-phi)/(denominator*denominator);
          r.driving_factor_derivative=p.m*(2.-6.*phi)/(denominator*denominator)-2.*r.driving_factor*denominator_derivative/denominator;
          const double old_e=2.*(old.cohesive_strain*S);
          const Tensor predictor=beta*(old.stress+2.*p.G*increment);
          r.force=predictor*S+p.G*beta*old_e-f;
          const double e=r.fraction*r.force/p.G, q=(d/denominator)*r.force;
          r.state={predictor-2.*p.G*(e-beta*old_e)*S, e*S, phi};
          r.Q=2.*q*S;
          r.rate=(e/beta-old_e)/dt;
          r.driving=r.driving_factor*r.force*r.force/(2.*p.G);
          r.bulk_energy=(r.state.stress*r.state.stress)/(4.*p.G);
          r.cohesive_energy=.5*e*q;
          r.bulk_viscosity=2.*p.G/p.eta*r.bulk_energy;
          r.cohesive_viscosity=2.*p.G/p.eta*r.cohesive_energy;
          return r;
        }

        State midpoint_endpoint(const State &old, const Response &stage)
        {
          return {2.*stage.state.stress-old.stress,
                  2.*stage.state.cohesive_strain-old.cohesive_strain,
                  2.*stage.state.phi-old.phi};
        }

        double cohesive_energy(const Parameters &p, const State &state)
        {
          if (state.phi==0.)
            {
              AssertThrow(state.cohesive_strain.norm()==0.,dealii::ExcMessage("Intact endpoint requires A=0."));
              return 0.;
            }
          return p.G*(state.cohesive_strain*state.cohesive_strain)
                 *(1.-state.phi)*(1.-state.phi)/(p.m*state.phi*(1.+state.phi));
        }

        StepAccounting midpoint_account(const Parameters &p, const State &old, const Response &stage,
                                        const Tensor &increment, const double dt, const double f)
        {
          const auto end=midpoint_endpoint(old,stage);
          StepAccounting a;
          const double change=cohesive_energy(p,end)-cohesive_energy(p,old);
          a.internal_work=stage.state.stress*increment;
          a.storage_change=(end.stress*end.stress-old.stress*old.stress)/(4.*p.G)+change;
          a.physical_dissipation=dt*(stage.bulk_viscosity+stage.cohesive_viscosity+f*stage.rate);
          a.compliance_work=stage.Q*(end.cohesive_strain-old.cohesive_strain)-change;
          a.error=a.internal_work-a.storage_change-a.physical_dissipation-a.compliance_work;
          return a;
        }

        Variation derivative(const Parameters &p, const Response &r, const Tensor &increment,
                             const double dphi, const double dt, const Tensor &S, const double df)
        {
          const double beta=1./(1.+dt*p.G/p.eta);
          const double dz=2.*p.G*beta*(increment*S)-df;
          const double de=(r.fraction*dz+r.fraction_derivative*r.force*dphi)/p.G;
          Variation a;
          a.cohesive_strain=de*S;
          a.stress=2.*p.G*beta*increment-2.*p.G*de*S;
          a.Q=2.*((1.-r.fraction)*dz-r.fraction_derivative*r.force*dphi)*S;
          a.rate=de/(beta*dt);
          a.driving=(r.driving_factor_derivative*r.force*r.force*dphi
                     +2.*r.driving_factor*r.force*dz)/(2.*p.G);
          return a;
        }

        StepAccounting account(const Parameters &p, const State &old, const Response &r,
                               const Tensor &increment, const double dt, const double f)
        {
          const auto inverse=[&](const double phi)
          {return (1.-phi)*(1.-phi)/(p.m*phi*(1.+phi));};
          const double rp=r.state.phi>0.?inverse(r.state.phi):0.;
          const double rn=old.phi>0.?inverse(old.phi):0.;
          const double old_cohesive=old.phi>0.?p.G*rn*(old.cohesive_strain*old.cohesive_strain):0.;
          StepAccounting a;
          a.internal_work=r.state.stress*increment;
          a.storage_change=r.bulk_energy+r.cohesive_energy
                           -(old.stress*old.stress)/(4.*p.G)-old_cohesive;
          a.physical_dissipation=dt*(r.bulk_viscosity+r.cohesive_viscosity+f*r.rate);
          const Tensor change=r.state.stress-old.stress;
          a.stress_loss=(change*change)/(4.*p.G);
          if(old.phi>0. && old.phi<1.)
            {
              const Tensor dq=r.Q-2.*p.G*rn*old.cohesive_strain;
              a.traction_loss=(dq*dq)/(4.*p.G*rn);
            }
          // Stable limits: h_old=0 gives E_coh+, h_new=infinity gives zero.
          a.compliance_work=r.cohesive_energy*(old.phi==0.?1.:(rn==0.?0.:1.-rp/rn));
          a.error=a.internal_work-a.storage_change-a.physical_dissipation
                  -a.stress_loss-a.traction_loss-a.compliance_work;
          return a;
        }
      }
    }
  }
}
