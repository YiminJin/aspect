/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
#ifndef aspect_cohesive_timestep_h
#define aspect_cohesive_timestep_h
#include <aspect/reconstructed_fault/cohesive_evolution.h>

namespace aspect { namespace CohesiveEvolution
{
  /** Linearized resolution of the assembled admissible problem, not a global
   * nonlinear error/stability certificate. In the nodal arclength metric M,
   * sym(C^T J C)>=alpha I implies ||delta j||_M<=||delta F||_{M^-1}/alpha.
   * Full-space evaluation norms conservatively bound constrained point values.
   */
  struct Resolution
  {
    std::vector<double> displacement;
    double maximum=0., force_dual_norm=0.;
  };
  inline Resolution linearized_resolution(const Operators &op,const Result &r)
  {
    const unsigned n=op.Q.n(),nq=op.Q.m(),na=op.affine_modes;
    Resolution out;out.displacement.resize(nq,0.);
    if(!r.free)return out;
    AssertThrow(r.curvature>0.,ExcMessage("Nonpositive admissible resolution tangent"));
    const double u=std::numeric_limits<double>::epsilon()/2.;
    const double count=2.*(n+na)+4.*nq+3.;
    const double gamma=count*u/(1-count*u);
    double dual2=0.;
    for(unsigned i=0;i<n;++i)
      {
        AssertThrow(op.mass[i]>0.,ExcMessage("Nonpositive resolution mass"));
        double force=op.force_offset.size()?op.force_offset[i]:0.;
        double operands=std::abs(force);
        for(unsigned j=0;j<n+na;++j)
          {const double term=op.K(i+na,j)*r.coefficients[j];force+=term;operands+=std::abs(term);}
        for(unsigned q=0;q<nq;++q)
          {
            const double w=op.Q(q,i)*op.weights[q];
            force+=w*r.point[q].resistance;
            operands+=std::abs(w)*(std::abs(r.point[q].q)+std::abs(r.point[q].resistance-r.point[q].q));
          }
        // Same force resolution as the unchanged complete weak certificate.
        // The gamma term covers this sum/product evaluation of assembled data;
        // bulk assembly/mesh and nonlinear active-set errors are separate checks.
        const double envelope=std::max(std::abs(force),weak_tolerance*op.mass[i])+gamma*operands;
        dual2+=envelope*envelope/op.mass[i];
      }
    out.force_dual_norm=std::sqrt(dual2);
    for(unsigned q=0;q<nq;++q)
      {
        double evaluation2=0.;
        for(unsigned j=0;j<n;++j)evaluation2+=op.Q(q,j)*op.Q(q,j)/op.mass[j];
        out.displacement[q]=std::sqrt(evaluation2)*out.force_dual_norm/r.curvature;
        out.maximum=std::max(out.maximum,out.displacement[q]);
      }
    AssertThrow(std::isfinite(out.maximum),ExcMessage("Nonfinite algebraic displacement resolution"));
    return out;
  }
  /** Minimum separation of log-speed intervals. Overlap means unresolved,
   * not a certificate that either instantaneous rate is accurate. */
  inline double resolved_log_change(double old_rate,double new_rate,
                                    double old_uncertainty,double new_uncertainty,double V0)
  {
    const double a=std::abs(old_rate),b=std::abs(new_rate);
    const double al=std::log(std::max(0.,a-old_uncertainty)+V0),au=std::log(a+old_uncertainty+V0);
    const double bl=std::log(std::max(0.,b-new_uncertainty)+V0),bu=std::log(b+new_uncertainty+V0);
    return std::max({0.,al-bu,bl-au});
  }
  struct HistoryDefect
  {
    double slip=0.,state=0.,damage=0.,resolution=0.;
    double maximum()const{return std::max({slip,state,damage,resolution});}
  };
  /** BE/trapezoidal defects in continuous physical histories. No alternate
   * state is published. The cohesive damage map is exactly Lipschitz 1/df;
   * the log aging update is Lipschitz 1/Dc in its slip increment. */
  inline HistoryDefect history_defect(const CohesiveInterface::Parameters &p,
      const CohesiveInterface::PointState &old,const CohesiveInterface::PointState &now,
      double dt,double increment,double old_rate,double new_rate,
      double displacement_uncertainty,double old_rate_uncertainty)
  {
    HistoryDefect out;
    const double df=2*p.Gc/p.peak,U=displacement_uncertainty;
    out.resolution=std::max(U/p.Dc,U/df);
    if(out.resolution>1.)return out; // Already unresolvable in continuous state.
    const double defect=.5*std::abs(increment-dt*old_rate);
    const double uncertainty=.5*(U+dt*old_rate_uncertainty);
    out.resolution=std::max({out.resolution,uncertainty/p.Dc,(U+uncertainty)/df});
    out.slip=std::max(0.,defect-uncertainty)/p.Dc;
    const double mid_jump=old.jump+.5*(increment+dt*old_rate);
    const double mid_damage=std::max(old.damage,std::min(std::abs(mid_jump-old.reference)/df,1.));
    out.damage=std::max(0.,std::abs(now.damage-mid_damage)-(U+uncertainty)/df);
    const double f0=(p.V0-std::abs(old_rate)*old.theta)/p.Dc;
    const double f1=(p.V0-std::abs(new_rate)*now.theta)/p.Dc;
    const double raw=std::abs(now.theta-old.theta-.5*dt*(f0+f1));
    const double theta_uncertainty=now.theta*std::expm1(U/p.Dc);
    const double state_uncertainty=theta_uncertainty*(1+.5*dt*std::abs(new_rate)/p.Dc)
        +.5*dt/p.Dc*(old.theta*old_rate_uncertainty+(now.theta+theta_uncertainty)*U/dt);
    const double state_scale=(1+.5*dt*std::abs(new_rate)/p.Dc)*std::min(old.theta,now.theta);
    out.resolution=std::max(out.resolution,state_uncertainty/state_scale);
    const double u=std::numeric_limits<double>::epsilon()/2.;
    const double roundoff=20*u/(1-20*u)*(std::abs(now.theta)+std::abs(old.theta)+.5*dt*(std::abs(f0)+std::abs(f1)));
    out.state=std::max(0.,raw-state_uncertainty-roundoff)
       /state_scale;
    return out;
  }
}}
#endif
