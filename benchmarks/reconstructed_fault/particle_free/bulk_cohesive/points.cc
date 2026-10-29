/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#define CATCH_CONFIG_RUNNER
#include "../../../../contrib/catch/catch.hpp"
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <deal.II/base/exceptions.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace BC=aspect::MaterialModel::BulkCohesive;

int main(int argc,char **argv)
{
  dealii::deal_II_exceptions::disable_abort_on_exception();
  const int result=Catch::Session().run(argc,argv);
  if(result)return result;
  const char *file=std::getenv("ASPECT_BULK_POINT_EXPORT");
  if(!file)return 0;
  std::ofstream out(file);out<<std::setprecision(17);
  out<<"family,dt,step,time,phi,G,eta,m,resistance,dgamma,Txy,a,q,w,k,L,stress,driving,bulk,cohesive,Db,Dc,Df,ledger_error\n";
  for(unsigned int family=0;family<3;++family)
    for(const double dt:{.01,.005})
      {
        const BC::Parameters p{1e6,1e6,128.,1000.};
        BC::State state;state.phi=family==0 ? 0. : .02;
        for(unsigned int step=1;step<=std::lround(4./dt);++step)
          {
            const double time=step*dt;
            const double rate=time<=1. ? .006 : time<=2. ? 0. : time<=3. ? -.012 : .006;
            BC::Tensor increment;increment[0][1]=.5*dt*rate;
            const double phi=family==2 ? std::min(.1,.02+.04*time) : state.phi;
            const auto response=BC::update(p,state,increment,phi,dt);
            const auto ledger=BC::account(p,state,response,increment,dt);
            out<<family<<','<<dt<<','<<step<<','<<time<<','<<phi<<','<<p.G<<','<<p.eta<<','<<p.m<<','<<p.resistance<<','<<dt*rate<<','
               <<response.state.T[0][1]<<','<<response.state.a<<','<<response.state.q<<','<<response.w<<','
               <<response.state.signed_shear<<','<<response.state.absolute_shear<<','<<response.stress[0][1]<<','<<response.driving<<','
               <<response.bulk_energy<<','<<response.cohesive_energy<<','<<response.bulk_dissipation<<','<<response.cohesive_dissipation<<','<<response.friction_dissipation<<','<<ledger.error<<'\n';
            state=response.state;
          }
      }
  if(const char *profile_file=std::getenv("ASPECT_BULK_PROFILE_EXPORT"))
    {
      std::ofstream profiles(profile_file);profiles<<std::setprecision(17);
      profiles<<"family,i,weight,phi,a,q,w,I,se,Ec,Es,Er,Dc,Ds,Dr,Df,Fs,Fr,moment,normalized\n";
      const BC::Parameters p{1e6,1e8,128.,1000.};
      const std::vector<double> weights{.02,.08,.13,.2,.2,.13,.08,.02};
      for(unsigned int family=0;family<3;++family)
        {
          std::vector<BC::Response> points(weights.size());
          for(unsigned int i=0;i<points.size();++i)
            {
              auto &r=points[i];r.state.phi=family==2 || i==0 ? 0. : .05+.05*i;
              const double h=BC::compliance(r.state.phi,p.m).h;
              r.state.q=family==0 ? 100. : 40.+13.*i;r.state.a=h*r.state.q/p.G;
              r.cohesive_energy=h*r.state.q*r.state.q/(2*p.G);
              r.cohesive_dissipation=h*r.state.q*r.state.q/p.eta;
              r.w=(family==1 && i%2 ? -1. : 1.)*h*1e-5;
            }
          const auto m=BC::observe_profile(p,weights,points);
          for(unsigned int i=0;i<points.size();++i)
            {
              const auto &r=points[i];profiles<<family<<','<<i<<','<<weights[i]<<','<<r.state.phi<<','<<r.state.a<<','<<r.state.q<<','<<r.w<<','
                <<m.I<<','<<m.elastic_slip<<','<<m.bulk_cohesive<<','<<m.surface_cohesive<<','<<m.residual_cohesive<<','
                <<m.bulk_viscous<<','<<m.surface_viscous<<','<<m.residual_viscous<<','<<m.bulk_friction<<','<<m.surface_friction<<','<<m.friction_excess<<','<<m.residual_moment<<','<<m.normalized<<'\n';
            }
        }
    }
}
