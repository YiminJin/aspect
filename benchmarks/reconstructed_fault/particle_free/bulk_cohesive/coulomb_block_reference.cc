// Independent homogeneous integration schedule; no native FE solver or history owner.
#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace BC=aspect::MaterialModel::BulkCohesive;
int main(int argc,char **argv)
{
  if(argc!=2)return 2;
  const BC::CoulombReturn::Parameters parameters{{1e6,1e8,4.,1000.},.1,0.};
  BC::TensorReturn::State state;state.phi=.2;
  std::ofstream out(argv[1]);out<<std::setprecision(17);
  out<<"step,time_s,Txx_Pa,Txy_Pa,tauxx_Pa,tauxy_Pa,Axx,Axy,Kxx,Kxy,Qxx_Pa,Qxy_Pa,L,Wxx_s-1,Wxy_s-1,Y_Pa,bulk_J_m3,cohesive_J_m3,Db_W_m3,Dc_W_m3,Df_W_m3,active\n";
  for(unsigned int step=0;step<=24;++step)
    {
      BC::Tensor increment;
      if(step>0 && step<=8)increment[0][1]=.5*.02*.0125;
      else if(step>8 && step<=20){increment[0][0]=.02*.0125;increment[1][1]=-increment[0][0];}
      const auto result=BC::CoulombReturn::update(parameters,state,increment,.2,.0125,10000.);
      if(step>0)state=result.point.state;
      const auto &r=result.point;
      out<<step<<','<<step*.0125<<','<<state.T[0][0]<<','<<state.T[0][1]<<','<<r.stress[0][0]<<','<<r.stress[0][1]<<','
         <<state.A[0][0]<<','<<state.A[0][1]<<','<<state.K[0][0]<<','<<state.K[0][1]<<','<<state.Q[0][0]<<','<<state.Q[0][1]<<','<<state.L<<','
         <<r.flow_rate[0][0]<<','<<r.flow_rate[0][1]<<','<<r.driving<<','<<r.bulk_energy<<','<<r.cohesive_energy<<','
         <<r.bulk_dissipation<<','<<r.cohesive_dissipation<<','<<r.friction_dissipation<<','<<result.active<<'\n';
    }
  std::cout<<"24 homogeneous point intervals exported\n";
}
