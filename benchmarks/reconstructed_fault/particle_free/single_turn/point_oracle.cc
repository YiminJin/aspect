/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_tip.h>
#include <cmath>
#include <iomanip>
#include <iostream>

// Rows: G eta m phi dt angle old_e tau_xx tau_yy tau_xy tau_zz
//       increment_xx increment_yy increment_xy increment_zz local_force.
// dt and increment are the full physical midpoint interval. Phase is fixed.
int main()
{
  namespace BT=aspect::MaterialModel::BulkCohesive;
  double G,eta,m,phi,dt,angle,e,xx,yy,xy,zz,dx,dy,dxy,dz,f;
  std::cout << std::setprecision(17);
  while (std::cin >> G >> eta >> m >> phi >> dt >> angle >> e
         >> xx >> yy >> xy >> zz >> dx >> dy >> dxy >> dz >> f)
    {
      BT::Tensor S;
      S[0][0]=-.5*std::sin(2*angle); S[1][1]=-S[0][0];
      S[0][1]=.5*std::cos(2*angle);
      BT::Tip::State old;
      old.phi=phi; old.cohesive_strain=e*S;
      old.stress[0][0]=xx; old.stress[1][1]=yy;
      old.stress[0][1]=xy; old.stress[2][2]=zz;
      BT::Tensor increment;
      increment[0][0]=dx; increment[1][1]=dy;
      increment[0][1]=dxy; increment[2][2]=dz;
      const BT::Parameters p{G,eta,m,0.};
      const auto stage=BT::Tip::update(p,old,.5*increment,phi,.5*dt,S,f);
      const auto end=BT::Tip::midpoint_endpoint(old,stage);
      const auto ledger=BT::Tip::midpoint_account(p,old,stage,increment,dt,f);
      std::cout << end.stress[0][0] << ' ' << end.stress[1][1] << ' '
                << end.stress[0][1] << ' ' << end.stress[2][2] << ' '
                << 2*(end.cohesive_strain*S) << ' ' << stage.rate << ' '
                << stage.driving << ' ' << stage.bulk_energy << ' '
                << stage.cohesive_energy << ' ' << ledger.error << '\n';
    }
  return std::cin.eof() ? 0 : 1;
}
